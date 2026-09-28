import os
import gc
import json
import argparse
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from LangGraph import app

REPO_ROOT = Path(__file__).resolve().parent

FINAL_DATA_DIR = Path(
    os.environ.get(
        "FINDETECT_FINAL_DATA_DIR",
        REPO_ROOT / "Data" / "processed"
    )
).expanduser()

FINAL_GLOB = (FINAL_DATA_DIR / "block_d_numerical_bucket_*.parquet").as_posix()

AUTOENCODER_MODEL_DIR = Path(
    os.environ.get(
        "AUTOENCODER_MODEL_DIR",
        REPO_ROOT / "models" / "autoencoder_final64"
    )
).expanduser().resolve()

OUTPUT_ROOT = Path(
    os.environ.get(
        "FINDETECT_OUTPUT_DIR",
        REPO_ROOT / "outputs" / "FINDETECT_Pipeline_Output"
    )
).expanduser().resolve()

TEMP_ROOT = Path(
    os.environ.get(
        "FINDETECT_TEMP_DIR",
        REPO_ROOT / ".tmp" / "findetect_daily"
    )
).expanduser().resolve()


def build_daily_parquet(run_date, input_path=None) -> Path:
    """
    Build the pipeline input from either a single parquet file (input_path)
    or all 64 Final buckets. If run_date is given, keep only that day.
    """
    if input_path:
        src_file = Path(input_path).expanduser().resolve()
        if not src_file.is_file():
            raise FileNotFoundError(f"Input file not found: {src_file}")
        source = src_file.as_posix()
        label = run_date or src_file.stem
        print(f"✅ Using single input file: {src_file}")
    else:
        parquet_files = sorted(
            FINAL_DATA_DIR.glob("block_d_numerical_bucket_*.parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(
                "\nFINDETECT Final dataset was not found.\n"
                f"Expected directory: {FINAL_DATA_DIR}\n\n"
                "Set the FINDETECT_FINAL_DATA_DIR environment variable, "
                "or use --input path/to/file.parquet"
            )
        source = FINAL_GLOB
        label = run_date
        print(f"✅ Final dataset directory: {FINAL_DATA_DIR}")
        print(f"✅ Final parquet files found: {len(parquet_files)}")

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    daily_path = TEMP_ROOT / f"final_{label}.parquet"
    if daily_path.exists():
        daily_path.unlink()

    print(f"\n--- PREPARING INPUT: {label} ---")

    con = duckdb.connect()
    con.execute("PRAGMA preserve_insertion_order=false")

    # Only add derived columns that the file does not already have
    existing = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{source}')"
        ).fetchall()
    }

    derived = {
        "hour_of_day":
            """CAST(date_part('hour', CAST("Timestamp" AS TIMESTAMP)) AS FLOAT)""",
        "currency_mismatch":
            """CAST("Receiving Currency" != "Payment Currency" AS FLOAT)""",
        "same_bank":
            """CAST("From Bank" = "To Bank" AS FLOAT)""",
        "payment_format_hash":
            """CAST(hash("Payment Format") % 10000000 AS FLOAT)""",
    }

    extra = ",\n".join(
        f"{expr} AS {name}"
        for name, expr in derived.items()
        if name not in existing
    )
    select_extra = f",\n{extra}" if extra else ""

    where = (
        f"""WHERE CAST("Timestamp" AS DATE) = DATE '{run_date}'"""
        if run_date else ""
    )

    con.execute(f"""
        COPY (
            SELECT *{select_extra}
            FROM read_parquet('{source}')
            {where}
        ) TO '{daily_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    stats = con.execute(f"""
        SELECT
            COUNT(*) AS rows,
            MIN("Timestamp") AS first_timestamp,
            MAX("Timestamp") AS last_timestamp
        FROM read_parquet('{daily_path.as_posix()}')
    """).df()

    con.close()

    print(stats.to_string(index=False))

    if stats.loc[0, "rows"] == 0:
        raise ValueError(f"No transactions found for {label}")

    print(f"✅ Input parquet created: {daily_path}")
    return daily_path


def save_outputs(run_label: str, final_state: dict):
    output_dir = OUTPUT_ROOT / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Account/day risk results
    # -----------------------------------------
    risk_scores = final_state.get("risk_scores")

    if isinstance(risk_scores, pd.DataFrame):
        risk_path = output_dir / "risk_scores.parquet"
        risk_scores.to_parquet(risk_path, index=False)
        print(f"✅ Risk scores saved: {risk_path}")

    # -----------------------------------------
    # SAR reports
    # -----------------------------------------
    reports = final_state.get("sar_reports", [])
    reports_path = output_dir / "sar_reports.jsonl"

    with open(reports_path, "w", encoding="utf-8") as f:
        for i, report in enumerate(reports, start=1):
            record = {
                "report_id": i,
                "graph_date": run_label,
                "report": report,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ SAR reports saved: {reports_path}")

    # -----------------------------------------
    # Small execution summary
    # -----------------------------------------
    summary = {
        "graph_date": run_label,
        "sar_report_count": len(reports),
    }

    if isinstance(risk_scores, pd.DataFrame):
        summary["account_day_rows"] = len(risk_scores)

        if "investigation_flag" in risk_scores.columns:
            summary["investigation_cases"] = int(
                risk_scores["investigation_flag"].sum()
            )

        if "is_high_risk_seed" in risk_scores.columns:
            summary["high_risk_seeds"] = int(
                risk_scores["is_high_risk_seed"].sum()
            )

        if "received_from_high_risk" in risk_scores.columns:
            summary["direct_exposed_receivers"] = int(
                risk_scores["received_from_high_risk"].sum()
            )

    summary_path = output_dir / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        default=None,
        help="Date in YYYY-MM-DD format (optional when --input is used)"
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Path to a single parquet file to use instead of the 64 buckets"
    )

    args = parser.parse_args()

    if not args.date and not args.input:
        parser.error(
            "Provide --date YYYY-MM-DD and/or --input path/to/file.parquet"
        )

    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            parser.error(
                f"--date must be a real date like 2022-10-24, got: {args.date}"
            )

    run_date = args.date
    run_label = args.date or Path(args.input).stem

    # Validate the frozen Final64 Autoencoder package.
    required_autoencoder_files = [
        "autoencoder_final64_fixed.keras",
        "preprocessing_and_scoring_final64.joblib",
        "metadata_final64.json",
    ]

    missing_autoencoder_files = [
        filename
        for filename in required_autoencoder_files
        if not (AUTOENCODER_MODEL_DIR / filename).is_file()
    ]

    if missing_autoencoder_files:
        raise FileNotFoundError(
            "\nAutoencoder Final64 package is incomplete.\n"
            f"Model directory: {AUTOENCODER_MODEL_DIR}\n"
            f"Missing files: {missing_autoencoder_files}\n"
        )

    # Pass the resolved path to the anomaly node.
    os.environ["AUTOENCODER_MODEL_DIR"] = str(AUTOENCODER_MODEL_DIR)

    print(f"✅ Autoencoder model directory: {AUTOENCODER_MODEL_DIR}")

    daily_path = build_daily_parquet(run_date, args.input)

    print("\n🚀 Starting FINDETECT pipeline...")

    initial_state = {"final_parquet_path": str(daily_path)}

    final_state = app.invoke(initial_state)

    print(f"\n✅ PIPELINE FINISHED FOR {run_label}")

    save_outputs(run_label, final_state)

    del final_state
    gc.collect()


if __name__ == "__main__":
    main()