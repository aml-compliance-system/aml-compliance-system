
import os
import gc
import json
import argparse
from pathlib import Path

import duckdb
import pandas as pd

from LangGraph import app


REPO_ROOT = Path(__file__).resolve().parent

FINAL_DATA_DIR = Path(
    os.environ.get(
        "FINDETECT_FINAL_DATA_DIR",
        REPO_ROOT
        / "Data"
        / "processed"
        / "FINDETECT_Block_D_Numerical_Final"
    )
).expanduser()

FINAL_GLOB = str(
    FINAL_DATA_DIR
    / "block_d_numerical_bucket_*.parquet"
)

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


def build_daily_parquet(run_date: str) -> Path:
    """
    Read one complete calendar day across all 64 Final buckets
    and create the same input structure currently expected
    by the pipeline.
    """

    parquet_files = sorted(
        FINAL_DATA_DIR.glob(
            "block_d_numerical_bucket_*.parquet"
        )
    )

    if not parquet_files:
        raise FileNotFoundError(
            "\nFINDETECT Final dataset was not found.\n"
            f"Expected directory: {FINAL_DATA_DIR}\n\n"
            "Set the FINDETECT_FINAL_DATA_DIR "
            "environment variable to the directory "
            "containing the 64 Block D parquet files."
        )

    print(
        f"✅ Final dataset directory: "
        f"{FINAL_DATA_DIR}"
    )

    print(
        f"✅ Final parquet files found: "
        f"{len(parquet_files)}"
    )

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    daily_path = TEMP_ROOT / f"final_{run_date}.parquet"

    if daily_path.exists():
        daily_path.unlink()

    print(f"\n--- PREPARING COMPLETE DAY: {run_date} ---")

    con = duckdb.connect()

    con.execute("PRAGMA preserve_insertion_order=false")

    query = f"""
    COPY (
        SELECT
            *,

            CAST(
                date_part(
                    'hour',
                    CAST("Timestamp" AS TIMESTAMP)
                ) AS FLOAT
            ) AS hour_of_day,

            CAST(
                "Receiving Currency" != "Payment Currency"
                AS FLOAT
            ) AS currency_mismatch,

            CAST(
                "From Bank" = "To Bank"
                AS FLOAT
            ) AS same_bank,

            CAST(
                hash("Payment Format") % 10000000
                AS FLOAT
            ) AS payment_format_hash

        FROM read_parquet('{FINAL_GLOB}')

        WHERE CAST("Timestamp" AS DATE)
              = DATE '{run_date}'
    )
    TO '{daily_path}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """

    con.execute(query)

    stats = con.execute(f"""
        SELECT
            COUNT(*) AS rows,
            MIN("Timestamp") AS first_timestamp,
            MAX("Timestamp") AS last_timestamp
        FROM read_parquet('{daily_path}')
    """).df()

    con.close()

    print(stats.to_string(index=False))

    if stats.loc[0, "rows"] == 0:
        raise ValueError(
            f"No transactions found for {run_date}"
        )

    print(
        f"✅ Daily parquet created: {daily_path}"
    )

    return daily_path


def save_outputs(run_date: str, final_state: dict):
    output_dir = OUTPUT_ROOT / run_date
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # Account/day risk results
    # -----------------------------------------
    risk_scores = final_state.get("risk_scores")

    if isinstance(risk_scores, pd.DataFrame):
        risk_path = (
            output_dir /
            "risk_scores.parquet"
        )

        risk_scores.to_parquet(
            risk_path,
            index=False
        )

        print(
            f"✅ Risk scores saved: {risk_path}"
        )

    # -----------------------------------------
    # SAR reports
    # -----------------------------------------
    reports = final_state.get(
        "sar_reports",
        []
    )

    reports_path = (
        output_dir /
        "sar_reports.jsonl"
    )

    with open(
        reports_path,
        "w",
        encoding="utf-8"
    ) as f:

        for i, report in enumerate(
            reports,
            start=1
        ):
            record = {
                "report_id": i,
                "graph_date": run_date,
                "report": report,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"✅ SAR reports saved: "
        f"{reports_path}"
    )

    # -----------------------------------------
    # Small execution summary
    # -----------------------------------------
    summary = {
        "graph_date": run_date,
        "sar_report_count": len(reports),
    }

    if isinstance(risk_scores, pd.DataFrame):

        summary[
            "account_day_rows"
        ] = len(risk_scores)

        if "investigation_flag" in risk_scores.columns:
            summary[
                "investigation_cases"
            ] = int(
                risk_scores[
                    "investigation_flag"
                ].sum()
            )

        if "is_high_risk_seed" in risk_scores.columns:
            summary[
                "high_risk_seeds"
            ] = int(
                risk_scores[
                    "is_high_risk_seed"
                ].sum()
            )

        if "received_from_high_risk" in risk_scores.columns:
            summary[
                "direct_exposed_receivers"
            ] = int(
                risk_scores[
                    "received_from_high_risk"
                ].sum()
            )

    summary_path = (
        output_dir /
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n=== DAILY SUMMARY ===")

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Date in YYYY-MM-DD format"
    )

    args = parser.parse_args()

    run_date = args.date

    # Validate the frozen Final64 Autoencoder package.
    required_autoencoder_files = [
        "autoencoder_final64.keras",
        "preprocessing_and_scoring_final64.joblib",
        "metadata_final64.json",
    ]

    missing_autoencoder_files = [
        filename
        for filename in required_autoencoder_files
        if not (
            AUTOENCODER_MODEL_DIR / filename
        ).is_file()
    ]

    if missing_autoencoder_files:
        raise FileNotFoundError(
            "\nAutoencoder Final64 package is incomplete.\n"
            f"Model directory: {AUTOENCODER_MODEL_DIR}\n"
            f"Missing files: {missing_autoencoder_files}\n"
        )

    # Pass the resolved path to the unchanged anomaly node.
    os.environ[
        "AUTOENCODER_MODEL_DIR"
    ] = str(AUTOENCODER_MODEL_DIR)

    print(
        f"✅ Autoencoder model directory: "
        f"{AUTOENCODER_MODEL_DIR}"
    )

    daily_path = build_daily_parquet(
        run_date
    )

    print(
        "\n🚀 Starting unchanged "
        "FINDETECT pipeline..."
    )

    initial_state = {
        "final_parquet_path":
            str(daily_path)
    }

    final_state = app.invoke(
        initial_state
    )

    print(
        "\n✅ PIPELINE FINISHED "
        f"FOR {run_date}"
    )

    save_outputs(
        run_date,
        final_state
    )

    del final_state
    gc.collect()


if __name__ == "__main__":
    main()
