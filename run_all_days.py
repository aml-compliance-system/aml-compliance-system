
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb


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


def get_all_dates():
    con = duckdb.connect()

    dates_df = con.execute(f"""
        SELECT DISTINCT
            CAST("Timestamp" AS DATE) AS run_date
        FROM read_parquet('{FINAL_GLOB}')
        ORDER BY run_date
    """).df()

    con.close()

    return [
        str(d)[:10]
        for d in dates_df["run_date"]
    ]


def is_completed(run_date):
    output_dir = OUTPUT_ROOT / run_date

    summary_path = output_dir / "summary.json"
    risk_path = output_dir / "risk_scores.parquet"
    sar_path = output_dir / "sar_reports.jsonl"

    if not (
        summary_path.exists()
        and risk_path.exists()
        and sar_path.exists()
    ):
        return False

    try:
        with open(
            summary_path,
            "r",
            encoding="utf-8"
        ) as f:
            summary = json.load(f)

        return (
            summary.get("graph_date")
            == run_date
        )

    except Exception:
        return False


def delete_temp(run_date):
    temp_file = (
        TEMP_ROOT /
        f"final_{run_date}.parquet"
    )

    if temp_file.exists():
        temp_file.unlink()

        print(
            f"🧹 Deleted temporary file: "
            f"{temp_file}"
        )


def main():
    dates = get_all_dates()

    print(
        f"FINDETECT full run: "
        f"{len(dates)} day(s)"
    )

    print(
        f"First date: {dates[0]}"
    )

    print(
        f"Last date : {dates[-1]}"
    )

    print("=" * 60)

    completed_before = sum(
        is_completed(d)
        for d in dates
    )

    print(
        f"Already completed: "
        f"{completed_before}/{len(dates)}"
    )

    print("=" * 60)

    for i, run_date in enumerate(
        dates,
        start=1
    ):

        print(
            "\n\n"
            + "=" * 60
        )

        print(
            f"DAY {i}/{len(dates)} "
            f"| {run_date}"
        )

        print(
            "=" * 60
        )

        if is_completed(run_date):
            print(
                f"⏭️ {run_date} already "
                "completed — skipping."
            )

            delete_temp(run_date)

            continue

        # Remove stale temporary input
        # from a previous interrupted run.
        delete_temp(run_date)

        result = subprocess.run(
            [
                sys.executable,
                "run_full_pipeline.py",
                "--date",
                run_date,
            ],
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:

            print(
                "\n❌ FULL RUN STOPPED"
            )

            print(
                f"Failed date: {run_date}"
            )

            print(
                "Fix the error, then run "
                "run_all_days.py again."
            )

            print(
                "Completed days will be "
                "skipped automatically."
            )

            sys.exit(
                result.returncode
            )

        if not is_completed(run_date):
            raise RuntimeError(
                f"{run_date} finished but "
                "expected output files "
                "were not found."
            )

        delete_temp(run_date)

        completed_now = sum(
            is_completed(d)
            for d in dates
        )

        print(
            f"\n✅ Progress: "
            f"{completed_now}/"
            f"{len(dates)} days complete"
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "🎉 ALL FINAL DATA DAYS COMPLETED"
    )

    print(
        f"Total days: {len(dates)}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
