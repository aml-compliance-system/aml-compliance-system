
"""
Build a production sample from bucketed Block D parquet files.

Unlike a row-count LIMIT (which just grabs whatever rows a file returns first,
usually in write/physical order), this pulls a COMPLETE time window across
ALL bucket files. That matters here because:

  1. Rolling/history features (outgoing_count_24h/72h, sender_total_sent_prev_week,
     hours_since_last_incoming, hop1/hop2/hop3 graph exposure, etc.) depend on an
     account's full activity within the window. A row-count cut can slice a day
     or a week in half, silently degrading those features to defaults/NaN.
  2. "numerical_bucket_NNN" files are a partitioning scheme, not a time-ordering
     guarantee. Reading a single bucket (or the first N rows of one bucket) does
     not guarantee you get real, contiguous transaction history for the accounts
     in it. Globbing all buckets and filtering by timestamp is what fixes that.
  3. This exactly mirrors how df_val / df_test were built during training
     evaluation (full date window, no downsampling) -- so the production
     distribution matches what the model was actually validated against,
     instead of matching the artificially rebalanced df_train distribution.

Example:
    python build_production_sample.py \\
        --input-glob "/data/block_d_numerical_bucket_*.parquet" \\
        --output "/data/prod_sample_oct17_19.parquet" \\
        --start "2022-10-17" \\
        --end "2022-10-19"
"""

import argparse
import sys

import duckdb


def main():
    parser = argparse.ArgumentParser(
        description="Extract a full time-window production sample across all bucketed parquet files."
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob pattern matching ALL bucket files, e.g. '/data/block_d_numerical_bucket_*.parquet'",
    )
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--start", required=True, help="Window start, e.g. '2022-10-17'")
    parser.add_argument("--end", required=True, help="Window end (inclusive), e.g. '2022-10-19'")
    parser.add_argument(
        "--timestamp-col",
        default="Timestamp",
        help="Column used to define the time window (default: Timestamp)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional safety cap. This ABORTS with a warning if exceeded rather than "
             "silently truncating -- truncating would reintroduce the same bug this "
             "script is meant to fix. Narrow --start/--end instead if you hit this.",
    )
    args = parser.parse_args()

    con = duckdb.connect()

    # Sanity check: confirm the glob actually matches multiple bucket files.
    file_count = con.execute(
        f"SELECT COUNT(*) FROM glob('{args.input_glob}')"
    ).fetchone()[0]
    if file_count == 0:
        print(f"❌ No files matched glob pattern: {args.input_glob}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {file_count} bucket file(s) matching pattern.")

    # Row count check BEFORE writing, so we never silently truncate a window.
    count_query = f"""
        SELECT COUNT(*)
        FROM read_parquet('{args.input_glob}')
        WHERE "{args.timestamp_col}" >= '{args.start}'
          AND "{args.timestamp_col}" <= '{args.end}'
    """
    row_count = con.execute(count_query).fetchone()[0]

    if row_count == 0:
        print(f"❌ Window {args.start} -> {args.end} matched 0 rows. Check the date range / column name.",
              file=sys.stderr)
        sys.exit(1)

    if args.max_rows is not None and row_count > args.max_rows:
        print(
            f"❌ Window {args.start} -> {args.end} has {row_count:,} rows, "
            f"exceeding --max-rows={args.max_rows:,}.\n"
            f"   Refusing to truncate (that would reintroduce the LIMIT bug). "
            f"Narrow the date window instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Extracting full window {args.start} -> {args.end}: {row_count:,} rows...")

    copy_query = f"""
        COPY (
            SELECT *
            FROM read_parquet('{args.input_glob}')
            WHERE "{args.timestamp_col}" >= '{args.start}'
              AND "{args.timestamp_col}" <= '{args.end}'
        ) TO '{args.output}' (FORMAT PARQUET);
    """
    con.execute(copy_query)

    written = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{args.output}')"
    ).fetchone()[0]

    print(f"✅ Done! Wrote {written:,} rows to {args.output}")
    if written != row_count:
        print(f"⚠️  Warning: written row count ({written:,}) != pre-check count ({row_count:,}).",
              file=sys.stderr)


if __name__ == "__main__":
    main()