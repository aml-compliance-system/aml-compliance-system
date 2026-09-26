import argparse
import duckdb

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--rows", type=int, default=1_000_000)
args = parser.parse_args()

print(f"Extracting {args.rows:,} rows...")
duckdb.sql(f"""
    COPY (SELECT * FROM read_parquet('{args.input}') LIMIT {args.rows})
    TO '{args.output}' (FORMAT PARQUET);
""")
print("Done!")