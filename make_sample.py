import duckdb

input_path = r"C:\Users\Analy\Desktop\Data bootcamp\project-dev\Data\processed\block_d_numerical_bucket_000.parquet"
output_path = r"C:\Users\Analy\Desktop\Data bootcamp\project-dev\Data\processed\sample_1000k.parquet"

print("Extracting 1000,000 rows...")
query = f"""
    COPY (
        SELECT * FROM read_parquet('{input_path}') LIMIT 1000000
    ) TO '{output_path}' (FORMAT PARQUET);
"""

duckdb.sql(query)
print("✅ Done! Created sample_1000k.parquet")