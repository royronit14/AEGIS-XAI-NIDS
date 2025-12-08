import pyarrow.parquet as pq

file = "datasets/botiot_merged.parquet"

pf = pq.ParquetFile(file)

print("Columns in BoT-IoT:")
print(pf.schema.names)

print("\nNumber of Row Groups:", pf.num_row_groups)

total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
print("Total Rows:", total_rows)
