from datasets.loader import load_dataset
import pyarrow.parquet as pq

# UNSW + CICIDS load fully
print("UNSW:", load_dataset("unsw").shape)
print("CICIDS:", load_dataset("cicids").shape)

# BoT-IoT should NOT load fully → use metadata
pf = pq.ParquetFile("datasets/botiot_merged.parquet")

print("\nBoT-IoT Columns:", pf.schema.names)
print("Row Groups:", pf.num_row_groups)
print("Total Rows:", pf.metadata.num_rows)