import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aegis.data.loaders.loader import load_dataset
import pyarrow.parquet as pq

# Base folder for processed data
BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", 
    "aegis", 
    "data", 
    "processed"
)
BASE = os.path.abspath(BASE)

def test_unsw_and_cicids():
    print("UNSW:", load_dataset("unsw").shape)
    print("CICIDS:", load_dataset("cicids").shape)

def test_botiot_metadata():
    file = os.path.join(BASE, "botiot_merged.parquet")

    if not os.path.exists(file):
        raise FileNotFoundError(f"BoT-IoT merged parquet not found: {file}")

    pf = pq.ParquetFile(file)

    print("\nBoT-IoT Columns:", pf.schema.names)
    print("Row Groups:", pf.num_row_groups)
    
    total_rows = sum(
        pf.metadata.row_group(i).num_rows
        for i in range(pf.num_row_groups)
    )

    print("Total Rows:", total_rows)

if __name__ == "__main__":
    print("=== TESTING DATA LOADERS ===\n")
    test_unsw_and_cicids()
    test_botiot_metadata()
