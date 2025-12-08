import pandas as pd
import glob
import os
import pyarrow.dataset as ds
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "botiot")
PARTS_DIR = os.path.join(BASE_DIR, "botiot_parts")
FINAL_FILE = os.path.join(BASE_DIR, "botiot_merged.parquet")

def merge_botiot():
    # Ensure folder exists for temporary parts
    os.makedirs(PARTS_DIR, exist_ok=True)

    # Load fixed column names
    names_file = os.path.join(RAW_DIR, "data_names_fixed.csv")
    col_names = pd.read_csv(names_file, header=None)[0].tolist()

    print("Loaded", len(col_names), "column names.")

    # Find all data files
    csv_files = sorted([
        f for f in glob.glob(os.path.join(RAW_DIR, "data_*.csv"))
        if "names" not in f.lower()
    ], key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

    print("Found", len(csv_files), "files.")

    part_count = 0

    for file in csv_files:
        print("Reading:", os.path.basename(file))

        # Read in chunks
        for chunk in pd.read_csv(file, names=col_names, chunksize=100000, low_memory=False):
            chunk = chunk.astype(str)  # Normalize dtypes

            part_count += 1
            part_path = os.path.join(PARTS_DIR, f"part_{part_count}.parquet")
            chunk.to_parquet(part_path)
            print(" → wrote", part_path)

    print("\nCombining all parts into final parquet...")

    # Load all parts & write final file
    dataset = ds.dataset(PARTS_DIR, format="parquet")
    table = dataset.to_table()
    pq.write_table(table, FINAL_FILE)

    print("MERGE COMPLETE →", FINAL_FILE)
    print("Total parts written:", part_count)
  
if __name__ == "__main__":
    merge_botiot()
