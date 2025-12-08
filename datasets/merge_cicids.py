import pandas as pd
import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "CICIDS2017")
OUTPUT_FILE = os.path.join(BASE_DIR, "cicids_merged.parquet")

def merge_cicids():
    # Get all CSV files in CICIDS2017 folder
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    print(f"Found {len(files)} CICIDS files.")
    dfs = []

    for f in files:
        print(f"Reading: {os.path.basename(f)}")
        df = pd.read_csv(f)
        dfs.append(df)

    print("Concatenating all files...")
    final_df = pd.concat(dfs, ignore_index=True)

    print(f"Saving merged file to {OUTPUT_FILE}...")
    final_df.to_parquet(OUTPUT_FILE)

    print("DONE! CICIDS merged successfully.")

if __name__ == "__main__":
    merge_cicids()
