import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import pyarrow.parquet as pq

from aegis.data.preprocess.preprocess import preprocess_dataframe


#   BOt-IoT SPECIAL HANDLING: CHUNKED SAMPLING (RAM-SAFE)
def load_botiot_sampled(file, frac=0.001):
    """
    Loads BoT-IoT using chunked parquet reading.
    Sampling happens inside each chunk so the full dataset
    is NEVER loaded into RAM.
    """

    print(f"[+] Chunk-loading BoT-IoT with frac={frac} ...")

    parquet_file = pq.ParquetFile(file)
    sampled_chunks = []

    rows_per_chunk = 200_000  # safe for 8GB RAM

    for batch in parquet_file.iter_batches(batch_size=rows_per_chunk):
        df_chunk = batch.to_pandas()

        # Sample inside the chunk
        chunk_sample = df_chunk.sample(frac=frac, random_state=42)
        sampled_chunks.append(chunk_sample)

        print(f"    - Sampled chunk: {chunk_sample.shape}")

    # Combine all tiny samples
    result = pd.concat(sampled_chunks, ignore_index=True)
    print(f"[+] Final BoT-IoT sampled shape: {result.shape}")

    return result

def load_dataset(name: str):
    base_path = Path(__file__).resolve().parent.parent / "aegis" / "data"


    if name == "botiot":
        file = base_path / "processed" / "botiot_merged.parquet"
        if not file.exists():
            raise FileNotFoundError(f"BoT-IoT parquet not found:\n{file}")

        return load_botiot_sampled(file, frac=0.1)


    if name == "unsw":
        file = base_path / "raw" / "UNSW-NB15" / "UNSW_NB15_training.csv"
        if not file.exists():
            raise FileNotFoundError(f"UNSW dataset not found:\n{file}")
        print(f"[+] Loading UNSW CSV: {file}")
        return pd.read_csv(file)


    else:
        file = base_path / "processed" / f"{name}_merged.parquet"
        if not file.exists():
            raise FileNotFoundError(f"Processed parquet not found:\n{file}")
        print(f"[+] Loading parquet: {file}")
        return pd.read_parquet(file)


def main():
    # OPTIONS: "cicids", "botiot", "unsw"
    DATASET = "unsw"

    print(f"\n🔥 Running preprocessing for: {DATASET.upper()}")

    df = load_dataset(DATASET)
    print(f"[+] Loaded dataset shape: {df.shape}")

    # PREPROCESS ENGINE
    X_train, X_test, y_train, y_test = preprocess_dataframe(df)

    print("\n✨ PREPROCESSING COMPLETE ✨")
    print("---------------------------------")
    print(f"X_train → {X_train.shape}")
    print(f"X_test  → {X_test.shape}")
    print(f"y_train → {y_train.shape}")
    print(f"y_test  → {y_test.shape}")
    print("---------------------------------")


if __name__ == "__main__":
    main()
