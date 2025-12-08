import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def load_cicids():
    file = os.path.join(BASE, "cicids_merged.parquet")
    df = pd.read_parquet(file)

    # Normalize columns first
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Find the correct label column
    possible_label_cols = ["label", "attack", "attack_cat", "class", "tag"]

    found = None
    for col in possible_label_cols:
        if col in df.columns:
            found = col
            break

    if found is None:
        raise ValueError(f"Could not find a label column. Found columns: {df.columns}")

    print(f"CICIDS label column detected: {found}")

    # Create unified label
    df["label"] = df[found].apply(
        lambda x: 0
        if str(x).lower() in ["benign", "normal", "0"]
        else 1
    )

    df["attack_cat"] = df[found].apply(
        lambda x: str(x)
        if str(x).lower() not in ["benign", "normal", "0"]
        else "Normal"
    )

    print("CICIDS loaded. Shape:", df.shape)
    return df
