import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def load_unsw():
    folder = os.path.join(BASE, "UNSW-NB15")

    # Your dataset ONLY has this file with real rows
    data_file = os.path.join(folder, "UNSW_NB15_training.csv")

    print("Loading UNSW-NB15 dataset...")
    df = pd.read_csv(data_file)

    # Normalize labels
    df["label"] = df["label"].astype(int)

    # Normalize attack_cat (replace '?' with normal)
    df["attack_cat"] = (
        df["attack_cat"]
        .replace("?", "Normal")
        .fillna("Normal")
    )

    # Clean column formatting
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    print("UNSW loaded. Shape:", df.shape)
    return df
