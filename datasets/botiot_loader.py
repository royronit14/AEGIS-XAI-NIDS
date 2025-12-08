import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def load_botiot():
    file = os.path.join(BASE, "botiot_merged.parquet")
    df = pd.read_parquet(file)

    # Clean columns
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Possible label column names for BoT-IoT
    possible_cols = [
        "label", "attack", "type", "target",
        "category", "class", "status"
    ]

    found = None
    for col in possible_cols:
        if col in df.columns:
            found = col
            break

    # If not found, detect binary-like columns
    if found is None:
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            # Check if column is basically binary
            if set(unique_vals).issubset({0, 1}):
                found = col
                print(f"Auto-detected binary label column: {found}")
                break

    if found is None:
        raise ValueError("No label-like column found in BoT-IoT dataset.")

    print(f"BoT-IoT label column detected: {found}")

    # Build final labels
    df["label"] = df[found].apply(
        lambda x: 0 if str(x).lower() in ["normal", "0"] else 1
    )

    df["attack_cat"] = df["label"].apply(
        lambda x: "Normal" if x == 0 else "Attack"
    )

    return df
