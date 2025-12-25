# aegis/drift/test.py

import pandas as pd
from sklearn.model_selection import train_test_split

from aegis.data.normalizer import normalize_dataset
from aegis.drift.feature_drift import compute_feature_drift

# -----------------------------
# CONFIG
# -----------------------------
PARQUET_PATH = "aegis/data/processed/cicids_merged.parquet"
DATASET_NAME = "cicids"

# -----------------------------
# LOAD & NORMALIZE DATA
# -----------------------------
df = pd.read_parquet(PARQUET_PATH)
df = normalize_dataset(df, DATASET_NAME)

print("✅ Data loaded & normalized")

# -----------------------------
# SPLIT FEATURES / LABEL
# -----------------------------
X = df.drop(columns=["__label__"])
y = df["__label__"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# DRIFT COMPUTATION
# -----------------------------
drift_df = compute_feature_drift(
    X_train=X_train,
    X_new=X_test
)

print("\n✅ Drift computation completed successfully")
print("\nTop drifting features:")
print(drift_df.head(10).to_string(index=False))
