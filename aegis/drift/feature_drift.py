import pandas as pd
from datetime import datetime

from aegis.drift.psi import calculate_psi
from aegis.mlops.mongo_client import drift_col
from aegis.mlops.schemas import drift_schema
from aegis.config import DriftConfig


def inject_synthetic_drift(X, factor=1.4, cols=3):
    X = X.copy()
    X.iloc[:, :cols] = X.iloc[:, :cols] * factor
    return X


def compute_feature_drift(
    X_train,
    X_new,
    dataset="unknown",
    simulate=False
):
    drift_report = []
    severe_features = []

    if simulate:
        X_new = inject_synthetic_drift(X_new)

    for col in X_train.columns:
        psi = float(calculate_psi(X_train[col], X_new[col]))

        # ✅ Correct status assignment
        if psi > DriftConfig.PSI_HIGH:
            status = "severe_drift"
        elif psi > DriftConfig.PSI_LOW:
            status = "moderate_drift"
        else:
            status = "no_drift"

        if status == "severe_drift":
            severe_features.append(col)

        drift_report.append({
            "feature": col,
            "psi": round(psi, 4),
            "status": status,
            "baseline_rows": len(X_train),
            "current_rows": len(X_new)
        })

    df = pd.DataFrame(drift_report).sort_values(by="psi", ascending=False)

    # 🔗 MongoDB snapshot (ONE per run)
    payload = drift_schema({
        "timestamp": datetime.utcnow(),
        "dataset": dataset,
        "psi": float(df["psi"].max()),
        "drift_level": (
            "high" if df["psi"].max() > DriftConfig.PSI_HIGH
            else "medium" if df["psi"].max() > DriftConfig.PSI_LOW
            else "low"
        ),
        "features": severe_features
    })

    drift_col.insert_one(payload)

    return df
