# aegis/drift/feature_drift.py

import pandas as pd
from aegis.drift.psi import calculate_psi


def compute_feature_drift(X_train, X_new, threshold=0.25):
    drift_report = []

    for col in X_train.columns:
        psi = calculate_psi(X_train[col], X_new[col])

        drift_report.append({
            "feature": col,
            "psi": round(float(psi), 4),
            "status": (
                "severe_drift" if psi > threshold
                else "moderate_drift" if psi > 0.1
                else "no_drift"
            )
        })

    return pd.DataFrame(drift_report).sort_values(
        by="psi", ascending=False
    )
