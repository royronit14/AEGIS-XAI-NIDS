import pandas as pd
from aegis.drift.psi import calculate_psi


def compute_processed_feature_drift(
    X_train_processed,
    X_new_processed,
    feature_names
):
    drift_report = []

    for idx, feature in enumerate(feature_names):
        psi = float(
            calculate_psi(
                X_train_processed[:, idx],
                X_new_processed[:, idx]
            )
        )

        if psi > 0.2:
            status = "severe_drift"
        elif psi > 0.1:
            status = "moderate_drift"
        else:
            status = "no_drift"

        drift_report.append({
            "feature": feature,
            "psi": round(psi, 4),
            "status": status
        })

    return pd.DataFrame(drift_report).sort_values(
        by="psi",
        ascending=False
    )
