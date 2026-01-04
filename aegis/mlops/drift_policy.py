from datetime import datetime
from aegis.config import DriftConfig

def apply_drift_policy(drift_df, dataset, model_version):
    alerts = []
    retrain_required = False
    max_psi = float(drift_df["psi"].max())

    for _, row in drift_df.iterrows():
        if row["psi"] > DriftConfig.PSI_LOW:
            alerts.append({
                "type": "warning" if row["psi"] <= DriftConfig.PSI_HIGH else "critical",
                "feature": row["feature"],
                "psi": row["psi"],
                "dataset": dataset,
                "model_version": model_version,
                "timestamp": datetime.utcnow().isoformat(),
                "message": (
                    "Moderate drift detected. Monitor closely."
                    if row["psi"] <= DriftConfig.PSI_HIGH
                    else "Severe drift detected. Retraining recommended."
                )
            })

        if row["psi"] > DriftConfig.PSI_HIGH:
            retrain_required = True

    return {
        "alerts": alerts,
        "retrain_required": retrain_required,
        "max_psi": max_psi,
        "model_version": model_version
    }
