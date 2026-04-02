from datetime import datetime
from aegis.config import DriftConfig
from aegis.mlops.mongo_client import health_col


class ModelHealth:
    def __init__(self):
        self.last_status = "unknown"

    def evaluate(self, policy_result):
        drift_level = policy_result.get("drift_level", "low")
        max_psi = policy_result.get("max_psi", 0)
        severe_ratio = policy_result.get("severe_ratio", 0)

        if drift_level == "high" or severe_ratio > 0.3:
            status = "critical"
            action = "retrain_required"
            reason = "widespread_severe_drift"

        elif drift_level == "medium":
            status = "degrading"
            action = "monitor_closely"
            reason = "moderate_distribution_shift"

        else:
            status = "healthy"
            action = "none"
            reason = "within_safe_limits"

        payload = {
            "timestamp": datetime.utcnow(),
            "model": policy_result.get("model_version", "unknown"),
            "status": status,
            "action": action,
            "reason": reason,
            "max_psi": max_psi,
            "drift_level": drift_level,
            "severe_ratio": round(severe_ratio, 3),
        }

        health_col.insert_one(payload)
        self.last_status = payload
        return payload
