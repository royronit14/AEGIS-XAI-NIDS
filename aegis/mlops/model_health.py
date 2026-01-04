from datetime import datetime
from aegis.mlops.mongo_client import health_col
from aegis.config import DriftConfig

class ModelHealth:
    def __init__(self):
        self.last_status = "unknown"

    def evaluate(self, policy_result):
        max_psi = policy_result.get("max_psi", 0)

        if max_psi > DriftConfig.PSI_HIGH:
            status = "stale"
            action = "schedule_retraining"
            reason = "psi_threshold_breached"
        else:
            status = "healthy"
            action = "none"
            reason = "within_threshold"

        payload = {
            "timestamp": datetime.utcnow(),
            "model": policy_result.get("model_version", "unknown"),
            "status": status,
            "action": action,
            "reason": reason,
            "max_psi": max_psi
        }

        health_col.insert_one(payload)
        self.last_status = payload
        return payload
