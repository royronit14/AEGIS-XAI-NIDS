from datetime import datetime
import uuid
from aegis.mlops.mongo_client import alert_col
from aegis.config import AlertConfig

class AlertManager:
    def __init__(self):
        # In-memory store (later DB)
        self.alerts = {}

    def create_alert(self, alert_payload):
        alert_id = str(uuid.uuid4())

        alert = {
            "alert_id": alert_id,
            "state": "open",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            **alert_payload
        }

        alert_col.insert_one(alert)
        self.alerts[alert_id] = alert
        return alert

    def update_state(self, alert_id, new_state):
        if alert_id not in self.alerts:
            raise ValueError("Alert not found")

        self.alerts[alert_id]["state"] = new_state
        self.alerts[alert_id]["updated_at"] = datetime.utcnow().isoformat()
        return self.alerts[alert_id]

    def list_alerts(self, state=None):
        if state:
            return [
                a for a in self.alerts.values()
                if a["state"] == state
            ]
        return list(self.alerts.values())
    
    def compute_alert_index(self, drift_level, confidence):
        drift_score = {"low": 0.2, "medium": 0.6, "high": 1.0}[drift_level]
        confidence_score = 1 - confidence

        return round(
            (AlertConfig.DRIFT_WEIGHT * drift_score)
            + (AlertConfig.CONF_WEIGHT * confidence_score),
            3
        )