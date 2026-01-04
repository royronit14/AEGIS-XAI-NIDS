from datetime import datetime
import uuid

class VersionRegistry:
    def __init__(self):
        self.current = {
            "model_version": None,
            "dataset_version": None,
            "run_id": None,
            "timestamp": None
        }

    def register(
        self,
        model_version: str,
        dataset_version: str
    ):
        self.current = {
            "model_version": model_version,
            "dataset_version": dataset_version,
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }
        return self.current

    def get(self):
        return self.current
