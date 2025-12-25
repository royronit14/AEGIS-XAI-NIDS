# aegis/explainability/serializer.py

import json
import os
from datetime import datetime

class ExplanationSerializer:
    def __init__(self, base_path="aegis/explainability/artifacts"):
        self.base_path = base_path

    def save(
        self,
        explanation,
        dataset,
        scope,
        model_name,
        identifier=None
    ):
        """
        scope: 'local' or 'global'
        identifier: alert_id (for local) or version tag (for global)
        """

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self.base_path, dataset, scope)
        os.makedirs(folder, exist_ok=True)

        filename_parts = [model_name, timestamp]
        if identifier:
            filename_parts.append(str(identifier))

        filename = "_".join(filename_parts) + ".json"
        path = os.path.join(folder, filename)

        explanation["_meta"] = {
            "dataset": dataset,
            "model": model_name,
            "scope": scope,
            "created_at": timestamp
        }

        with open(path, "w") as f:
            json.dump(explanation, f, indent=2)

        return path
