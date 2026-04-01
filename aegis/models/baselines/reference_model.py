import json
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score


class ReferenceModel:
    """
    Reference (baseline) model.
    - Fixed preprocessing
    - Fixed model
    - Fixed evaluation
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            n_jobs=-1,
            random_state=42
        )

    def fit(self, df):
        X = df.drop(columns=["label"])
        y = df["label"]

        self.preprocessor.fit(df)
        X_t = self.preprocessor.transform(df)

        self.model.fit(X_t, y)
        return self

    def evaluate(self, df):
        X = df.drop(columns=["label"])
        y = df["label"]

        X_t = self.preprocessor.transform(df)
        preds = self.model.predict(X_t)

        f1 = f1_score(y, preds)
        report = classification_report(y, preds, output_dict=True)

        return {
            "f1": f1,
            "report": report
        }

    def save_metrics(
        self,
        dataset_name,
        metrics,
        path="aegis/metrics/baseline_metrics.json"
    ):
        # Resolve path relative to PROJECT ROOT (not notebook CWD)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )

        full_path = os.path.join(project_root, path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        record = {
            "dataset": dataset_name,
            "model": "RandomForest_v1",
            "timestamp": datetime.utcnow().isoformat(),
            "f1": metrics["f1"]
        }

        try:
            with open(full_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []

        data.append(record)

        with open(full_path, "w") as f:
            json.dump(data, f, indent=2)
