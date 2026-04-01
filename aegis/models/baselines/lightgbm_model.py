from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
import json
from datetime import datetime


class LightGBMBaseline:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
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

    def save_metrics(self, dataset_name, metrics,
                     path="aegis/metrics/baseline_metrics.json"):
        record = {
            "dataset": dataset_name,
            "model": "LightGBM_v1",
            "timestamp": datetime.utcnow().isoformat(),
            "f1": metrics["f1"]
        }

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []

        data.append(record)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
