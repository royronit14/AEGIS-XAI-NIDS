import yaml
import pandas as pd

class SchemaValidator:
    def __init__(self, schema_path: str):
        with open(schema_path, "r") as f:
            self.schema = yaml.safe_load(f)

        self.feature_map = self.schema["features"]
        self.feature_names = list(self.feature_map.keys())

        self.label_name = self.schema["label"]["name"]

    def validate_dataframe(self, df, require_label=False):
        df = df.copy()

        missing = [
            f for f in self.feature_names
            if f not in df.columns
        ]

        # Allow missing OPTIONAL categorical features (e.g., protocol)
        optional_features = [
            f for f, t in self.feature_map.items()
            if t == "category"
        ]

        hard_missing = [
            f for f in missing
            if f not in optional_features
        ]

        if hard_missing:
            raise ValueError(
                f"Schema validation failed. Missing features: {hard_missing}"
            )

        # Fill optional missing categorical features with placeholder
        for f in optional_features:
            if f not in df.columns:
                df[f] = "unknown"

        if require_label and self.label_name not in df.columns:
            raise ValueError(
                f"Label column `{self.label_name}` missing"
            )

        ordered_cols = self.feature_names.copy()
        if require_label:
            ordered_cols.append(self.label_name)

        return df[ordered_cols]


    def get_feature_names(self):
        return self.feature_names
