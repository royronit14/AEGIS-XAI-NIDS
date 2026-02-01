# import yaml
# import pandas as pd

# class BaseLoader:
#     def __init__(self, schema_path: str):
#         with open(schema_path, "r") as f:
#             self.schema = yaml.safe_load(f)

#         self.features = self.schema["features"]
#         self.label_name = self.schema["label"]["name"]

#     def enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
#         # Ensure all schema features exist
#         for feature, dtype in self.features.items():
#             if feature not in df.columns:
#                 df[feature] = "unknown" if dtype == "category" else 0

#         # Ensure label exists
#         if self.label_name not in df.columns:
#             raise ValueError("Label column missing")

#         allowed_cols = list(self.features.keys()) + [self.label_name]
#         return df[allowed_cols]


import yaml
import pandas as pd

class BaseLoader:
    def __init__(self, schema_path: str):
        with open(schema_path, "r") as f:
            self.schema = yaml.safe_load(f)

        self.features = self.schema["features"]
        self.label_name = self.schema["label"]["name"]

    def enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure all schema features exist with correct type
        for feature, dtype in self.features.items():
            if feature not in df.columns:
                df[feature] = "unknown" if dtype == "category" else 0
            else:
                if dtype != "category":
                    # Force numeric coercion
                    df[feature] = pd.to_numeric(df[feature], errors="coerce")

        # Ensure label exists and is numeric
        if self.label_name not in df.columns:
            raise ValueError("Label column missing")

        df[self.label_name] = pd.to_numeric(
            df[self.label_name], errors="coerce"
        )

        allowed_cols = list(self.features.keys()) + [self.label_name]
        return df[allowed_cols]
