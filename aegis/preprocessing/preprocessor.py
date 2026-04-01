import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self, schema: dict):
        self.schema = schema
        self.feature_map = schema["features"]

        self.categorical_features = [
            f for f, t in self.feature_map.items()
            if t == "category"
        ]

        self.numeric_features = [
            f for f, t in self.feature_map.items()
            if t in ("int", "float")
        ]

        self.pipeline = None
        self.feature_names_ = None

    def fit(self, df: pd.DataFrame):
        """
        Fit preprocessing ONLY on training data.
        """
        X = df[self.categorical_features + self.numeric_features]

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ("cat", categorical_pipeline, self.categorical_features),
                ("num", numeric_pipeline, self.numeric_features)
            ]
        )

        X_transformed = self.pipeline.fit_transform(X)

        # Save feature names for explainability later
        cat_features = (
            self.pipeline.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(self.categorical_features)
        )

        self.feature_names_ = list(cat_features) + self.numeric_features

        return self

    def transform(self, df: pd.DataFrame):
        """
        Transform data using already-fitted pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Preprocessor not fitted yet")

        X = df[self.categorical_features + self.numeric_features]
        X_transformed = self.pipeline.transform(X)

        return X_transformed

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
