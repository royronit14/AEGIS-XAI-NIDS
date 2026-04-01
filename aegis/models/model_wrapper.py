import joblib
import numpy as np

class ModelWrapper:
    def __init__(self, model):
        self.model = model

        # Feature names the model was trained on
        self.expected_features = list(model.feature_names_in_)

    def _align(self, df):
        """
        Align incoming dataframe EXACTLY to training-time features.
        """
        return df[self.expected_features]

    def predict(self, df):
        X = self._align(df)
        return self.model.predict(X)

    def predict_proba(self, df):
        X = self._align(df)
        return self.model.predict_proba(X)

    def get_confidence(self, df):
        probs = self.predict_proba(df)
        return probs.max(axis=1)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
