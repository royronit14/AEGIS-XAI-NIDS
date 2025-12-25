# aegis/explainability/engine.py

import shap
import numpy as np
import joblib

class XAIEngine:
    def __init__(self, model_path, model_type):
        """
        model_type: 'lr' or 'rf'
        """
        self.model = joblib.load(model_path)
        self.model_type = model_type
        self.explainer = None

    def build_explainer(self, X_background):
        """
        X_background: small representative sample (e.g. 100 rows)
        """
        if self.model_type == "lr":
            self.explainer = shap.LinearExplainer(
                self.model,
                X_background,
                feature_dependence="independent"
            )

        elif self.model_type == "rf":
            self.explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation="tree_path_dependent"
            )

        else:
            raise ValueError("Unsupported model type")

    def explain_instance(self, X_instance):
        """
        X_instance: single row (DataFrame or ndarray)
        """
        shap_values = self.explainer.shap_values(X_instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # attack class

        return shap_values[0]

    def explain_batch(self, X_batch):
        """
        Batch explanations (used for global analysis)
        """
        shap_values = self.explainer.shap_values(X_batch)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values
