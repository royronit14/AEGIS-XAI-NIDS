# aegis/explainability/global_xai.py

import numpy as np

class GlobalExplainer:
    def __init__(self, xai_engine, feature_names):
        self.engine = xai_engine
        self.feature_names = feature_names

    def compute_global_importance(self, X_sample, top_k=10):
        shap_values = self.engine.explain_batch(X_sample)

        # 🔧 Ensure numpy array
        shap_values = np.array(shap_values)

        # 🔧 Handle multi-class / multi-dim SHAP
        if shap_values.ndim > 2:
            shap_values = shap_values.mean(axis=0)

        # Mean absolute impact per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        mean_abs_shap = mean_abs_shap.reshape(-1)

        feature_scores = list(zip(self.feature_names, mean_abs_shap))
        feature_scores.sort(key=lambda x: float(x[1]), reverse=True)

        return feature_scores[:top_k]

    def build_policy_report(self, X_sample, dataset_name, model_name):
        top_features = self.compute_global_importance(X_sample)

        report = {
            "dataset": dataset_name,
            "model": model_name,
            "top_global_features": [
                {
                    "feature": f,
                    "mean_shap_impact": round(float(v), 6)
                }
                for f, v in top_features
            ],
            "interpretation": self._interpret(top_features)
        }

        return report

    def _interpret(self, top_features):
        features = [f for f, _ in top_features[:5]]
        return (
            "Model decisions are primarily influenced by traffic volume, "
            "connection behavior, and destination activity features such as "
            + ", ".join(features)
        )
