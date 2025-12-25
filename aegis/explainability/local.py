# aegis/explainability/local.py

import numpy as np

class LocalExplainer:
    def __init__(self, xai_engine, feature_names):
        self.engine = xai_engine
        self.feature_names = feature_names

    def explain(
        self,
        X_instance,
        prediction,
        confidence,
        top_k=5,
        alert_id=None
    ):
        shap_values = self.engine.explain_instance(X_instance)

        # 🔧 Ensure 1D array (critical fix)
        shap_values = np.array(shap_values).reshape(-1)

        feature_impacts = list(zip(self.feature_names, shap_values))
        feature_impacts.sort(key=lambda x: abs(float(x[1])), reverse=True)

        top_features = feature_impacts[:top_k]

        explanation = {
            "alert_id": alert_id,
            "prediction": str(prediction),
            "confidence": round(float(confidence), 4),
            "top_contributors": [
                {
                    "feature": f,
                    "impact": round(float(v), 6),
                    "direction": "increase_risk" if v > 0 else "decrease_risk"
                }
                for f, v in top_features
            ]
        }

        explanation["summary"] = self._generate_summary(top_features)

        return explanation

    def _generate_summary(self, top_features):
        positives = [f for f, v in top_features if v > 0]
        negatives = [f for f, v in top_features if v < 0]

        parts = []
        if positives:
            parts.append(
                f"High influence from {', '.join(positives[:2])} increased attack likelihood"
            )
        if negatives:
            parts.append(
                f"Features like {', '.join(negatives[:2])} reduced risk score"
            )

        return ". ".join(parts) + "."
