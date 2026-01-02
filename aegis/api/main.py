# aegis/api/main.py

# uvicorn aegis.api.main:app --reload   (To run it babua)


from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware

from aegis.api.schemas import ExplainRequest, ExplainResponse
from aegis.data.normalizer import normalize_dataset
from aegis.explainability.engine import XAIEngine
from aegis.explainability.local import LocalExplainer
from aegis.drift.feature_drift import compute_feature_drift
from aegis.mlops.drift_policy import apply_drift_policy
from aegis.mlops.logger import log_event
from aegis.mlops.alert_manager import AlertManager
from aegis.mlops.model_health import ModelHealth
from aegis.mlops.versioning import VersionRegistry


app = FastAPI(title="AEGIS Explainability API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# LOAD DATA & MODEL (CICIDS demo)
# -----------------------------
DATASET_NAME = "cicids"
PARQUET_PATH = "aegis/data/processed/cicids_merged.parquet"
MODEL_PATH = "aegis/models/cicids_rf_model.joblib"

df = pd.read_parquet(PARQUET_PATH)
df = normalize_dataset(df, DATASET_NAME)

X = df.drop(columns=["__label__"])
y = df["__label__"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load(MODEL_PATH)

# -----------------------------
# XAI ENGINE
# -----------------------------
xai_engine = XAIEngine(MODEL_PATH, model_type="rf")
xai_engine.build_explainer(X_train.sample(100, random_state=42))

local_xai = LocalExplainer(xai_engine, feature_names=X.columns)

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/explain/local", response_model=ExplainResponse)
def explain_local(req: ExplainRequest):
    X_instance = X_test.iloc[[req.index]]

    prediction = str(model.predict(X_instance)[0])
    confidence = float(model.predict_proba(X_instance)[0].max())

    explanation = local_xai.explain(
        X_instance=X_instance,
        prediction=prediction,
        confidence=confidence,
        alert_id=f"{req.dataset.upper()}-{req.index}"
    )

    log_event(
        event_type="explanation",
        payload={
            "dataset": req.dataset,
            "alert_id": explanation["alert_id"],
            "prediction": explanation["prediction"],
            "confidence": explanation["confidence"]
        }
    )

    explanation["metadata"] = version_registry.get()

    return explanation


alert_manager = AlertManager()  # Created a Global Manager
@app.get("/alerts")
def list_alerts(state: str | None = None):
    return alert_manager.list_alerts(state)
@app.post("/alerts/{alert_id}/state")
def update_alert_state(alert_id: str, new_state: str):
    return alert_manager.update_state(alert_id, new_state)


model_health = ModelHealth()  #Created a Global Instance Boi
@app.get("/model/health")   # Added a Model Health Endpoint
def get_model_health():
    return model_health.__dict__


version_registry = VersionRegistry()   #Creating Registry
version_registry.register(
    model_version="rf_v1",
    dataset_version="cicids_v1_normalized"
)


@app.get("/drift/features")
def drift_features():
    drift_df = compute_feature_drift(X_train, X_test)

    policy_result = apply_drift_policy(
        drift_df,
        dataset=DATASET_NAME,
        model_version="rf_v1"
    )

    log_event(
        event_type="drift_check",
        payload={
            "dataset": DATASET_NAME,
            "alerts": policy_result["alerts"],
            "retrain_required": policy_result["retrain_required"]
        }
    )

    created_alerts = []

    for alert in policy_result["alerts"]:
        created = alert_manager.create_alert(alert)
        created_alerts.append(created)

    health_status = model_health.evaluate(policy_result)
    
    return {
        "metadata": version_registry.get(),
        "drift_metrics": drift_df.head(10).to_dict(orient="records"),
        "alerts": created_alerts,
        "retrain_required": policy_result["retrain_required"],
        "model_health": health_status
    }