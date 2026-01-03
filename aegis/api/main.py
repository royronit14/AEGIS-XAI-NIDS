# aegis/api/main.py

# python -m uvicorn aegis.api.main:app --reload   (To run it babua)
# python -m http.server 5500

from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
from fastapi import Request
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request


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
from aegis.mlops.mongo_client import drift_col, health_col, alert_col
from aegis.api.history import router as history_router
from aegis.utils.serialize import safe_serialize
from aegis.api.middleware import global_exception_handler
from aegis.api.ratelimit import limiter

def serialize(doc):
    doc["_id"] = str(doc["_id"])
    return doc

app = FastAPI(title="AEGIS Explainability API")

app.include_router(history_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(Exception, global_exception_handler)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, lambda r,e: JSONResponse(
    status_code=429,
    content={"error": "Rate limit exceeded"}
))

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

version_registry = VersionRegistry()   #Creating Registry
version_registry.register(
    model_version="rf_v1",
    dataset_version="cicids_v1_normalized"
)
# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/explain/local", response_model=ExplainResponse)
@limiter.limit("20/minute")
def explain_local(request: Request, req: ExplainRequest):
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
@limiter.limit("30/minute")
def list_alerts(request: Request, state: str | None = None):
    return alert_manager.list_alerts(state)
@app.post("/alerts/{alert_id}/state")
def update_alert_state(alert_id: str, new_state: str):
    return alert_manager.update_state(alert_id, new_state)


model_health = ModelHealth()  #Created a Global Instance Boi
@app.get("/model/health")   # Added a Model Health Endpoint
def get_model_health():
    return model_health.__dict__

    
@app.get("/drift/features")
@limiter.limit("10/minute")
def drift_features(request: Request, simulate: bool = False):
    drift_df = compute_feature_drift(
        X_train, X_test, dataset=DATASET_NAME, simulate=simulate
    )

    policy_result = apply_drift_policy(
        drift_df,
        dataset=DATASET_NAME,
        model_version="rf_v1"
    )

    # ---- Alerts ----
    created_alerts = []
    for alert in policy_result["alerts"]:
        created = alert_manager.create_alert(alert)
        created_alerts.append(created)

    # ---- Model Health (persisted) ----
    policy_result["model_version"] = "rf_v1"
    model_health_status = model_health.evaluate(policy_result)

    # ---- Alert Index ----
    drift_level = (
        "high" if policy_result["retrain_required"]
        else "medium" if len(created_alerts) > 0
        else "low"
    )

    avg_confidence = float(model.predict_proba(X_test[:50]).max(axis=1).mean())
    alert_index = alert_manager.compute_alert_index(
        drift_level, avg_confidence
    )

    return safe_serialize({
        "metadata": version_registry.get(),
        "alert_index": alert_index,
        "drift_level": drift_level,
        "drift_metrics": drift_df.head(10).to_dict(orient="records"),
        "alerts": created_alerts,
        "model_health": model_health_status
    })
    

@app.get("/history/drift")
@limiter.limit("60/minute")
def drift_history(request: Request, limit: int = 50):
    docs = (
        drift_col
        .find({})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return [serialize(d) for d in docs]

@app.get("/history/performance")
@limiter.limit("60/minute")
def performance_history(request: Request, limit: int = 50):
    docs = (
        health_col
        .find({})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return [serialize(d) for d in docs]

@app.get("/history/retraining")
@limiter.limit("60/minute")
def retraining_history(request: Request, limit: int = 50):
    docs = (
        alert_col
        .find({"state": "retrain_scheduled"})
        .sort("created_at", -1)
        .limit(limit)
    )
    return [serialize(d) for d in docs]