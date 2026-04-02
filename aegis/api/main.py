# aegis/api/main.py
# python -m uvicorn aegis.api.main:app --reload
# python -m http.server 5500

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# INTERNAL IMPORTS
# -----------------------------
from aegis.api.schemas import ExplainRequest, ExplainResponse
from aegis.api.history import router as history_router
from aegis.api.middleware import global_exception_handler
from aegis.api.ratelimit import limiter

from aegis.schema.schema_validator import SchemaValidator
from aegis.preprocessing.preprocessor import Preprocessor
from aegis.models.model_wrapper import ModelWrapper

from aegis.explainability.engine import XAIEngine
from aegis.explainability.local import LocalExplainer
from aegis.explainability.context import enrich_explanation_with_drift

from aegis.drift.feature_drift import compute_feature_drift
from aegis.drift.processed_drift import compute_processed_feature_drift
from aegis.drift.drift_summary import summarize_drift

from aegis.mlops.drift_policy import apply_drift_policy
from aegis.mlops.alert_manager import AlertManager
from aegis.mlops.model_health import ModelHealth
from aegis.mlops.versioning import VersionRegistry
from aegis.mlops.logger import log_event
from aegis.mlops.mongo_client import drift_col, health_col, alert_col

from aegis.utils.serialize import safe_serialize
from aegis.data.normalizer import normalize_dataset


# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="AEGIS Explainability API")

app.include_router(history_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(
    RateLimitExceeded,
    lambda r, e: JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded"}
    ),
)

# -----------------------------
# CONFIG
# -----------------------------
DATASET_NAME = "cicids"
PARQUET_PATH = "aegis/data/processed/cicids_merged.parquet"
MODEL_PATH = "aegis/models/cicids_rf_model.joblib"
SCHEMA_PATH = "aegis/schema/universal_schema_v1.yaml"


# -----------------------------
# LOAD & NORMALIZE DATA
# -----------------------------
df = pd.read_parquet(PARQUET_PATH)
# Phase-1 NOTE:
# normalize_dataset() modifies column names (strip),
# which breaks alignment with legacy-trained model.
# It will be re-enabled in Phase-2 retraining.

# df = normalize_dataset(df, DATASET_NAME)

# Manually create internal label ONLY
df["__label__"] = df["Label"]
df.drop(columns=["Label"], inplace=True)

# -----------------------------
# SCHEMA VALIDATION (STEP 1)
# -----------------------------
schema_validator = SchemaValidator(SCHEMA_PATH)

# Phase-1 NOTE:
# SchemaValidator is NOT enforcing feature selection at runtime.
# It is kept for design-time validation & Phase-2 retraining.
# We only ensure label exists via normalize_dataset.

# df = schema_validator.validate_dataframe(df, require_label=True)

# # =============================
# # PREPROCESSOR (TRAINING-ALIGNED, SAFE)
# # =============================
# # NOTE:
# # Preprocessor is NOT used for model inference in Phase-1.
# # Model inference uses training-time feature alignment (ModelWrapper).

# preprocessor = Preprocessor(schema_validator.schema)

# # IMPORTANT:
# # We do NOT restrict to schema features for Phase-1 demo
# # We fit on full dataframe so feature space matches trained model
# preprocessor.fit(df)

# feature_names = preprocessor.feature_names_

# -----------------------------
# TRAIN / TEST SPLIT (RAW DF)
# -----------------------------
X = df.drop(columns=["__label__"])
y = df["__label__"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# MODEL WRAPPER (STEP 3)
# -----------------------------
raw_model = joblib.load(MODEL_PATH)

model_wrapper = ModelWrapper(model=raw_model)


# -----------------------------
# XAI ENGINE (USES PROCESSED SPACE)
# -----------------------------
xai_engine = XAIEngine(MODEL_PATH, model_type="rf")
xai_engine.build_explainer(
    X_train.sample(100, random_state=42)
)

local_xai = LocalExplainer(
    xai_engine,
    feature_names=list(raw_model.feature_names_in_)
)

# -----------------------------
# MLOPS GLOBALS
# -----------------------------
LAST_DRIFT_DF = None

alert_manager = AlertManager()
model_health = ModelHealth()

version_registry = VersionRegistry()
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
    # Raw instance (for model wrapper)
# Raw → Processed
    X_instance_raw = X_test.iloc[[req.index]]
    X_instance_processed = X_instance_raw

    prediction = str(model_wrapper.predict(X_instance_raw)[0])
    confidence = float(model_wrapper.get_confidence(X_instance_raw)[0])

    explanation = local_xai.explain(
        X_instance=X_instance_processed,
        prediction=prediction,
        confidence=confidence,
        alert_id=f"{req.dataset.upper()}-{req.index}",
    )


    log_event(
        event_type="explanation",
        payload={
            "dataset": req.dataset,
            "alert_id": explanation["alert_id"],
            "prediction": explanation["prediction"],
            "confidence": explanation["confidence"],
        },
    )

    explanation["metadata"] = version_registry.get()
    global LAST_DRIFT_DF
    explanation = enrich_explanation_with_drift(
        explanation,
        LAST_DRIFT_DF
    )

    return explanation



@app.get("/alerts")
@limiter.limit("30/minute")
def list_alerts(request: Request, state: str | None = None):
    return alert_manager.list_alerts(state)


@app.post("/alerts/{alert_id}/state")
def update_alert_state(alert_id: str, new_state: str):
    return alert_manager.update_state(alert_id, new_state)


@app.get("/model/health")
def get_model_health():
    return model_health.__dict__


@app.get("/drift/features")
@limiter.limit("10/minute")
def drift_features(request: Request, simulate: bool = False):
    # X_train_p = preprocessor.transform(X_train)
    # X_test_p = preprocessor.transform(X_test)

    # drift_df = compute_processed_feature_drift(
    #     X_train_p,
    #     X_test_p,
    #     feature_names
    # ) 
    #
    # Phase-1 NOTE:
# Processed-space drift is disabled.
# Will be enabled in Phase-2 after schema-aligned retraining.

   
    # global LAST_DRIFT_DF
    # LAST_DRIFT_DF = drift_df

    # policy_result = apply_drift_policy(
    #     drift_df,
    #     dataset=DATASET_NAME,
    #     model_version="rf_v1",
    # )

    # created_alerts = [
    #     alert_manager.create_alert(a)
    #     for a in policy_result["alerts"]
    # ]

    # policy_result["model_version"] = "rf_v1"
    # model_health_status = model_health.evaluate(policy_result)

    # drift_summary = summarize_drift(drift_df)
    # drift_level = drift_summary["drift_level"]
    
    # policy_result["drift_level"] = drift_level
    # policy_result["severe_ratio"] = drift_summary["severe_ratio"]
    # policy_result["max_psi"] = float(drift_df["psi"].max())


    # avg_confidence = float(
    #     model_wrapper.get_confidence(X_test[:50]).mean()
    # )

    # alert_index = alert_manager.compute_alert_index(
    #     drift_level,
    #     avg_confidence
    # ) + round(drift_summary["severe_ratio"], 3)


    return safe_serialize({
        "status": "drift disabled in Phase-1",
        "metadata": version_registry.get()
        # "alert_index": min(alert_index, 1.0),
    #     "drift_level": drift_level,
    #     "drift_summary": drift_summary,
    #     "drift_metrics": drift_df.head(10).to_dict(orient="records"),
    #     "alerts": created_alerts,
    #     "model_health": model_health_status,
    })


@app.get("/history/drift")
def drift_history(limit: int = 50):
    return list(
        drift_col.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )


@app.get("/history/performance")
def performance_history(limit: int = 50):
    return list(
        health_col.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )


@app.get("/history/retraining")
def retraining_history(limit: int = 50):
    return list(
        alert_col.find(
            {"state": "retrain_scheduled"},
            {"_id": 0},
        )
        .sort("created_at", -1)
        .limit(limit)
    )
