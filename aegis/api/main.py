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

    return explanation


@app.get("/drift/features")
def drift_features():
    drift_df = compute_feature_drift(X_train, X_test)
    return drift_df.head(10).to_dict(orient="records")
