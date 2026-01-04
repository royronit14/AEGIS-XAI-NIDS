# # aegis/tests/test_xai_sanity.py

# import joblib
# import pandas as pd
# from sklearn.model_selection import train_test_split

# from aegis.explainability.engine import XAIEngine
# from aegis.explainability.local import LocalExplainer
# from aegis.explainability.global_xai import GlobalExplainer
# from aegis.explainability.serializer import ExplanationSerializer
# from aegis.data.normalizer import normalize_dataset


# # -----------------------------
# # CONFIG
# # -----------------------------
# MODEL_PATH = "aegis/models/cicids_rf_model.joblib"
# DATASET_NAME = "cicids"
# MODEL_NAME = "random_forest"
# PARQUET_PATH = "aegis/data/processed/cicids_merged.parquet"

# # -----------------------------
# # LOAD DATA
# # -----------------------------
# df = pd.read_parquet(PARQUET_PATH)

# df = normalize_dataset(df, "cicids")

# X = df.drop(columns=["__label__"])
# y = df["__label__"]


# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = joblib.load(MODEL_PATH)

# print("✅ CICIDS data & model loaded")

# # -----------------------------
# # STEP 1: ENGINE
# # -----------------------------
# xai = XAIEngine(
#     model_path=MODEL_PATH,
#     model_type="rf"
# )

# xai.build_explainer(X_train.sample(100, random_state=42))
# print("✅ XAI Engine initialized")

# # -----------------------------
# # STEP 2: LOCAL EXPLAINABILITY
# # -----------------------------
# local_xai = LocalExplainer(
#     xai_engine=xai,
#     feature_names=X_test.columns
# )

# idx = 0
# X_instance = X_test.iloc[[idx]]

# prediction = model.predict(X_instance)[0]
# confidence = model.predict_proba(X_instance)[0].max()

# local_result = local_xai.explain(
#     X_instance=X_instance,
#     prediction=prediction,
#     confidence=confidence,
#     alert_id="SANITY-CICIDS-001"
# )

# print("✅ Local explanation generated")

# # -----------------------------
# # STEP 3: GLOBAL EXPLAINABILITY
# # -----------------------------
# global_xai = GlobalExplainer(
#     xai_engine=xai,
#     feature_names=X_test.columns
# )

# global_result = global_xai.build_policy_report(
#     X_sample=X_test.sample(500, random_state=42),
#     dataset_name="CICIDS",
#     model_name="RandomForest"
# )

# print("✅ Global explanation generated")

# # -----------------------------
# # STEP 4: SERIALIZATION
# # -----------------------------
# serializer = ExplanationSerializer()

# local_path = serializer.save(
#     explanation=local_result,
#     dataset=DATASET_NAME,
#     scope="local",
#     model_name=MODEL_NAME,
#     identifier="SANITY"
# )

# global_path = serializer.save(
#     explanation=global_result,
#     dataset=DATASET_NAME,
#     scope="global",
#     model_name=MODEL_NAME,
#     identifier="v1"
# )

# print("✅ Explanations saved")
# print("Local explanation path:", local_path)
# print("Global explanation path:", global_path)

# print("\n🎯 SANITY TEST PASSED — XAI SYSTEM IS WORKING")



# # To run the program "python -m aegis.tests.test_xai_sanity"







from aegis.mlops.mongo_client import drift_col
from datetime import datetime

drift_col.insert_one({
    "timestamp": datetime.utcnow(),
    "dataset": "UNSW",
    "psi": 0.42,
    "drift_level": "medium",
    "features": ["src_bytes", "dst_bytes"]
})

print("Inserted")
