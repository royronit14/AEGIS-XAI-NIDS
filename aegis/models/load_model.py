import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def load_model(name="cicids"):
    model_path = os.path.join(BASE, f"{name}_rf_model.joblib")
    print("Loading trained model:", model_path)
    return joblib.load(model_path)
