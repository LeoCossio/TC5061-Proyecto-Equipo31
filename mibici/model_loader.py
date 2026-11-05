import os
import mlflow.sklearn

MODELS = {
    "p1": "models:/mibici-p1/1",
    "p2": "models:/mibici-p2/1",
    "p3": "models:/mibici-p3/1",
}

def load_model(question="p3"):
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )

    mlflow.set_tracking_uri(tracking_uri)

    if question not in MODELS:
        raise ValueError(f"Modelo no disponible: {question}")

    return mlflow.sklearn.load_model(MODELS[question])

