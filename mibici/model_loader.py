
"""Carga de modelos de MiBici desde MLflow Registry o artefactos locales."""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn


MODELS = {
    "p1": "models:/mibici-equipo31-p1-demanda/1",
    "p2": "models:/mibici-equipo31-p2-riesgo/1",
    "p3": "models:/mibici-equipo31-p3-grupos/1",
}


def load_model(question="p3"):
    """Carga un modelo del Registry o de una ruta local configurada."""

    if question not in MODELS:
        raise ValueError(f"Modelo no disponible: {question}")

    local_path = os.getenv(f"MIBICI_MODEL_{question.upper()}_PATH")

    if local_path:
        path = Path(local_path)

        if not (path / "MLmodel").is_file():
            raise FileNotFoundError(
                f"No se encontró MLmodel en la ruta: {path}"
            )

        return mlflow.sklearn.load_model(str(path))

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(MODELS[question])
