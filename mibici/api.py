
"""API REST para predicciones de los modelos MiBici."""

import pandas as pd
from fastapi import FastAPI

from mibici.esquemas_api import (
    P1Input,
    P1Output,
    P2Input,
    P2Output,
    P3Input,
    P3Output,
)
from mibici.model_loader import load_model


# =========================
# Inicializacion de la API
# =========================

app = FastAPI(
    title="MiBici - API de predicción",
    description="Servicio REST de predicción para los modelos del Equipo 31.",
    version="1.0.0",
)


# =========================
# Carga de modelos MLflow
# =========================

from functools import lru_cache


@lru_cache(maxsize=3)
def obtener_modelo(pregunta: str):
    """Carga el modelo solicitado y lo conserva en memoria."""
    return load_model(pregunta)


# =========================
# Estado del servicio
# =========================

@app.get("/health")
def health():
    """Verifica que la API está activa."""
    return {"status": "ok"}


# =========================
# Prediccion P1: demanda
# =========================

@app.post("/predict/p1", response_model=P1Output)
def predict_p1(data: P1Input):
    """Predice la demanda utilizando el modelo P1."""
    df = pd.DataFrame([data.model_dump()])
    prediction = float(obtener_modelo("p1").predict(df)[0])

    return {"demanda_predicha": prediction}


# =========================
# Prediccion P2: riesgo
# =========================

@app.post("/predict/p2", response_model=P2Output)
def predict_p2(data: P2Input):
    """Predice la clase de riesgo utilizando el modelo P2."""
    df = pd.DataFrame([data.model_dump()])
    prediction = int(obtener_modelo("p2").predict(df)[0])

    return {"clase_predicha": prediction}


# =========================
# Prediccion P3: clustering
# =========================

@app.post("/predict/p3", response_model=P3Output)
def predict_p3(data: P3Input):
    """Predice el cluster utilizando el modelo P3."""
    df = pd.DataFrame([data.model_dump()])
    cluster = int(obtener_modelo("p3").predict(df)[0])

    return {"cluster": cluster}


# =========================
# Endpoint de compatibilidad
# =========================

@app.post("/predict", response_model=P3Output)
def predict(data: P3Input):
    """Mantiene compatibilidad con el endpoint anterior de P3."""
    return predict_p3(data)
