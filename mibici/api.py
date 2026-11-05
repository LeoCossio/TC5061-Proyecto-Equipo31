from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from mibici.model_loader import load_model

app = FastAPI(title="MiBici - API de predicción")

# Cargar modelos registrados en MLflow
model_p1 = load_model("p1")
model_p2 = load_model("p2")
model_p3 = load_model("p3")


# =========================
# Esquemas de entrada
# =========================

class P1Input(BaseModel):
    hora: int
    dia_semana: int
    fin_semana: int
    lag_1: float
    lag_2: float
    lag_3: float
    lag_24: float
    lag_168: float
    media_movil_24: float
    media_movil_168: float
    media_hist_estacion_dow_hora: float
    media_hist_estacion: float
    media_hist_estacion_hora: float
    tendencia_24_168: float
    hora_sin: float
    hora_cos: float
    prop_horas_cero_24: float


class P2Input(BaseModel):
    hora: int
    dia_semana: int
    fin_semana: int
    capacidad: float
    salidas_lag_1: float
    llegadas_lag_1: float
    flujo_neto_lag_1: float
    presion_lag_1: float
    presion_lag_2: float
    presion_lag_24: float
    presion_lag_168: float
    presion_media_24: float
    presion_media_168: float


class P3Input(BaseModel):
    viajes_diarios: float
    balance_relativo: float
    proporcion_fin_semana: float
    manana: float
    mediodia: float
    tarde: float


# =========================
# Endpoints
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/p1")
def predict_p1(data: P1Input):
    df = pd.DataFrame([data.model_dump()])
    prediction = float(model_p1.predict(df)[0])
    return {"demanda_predicha": prediction}


@app.post("/predict/p2")
def predict_p2(data: P2Input):
    df = pd.DataFrame([data.model_dump()])
    prediction = model_p2.predict(df)[0]
    return {"clase_predicha": int(prediction)}


@app.post("/predict/p3")
def predict_p3(data: P3Input):
    df = pd.DataFrame([data.model_dump()])
    cluster = int(model_p3.predict(df)[0])
    return {"cluster": cluster}


# Compatibilidad con el endpoint anterior
@app.post("/predict")
def predict(data: P3Input):
    return predict_p3(data)