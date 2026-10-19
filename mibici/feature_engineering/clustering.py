
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def crear_features_p3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye características por estación para clustering.

    Incluye intensidad de demanda, balance relativo,
    proporción de viajes en fines de semana y
    distribución de salidas por franjas horarias.

    Cada fila representa una estación.
    """
    df = df.copy()

    # Conteos de viajes por estación
    salidas = df.groupby("origen_id").size()
    llegadas = df.groupby("destino_id").size()

    features = pd.DataFrame({
        "salidas": salidas,
        "llegadas": llegadas,
    }).fillna(0)

    # Intensidad de demanda
    dias = df["inicio"].dt.normalize().nunique()
    features["viajes_diarios"] = features["salidas"] / dias

    # Balance relativo
    movimientos = features["salidas"] + features["llegadas"]
    features["balance_relativo"] = (
        (features["llegadas"] - features["salidas"]) / movimientos
    )

    # Patrones de fines de semana
    df["fin_semana"] = df["inicio"].dt.dayofweek >= 5
    features["proporcion_fin_semana"] = (
        df.groupby("origen_id")["fin_semana"].mean()
    )

    # Patrones horarios
    df["hora"] = df["inicio"].dt.hour

    horas = (
        df.groupby(["origen_id", "hora"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )

    proporciones = horas.div(horas.sum(axis=1), axis=0)

    franjas = {
        "manana": range(6, 11),
        "mediodia": range(11, 16),
        "tarde": range(16, 21),
        "noche": [21, 22, 23, 0, 1, 2, 3, 4, 5],
    }

    for nombre, horas_franja in franjas.items():
        features[nombre] = proporciones[list(horas_franja)].sum(axis=1)

    # Excluir estaciones con actividad insuficiente
    features = features[features["salidas"] >= 100]

    # Conservar únicamente las siete características
    columnas = [
        "viajes_diarios",
        "balance_relativo",
        "proporcion_fin_semana",
        "manana",
        "mediodia",
        "tarde",
        "noche",
    ]

    return features[columnas].fillna(0)

def preprocesar_features_p3(features: pd.DataFrame):
    """
    Transforma y estandariza las características de P3.

    Reduce la asimetría de la demanda con log1p y elimina
    una franja horaria redundante. Devuelve las características
    escaladas y el escalador utilizado.
    """
    datos = features.drop(columns=["noche"]).copy()

    datos["viajes_diarios"] = np.log1p(datos["viajes_diarios"])

    scaler = StandardScaler()
    escalados = pd.DataFrame(
        scaler.fit_transform(datos),
        index=datos.index,
        columns=datos.columns,
    )

    return escalados, scaler


def baseline_p3(features: pd.DataFrame) -> pd.Series:
    """
    Agrupa estaciones en tres niveles de demanda.

    Utiliza terciles de viajes diarios como baseline
    para comparar posteriormente con K-Means.
    """
    grupos = pd.qcut(
        features["viajes_diarios"],
        q=3,
        labels=[0, 1, 2],
    )

    return grupos.astype(int).rename("cluster")

def entrenar_kmeans_p3(features_escaladas: pd.DataFrame, n_clusters: int = 3):
    """
    Agrupa estaciones según sus patrones de uso.

    Utiliza K-Means sobre las características estandarizadas.
    Devuelve las etiquetas y el modelo entrenado.
    """
    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    grupos = pd.Series(
        modelo.fit_predict(features_escaladas),
        index=features_escaladas.index,
        name="cluster",
    )

    return grupos, modelo
