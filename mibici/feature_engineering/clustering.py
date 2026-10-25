
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


COLUMNAS_P3_AMPLIADAS_EXTRA = ["entropia_horaria", "cv_diario", "prop_hora_pico"]


def crear_features_p3_ampliadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega tres características de comportamiento a las de `crear_features_p3`.

    - `entropia_horaria`: entropía (en bits) del perfil de salidas por hora.
      Es baja si la estación concentra su uso en pocas horas y alta si su uso
      es uniforme durante el día.
    - `cv_diario`: coeficiente de variación de las salidas diarias, desde el
      primer día con actividad. Distingue estaciones estables de estaciones
      con uso irregular.
    - `prop_hora_pico`: proporción de salidas en horas pico entre semana
      (7-9 y 17-20 h). Separa estaciones de movilidad laboral o escolar de las
      de uso recreativo.

    Mantiene las mismas estaciones que `crear_features_p3` para que ambas
    versiones del espacio de características sean comparables.
    """
    base = crear_features_p3(df)

    datos = df[["origen_id", "inicio"]].copy()
    datos = datos[datos["origen_id"].isin(base.index)]
    hora = datos["inicio"].dt.hour
    entre_semana = datos["inicio"].dt.dayofweek < 5

    # Entropía del perfil horario.
    horas = (
        datos.assign(hora=hora)
        .groupby(["origen_id", "hora"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    proporciones = horas.div(horas.sum(axis=1), axis=0)
    logaritmo = np.log2(proporciones.where(proporciones > 0, 1.0))
    entropia = -(proporciones * logaritmo).sum(axis=1)

    # Variabilidad de las salidas diarias desde la primera actividad.
    por_dia = (
        datos.assign(dia=datos["inicio"].dt.normalize())
        .groupby(["origen_id", "dia"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pd.date_range(
            datos["inicio"].min().normalize(),
            datos["inicio"].max().normalize(),
            freq="D",
        ), fill_value=0)
    )
    matriz = por_dia.to_numpy(dtype="float64")
    primer_dia = (matriz > 0).argmax(axis=1)
    mascara = np.arange(matriz.shape[1])[None, :] >= primer_dia[:, None]
    dias_activos = mascara.sum(axis=1)
    media = (matriz * mascara).sum(axis=1) / dias_activos
    varianza = (((matriz - media[:, None]) ** 2) * mascara).sum(axis=1) / dias_activos
    cv_diario = pd.Series(np.sqrt(varianza) / media, index=por_dia.index)

    # Participación de las horas pico entre semana.
    pico = entre_semana & hora.isin([7, 8, 9, 17, 18, 19, 20])
    prop_pico = pico.groupby(datos["origen_id"]).mean()

    base = base.copy()
    base["entropia_horaria"] = entropia
    base["cv_diario"] = cv_diario
    base["prop_hora_pico"] = prop_pico

    return base.fillna(0)
