
"""Preparación de datos para P2: riesgo estimado de desbalance."""

import numpy as np
import pandas as pd


def cargar_capacidades(ruta_catalogo):
    """Carga las capacidades válidas de la nomenclatura de estaciones."""
    catalogo = pd.read_csv(
        ruta_catalogo,
        usecols=["id", "dpcapacity"],
    )
    catalogo = catalogo.rename(
        columns={"id": "estacion_id", "dpcapacity": "capacidad"}
    )
    catalogo["estacion_id"] = pd.to_numeric(
        catalogo["estacion_id"], errors="coerce"
    )
    catalogo["capacidad"] = pd.to_numeric(
        catalogo["capacidad"], errors="coerce"
    )

    catalogo = catalogo.dropna(subset=["estacion_id", "capacidad"])
    catalogo = catalogo[catalogo["capacidad"] > 0].copy()
    catalogo["estacion_id"] = catalogo["estacion_id"].astype("int64")

    if catalogo["estacion_id"].duplicated().any():
        raise ValueError("La nomenclatura contiene estaciones duplicadas.")

    return catalogo


def construir_panel_p2(viajes, catalogo):
    """
    Construye un panel horario de entradas y salidas por estación.

    Las horas sin viajes se representan con cero. Se generan registros
    desde la primera actividad de cada estación, sin asumir que
    operaba antes de esa fecha.

    La capacidad proviene del catálogo. Se excluyen estaciones sin
    capacidad conocida, sin inventar información faltante.
    """
    salidas = (
        viajes.assign(hora_inicio=viajes["inicio"].dt.floor("h"))
        .groupby(["origen_id", "hora_inicio"])
        .size()
        .rename("salidas")
        .reset_index()
        .rename(columns={"origen_id": "estacion_id", "hora_inicio": "fecha_hora"})
    )

    llegadas = (
        viajes.assign(hora_fin=viajes["fin"].dt.floor("h"))
        .groupby(["destino_id", "hora_fin"])
        .size()
        .rename("llegadas")
        .reset_index()
        .rename(columns={"destino_id": "estacion_id", "hora_fin": "fecha_hora"})
    )

    actividad = salidas.merge(
        llegadas, on=["estacion_id", "fecha_hora"], how="outer"
    )
    actividad[["salidas", "llegadas"]] = (
        actividad[["salidas", "llegadas"]].fillna(0).astype("int32")
    )

    actividad = actividad.merge(catalogo, on="estacion_id", how="inner")

    if actividad.empty:
        raise ValueError("No hay estaciones con actividad y capacidad conocida.")

    # Última hora completa del periodo de salidas observado.
    limite = viajes["inicio"].max().floor("h")

    actividad = actividad[actividad["fecha_hora"] <= limite].copy()
    primera_hora = actividad.groupby("estacion_id")["fecha_hora"].min()

    estaciones = []
    fechas = []

    for estacion, inicio in primera_hora.items():
        rango = pd.date_range(inicio, limite, freq="h")
        estaciones.append(np.full(len(rango), estacion, dtype="int64"))
        fechas.append(rango.to_numpy())

    panel = pd.DataFrame({
        "estacion_id": np.concatenate(estaciones),
        "fecha_hora": np.concatenate(fechas),
    })

    panel = panel.merge(
        actividad,
        on=["estacion_id", "fecha_hora"],
        how="left",
    )

    panel[["salidas", "llegadas"]] = (
        panel[["salidas", "llegadas"]].fillna(0).astype("int32")
    )

    panel["capacidad"] = (
        panel["estacion_id"].map(
            catalogo.set_index("estacion_id")["capacidad"]
        ).astype("float32")
    )

    panel["flujo_neto"] = panel["llegadas"] - panel["salidas"]
    panel["presion_desbalance"] = (
        panel["flujo_neto"].abs() / panel["capacidad"]
    ).astype("float32")

    return panel.sort_values(
        ["estacion_id", "fecha_hora"], ignore_index=True
    )

# Variables de predicción de P2.
COLUMNAS_P2 = [
    "hora",
    "dia_semana",
    "fin_semana",
    "capacidad",
    "salidas_lag_1",
    "llegadas_lag_1",
    "flujo_neto_lag_1",
    "presion_lag_1",
    "presion_lag_2",
    "presion_lag_24",
    "presion_lag_168",
    "presion_media_24",
    "presion_media_168",
]

OBJETIVO_P2 = "riesgo_siguiente_hora"
HORAS_CALENTAMIENTO_P2 = 168


def crear_features_p2(panel):
    """Crea variables históricas para predecir el riesgo de la hora H."""
    datos = panel.sort_values(
        ["estacion_id", "fecha_hora"], ignore_index=True
    ).copy()

    datos["hora"] = datos["fecha_hora"].dt.hour
    datos["dia_semana"] = datos["fecha_hora"].dt.dayofweek
    datos["fin_semana"] = (datos["dia_semana"] >= 5).astype("int8")

    por_estacion = datos.groupby("estacion_id", sort=False)

    for columna in ("salidas", "llegadas", "flujo_neto"):
        datos[f"{columna}_lag_1"] = por_estacion[columna].shift(1)

    for rezago in (1, 2, 24, 168):
        datos[f"presion_lag_{rezago}"] = (
            por_estacion["presion_desbalance"].shift(rezago)
        )

    for ventana in (24, 168):
        datos[f"presion_media_{ventana}"] = (
            por_estacion["presion_desbalance"]
            .transform(
                lambda serie: serie.shift(1).rolling(
                    ventana, min_periods=ventana
                ).mean()
            )
        )

    posicion = por_estacion.cumcount()
    datos = datos[posicion >= HORAS_CALENTAMIENTO_P2].copy()

    datos[COLUMNAS_P2] = datos[COLUMNAS_P2].astype("float32")

    # Guardamos la presión observada de la hora objetivo para crear
    # la etiqueta posteriormente, usando solo entrenamiento para el umbral.
    return datos.reset_index(drop=True)


def dividir_temporal_p2(datos, inicio_validacion, inicio_prueba):
    """Separa las observaciones cronológicamente, como en P1."""
    fechas = datos["fecha_hora"]

    entrenamiento = datos[
        fechas < pd.Timestamp(inicio_validacion)
    ].copy()

    validacion = datos[
        (fechas >= pd.Timestamp(inicio_validacion))
        & (fechas < pd.Timestamp(inicio_prueba))
    ].copy()

    prueba = datos[
        fechas >= pd.Timestamp(inicio_prueba)
    ].copy()

    return entrenamiento, validacion, prueba


def definir_umbral_p2(entrenamiento, cuantíl=0.90):
    """Calcula el umbral de presión exclusivamente con entrenamiento."""
    if not 0 < cuantíl < 1:
        raise ValueError("El cuantil debe estar entre 0 y 1.")

    return float(
        entrenamiento["presion_desbalance"].quantile(cuantíl)
    )


def asignar_etiqueta_p2(datos, umbral):
    """Clasifica cada estación-hora según su presión observada."""
    resultado = datos.copy()
    resultado[OBJETIVO_P2] = (
        resultado["presion_desbalance"] >= umbral
    ).astype("int8")
    return resultado

from sklearn.ensemble import RandomForestClassifier


def baseline_p2(entrenamiento, evaluacion):
    """
    Baseline: predice riesgo alto cuando la presión de la hora
    anterior supera el umbral aprendido durante entrenamiento.
    """
    umbral = definir_umbral_p2(entrenamiento)

    return (
        evaluacion["presion_lag_1"] >= umbral
    ).astype("int8")



def entrenar_random_forest_p2(
    entrenamiento,
    semilla=42,
    n_estimators=100,
    min_samples_leaf=20,
    max_samples=0.3,
    max_depth=None,
    max_features="sqrt",
):
    """Entrena el clasificador de riesgo estimado de desbalance."""
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        max_features=max_features,
        random_state=semilla,
        n_jobs=2,
        class_weight="balanced_subsample",
    )

    modelo.fit(
        entrenamiento[COLUMNAS_P2],
        entrenamiento[OBJETIVO_P2],
    )

    return modelo

