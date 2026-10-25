import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Variable objetivo de P1: salidas de la estación en la hora que se quiere predecir.
OBJETIVO_P1 = "viajes_siguiente_hora"

# Características que recibe el modelo. Todas se calculan con información
# disponible ANTES de que empiece la hora objetivo.
COLUMNAS_P1 = [
    "hora",
    "dia_semana",
    "fin_semana",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "media_movil_24",
    "media_movil_168",
    "media_hist_estacion_dow_hora",
    "media_hist_estacion",
]

# Características adicionales de la versión ampliada (experimentación).
COLUMNAS_P1_AMPLIADAS = COLUMNAS_P1 + [
    "media_hist_estacion_hora",
    "tendencia_24_168",
    "hora_sin",
    "hora_cos",
    "prop_horas_cero_24",
]

# Horas pico entre semana (mañana y tarde) para medir el error donde más importa.
HORAS_PICO = [7, 8, 9, 17, 18, 19, 20]

# Un rezago de una semana es el más largo; las primeras 168 horas de cada
# estación no tienen historia completa y no se usan como ejemplos.
HORAS_CALENTAMIENTO = 168


def construir_panel_horario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte los viajes en una serie horaria de salidas por estación.

    Cuenta los viajes por estación de origen y hora de inicio, y completa las
    horas sin viajes con cero, desde la primera salida de cada estación hasta
    el final del periodo. Así las horas vacías no desaparecen y el modelo
    puede aprender que una estación puede tener demanda cero. No se generan
    filas anteriores a la primera salida de una estación para no inventar
    ceros de estaciones que todavía no operaban.

    Cada fila representa una estación-hora. Es la base de P1 porque la
    pregunta es cuántos viajes iniciarán en una estación en una hora futura.
    """
    horas = df["inicio"].dt.floor("h")

    conteos = (
        df.assign(hora_inicio=horas)
        .groupby(["origen_id", "hora_inicio"])
        .size()
        .rename(OBJETIVO_P1)
        .reset_index()
    )

    ultima_hora = horas.max()
    primera_hora = conteos.groupby("origen_id")["hora_inicio"].min()

    estaciones = []
    marcas = []
    for estacion, inicio in primera_hora.items():
        rango = pd.date_range(inicio, ultima_hora, freq="h")
        estaciones.append(np.full(len(rango), estacion))
        marcas.append(rango)

    panel = pd.DataFrame({
        "origen_id": np.concatenate(estaciones),
        "hora_inicio": np.concatenate([m.to_numpy() for m in marcas]),
    })

    panel = panel.merge(conteos, on=["origen_id", "hora_inicio"], how="left")
    panel[OBJETIVO_P1] = panel[OBJETIVO_P1].fillna(0).astype("int32")

    return panel.sort_values(["origen_id", "hora_inicio"], ignore_index=True)


def crear_features_p1(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las características de P1 sin usar información futura.

    Para la fila de la hora H, todo lo que se calcula sale de horas
    anteriores a H: salidas de las 3 horas previas, del mismo momento un día
    y una semana antes, promedios móviles de 24 y 168 horas, y promedios
    históricos de la estación (en general y para el mismo día de la semana y
    hora). Los promedios históricos son causales: en cada fila solo acumulan
    observaciones pasadas, nunca la de la hora objetivo ni las posteriores.
    Se agregan también hora, día de la semana y fin de semana.

    Además calcula cinco variables de la versión ampliada
    (COLUMNAS_P1_AMPLIADAS): promedio histórico de la estación para esa hora
    sin distinguir día, diferencia entre el promedio de 24 h y el de 168 h
    (tendencia reciente), hora codificada como seno y coseno, y proporción de
    horas sin salidas en las últimas 24 h (qué tan inactiva ha estado la
    estación). Permiten probar si captar el nivel reciente y la continuidad
    del ciclo diario mejora la predicción.

    Son pertinentes para la pregunta porque la demanda de una estación
    depende del ciclo diario y semanal de movilidad (calendario), de la
    intensidad reciente (rezagos y promedios móviles) y de su nivel habitual
    (promedios históricos). Se descartan las primeras 168 horas de cada
    estación, que no tienen historia suficiente.

    Devuelve una fila por estación-hora con las columnas de COLUMNAS_P1, la
    variable objetivo, la estación y la hora.
    """
    datos = panel.sort_values(["origen_id", "hora_inicio"], ignore_index=True)
    y = datos[OBJETIVO_P1].astype("float64")
    por_estacion = y.groupby(datos["origen_id"])

    # Calendario de la hora que se quiere predecir.
    datos["hora"] = datos["hora_inicio"].dt.hour
    datos["dia_semana"] = datos["hora_inicio"].dt.dayofweek
    datos["fin_semana"] = (datos["dia_semana"] >= 5).astype("int8")

    # Rezagos: el valor de horas que ya ocurrieron al momento de predecir.
    for horas in (1, 2, 3, 24, 168):
        datos[f"lag_{horas}"] = por_estacion.shift(horas)

    # Promedios móviles de las N horas previas: (acumulado hasta H-1) - (hasta H-1-N).
    acumulado = por_estacion.cumsum()
    por_acumulado = acumulado.groupby(datos["origen_id"])
    previo = por_acumulado.shift(1)
    for ventana in (24, 168):
        inicio_ventana = por_acumulado.shift(ventana + 1).fillna(0)
        datos[f"media_movil_{ventana}"] = (previo - inicio_ventana) / ventana

    posicion = por_estacion.cumcount()

    # Promedio histórico de la estación hasta la hora anterior.
    datos["media_hist_estacion"] = (acumulado - y) / posicion.where(posicion > 0)

    # Promedio histórico de la estación para el mismo día de la semana y hora.
    clave = [datos["origen_id"], datos["dia_semana"], datos["hora"]]
    por_momento = y.groupby(clave)
    previas = por_momento.cumcount()
    datos["media_hist_estacion_dow_hora"] = (
        (por_momento.cumsum() - y) / previas.where(previas > 0)
    )

    # Variables adicionales de la versión ampliada. Siguen siendo causales:
    # solo acumulan o promedian horas anteriores a la hora objetivo.
    por_hora = y.groupby([datos["origen_id"], datos["hora"]])
    previas_hora = por_hora.cumcount()
    datos["media_hist_estacion_hora"] = (
        (por_hora.cumsum() - y) / previas_hora.where(previas_hora > 0)
    )

    datos["tendencia_24_168"] = datos["media_movil_24"] - datos["media_movil_168"]

    angulo = 2 * np.pi * datos["hora"] / 24
    datos["hora_sin"] = np.sin(angulo)
    datos["hora_cos"] = np.cos(angulo)

    ceros = (y == 0).astype("float64").groupby(datos["origen_id"]).cumsum()
    por_ceros = ceros.groupby(datos["origen_id"])
    datos["prop_horas_cero_24"] = (
        por_ceros.shift(1) - por_ceros.shift(25).fillna(0)
    ) / 24

    datos = datos[posicion >= HORAS_CALENTAMIENTO].copy()
    datos[COLUMNAS_P1_AMPLIADAS] = datos[COLUMNAS_P1_AMPLIADAS].astype("float32")

    return datos.reset_index(drop=True)


def dividir_temporal_p1(
    features: pd.DataFrame,
    inicio_validacion: str,
    inicio_prueba: str,
):
    """
    Separa las filas en entrenamiento, validación y prueba por fecha.

    La separación es cronológica y nunca aleatoria: entrenamiento antes de
    `inicio_validacion`, validación entre ambas fechas y prueba desde
    `inicio_prueba`. Evita que el modelo vea meses futuros al entrenar.
    """
    hora = features["hora_inicio"]
    entrenamiento = features[hora < pd.Timestamp(inicio_validacion)]
    validacion = features[
        (hora >= pd.Timestamp(inicio_validacion))
        & (hora < pd.Timestamp(inicio_prueba))
    ]
    prueba = features[hora >= pd.Timestamp(inicio_prueba)]

    return entrenamiento, validacion, prueba


def baseline_p1(features: pd.DataFrame, media_global: float) -> pd.Series:
    """
    Predice la media histórica de la estación para el mismo día y hora.

    Es el baseline de P1: sin entrenar ningún modelo, la demanda esperada es
    lo que históricamente salió de esa estación en ese día de la semana y
    esa hora, usando solo semanas anteriores. Si la estación no tiene
    historia para ese momento se usa la media de la estación, y como último
    recurso la media global calculada con entrenamiento.
    """
    prediccion = (
        features["media_hist_estacion_dow_hora"]
        .fillna(features["media_hist_estacion"])
        .fillna(media_global)
    )

    return prediccion.rename("prediccion")


def entrenar_random_forest_p1(
    entrenamiento: pd.DataFrame,
    semilla: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_samples: float,
):
    """
    Entrena un RandomForestRegressor para predecir las salidas por hora.

    Aprende la relación entre calendario, rezagos y promedios históricos con
    las filas de entrenamiento. Cada árbol se entrena con una fracción
    aleatoria de las filas (`max_samples`), lo que mantiene el costo bajo con
    millones de registros. Devuelve el modelo entrenado.
    """
    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        n_jobs=-1,
        random_state=semilla,
    )
    modelo.fit(entrenamiento[COLUMNAS_P1], entrenamiento[OBJETIVO_P1])

    return modelo


def es_hora_pico(features: pd.DataFrame) -> pd.Series:
    """Marca las horas pico entre semana, donde un error de predicción cuesta más."""
    return (features["fin_semana"] == 0) & features["hora"].isin(HORAS_PICO)
