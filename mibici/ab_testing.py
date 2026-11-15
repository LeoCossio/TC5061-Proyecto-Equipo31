"""Comparación A/B de dos versiones del modelo P1 (demanda por estación-hora).

El modelo A es la versión vigente (entrenada con octubre de 2025 a mayo de 2026). El
modelo B se reentrena con los mismos hiperparámetros pero con todos los datos anteriores
al mes de evaluación (octubre a agosto). Ambos se evalúan sobre las mismas filas del mes
de evaluación (septiembre de 2026), que ninguno de los dos vio al entrenar.

La decisión sobre cuál versión queda vigente se toma con una regla fijada antes de
mirar los resultados (ver `decidir`).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone

from mibici.config import (
    AB_FECHA_INICIO_EVALUACION,
    AB_NIVEL_CONFIANZA,
    AB_TOLERANCIA_EMPEORAMIENTO,
    P1_FECHA_INICIO_VALIDACION,
    PROCESSED_DATA_DIR,
)
from mibici.features import (
    COLUMNAS_P1_AMPLIADAS,
    OBJETIVO_P1,
    baseline_p1,
    construir_panel_horario,
    crear_features_p1,
    es_hora_pico,
)


@dataclass
class DatosAB:
    """Filas usadas por el experimento."""

    entrenamiento_b: pd.DataFrame  # todo lo anterior al mes de evaluación
    evaluacion: pd.DataFrame  # mes de evaluación (común a A y B)
    historia_a: pd.DataFrame  # periodo con el que se entrenó A (antes de la validación)
    todo_posterior: pd.DataFrame  # desde la validación de A hasta el final


def preparar_datos_ab(inicio_evaluacion=AB_FECHA_INICIO_EVALUACION):
    """Construye la tabla estación-hora y la divide cronológicamente.

    Reutiliza `crear_features_p1`, de modo que las variables son las mismas con las que se
    entrenó el modelo A y solo usan información anterior a cada hora.
    """
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    if len(archivos) != 12:
        raise FileNotFoundError(
            f"Se esperaban 12 Parquet en {PROCESSED_DATA_DIR} y hay {len(archivos)}. "
            "Ejecuta primero: python -m mibici.dataset")
    viajes = pd.concat(
        [pd.read_parquet(a, columns=["origen_id", "inicio"]) for a in archivos],
        ignore_index=True,
    )
    features = crear_features_p1(construir_panel_horario(viajes))
    hora = features["hora_inicio"]
    corte = pd.Timestamp(inicio_evaluacion)
    return DatosAB(
        entrenamiento_b=features[hora < corte],
        evaluacion=features[hora >= corte],
        historia_a=features[hora < pd.Timestamp(P1_FECHA_INICIO_VALIDACION)],
        todo_posterior=features[hora >= pd.Timestamp(P1_FECHA_INICIO_VALIDACION)],
    )


def entrenar_alternativa(modelo_a, entrenamiento_b):
    """Reentrena una copia de A (mismos hiperparámetros y semilla) con más datos.

    Se aísla así el efecto de usar datos más recientes: lo único que cambia respecto de A
    es el periodo de entrenamiento.
    """
    modelo_b = clone(modelo_a)
    return modelo_b.fit(entrenamiento_b[COLUMNAS_P1_AMPLIADAS], entrenamiento_b[OBJETIVO_P1])


def metricas(conjunto, prediccion):
    """MAE, RMSE y MAE en horas pico (en viajes por estación-hora)."""
    real = conjunto[OBJETIVO_P1].to_numpy(dtype="float64")
    prediccion = np.asarray(prediccion, dtype="float64")
    error = real - prediccion
    pico = es_hora_pico(conjunto).to_numpy()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mae_horas_pico": float(np.abs(error[pico]).mean()),
    }


def comparar_pareado(evaluacion, pred_a, pred_b, nivel=AB_NIVEL_CONFIANZA):
    """Compara el error de A y B día por día (observaciones pareadas).

    Las filas de un mismo día no son independientes (comparten clima, eventos, etc.), así
    que se calcula el MAE de cada día y se compara la serie diaria. Devuelve la diferencia
    media de MAE (A menos B; positiva si B es mejor), su intervalo de confianza con la
    distribución t, el valor p de la prueba de Wilcoxon y los días en que B gana.
    """
    real = evaluacion[OBJETIVO_P1].to_numpy(dtype="float64")
    tabla = pd.DataFrame({
        "dia": evaluacion["hora_inicio"].dt.normalize().to_numpy(),
        "error_a": np.abs(real - np.asarray(pred_a, dtype="float64")),
        "error_b": np.abs(real - np.asarray(pred_b, dtype="float64")),
    })
    diario = tabla.groupby("dia")[["error_a", "error_b"]].mean()
    diferencia = diario["error_a"] - diario["error_b"]
    n = len(diferencia)
    media = float(diferencia.mean())
    if n > 1 and diferencia.std(ddof=1) > 0:
        error_estandar = diferencia.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.5 + nivel / 2, n - 1)
        intervalo = (media - t * error_estandar, media + t * error_estandar)
        p_wilcoxon = float(stats.wilcoxon(diferencia).pvalue)
    else:
        intervalo, p_wilcoxon = (media, media), 1.0
    return {
        "dias": int(n),
        "diferencia_mae_diario": media,
        "ic_inferior": float(intervalo[0]),
        "ic_superior": float(intervalo[1]),
        "p_wilcoxon": p_wilcoxon,
        "dias_en_que_gana_b": int((diferencia > 0).sum()),
        "diario": diario.assign(diferencia=diferencia),
    }


def decidir(metricas_a, metricas_b, pareado, tolerancia=AB_TOLERANCIA_EMPEORAMIENTO):
    """Regla de decisión, fijada antes de ver los resultados.

    B pasa a ser la versión vigente solo si cumple las tres condiciones:
      1. Reduce el MAE y el intervalo de confianza de la diferencia diaria excluye el cero.
      2. No empeora el RMSE en más de `tolerancia` (1 %).
      3. No empeora el MAE en hora pico en más de `tolerancia` (1 %).
    En cualquier otro caso se conserva A, porque cambiar de versión tiene un costo y solo se
    justifica con una mejora comprobada.
    """
    mejora_significativa = pareado["ic_inferior"] > 0
    rmse_ok = metricas_b["rmse"] <= metricas_a["rmse"] * (1 + tolerancia)
    pico_ok = metricas_b["mae_horas_pico"] <= metricas_a["mae_horas_pico"] * (1 + tolerancia)
    condiciones = {
        "mejora_mae_significativa": bool(mejora_significativa),
        "rmse_no_empeora": bool(rmse_ok),
        "mae_pico_no_empeora": bool(pico_ok),
    }
    return ("B" if all(condiciones.values()) else "A"), condiciones


def mae_semanal(conjunto, prediccion):
    """MAE por semana (lunes a domingo) de un conjunto de filas."""
    real = conjunto[OBJETIVO_P1].to_numpy(dtype="float64")
    error = np.abs(real - np.asarray(prediccion, dtype="float64"))
    semana = conjunto["hora_inicio"].dt.to_period("W-SUN").dt.start_time
    return pd.Series(error, index=semana.to_numpy()).groupby(level=0).mean()


def mae_por_hora(conjunto, prediccion):
    """MAE por hora del día."""
    real = conjunto[OBJETIVO_P1].to_numpy(dtype="float64")
    error = np.abs(real - np.asarray(prediccion, dtype="float64"))
    return pd.Series(error, index=conjunto["hora"].to_numpy()).groupby(level=0).mean()


def mae_por_grupo_de_estacion(conjunto, prediccion, estaciones_conocidas):
    """MAE separando estaciones con historia en el entrenamiento de A de las nuevas."""
    real = conjunto[OBJETIVO_P1].to_numpy(dtype="float64")
    error = np.abs(real - np.asarray(prediccion, dtype="float64"))
    conocida = conjunto["origen_id"].isin(estaciones_conocidas).to_numpy()
    return {
        "conocidas": {"filas": int(conocida.sum()), "mae": float(error[conocida].mean())},
        "nuevas": {"filas": int((~conocida).sum()),
                   "mae": float(error[~conocida].mean()) if (~conocida).any() else float("nan")},
    }


def mae_baseline_semanal(conjunto, media_global):
    """MAE semanal del baseline (media histórica de la estación para ese día y hora)."""
    return mae_semanal(conjunto, baseline_p1(conjunto, media_global).to_numpy())


def resumen_mensual(conjunto, prediccion, media_global):
    """Demanda media y error por mes del modelo vigente y del baseline.

    La ventaja del modelo sobre el baseline (media histórica, que se actualiza con cada hora
    nueva) indica si el modelo conserva su valor con el paso del tiempo.
    """
    real = conjunto[OBJETIVO_P1].to_numpy(dtype="float64")
    error_modelo = np.abs(real - np.asarray(prediccion, dtype="float64"))
    error_baseline = np.abs(real - baseline_p1(conjunto, media_global).to_numpy(dtype="float64"))
    mes = conjunto["hora_inicio"].dt.to_period("M").astype(str).to_numpy()
    tabla = pd.DataFrame({"mes": mes, "demanda_media": real, "mae_modelo": error_modelo,
                          "mae_baseline": error_baseline}).groupby("mes").mean()
    tabla["ventaja_sobre_baseline_pct"] = 100 * (1 - tabla["mae_modelo"] / tabla["mae_baseline"])
    return tabla
