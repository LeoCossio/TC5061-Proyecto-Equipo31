"""Experimentos de P1 (regresión de demanda por estación-hora).

Cada ejecución registra, dentro de la corrida activa:
  1. Una búsqueda de hiperparámetros de HistGradientBoostingRegressor como
     nested runs: una corrida padre y una hija por combinación.
  2. Cuatro candidatos que varían algoritmo y características:
     Random Forest y HistGradientBoosting, con características base y ampliadas.

La partición temporal y las métricas son las mismas del baseline
(entrenamiento hasta mayo, validación en junio-julio, prueba en agosto-septiembre).
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import ParameterGrid

from mibici.config import (
    P1_FECHA_FIN,
    P1_FECHA_INICIO,
    P1_FECHA_INICIO_PRUEBA,
    P1_FECHA_INICIO_VALIDACION,
    P1_HGB_ESPACIO,
    P1_HGB_MAX_ITER,
    P1_HGB_MIN_SAMPLES_LEAF,
    P1_RF_MAX_SAMPLES,
    P1_RF_MIN_SAMPLES_LEAF,
    P1_RF_N_ESTIMADORES,
    P1_SEMILLA,
    PROCESSED_DATA_DIR,
)
from mibici.experimentos.comun import (
    archivos_y_version,
    commit_actual,
    etiquetas,
    medir,
    registrar_candidato,
    resumen_json,
)
from mibici.features import (
    COLUMNAS_P1,
    COLUMNAS_P1_AMPLIADAS,
    OBJETIVO_P1,
    baseline_p1,
    construir_panel_horario,
    crear_features_p1,
    dividir_temporal_p1,
)
from mibici.train_p1 import calcular_metricas

PREGUNTA = "p1"


@dataclass
class DatosP1:
    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    prueba: pd.DataFrame
    media_global: float
    parametros: dict
    metricas_baseline: dict


def preparar_datos(version, commit):
    """Construye el panel, las características y la partición cronológica."""
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    viajes = pd.concat(
        [pd.read_parquet(a, columns=["origen_id", "inicio"]) for a in archivos],
        ignore_index=True,
    )
    panel = construir_panel_horario(viajes)
    features = crear_features_p1(panel)
    entrenamiento, validacion, prueba = dividir_temporal_p1(
        features, P1_FECHA_INICIO_VALIDACION, P1_FECHA_INICIO_PRUEBA
    )
    media_global = float(entrenamiento[OBJETIVO_P1].mean())

    metricas_baseline = {
        **calcular_metricas("val", validacion, baseline_p1(validacion, media_global)),
        **calcular_metricas("test", prueba, baseline_p1(prueba, media_global)),
    }

    parametros = {
        "fecha_inicio": P1_FECHA_INICIO,
        "fecha_fin": P1_FECHA_FIN,
        "fecha_inicio_validacion": P1_FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": P1_FECHA_INICIO_PRUEBA,
        "data_version": version,
        "semilla": P1_SEMILLA,
        "granularidad": "estacion-hora",
        "horizonte": "1 hora",
        "n_estaciones": int(panel["origen_id"].nunique()),
        "filas_entrenamiento": len(entrenamiento),
        "filas_validacion": len(validacion),
        "filas_prueba": len(prueba),
    }
    return DatosP1(
        entrenamiento, validacion, prueba, media_global, parametros, metricas_baseline
    )


def entrenar_hgb(entrenamiento, columnas, hiperparametros):
    """Entrena un HistGradientBoostingRegressor con los hiperparámetros dados."""
    modelo = HistGradientBoostingRegressor(
        max_iter=P1_HGB_MAX_ITER,
        min_samples_leaf=P1_HGB_MIN_SAMPLES_LEAF,
        early_stopping=False,
        random_state=P1_SEMILLA,
        **hiperparametros,
    )
    return modelo.fit(entrenamiento[columnas], entrenamiento[OBJETIVO_P1])


def evaluar(modelo, columnas, datos, conjuntos=("val", "test")):
    """Calcula las métricas del modelo y el tiempo de predicción."""
    metricas, predicciones, segundos = {}, {}, 0.0
    for prefijo in conjuntos:
        conjunto = datos.validacion if prefijo == "val" else datos.prueba
        prediccion, t = medir(modelo.predict, conjunto[columnas])
        segundos += t
        predicciones[prefijo] = prediccion
        metricas.update(calcular_metricas(prefijo, conjunto, prediccion))
    return metricas, predicciones, segundos


def buscar_hgb(datos, commit):
    """Búsqueda en malla con nested runs; elige la combinación de menor val_mae."""
    combinaciones = list(ParameterGrid(P1_HGB_ESPACIO))
    mejor = None

    with mlflow.start_run(run_name="P1-busqueda-hgb", nested=True):
        mlflow.set_tags(etiquetas(PREGUNTA, "hist_gradient_boosting", "search", commit))
        mlflow.log_params({
            **datos.parametros,
            "algoritmo": "HistGradientBoostingRegressor",
            "espacio_busqueda": resumen_json(P1_HGB_ESPACIO),
            "n_combinaciones": len(combinaciones),
            "criterio_seleccion": "val_mae",
            "caracteristicas": "base",
        })

        for i, hiperparametros in enumerate(combinaciones):
            with mlflow.start_run(run_name=f"P1-hgb-comb-{i:02d}", nested=True):
                modelo, entrenamiento_s = medir(
                    entrenar_hgb, datos.entrenamiento, COLUMNAS_P1, hiperparametros
                )
                metricas, _, _ = evaluar(modelo, COLUMNAS_P1, datos, ("val",))
                mlflow.log_params({**hiperparametros, "caracteristicas": "base"})
                mlflow.log_metrics({**metricas, "tiempo_entrenamiento_s": entrenamiento_s})
                mlflow.set_tags(
                    etiquetas(PREGUNTA, "hist_gradient_boosting", "search", commit)
                )
            print(f"  [busqueda] {i:02d} {hiperparametros} val_mae={metricas['val_mae']:.4f}",
                  flush=True)
            if mejor is None or metricas["val_mae"] < mejor[1]:
                mejor = (hiperparametros, metricas["val_mae"])

        mlflow.log_metric("mejor_val_mae", mejor[1])
        mlflow.log_dict(mejor[0], "config/mejores_hiperparametros.json")
        mlflow.log_params({f"mejor_{k}": v for k, v in mejor[0].items()})

    return mejor[0]


def entrenar_rf(entrenamiento, columnas):
    """Entrena el RandomForest de P1 con las columnas indicadas.

    Usa los mismos hiperparámetros de `config.py` que el modelo de la primera
    entrega, pero permite comparar la versión base con la ampliada.
    """
    modelo = RandomForestRegressor(
        n_estimators=P1_RF_N_ESTIMADORES,
        min_samples_leaf=P1_RF_MIN_SAMPLES_LEAF,
        max_samples=P1_RF_MAX_SAMPLES,
        n_jobs=-1,
        random_state=P1_SEMILLA,
    )
    return modelo.fit(entrenamiento[columnas], entrenamiento[OBJETIVO_P1])


def figuras_p1(titulo, prueba, prediccion, modelo=None, columnas=None):
    """Genera perfil horario, real vs. predicho, residuales e importancias."""
    real = prueba[OBJETIVO_P1].to_numpy()
    prediccion = np.asarray(prediccion)
    residual = real - prediccion
    figuras = []

    perfil = pd.DataFrame({
        "hora": prueba["hora"].to_numpy(), "real": real, "predicho": prediccion,
    }).groupby("hora").mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(perfil.index, perfil["real"], marker="o", label="Real")
    ax.plot(perfil.index, perfil["predicho"], marker="s", label="Predicho")
    ax.set(title=f"{titulo}: perfil horario (prueba)", xlabel="Hora del día",
           ylabel="Salidas promedio por estación-hora", xticks=range(0, 24, 2))
    ax.grid(alpha=0.2)
    ax.legend()
    figuras.append((fig, "perfil_horario.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    muestra = np.random.default_rng(P1_SEMILLA).choice(
        len(real), size=min(50_000, len(real)), replace=False
    )
    ax.hexbin(real[muestra], prediccion[muestra], gridsize=40, bins="log",
              mincnt=1, cmap="viridis")
    limite = max(real[muestra].max(), prediccion[muestra].max())
    ax.plot([0, limite], [0, limite], color="red", linewidth=1, label="Ideal")
    ax.set(title=f"{titulo}: real vs. predicho (prueba)", xlabel="Real",
           ylabel="Predicho")
    ax.legend()
    figuras.append((fig, "real_vs_predicho.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.clip(residual, -8, 8), bins=65, color="#4C78A8")
    ax.axvline(0, color="red", linewidth=1)
    ax.set(title=f"{titulo}: residuales (real - predicho, prueba)",
           xlabel="Residual (viajes por estación-hora)", ylabel="Frecuencia")
    figuras.append((fig, "residuales.png"))

    if modelo is not None and hasattr(modelo, "feature_importances_"):
        importancias = pd.Series(modelo.feature_importances_, index=columnas).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        importancias.plot.barh(ax=ax)
        ax.set(title=f"{titulo}: importancia de características", xlabel="Importancia")
        figuras.append((fig, "importancias.png"))

    return figuras


def registrar(nombre, familia, parametros, modelo, columnas, datos, commit, entrenamiento_s):
    """Evalúa un modelo ya entrenado y lo registra como candidato."""
    metricas, predicciones, prediccion_s = evaluar(modelo, columnas, datos)
    base = datos.metricas_baseline
    mejoras = {
        "mejora_val_mae_pct_vs_baseline":
            100 * (base["val_mae"] - metricas["val_mae"]) / base["val_mae"],
        "mejora_test_mae_pct_vs_baseline":
            100 * (base["test_mae"] - metricas["test_mae"]) / base["test_mae"],
    }
    registrar_candidato(
        nombre=nombre,
        pregunta=PREGUNTA,
        familia=familia,
        parametros={**datos.parametros, **parametros, "n_features": len(columnas)},
        metricas={**metricas, **mejoras},
        commit=commit,
        modelo=modelo,
        ejemplo=datos.entrenamiento[columnas],
        salida=modelo.predict(datos.entrenamiento[columnas].head(5)),
        entrenamiento=datos.entrenamiento[columnas + [OBJETIVO_P1]],
        objetivo=OBJETIVO_P1,
        figuras=figuras_p1(nombre, datos.prueba, predicciones["test"], modelo, columnas),
        tiempos={"tiempo_entrenamiento_s": entrenamiento_s,
                 "tiempo_prediccion_s": prediccion_s},
    )
    print(f"    val_mae={metricas['val_mae']:.4f} test_mae={metricas['test_mae']:.4f} "
          f"(baseline test_mae={base['test_mae']:.4f})", flush=True)


def ejecutar():
    """Corre la búsqueda y los cuatro candidatos de P1."""
    _, version = archivos_y_version()
    commit = commit_actual()
    print("P1: preparando datos...", flush=True)
    datos = preparar_datos(version, commit)
    print(f"P1: baseline val_mae={datos.metricas_baseline['val_mae']:.4f} "
          f"test_mae={datos.metricas_baseline['test_mae']:.4f}", flush=True)

    with mlflow.start_run(run_name="P1-experimentos") as umbrella:
        mlflow.set_tag("mlflow.runName", "P1-experimentos")
        mlflow.set_tags(etiquetas(PREGUNTA, "experimentos", "search", commit))
        mlflow.log_params(datos.parametros)

        print("P1: búsqueda de hiperparámetros (HistGradientBoosting)...", flush=True)
        mejor = buscar_hgb(datos, commit)

        hgb_parametros = {"algoritmo": "HistGradientBoostingRegressor",
                          "max_iter": P1_HGB_MAX_ITER,
                          "min_samples_leaf": P1_HGB_MIN_SAMPLES_LEAF, **mejor}
        rf_parametros = {"algoritmo": "RandomForestRegressor",
                         "n_estimators": P1_RF_N_ESTIMADORES,
                         "min_samples_leaf": P1_RF_MIN_SAMPLES_LEAF,
                         "max_samples": P1_RF_MAX_SAMPLES}

        print("P1: candidatos...", flush=True)
        for nombre, familia, parametros, columnas, tipo in [
            ("P1-rf-caracteristicas-base", "random_forest", rf_parametros,
             COLUMNAS_P1, "rf"),
            ("P1-rf-caracteristicas-ampliadas", "random_forest", rf_parametros,
             COLUMNAS_P1_AMPLIADAS, "rf"),
            ("P1-hgb-mejor-busqueda", "hist_gradient_boosting", hgb_parametros,
             COLUMNAS_P1, "hgb"),
            ("P1-hgb-caracteristicas-ampliadas", "hist_gradient_boosting",
             hgb_parametros, COLUMNAS_P1_AMPLIADAS, "hgb"),
        ]:
            if tipo == "rf":
                modelo, t = medir(entrenar_rf, datos.entrenamiento, columnas)
            else:
                modelo, t = medir(entrenar_hgb, datos.entrenamiento, columnas, mejor)
            registrar(nombre, familia,
                      {**parametros, "caracteristicas":
                       "ampliadas" if columnas is COLUMNAS_P1_AMPLIADAS else "base"},
                      modelo, columnas, datos, commit, t)

    print(f"P1 listo. Corrida principal: {umbrella.info.run_id}", flush=True)

