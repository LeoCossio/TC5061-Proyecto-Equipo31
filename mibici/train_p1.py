"""Ejecuta y registra el baseline y el RandomForest de P1 en MLflow.

Ejecutar desde la raíz del proyecto:
    python -m mibici.train_p1
"""

import subprocess

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from mibici.config import (
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    P1_FECHA_FIN,
    P1_FECHA_INICIO,
    P1_FECHA_INICIO_PRUEBA,
    P1_FECHA_INICIO_VALIDACION,
    P1_RF_MAX_SAMPLES,
    P1_RF_MIN_SAMPLES_LEAF,
    P1_RF_N_ESTIMADORES,
    P1_SEMILLA,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
)
from mibici.features import (
    COLUMNAS_P1,
    OBJETIVO_P1,
    baseline_p1,
    construir_panel_horario,
    crear_features_p1,
    dividir_temporal_p1,
    entrenar_random_forest_p1,
    es_hora_pico,
)
from mibici.train_p3 import version_datos


def calcular_metricas(prefijo, conjunto, prediccion):
    """Calcula MAE, RMSE y MAE en horas pico para un conjunto evaluado."""
    real = conjunto[OBJETIVO_P1]
    pico = es_hora_pico(conjunto)

    return {
        f"{prefijo}_mae": mean_absolute_error(real, prediccion),
        f"{prefijo}_rmse": root_mean_squared_error(real, prediccion),
        f"{prefijo}_mae_horas_pico": mean_absolute_error(
            real[pico], prediccion[pico]
        ),
    }


def guardar_grafica_perfil_horario(nombre, prueba, prediccion, titulo):
    """Compara la demanda real y la predicha por hora del día en prueba."""
    ruta = FIGURES_DIR / nombre
    perfil = (
        pd.DataFrame({
            "hora": prueba["hora"].to_numpy(),
            "real": prueba[OBJETIVO_P1].to_numpy(),
            "predicho": np.asarray(prediccion),
        })
        .groupby("hora")
        .mean()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(perfil.index, perfil["real"], marker="o", label="Real")
    ax.plot(perfil.index, perfil["predicho"], marker="s", label="Predicho")
    ax.set_title(titulo)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Salidas promedio por estación-hora")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def guardar_grafica_importancias(modelo):
    """Guarda la importancia de cada característica del RandomForest."""
    ruta = FIGURES_DIR / "p1_rf_importancias.png"
    importancias = pd.Series(
        modelo.feature_importances_, index=COLUMNAS_P1
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importancias.plot.barh(ax=ax)
    ax.set_title("P1 - Importancia de características (RandomForest)")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def main():
    # 1. Cargar los mismos 12 meses de datos que usa el resto del equipo.
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    if len(archivos) != 12:
        raise ValueError(f"Se esperaban 12 Parquet, hay {len(archivos)}")

    df = pd.concat(
        [pd.read_parquet(archivo, columns=["origen_id", "inicio"]) for archivo in archivos],
        ignore_index=True,
    )

    # 2. Serie horaria por estación, características y partición cronológica.
    panel = construir_panel_horario(df)
    features = crear_features_p1(panel)
    entrenamiento, validacion, prueba = dividir_temporal_p1(
        features, P1_FECHA_INICIO_VALIDACION, P1_FECHA_INICIO_PRUEBA
    )
    media_global = float(entrenamiento[OBJETIVO_P1].mean())

    # 3. Versión de datos y commit para los dos runs.
    version = version_datos(archivos)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        text=True,
    ).strip()

    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    parametros_comunes = {
        "fecha_inicio": P1_FECHA_INICIO,
        "fecha_fin": P1_FECHA_FIN,
        "fecha_inicio_validacion": P1_FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": P1_FECHA_INICIO_PRUEBA,
        "data_version": version,
        "semilla": P1_SEMILLA,
        "granularidad": "estacion-hora",
        "horizonte": "1 hora",
        "n_estaciones": panel["origen_id"].nunique(),
        "filas_entrenamiento": len(entrenamiento),
        "filas_validacion": len(validacion),
        "filas_prueba": len(prueba),
        "n_features": len(COLUMNAS_P1),
    }
    etiquetas = {
        "equipo": "31",
        "pregunta": "P1",
        "git_commit": commit,
    }

    # 4. Baseline: media histórica por estación, día de la semana y hora.
    pred_val_base = baseline_p1(validacion, media_global)
    pred_test_base = baseline_p1(prueba, media_global)
    metricas_base = {
        **calcular_metricas("val", validacion, pred_val_base),
        **calcular_metricas("test", prueba, pred_test_base),
    }
    figura_base = guardar_grafica_perfil_horario(
        "p1_baseline_perfil_horario.png",
        prueba,
        pred_test_base,
        "P1 - Baseline: demanda real vs. media histórica (prueba)",
    )

    with mlflow.start_run(run_name="P1-baseline-media-historica"):
        mlflow.log_params({
            **parametros_comunes,
            "modelo": "media_historica_estacion_dow_hora",
        })
        mlflow.log_metrics(metricas_base)
        mlflow.set_tags(etiquetas)
        mlflow.log_artifact(str(figura_base))

    # 5. RandomForestRegressor con las mismas características y particiones.
    modelo = entrenar_random_forest_p1(
        entrenamiento,
        semilla=P1_SEMILLA,
        n_estimators=P1_RF_N_ESTIMADORES,
        min_samples_leaf=P1_RF_MIN_SAMPLES_LEAF,
        max_samples=P1_RF_MAX_SAMPLES,
    )
    pred_val_rf = modelo.predict(validacion[COLUMNAS_P1])
    pred_test_rf = modelo.predict(prueba[COLUMNAS_P1])
    metricas_rf = {
        **calcular_metricas("val", validacion, pred_val_rf),
        **calcular_metricas("test", prueba, pred_test_rf),
    }
    figura_rf = guardar_grafica_perfil_horario(
        "p1_rf_perfil_horario.png",
        prueba,
        pred_test_rf,
        "P1 - RandomForest: demanda real vs. predicha (prueba)",
    )
    figura_importancias = guardar_grafica_importancias(modelo)

    with mlflow.start_run(run_name="P1-random-forest"):
        mlflow.log_params({
            **parametros_comunes,
            "modelo": "RandomForestRegressor",
            "n_estimators": P1_RF_N_ESTIMADORES,
            "min_samples_leaf": P1_RF_MIN_SAMPLES_LEAF,
            "max_samples": P1_RF_MAX_SAMPLES,
        })
        mlflow.log_metrics(metricas_rf)
        mlflow.set_tags(etiquetas)
        mlflow.log_artifact(str(figura_rf))
        mlflow.log_artifact(str(figura_importancias))

    print(f"Filas entrenamiento/validación/prueba: "
          f"{len(entrenamiento):,} / {len(validacion):,} / {len(prueba):,}")
    for nombre, metricas in (("baseline", metricas_base), ("RandomForest", metricas_rf)):
        detalle = ", ".join(f"{k}={v:.4f}" for k, v in metricas.items())
        print(f"{nombre}: {detalle}")
    print("Ambos runs registrados en MLflow")


if __name__ == "__main__":
    main()
