
"""Entrenamiento de P2: clasificación del riesgo estimado de desbalance.

Ejecutar desde la raíz del proyecto:
    python -m mibici.train_p2
"""

import subprocess

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from mibici.config import (
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    P2_RF_N_ESTIMATORS,
    P2_RF_MAX_DEPTH,
    P2_RF_MIN_SAMPLES_LEAF,
    P2_RF_MAX_SAMPLES,
    P2_RF_MAX_FEATURES,
)
from mibici.feature_engineering.classification import (
    COLUMNAS_P2,
    OBJETIVO_P2,
    asignar_etiqueta_p2,
    baseline_p2,
    cargar_capacidades,
    construir_panel_p2,
    crear_features_p2,
    definir_umbral_p2,
    dividir_temporal_p2,
    entrenar_random_forest_p2,
)
from mibici.train_p3 import version_datos


# ---------------------------------------------------------
# Configuración del experimento P2
# ---------------------------------------------------------

SEMILLA = 42

FECHA_INICIO = "2025-10-01"
FECHA_FIN = "2026-09-30"

INICIO_VALIDACION = "2026-06-01"
INICIO_PRUEBA = "2026-08-01"

CUANTIL_RIESGO = 0.90
UMBRAL_PREDICCION = 0.5

MAX_FILAS_ENTRENAMIENTO_RF = 250_000

# Mejores hiperparámetros encontrados:
# Grid Search de 72 combinaciones + evaluación de 5 finalistas.
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 30
RF_MIN_SAMPLES_LEAF = 10
RF_MAX_SAMPLES = 0.3
RF_MAX_FEATURES = 0.8


# ---------------------------------------------------------
# Métricas
# ---------------------------------------------------------

def calcular_metricas(prefijo, reales, predicciones, probabilidades):
    """Calcula las métricas de clasificación binaria."""

    return {
        f"{prefijo}_precision": float(
            precision_score(reales, predicciones, zero_division=0)
        ),
        f"{prefijo}_recall": float(
            recall_score(reales, predicciones, zero_division=0)
        ),
        f"{prefijo}_f1": float(
            f1_score(reales, predicciones, zero_division=0)
        ),
        f"{prefijo}_pr_auc": float(
            average_precision_score(reales, probabilidades)
        ),
    }


# ---------------------------------------------------------
# Visualización
# ---------------------------------------------------------

def guardar_confusion(reales, predicciones, nombre, titulo):
    """Genera y guarda la matriz de confusión."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / nombre

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=[0, 1],
    )

    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Sin riesgo", "Riesgo alto"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    visualizacion.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

    return ruta


# ---------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------

def main(
    model="rf",
    n_estimators=P2_RF_N_ESTIMATORS,
    max_depth=P2_RF_MAX_DEPTH,
    min_samples_leaf=P2_RF_MIN_SAMPLES_LEAF,
    max_samples=P2_RF_MAX_SAMPLES,
    max_features=P2_RF_MAX_FEATURES,
):

    # 1. Verificar los archivos de datos.
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))

    if len(archivos) != 12:
        raise ValueError(
            f"Se esperaban 12 archivos Parquet y se encontraron "
            f"{len(archivos)}."
        )

    # 2. Cargar viajes.
    print("P2: cargando viajes...", flush=True)

    viajes = pd.concat(
        [
            pd.read_parquet(
                archivo,
                columns=[
                    "inicio",
                    "fin",
                    "origen_id",
                    "destino_id",
                ],
            )
            for archivo in archivos
        ],
        ignore_index=True,
    )

    print(f"Viajes cargados: {len(viajes):,}", flush=True)

    # 3. Cargar nomenclatura de estaciones.
    catalogo = cargar_capacidades(
        EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"
    )

    # 4. Crear panel estación-hora.
    print("P2: construyendo panel horario...", flush=True)

    panel = construir_panel_p2(viajes, catalogo)
    del viajes

    # 5. Crear características históricas.
    print("P2: generando características...", flush=True)

    features = crear_features_p2(panel)
    del panel

    if features[COLUMNAS_P2].isna().any().any():
        raise ValueError(
            "Se encontraron valores faltantes en las features de P2."
        )

    # 6. Separación temporal.
    entrenamiento, validacion, prueba = dividir_temporal_p2(
        features,
        INICIO_VALIDACION,
        INICIO_PRUEBA,
    )

    del features

    print(
        "Filas entrenamiento/validación/prueba: "
        f"{len(entrenamiento):,} / "
        f"{len(validacion):,} / "
        f"{len(prueba):,}",
        flush=True,
    )

    # 7. Calcular el umbral de riesgo solo con entrenamiento.
    umbral_riesgo = definir_umbral_p2(
        entrenamiento,
        CUANTIL_RIESGO,
    )

    if umbral_riesgo <= 0:
        raise ValueError(
            "El umbral de riesgo no es positivo. "
            "Revisar la distribución del flujo neto."
        )

    # 8. Asignar etiquetas.
    entrenamiento = asignar_etiqueta_p2(
        entrenamiento,
        umbral_riesgo,
    )
    validacion = asignar_etiqueta_p2(
        validacion,
        umbral_riesgo,
    )
    prueba = asignar_etiqueta_p2(
        prueba,
        umbral_riesgo,
    )

    if entrenamiento[OBJETIVO_P2].nunique() != 2:
        raise ValueError(
            "El conjunto de entrenamiento necesita ambas clases."
        )

    # 9. Obtener la huella de los datos y el commit.
    print("P2: preparando registro en MLflow...", flush=True)

    version = version_datos(archivos)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        text=True,
    ).strip()

    # 10. Configurar MLflow.
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    parametros_comunes = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "inicio_validacion": INICIO_VALIDACION,
        "inicio_prueba": INICIO_PRUEBA,
        "semilla": SEMILLA,
        "cuantil_riesgo": CUANTIL_RIESGO,
        "umbral_riesgo": round(umbral_riesgo, 6),
        "data_version": version,
        "granularidad": "estacion-hora",
        "horizonte": "1 hora",
        "n_features": len(COLUMNAS_P2),
        "filas_entrenamiento": len(entrenamiento),
        "filas_validacion": len(validacion),
        "filas_prueba": len(prueba),
    }

    etiquetas = {
        "equipo": "31",
        "pregunta": "P2",
        "git_commit": commit,
    }

    # =====================================================
    # 11. BASELINE
    # =====================================================
    if model == "baseline":
        print("P2: evaluando baseline...", flush=True)

        pred_val_baseline = baseline_p2(
            entrenamiento,
            validacion,
        )

        pred_test_baseline = baseline_p2(
            entrenamiento,
            prueba,
        )

        metricas_baseline = {
            **calcular_metricas(
                "val",
                validacion[OBJETIVO_P2],
                pred_val_baseline,
                validacion["presion_lag_1"],
            ),
            **calcular_metricas(
                "test",
                prueba[OBJETIVO_P2],
                pred_test_baseline,
                prueba["presion_lag_1"],
            ),
        }

        figura_baseline = guardar_confusion(
            prueba[OBJETIVO_P2],
            pred_test_baseline,
            "p2_baseline_confusion.png",
            "P2 - Baseline de persistencia (prueba)",
        )

        with mlflow.start_run(
            run_name="P2-baseline-persistencia"
        ):
            mlflow.log_params({
                **parametros_comunes,
                "modelo": "persistencia_presion",
                "umbral_prediccion": umbral_riesgo,
            })

            mlflow.log_metrics(metricas_baseline)
            mlflow.set_tags({
                **etiquetas,
                "question": "p2",
                "model_family": "baseline",
                "stage_of_work": "baseline",
            })
            mlflow.log_artifact(str(figura_baseline))

        print("P2: baseline registrado en MLflow.", flush=True)

    # =====================================================
    # 12. RANDOM FOREST OPTIMIZADO
    # =====================================================
    elif model == "rf":
        print("P2: preparando muestra de entrenamiento...", flush=True)

        if len(entrenamiento) > MAX_FILAS_ENTRENAMIENTO_RF:
            muestra = entrenamiento.sample(
                n=MAX_FILAS_ENTRENAMIENTO_RF,
                random_state=SEMILLA,
            )
        else:
            muestra = entrenamiento

        print(
            f"P2: entrenando Random Forest con "
            f"{len(muestra):,} registros...",
            flush=True,
        )

        modelo = entrenar_random_forest_p2(
            muestra,
            semilla=SEMILLA,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            max_depth=max_depth,
            max_features=max_features,
        )

        # 13. Obtener probabilidades.
        print("P2: evaluando Random Forest...", flush=True)

        prob_val = modelo.predict_proba(
            validacion[COLUMNAS_P2]
        )[:, 1]

        prob_test = modelo.predict_proba(
            prueba[COLUMNAS_P2]
        )[:, 1]

        # 14. Aplicar el umbral fijo de clasificación.
        pred_val_rf = (
            prob_val >= UMBRAL_PREDICCION
        ).astype("int8")

        pred_test_rf = (
            prob_test >= UMBRAL_PREDICCION
        ).astype("int8")

        # 15. Calcular métricas.
        metricas_rf = {
            **calcular_metricas(
                "val",
                validacion[OBJETIVO_P2],
                pred_val_rf,
                prob_val,
            ),
            **calcular_metricas(
                "test",
                prueba[OBJETIVO_P2],
                pred_test_rf,
                prob_test,
            ),
        }

        # 16. Guardar gráfica.
        figura_rf = guardar_confusion(
            prueba[OBJETIVO_P2],
            pred_test_rf,
            "p2_rf_optimizado_confusion.png",
            "P2 - Random Forest optimizado (prueba)",
        )

        # 17. Registrar experimento optimizado.
        with mlflow.start_run(
            run_name="P2-random-forest-optimizado"
        ):
            mlflow.log_params({
                **parametros_comunes,
                "modelo": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_samples": max_samples,
                "max_features": max_features,
                "class_weight": "balanced_subsample",
                "n_jobs": 2,
                "filas_entrenamiento_rf": len(muestra),
                "umbral_prediccion": UMBRAL_PREDICCION,
                "metodo_seleccion": (
                    "grid_72_combinaciones_5_finalistas"
                ),
            })

            mlflow.log_metrics(metricas_rf)
            mlflow.set_tags({
                **etiquetas,
                "question": "p2",
                "model_family": "random_forest_classifier",
                "stage_of_work": "candidate",
            })
            mlflow.log_artifact(str(figura_rf))

    else:
        raise ValueError(f"Modelo no soportado para P2: {model}")

    # =====================================================
    # 18. RESULTADOS
    # =====================================================
    
    print("\n" + "=" * 55)
    print(f"RESULTADOS P2 - {model.upper()}")
    print("=" * 55)

    print(f"Umbral de presión: {umbral_riesgo:.4f}")

    if model == "baseline":
        metricas = metricas_baseline
    else:
        metricas = metricas_rf
        print(f"Registros utilizados por RF: {len(muestra):,}")

    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")

    print(f"\nModelo {model} registrado en MLflow")
    print("Experimento:", MLFLOW_EXPERIMENT)


if __name__ == "__main__":
    main()
