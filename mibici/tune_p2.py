
"""Búsqueda controlada de hiperparámetros para P2.

Ejecutar:
    python -m mibici.tune_p2
"""

import gc
import itertools
import json
import subprocess

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from mibici.config import (
    EXTERNAL_DATA_DIR,
    MLFLOW_EXPERIMENT,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
)
from mibici.feature_engineering.classification import (
    COLUMNAS_P2,
    OBJETIVO_P2,
    asignar_etiqueta_p2,
    cargar_capacidades,
    construir_panel_p2,
    crear_features_p2,
    definir_umbral_p2,
    dividir_temporal_p2,
)
from mibici.train_p3 import version_datos


SEMILLA = 42
MAX_FILAS = 100_000
UMBRAL_PREDICCION = 0.5


GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 15, 20, 30],
    "min_samples_leaf": [5, 10, 20],
    "max_features": ["sqrt", 0.8],
}



def main():
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    if len(archivos) != 12:
        raise ValueError("Se necesitan los 12 archivos Parquet.")

    print("Cargando datos P2...", flush=True)

    viajes = pd.concat(
        [
            pd.read_parquet(
                archivo,
                columns=["inicio", "fin", "origen_id", "destino_id"],
            )
            for archivo in archivos
        ],
        ignore_index=True,
    )

    catalogo = cargar_capacidades(
        EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"
    )
    panel = construir_panel_p2(viajes, catalogo)
    del viajes

    features = crear_features_p2(panel)
    del panel
    gc.collect()

    entrenamiento, validacion, prueba = dividir_temporal_p2(
        features, "2026-06-01", "2026-08-01"
    )
    del features

    umbral_riesgo = definir_umbral_p2(entrenamiento, 0.90)
    if umbral_riesgo <= 0:
        raise ValueError("El umbral de riesgo debe ser positivo.")

    entrenamiento = asignar_etiqueta_p2(entrenamiento, umbral_riesgo)
    validacion = asignar_etiqueta_p2(validacion, umbral_riesgo)
    prueba = asignar_etiqueta_p2(prueba, umbral_riesgo)

    muestra = entrenamiento.sample(
        n=min(MAX_FILAS, len(entrenamiento)),
        random_state=SEMILLA,
    )

    X_train = muestra[COLUMNAS_P2].to_numpy(dtype=np.float32)
    y_train = muestra[OBJETIVO_P2].to_numpy()
    X_val = validacion[COLUMNAS_P2].to_numpy(dtype=np.float32)
    y_val = validacion[OBJETIVO_P2].to_numpy()

    # No evaluamos prueba durante la búsqueda.
    del entrenamiento, validacion, prueba, muestra
    gc.collect()

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

 
    combinaciones = list(itertools.product(
        GRID["n_estimators"],
        GRID["max_depth"],
        GRID["min_samples_leaf"],
        GRID["max_features"],
    ))


    mejor_f1 = -1.0
    mejor_configuracion = None

    for numero, (arboles, profundidad, hojas, variables) in enumerate(
        combinaciones, start=1
    ):
        print(
            f"\n[{numero}/{len(combinaciones)}] "
            f"Árboles={arboles}, profundidad={profundidad}, "
            f"hojas={hojas}, max_features={variables}",
            flush=True,
        )

        modelo = RandomForestClassifier(
            n_estimators=arboles,
            max_depth=profundidad,
            min_samples_leaf=hojas,
            max_samples=0.3,
            class_weight="balanced_subsample",
            random_state=SEMILLA,
            n_jobs=2,
            max_features=variables,
        )

        modelo.fit(X_train, y_train)

        probabilidades = modelo.predict_proba(X_val)[:, 1]
        predicciones = (
            probabilidades >= UMBRAL_PREDICCION
        ).astype("int8")

        precision = precision_score(
            y_val, predicciones, zero_division=0
        )
        recall = recall_score(
            y_val, predicciones, zero_division=0
        )
        f1 = f1_score(y_val, predicciones, zero_division=0)
        pr_auc = average_precision_score(y_val, probabilidades)

        with mlflow.start_run(run_name=f"P2-grid-{numero:02d}"):
            mlflow.log_params({
                "modelo": "RandomForestClassifier",
                "n_estimators": arboles,
                "max_depth": profundidad,
                "min_samples_leaf": hojas,
                "max_samples": 0.3,
                "class_weight": "balanced_subsample",
                "n_jobs": 2,
                "semilla": SEMILLA,
                "filas_entrenamiento": len(X_train),
                "umbral_riesgo": umbral_riesgo,
                "umbral_prediccion": UMBRAL_PREDICCION,
                "data_version": version,
                "metodo_busqueda": "grid_temporal",
                "max_features": variables,
            })
            mlflow.log_metrics({
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "val_pr_auc": pr_auc,
            })
            mlflow.set_tags({
                "equipo": "31",
                "pregunta": "P2",
                "fase": "hyperparameter_tuning",
                "git_commit": commit,
            })

        print(
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"PR-AUC={pr_auc:.4f}",
            flush=True,
        )

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_configuracion = {
                "n_estimators": arboles,
                "max_depth": profundidad,
                "min_samples_leaf": hojas,
                "max_samples": 0.3,
                "max_features": variables,
            }

        del modelo, probabilidades, predicciones
        gc.collect()

    print("\nBÚSQUEDA COMPLETADA")
    print("Mejor F1 en validación:", round(mejor_f1, 4))
    print("Mejores hiperparámetros:", mejor_configuracion)

    # Guardamos el resultado fuera de Git para usarlo en el entrenamiento final.
    ruta = PROCESSED_DATA_DIR / "p2_best_params.json"
    ruta.write_text(
        json.dumps(
            {
                **mejor_configuracion,
                "val_f1": mejor_f1,
                "umbral_prediccion": UMBRAL_PREDICCION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Configuración guardada en:", ruta)


if __name__ == "__main__":
    main()
