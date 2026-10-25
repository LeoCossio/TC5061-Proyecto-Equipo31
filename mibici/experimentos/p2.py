"""Experimentos de P2 (riesgo estimado de desbalance, clasificación).

Cada ejecución registra, dentro de la corrida activa:
  1. Una búsqueda de hiperparámetros de HistGradientBoostingClassifier como
     nested runs (corrida padre y una hija por combinación).
  2. Cuatro candidatos: Random Forest de la primera entrega, regresión
     logística, HistGradientBoosting con la mejor combinación y la misma
     combinación con el umbral de decisión elegido en validación.

La etiqueta, el umbral de riesgo, la partición temporal y las métricas son
los mismos del baseline de persistencia.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    EXTERNAL_DATA_DIR,
    P2_HGB_ESPACIO,
    P2_HGB_MAX_ITER,
    P2_HGB_MIN_SAMPLES_LEAF,
    P2_LR_C,
    P2_RF_MAX_DEPTH,
    P2_RF_MAX_FEATURES,
    P2_RF_MAX_SAMPLES,
    P2_RF_MIN_SAMPLES_LEAF,
    P2_RF_N_ESTIMATORS,
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
from mibici.train_p2 import (
    CUANTIL_RIESGO,
    FECHA_FIN,
    FECHA_INICIO,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    MAX_FILAS_ENTRENAMIENTO_RF,
    SEMILLA,
    UMBRAL_PREDICCION,
    calcular_metricas,
)

PREGUNTA = "p2"


@dataclass
class DatosP2:
    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    prueba: pd.DataFrame
    parametros: dict
    metricas_baseline: dict


def preparar_datos(version):
    """Construye el panel, las características, el umbral y las etiquetas."""
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    viajes = pd.concat(
        [pd.read_parquet(a, columns=["inicio", "fin", "origen_id", "destino_id"])
         for a in archivos],
        ignore_index=True,
    )
    catalogo = cargar_capacidades(EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv")
    panel = construir_panel_p2(viajes, catalogo)
    del viajes
    features = crear_features_p2(panel)
    del panel

    entrenamiento, validacion, prueba = dividir_temporal_p2(
        features, INICIO_VALIDACION, INICIO_PRUEBA
    )
    del features

    umbral_riesgo = definir_umbral_p2(entrenamiento, CUANTIL_RIESGO)
    entrenamiento = asignar_etiqueta_p2(entrenamiento, umbral_riesgo)
    validacion = asignar_etiqueta_p2(validacion, umbral_riesgo)
    prueba = asignar_etiqueta_p2(prueba, umbral_riesgo)

    metricas_baseline = {
        **calcular_metricas("val", validacion[OBJETIVO_P2],
                            baseline_p2(entrenamiento, validacion),
                            validacion["presion_lag_1"]),
        **calcular_metricas("test", prueba[OBJETIVO_P2],
                            baseline_p2(entrenamiento, prueba),
                            prueba["presion_lag_1"]),
    }

    parametros = {
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
        "proporcion_riesgo_entrenamiento": round(float(entrenamiento[OBJETIVO_P2].mean()), 4),
    }
    return DatosP2(entrenamiento, validacion, prueba, parametros, metricas_baseline)


def entrenar_hgb(entrenamiento, hiperparametros):
    """Entrena un HistGradientBoostingClassifier con clases balanceadas."""
    modelo = HistGradientBoostingClassifier(
        max_iter=P2_HGB_MAX_ITER,
        min_samples_leaf=P2_HGB_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        early_stopping=False,
        random_state=SEMILLA,
        **hiperparametros,
    )
    return modelo.fit(entrenamiento[COLUMNAS_P2], entrenamiento[OBJETIVO_P2])


def entrenar_logistica(entrenamiento):
    """Regresión logística con variables estandarizadas y clases balanceadas."""
    modelo = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=P2_LR_C, class_weight="balanced", max_iter=300,
                           random_state=SEMILLA),
    )
    return modelo.fit(entrenamiento[COLUMNAS_P2], entrenamiento[OBJETIVO_P2])


def entrenar_rf(entrenamiento):
    """Random Forest de la primera entrega, con la muestra de entrenamiento original."""
    muestra = entrenamiento
    if len(entrenamiento) > MAX_FILAS_ENTRENAMIENTO_RF:
        muestra = entrenamiento.sample(n=MAX_FILAS_ENTRENAMIENTO_RF, random_state=SEMILLA)
    return entrenar_random_forest_p2(
        muestra,
        semilla=SEMILLA,
        n_estimators=P2_RF_N_ESTIMATORS,
        min_samples_leaf=P2_RF_MIN_SAMPLES_LEAF,
        max_samples=P2_RF_MAX_SAMPLES,
        max_depth=P2_RF_MAX_DEPTH,
        max_features=P2_RF_MAX_FEATURES,
    )


def probabilidades(modelo, conjunto):
    """Probabilidad de la clase de riesgo y segundos que tardó en calcularse."""
    return medir(lambda: modelo.predict_proba(conjunto[COLUMNAS_P2])[:, 1])


def umbral_maximo_f1(real, probabilidad):
    """Umbral que maximiza F1 en el conjunto dado (se usa solo con validación)."""
    precision, recall, umbrales = precision_recall_curve(real, probabilidad)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(umbrales[int(np.argmax(f1))])


def buscar_hgb(datos, commit):
    """Búsqueda en malla con nested runs; elige la combinación de mayor val_pr_auc.

    PR-AUC no depende del umbral de decisión, que se decide después en validación.
    """
    combinaciones = list(ParameterGrid(P2_HGB_ESPACIO))
    mejor = None

    with mlflow.start_run(run_name="P2-busqueda-hgb", nested=True):
        mlflow.set_tags(etiquetas(PREGUNTA, "hist_gradient_boosting", "search", commit))
        mlflow.log_params({
            **datos.parametros,
            "algoritmo": "HistGradientBoostingClassifier",
            "espacio_busqueda": resumen_json(P2_HGB_ESPACIO),
            "n_combinaciones": len(combinaciones),
            "criterio_seleccion": "val_pr_auc",
            "class_weight": "balanced",
        })

        for i, hiperparametros in enumerate(combinaciones):
            with mlflow.start_run(run_name=f"P2-hgb-comb-{i:02d}", nested=True):
                modelo, entrenamiento_s = medir(entrenar_hgb, datos.entrenamiento,
                                                hiperparametros)
                prob_val, _ = probabilidades(modelo, datos.validacion)
                metricas = calcular_metricas(
                    "val", datos.validacion[OBJETIVO_P2],
                    (prob_val >= UMBRAL_PREDICCION).astype("int8"), prob_val,
                )
                mlflow.log_params(hiperparametros)
                mlflow.log_metrics({**metricas, "tiempo_entrenamiento_s": entrenamiento_s})
                mlflow.set_tags(
                    etiquetas(PREGUNTA, "hist_gradient_boosting", "search", commit)
                )
            print(f"  [busqueda] {i:02d} {hiperparametros} "
                  f"val_pr_auc={metricas['val_pr_auc']:.4f} val_f1={metricas['val_f1']:.4f}",
                  flush=True)
            if mejor is None or metricas["val_pr_auc"] > mejor[1]:
                mejor = (hiperparametros, metricas["val_pr_auc"])

        mlflow.log_metric("mejor_val_pr_auc", mejor[1])
        mlflow.log_dict(mejor[0], "config/mejores_hiperparametros.json")
        mlflow.log_params({f"mejor_{k}": v for k, v in mejor[0].items()})

    return mejor[0]


def figuras_p2(titulo, real_test, prob_test, umbral):
    """Matriz de confusión y curva precisión-recall sobre el conjunto de prueba."""
    prediccion = (prob_test >= umbral).astype("int8")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(real_test, prediccion, labels=[0, 1]),
        display_labels=["Sin riesgo", "Riesgo alto"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{titulo}: matriz de confusión (prueba, umbral {umbral:.2f})")
    matriz = (fig, "matriz_confusion.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(real_test, prob_test, ax=ax)
    precision = float(((prediccion == 1) & (real_test == 1)).sum() / max(prediccion.sum(), 1))
    recall = float(((prediccion == 1) & (real_test == 1)).sum() / max(real_test.sum(), 1))
    ax.scatter([recall], [precision], color="red", zorder=3, label=f"Umbral {umbral:.2f}")
    ax.axhline(real_test.mean(), color="gray", linestyle="--", linewidth=1,
               label="Proporción de riesgo")
    ax.set_title(f"{titulo}: curva precisión-recall (prueba)")
    ax.legend()
    curva = (fig, "curva_precision_recall.png")

    return [matriz, curva]


def registrar(nombre, familia, parametros, modelo, datos, commit, entrenamiento_s,
              umbral=UMBRAL_PREDICCION, umbral_origen="fijo"):
    """Evalúa un modelo ya entrenado y lo registra como candidato."""
    prob_val, t_val = probabilidades(modelo, datos.validacion)
    prob_test, t_test = probabilidades(modelo, datos.prueba)
    metricas = {
        **calcular_metricas("val", datos.validacion[OBJETIVO_P2],
                            (prob_val >= umbral).astype("int8"), prob_val),
        **calcular_metricas("test", datos.prueba[OBJETIVO_P2],
                            (prob_test >= umbral).astype("int8"), prob_test),
    }
    base = datos.metricas_baseline
    deltas = {
        "delta_val_f1_vs_baseline": metricas["val_f1"] - base["val_f1"],
        "delta_test_f1_vs_baseline": metricas["test_f1"] - base["test_f1"],
        "delta_test_recall_vs_baseline": metricas["test_recall"] - base["test_recall"],
        "delta_test_precision_vs_baseline": metricas["test_precision"] - base["test_precision"],
    }
    registrar_candidato(
        nombre=nombre,
        pregunta=PREGUNTA,
        familia=familia,
        parametros={**datos.parametros, **parametros, "umbral_prediccion": round(umbral, 4),
                    "umbral_origen": umbral_origen},
        metricas={**metricas, **deltas},
        commit=commit,
        modelo=modelo,
        ejemplo=datos.entrenamiento[COLUMNAS_P2],
        salida=modelo.predict(datos.entrenamiento[COLUMNAS_P2].head(5)),
        entrenamiento=datos.entrenamiento[COLUMNAS_P2 + [OBJETIVO_P2]],
        objetivo=OBJETIVO_P2,
        figuras=figuras_p2(nombre, datos.prueba[OBJETIVO_P2].to_numpy(), prob_test, umbral),
        tiempos={"tiempo_entrenamiento_s": entrenamiento_s,
                 "tiempo_prediccion_s": t_val + t_test},
    )
    print(f"    val_f1={metricas['val_f1']:.4f} test_f1={metricas['test_f1']:.4f} "
          f"test_recall={metricas['test_recall']:.4f} test_precision={metricas['test_precision']:.4f} "
          f"(baseline test_f1={base['test_f1']:.4f})", flush=True)
    return prob_val


def ejecutar():
    """Corre la búsqueda y los cuatro candidatos de P2."""
    _, version = archivos_y_version()
    commit = commit_actual()
    print("P2: preparando datos...", flush=True)
    datos = preparar_datos(version)
    b = datos.metricas_baseline
    print(f"P2: baseline val_f1={b['val_f1']:.4f} test_f1={b['test_f1']:.4f}", flush=True)

    with mlflow.start_run(run_name="P2-experimentos") as umbrella:
        mlflow.set_tag("mlflow.runName", "P2-experimentos")
        mlflow.set_tags(etiquetas(PREGUNTA, "experimentos", "search", commit))
        mlflow.log_params(datos.parametros)

        print("P2: búsqueda de hiperparámetros (HistGradientBoosting)...", flush=True)
        mejor = buscar_hgb(datos, commit)

        hgb_parametros = {"algoritmo": "HistGradientBoostingClassifier",
                          "max_iter": P2_HGB_MAX_ITER,
                          "min_samples_leaf": P2_HGB_MIN_SAMPLES_LEAF,
                          "class_weight": "balanced", **mejor}

        print("P2: candidatos...", flush=True)
        modelo, t = medir(entrenar_rf, datos.entrenamiento)
        registrar("P2-rf-primera-entrega", "random_forest_classifier",
                  {"algoritmo": "RandomForestClassifier",
                   "n_estimators": P2_RF_N_ESTIMATORS, "max_depth": P2_RF_MAX_DEPTH,
                   "min_samples_leaf": P2_RF_MIN_SAMPLES_LEAF,
                   "max_samples": P2_RF_MAX_SAMPLES, "max_features": P2_RF_MAX_FEATURES,
                   "class_weight": "balanced_subsample",
                   "filas_entrenamiento_modelo": MAX_FILAS_ENTRENAMIENTO_RF},
                  modelo, datos, commit, t)

        modelo, t = medir(entrenar_logistica, datos.entrenamiento)
        registrar("P2-regresion-logistica", "logistic_regression",
                  {"algoritmo": "LogisticRegression", "C": P2_LR_C,
                   "class_weight": "balanced", "estandarizacion": "StandardScaler"},
                  modelo, datos, commit, t)

        modelo, t = medir(entrenar_hgb, datos.entrenamiento, mejor)
        prob_val = registrar("P2-hgb-mejor-busqueda", "hist_gradient_boosting",
                             hgb_parametros, modelo, datos, commit, t)

        umbral = umbral_maximo_f1(datos.validacion[OBJETIVO_P2].to_numpy(), prob_val)
        registrar("P2-hgb-umbral-validacion", "hist_gradient_boosting",
                  hgb_parametros, modelo, datos, commit, t,
                  umbral=umbral, umbral_origen="maximiza_f1_en_validacion")

    print(f"P2 listo. Corrida principal: {umbrella.info.run_id}", flush=True)
