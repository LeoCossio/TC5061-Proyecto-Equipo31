"""Compara baselines y candidatos del experimento y elige el mejor por pregunta.

Lee las corridas de MLflow (servidor indicado en MLFLOW_TRACKING_URI, o la base
local `mlflow.db`) y genera `reports/comparacion_experimentos.md` con una tabla
por pregunta. Los criterios de selección se definieron antes de ver los
resultados de prueba y usan solo métricas de validación:

  P1: menor val_mae (MAE es la métrica ligada a la decisión de planeación).
  P2: mayor val_f1 (equilibra recall y precision de la clase de riesgo).
  P3: mayor silhouette_espacio_base (mismo espacio que el baseline), con el
      grupo más pequeño con al menos 5 % de las estaciones.

Uso desde la raíz del repositorio:
  python scripts/comparar_experimentos.py            # solo genera el reporte
  python scripts/comparar_experimentos.py --marcar   # además etiqueta al elegido
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mibici.config import MLFLOW_EXPERIMENT  # noqa: E402

SALIDA = ROOT / "reports" / "comparacion_experimentos.md"

PREGUNTAS = {
    "p1": {
        "titulo": "P1 - Regresión de demanda (viajes por estación-hora)",
        "criterio": "menor `val_mae`",
        "metricas": ["val_mae", "test_mae", "test_rmse", "test_mae_horas_pico"],
        "seleccion": ("val_mae", "min"),
        "mejora": ("test_mae", "min"),
        "hiper": ["algoritmo", "caracteristicas", "loss", "learning_rate",
                  "max_leaf_nodes", "n_estimators", "min_samples_leaf", "max_samples"],
    },
    "p2": {
        "titulo": "P2 - Clasificación de riesgo estimado de desbalance",
        "criterio": "mayor `val_f1`",
        "metricas": ["val_f1", "test_precision", "test_recall", "test_f1", "test_pr_auc"],
        "seleccion": ("val_f1", "max"),
        "mejora": ("test_f1", "max"),
        "hiper": ["algoritmo", "learning_rate", "max_leaf_nodes", "l2_regularization",
                  "n_estimators", "max_depth", "C", "umbral_prediccion"],
    },
    "p3": {
        "titulo": "P3 - Clustering de estaciones",
        "criterio": "mayor `silhouette_espacio_base` con grupo mínimo >= 5 %",
        "metricas": ["silhouette_espacio_base", "train_silhouette_score",
                     "davies_bouldin", "grupo_mas_pequeno_pct"],
        "seleccion": ("silhouette_espacio_base", "max"),
        "mejora": ("silhouette_espacio_base", "max"),
        "hiper": ["algoritmo", "n_grupos", "caracteristicas", "covariance_type"],
    },
}
COSTO = ["tiempo_entrenamiento_s", "tiempo_prediccion_s"]


def cargar_corridas():
    """Descarga las corridas del experimento como DataFrame."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    corridas = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT], max_results=5000, output_format="pandas"
    )
    return corridas


def valor(fila, columna):
    """Lee una métrica, parámetro o etiqueta; devuelve NaN si no existe."""
    for prefijo in ("metrics.", "params.", "tags."):
        if prefijo + columna in fila.index and pd.notna(fila[prefijo + columna]):
            return fila[prefijo + columna]
    return float("nan")


def formato(x):
    if isinstance(x, float):
        return "—" if pd.isna(x) else f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}"
    return "—" if x is None else str(x)


def tabla_markdown(filas, columnas):
    """Convierte una lista de diccionarios en una tabla Markdown."""
    encabezado = "| " + " | ".join(columnas) + " |"
    separador = "|" + "|".join("---" for _ in columnas) + "|"
    cuerpo = ["| " + " | ".join(formato(f.get(c)) for c in columnas) + " |" for f in filas]
    return "\n".join([encabezado, separador, *cuerpo])


def principal(marcar):
    corridas = cargar_corridas()
    corridas = corridas[corridas["status"] == "FINISHED"]
    partes = [
        "# Comparación de experimentos — Equipo 31\n",
        "Generado con `python scripts/comparar_experimentos.py`. "
        "Las métricas `val_` se usan para elegir; `test_` se reporta una sola vez.\n",
    ]
    elegidos = {}

    for pregunta, cfg in PREGUNTAS.items():
        base = corridas[corridas["tags.question"] == pregunta]
        baseline = base[base["tags.stage_of_work"] == "baseline"].sort_values(
            "start_time", ascending=False)
        candidatos = base[base["tags.stage_of_work"] == "candidate"].sort_values(
            "start_time", ascending=False).drop_duplicates("tags.mlflow.runName")
        if baseline.empty or candidatos.empty:
            partes.append(f"## {cfg['titulo']}\n\nSin corridas suficientes.\n")
            continue
        baseline = baseline.iloc[0]

        filas = [{"corrida": "BASELINE " + str(baseline["tags.mlflow.runName"]),
                  **{c: valor(baseline, c) for c in cfg["hiper"] + cfg["metricas"] + COSTO},
                  "mejora_vs_baseline_%": 0.0, "brecha": float("nan")}]

        metrica_sel, sentido = cfg["seleccion"]
        metrica_mejora, sentido_mejora = cfg["mejora"]
        if pregunta == "p3":
            # El baseline registró su silhouette con el nombre de la primera entrega.
            filas[0]["silhouette_espacio_base"] = valor(baseline, "train_silhouette_score")
        if pregunta == "p2":
            # El umbral del baseline es de presión, no de probabilidad: no es comparable.
            filas[0]["umbral_prediccion"] = float("nan")
        ref = float(filas[0][metrica_mejora])
        for _, fila in candidatos.iterrows():
            actual = float(valor(fila, metrica_mejora))
            mejora = 100 * ((ref - actual) if sentido_mejora == "min" else (actual - ref)) / abs(ref)
            val = float(valor(fila, metrica_sel))
            brecha = float("nan")
            if pregunta in ("p1", "p2"):
                prefijo = metrica_sel.replace("val_", "")
                brecha = float(valor(fila, "test_" + prefijo)) - val
            filas.append({"corrida": fila["tags.mlflow.runName"],
                          "run_id": fila["run_id"],
                          **{c: valor(fila, c) for c in cfg["hiper"] + cfg["metricas"] + COSTO},
                          "mejora_vs_baseline_%": mejora, "brecha": brecha})

        elegibles = [f for f in filas[1:]
                     if pregunta != "p3" or float(f["grupo_mas_pequeno_pct"]) >= 5]
        clave = lambda f: float(f[metrica_sel])  # noqa: E731
        elegido = (min if sentido == "min" else max)(elegibles, key=clave)
        elegidos[pregunta] = elegido

        columnas = ["corrida", *[c for c in cfg["hiper"]
                                 if any(pd.notna(f.get(c)) for f in filas)],
                    *cfg["metricas"], "mejora_vs_baseline_%", *COSTO]
        partes.append(f"## {cfg['titulo']}\n")
        partes.append(f"Criterio de selección: {cfg['criterio']}.\n")
        if pregunta == "p2":
            partes.append("Nota: el umbral de `P2-hgb-umbral-validacion` se ajusta sobre "
                          "validación, así que su `val_f1` es optimista; la comparación "
                          "imparcial está en las columnas `test_`.\n")
        partes.append(tabla_markdown(filas, columnas) + "\n")
        partes.append(f"**Elegido:** `{elegido['corrida']}` "
                      f"({metrica_sel} = {formato(float(elegido[metrica_sel]))}, "
                      f"mejora relativa en {metrica_mejora} vs. baseline = {elegido['mejora_vs_baseline_%']:.1f} %).\n")

    SALIDA.write_text("\n".join(partes), encoding="utf-8")
    print(f"Reporte escrito en {SALIDA.relative_to(ROOT)}")

    if marcar:
        cliente = mlflow.MlflowClient()
        for pregunta, elegido in elegidos.items():
            cliente.set_tag(elegido["run_id"], "selected_for_production", "true")
            print(f"Etiquetado {pregunta}: {elegido['corrida']} ({elegido['run_id']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--marcar", action="store_true",
                        help="Etiqueta la corrida elegida con selected_for_production=true")
    principal(parser.parse_args().marcar)
