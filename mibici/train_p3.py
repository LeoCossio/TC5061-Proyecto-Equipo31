"""Ejecuta y registra el baseline y K-Means de P3 en MLflow.

Ejecutar desde la raíz del proyecto:
    python -m mibici.train_p3
"""

import hashlib
import subprocess

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mibici.config import (
    PROJ_ROOT,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    P3_FECHA_INICIO,
    P3_FECHA_FIN,
    P3_SEMILLA,
    P3_MIN_SALIDAS,
    P3_N_CLUSTERS,
)
from mibici.features import (
    crear_features_p3,
    preprocesar_features_p3,
    baseline_p3,
    entrenar_kmeans_p3,
)


def version_datos(archivos):
    """Calcula la huella SHA-256 de los archivos Parquet usados en P3."""
    resumen = hashlib.sha256()

    for archivo in archivos:
        resumen.update(archivo.name.encode())
        with archivo.open("rb") as f:
            for bloque in iter(lambda: f.read(1024 * 1024), b""):
                resumen.update(bloque)

    return resumen.hexdigest()


def guardar_grafica_baseline(grupos):
    """Guarda la cantidad de estaciones por tercil de demanda."""
    ruta = FIGURES_DIR / "p3_baseline.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    grupos.value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("P3 - Estaciones por grupo de demanda")
    ax.set_xlabel("Grupo (0: baja, 1: media, 2: alta)")
    ax.set_ylabel("Número de estaciones")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def guardar_grafica_pca(escaladas, grupos):
    """Proyecta los grupos de K-Means en dos componentes para visualizarlos."""
    ruta = FIGURES_DIR / "p3_kmeans_pca.png"
    pca = PCA(n_components=2)
    coordenadas = pca.fit_transform(escaladas)

    fig, ax = plt.subplots(figsize=(9, 6))
    grafica = ax.scatter(
        coordenadas[:, 0],
        coordenadas[:, 1],
        c=grupos.to_numpy(),
        cmap="viridis",
        s=30,
        alpha=0.7,
    )
    ax.set_title("P3 - Segmentación de estaciones MiBici (K-Means)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.grid(alpha=0.2)
    fig.colorbar(grafica, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta, pca.explained_variance_ratio_.sum()


def main(model="kmeans", n_clusters=P3_N_CLUSTERS):
    # 1. Cargar los mismos 12 meses de datos para ambos métodos.
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    if len(archivos) != 12:
        raise ValueError(f"Se esperaban 12 Parquet, hay {len(archivos)}")

    df = pd.concat(
        [pd.read_parquet(archivo) for archivo in archivos],
        ignore_index=True,
    )

    # 2. Construir características una sola vez para comparaciones justas.
    features = crear_features_p3(df)
    escaladas, _scaler = preprocesar_features_p3(features)

    # 3. Obtener la versión de datos y el commit para los dos runs.
    version = version_datos(archivos)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        text=True,
    ).strip()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    parametros_comunes = {
        "fecha_inicio": P3_FECHA_INICIO,
        "fecha_fin": P3_FECHA_FIN,
        "data_version": version,
        "min_salidas": P3_MIN_SALIDAS,
        "semilla": P3_SEMILLA,
        "scaler": "StandardScaler",
        "transformacion": "log1p",
    }
    etiquetas = {
        "equipo": "31",
        "pregunta": "P3",
        "git_commit": commit,
    }


    # 4. Baseline: terciles de demanda
    if model == "baseline":
        grupos_baseline = baseline_p3(features)
        score_baseline = silhouette_score(escaladas, grupos_baseline)
        figura_baseline = guardar_grafica_baseline(grupos_baseline)

        with mlflow.start_run(run_name="P3-baseline-demanda"):
            mlflow.log_params({
                **parametros_comunes,
                "modelo": "terciles_demanda",
                "n_grupos": 3,
            })
            mlflow.log_metric("train_silhouette_score", score_baseline)
            mlflow.log_metric("n_estaciones", len(features))
            mlflow.set_tags({
                **etiquetas,
                "question": "p3",
                "model_family": "baseline",
                "stage_of_work": "baseline",
            })
            mlflow.log_artifact(str(figura_baseline))

        print(f"Silhouette baseline: {score_baseline:.4f}")

    # 5. K-Means
    elif model == "kmeans":
        grupos_kmeans, modelo_kmeans = entrenar_kmeans_p3(
            escaladas, n_clusters=n_clusters
        )

        score_kmeans = silhouette_score(escaladas, grupos_kmeans)
        figura_pca, varianza_pca = guardar_grafica_pca(
            escaladas, grupos_kmeans
        )

        with mlflow.start_run(run_name="...") as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            mlflow.log_params({
                **parametros_comunes,
                "modelo": "KMeans",
                "n_grupos": n_clusters,
                "n_init": modelo_kmeans.n_init,
            })
            mlflow.log_metric("train_silhouette_score", score_kmeans)
            mlflow.log_metric("n_estaciones", len(features))
            mlflow.log_metric("inertia", modelo_kmeans.inertia_)
            mlflow.log_metric("pca_2d_varianza_explicada", varianza_pca)
            mlflow.set_tags({
                **etiquetas,
                "question": "p3",
                "model_family": "kmeans",
                "stage_of_work": "candidate",
            })
            mlflow.log_artifact(str(figura_pca))

        print(f"Silhouette K-Means: {score_kmeans:.4f}")

    else:
        raise ValueError(f"Modelo no soportado para P3: {model}")

    print(f"Estaciones: {len(features)}")
    print("Corrida registrada en MLflow")

if __name__ == "__main__":
    main()
