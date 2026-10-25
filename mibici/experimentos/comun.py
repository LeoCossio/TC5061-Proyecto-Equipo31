"""Utilidades compartidas por los experimentos de P1, P2 y P3.

Centraliza la conexión con MLflow, las etiquetas obligatorias y el registro
de artefactos de las corridas candidatas, para que las tres preguntas
documenten sus resultados con la misma estructura:

    plots/    gráficas de desempeño
    data/     manifiesto con la versión de los datos
    config/   configuración y parámetros con que se ejecutó la corrida
    model/    modelo con firma (signature) y ejemplo de entrada
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from mibici.config import (  # noqa: E402
    EQUIPO_ID,
    MLFLOW_EXPERIMENT,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
)
from mibici.train_p3 import version_datos  # noqa: E402

MANIFIESTO = PROJ_ROOT / "references" / "manifest_datos.json"
CONFIG_PY = PROJ_ROOT / "mibici" / "config.py"


def conectar_mlflow():
    """Selecciona el experimento del equipo en el servidor de tracking.

    Usa `MLFLOW_TRACKING_URI` si está definida (por ejemplo, el servidor local
    en http://127.0.0.1:5000). Si no, usa la misma base `mlflow.db` del
    proyecto, para que las corridas queden en el mismo lugar.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def commit_actual():
    """Devuelve el hash del commit con el que se ejecuta la corrida."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
    ).strip()


def archivos_y_version():
    """Lista los Parquet del periodo y calcula la huella SHA-256 de los datos."""
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    if len(archivos) != 12:
        raise ValueError(f"Se esperaban 12 Parquet, hay {len(archivos)}")
    return archivos, version_datos(archivos)


def etiquetas(pregunta, familia, etapa, commit):
    """Etiquetas para filtrar y comparar corridas en la interfaz de MLflow."""
    return {
        "equipo": EQUIPO_ID,
        "pregunta": pregunta.upper(),
        "question": pregunta.lower(),
        "model_family": familia,
        "stage_of_work": etapa,
        "git_commit": commit,
    }


def medir(funcion, *args, **kwargs):
    """Ejecuta una función y devuelve su resultado y los segundos que tardó."""
    inicio = time.perf_counter()
    resultado = funcion(*args, **kwargs)
    return resultado, time.perf_counter() - inicio


def guardar_figura(fig, nombre, carpeta="plots"):
    """Guarda una figura como artefacto de la corrida activa y la cierra."""
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / nombre
        fig.tight_layout()
        fig.savefig(ruta, dpi=130)
        mlflow.log_artifact(str(ruta), artifact_path=carpeta)
    plt.close(fig)


def registrar_artefactos_base(parametros):
    """Registra la versión de los datos y la configuración de la corrida."""
    if MANIFIESTO.exists():
        mlflow.log_artifact(str(MANIFIESTO), artifact_path="data")
    mlflow.log_artifact(str(CONFIG_PY), artifact_path="config")
    mlflow.log_dict(
        {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
         for k, v in parametros.items()},
        "config/parametros_corrida.json",
    )


def registrar_dataset(entrenamiento, nombre, objetivo):
    """Asocia el conjunto de entrenamiento a la corrida con `log_input`.

    MLflow guarda el esquema, el resumen estadístico y una huella del
    contenido, no las filas, de modo que no se copian datos al servidor.
    """
    dataset = mlflow.data.from_pandas(
        entrenamiento,
        source=str(PROCESSED_DATA_DIR),
        name=nombre,
        targets=objetivo,
    )
    mlflow.log_input(dataset, context="training")


def registrar_modelo(modelo, ejemplo, salida):
    """Registra el modelo con su firma y un ejemplo de entrada."""
    ejemplo = ejemplo.head(5).astype("float64")
    firma = infer_signature(ejemplo, salida[:5])
    # cloudpickle: el formato skops (por defecto) rechaza los árboles de scikit-learn
    # como tipo no confiable, así que no puede guardar Random Forest ni Boosting.
    mlflow.sklearn.log_model(
        modelo,
        name="model",
        signature=firma,
        input_example=ejemplo,
        serialization_format="cloudpickle",
    )


def registrar_candidato(
    nombre,
    pregunta,
    familia,
    parametros,
    metricas,
    commit,
    modelo,
    ejemplo,
    salida,
    entrenamiento,
    objetivo,
    figuras,
    tiempos=None,
):
    """Registra una corrida candidata completa dentro de la corrida activa.

    `figuras` es una lista de (figura, nombre_de_archivo) que se guarda en
    `plots/`. `tiempos` permite registrar el costo de entrenamiento e inferencia.
    """
    with mlflow.start_run(run_name=nombre, nested=True) as corrida:
        mlflow.log_params(parametros)
        mlflow.log_metrics({**metricas, **(tiempos or {})})
        mlflow.set_tags(etiquetas(pregunta, familia, "candidate", commit))
        registrar_artefactos_base({**parametros, "modelo_nombre": nombre})
        registrar_dataset(entrenamiento, f"{pregunta}-entrenamiento", objetivo)
        for figura, archivo in figuras:
            guardar_figura(figura, archivo)
        registrar_modelo(modelo, ejemplo, salida)

    print(f"  [candidato] {nombre} -> {corrida.info.run_id}", flush=True)
    return corrida.info.run_id


def resumen_json(datos):
    """Serializa un diccionario de forma estable para guardarlo como parámetro."""
    return json.dumps(datos, sort_keys=True, default=str)

