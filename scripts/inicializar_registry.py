"""Publica las versiones de inferencia incluidas con Docker Compose."""

from pathlib import Path

import mlflow
from mlflow import MlflowClient

MODELOS = {
    "p2": "mibici-equipo31-p2-riesgo",
    "p3": "mibici-equipo31-p3-grupos",
}


def main():
    cliente = MlflowClient()

    for clave, nombre in MODELOS.items():
        versiones = cliente.search_model_versions(f"name = '{nombre}'")
        if any(v.version == "1" and v.status == "READY" for v in versiones):
            print(f"{nombre} v1 disponible")
            continue
        if versiones:
            raise RuntimeError(f"{nombre} ya tiene versiones, pero v1 no está lista")

        ruta = Path("/modelos") / clave
        if not (ruta / "MLmodel").is_file() or not (ruta / "model.pkl").is_file():
            raise FileNotFoundError(f"Artefacto MLflow incompleto: {ruta}")

        mlflow.set_experiment("mibici-equipo31-bootstrap")
        with mlflow.start_run(run_name=f"importar-{clave}-v1") as run:
            mlflow.log_artifacts(str(ruta), artifact_path="model")
            version = mlflow.register_model(f"runs:/{run.info.run_id}/model", nombre)

        if version.version != "1" or version.status != "READY":
            raise RuntimeError(f"{nombre} v1 no quedó disponible")


if __name__ == "__main__":
    main()