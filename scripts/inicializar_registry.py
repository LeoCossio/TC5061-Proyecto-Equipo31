"""Publica los modelos incluidos con Docker Compose."""

from pathlib import Path

import mlflow
from mlflow import MlflowClient

MODELOS = [
    ("p1_v1", "mibici-equipo31-p1-demanda", 1),
    ("p1_v2", "mibici-equipo31-p1-demanda", 2),
    ("p2", "mibici-equipo31-p2-riesgo", 1),
    ("p3", "mibici-equipo31-p3-grupos", 1),
]


def main():
    cliente = MlflowClient()

    for clave, nombre, numero in MODELOS:
        versiones = cliente.search_model_versions(
            f"name = '{nombre}'"
        )

        if any(
            int(v.version) == numero and v.status == "READY"
            for v in versiones
        ):
            print(f"{nombre} v{numero} disponible")
            continue

        siguiente = max(
            (int(v.version) for v in versiones),
            default=0,
        ) + 1

        if siguiente != numero:
            raise RuntimeError(
                f"Version inesperada para {nombre}: {siguiente}"
            )

        ruta = Path("/modelos") / clave

        if not (ruta / "MLmodel").is_file():
            raise FileNotFoundError(ruta)

        mlflow.set_experiment("mibici-equipo31-bootstrap")

        with mlflow.start_run(run_name=f"importar-{clave}"):
            mlflow.log_artifacts(str(ruta), artifact_path="model")
            run_id = mlflow.active_run().info.run_id

            version = mlflow.register_model(
                f"runs:/{run_id}/model",
                nombre,
            )

        if int(version.version) != numero or version.status != "READY":
            raise RuntimeError(f"No se pudo registrar {nombre}")

    nombre_p1 = "mibici-equipo31-p1-demanda"
    aliases = cliente.get_registered_model(nombre_p1).aliases

    if "champion" not in aliases:
        cliente.set_registered_model_alias(
            nombre_p1, "champion", "2"
        )

    if "challenger" not in aliases:
        cliente.set_registered_model_alias(
            nombre_p1, "challenger", "1"
        )

    etiquetas = {
        "git_commit": "93a866e7df3a1ff1d011d4da6320b76e9336f799",
        "data_version": "d78c6bcfdf6ecb0389084c4c8f116d319fb0f225665a1cb850e34b83dc524454",
        "ventana_entrenamiento": "2025-10-08 a 2026-08-31",
        "evaluacion": "septiembre de 2026",
        "run_id_entrenamiento_original": "cc62cc1d5555443eb6698910a0a7ef22",
        "version_comparada": "1",
        "origen": "modelos_registry/p1_v2",
    }

    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(
            nombre_p1, "2", clave, valor
        )

    print("Registry inicializado correctamente")


if __name__ == "__main__":
    main()