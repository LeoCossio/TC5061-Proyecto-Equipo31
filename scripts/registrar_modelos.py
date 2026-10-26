
"""Registro reproducible de los modelos seleccionados del Equipo 31.

Uso:
    python scripts/registrar_modelos.py

Requiere que los experimentos candidatos se hayan ejecutado previamente
y que MLFLOW_TRACKING_URI apunte al servidor correspondiente.
"""

import os

import mlflow
from mlflow import MlflowClient

EXPERIMENTO = "mibici-equipo31"

GANADORES = {
    "p1": {
        "run_name": "P1-hgb-caracteristicas-ampliadas",
        "registry_name": "mibici-equipo31-p1-demanda",
        "metric": "val_mae",
        "description": (
            "Prediccion de demanda horaria por estacion mediante "
            "HistGradientBoostingRegressor con caracteristicas ampliadas. "
            "Seleccionado por menor MAE de validacion."
        ),
    },
    "p2": {
        "run_name": "P2-hgb-umbral-validacion",
        "registry_name": "mibici-equipo31-p2-riesgo",
        "metric": "val_f1",
        "description": (
            "Clasificacion del riesgo estimado de desbalance por estacion-hora "
            "mediante HistGradientBoostingClassifier. El umbral de decision "
            "se ajusto sobre validacion. El riesgo es un indicador indirecto, "
            "no una medicion del inventario real de bicicletas."
        ),
    },
    "p3": {
        "run_name": "P3-kmeans-mejor-k",
        "registry_name": "mibici-equipo31-p3-grupos",
        "metric": "silhouette_espacio_base",
        "description": (
            "Segmentacion historica de estaciones MiBici mediante KMeans "
            "con cuatro grupos. Permite identificar perfiles operativos "
            "de uso de las estaciones."
        ),
    },
}


def buscar_run(client, experiment_id, nombre):
    """Busca la ejecucion candidata mas reciente que haya terminado bien."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{nombre}'",
        output_format="list",
        max_results=1000,
    )

    runs = [
        run for run in runs
        if run.info.status == "FINISHED"
        and run.data.tags.get("stage_of_work") == "candidate"
    ]

    if not runs:
        raise RuntimeError(
            f"No se encontro una corrida candidata finalizada: {nombre}"
        )

    runs.sort(key=lambda run: run.info.start_time, reverse=True)
    return runs[0]


def buscar_modelo_guardado(client, experiment_id, run_id):
    """Localiza el LoggedModel de MLflow 3 asociado al run."""
    encontrados = []
    page_token = None

    while True:
        pagina = client.search_logged_models(
            experiment_ids=[experiment_id],
            page_token=page_token,
        )

        encontrados.extend(
            modelo for modelo in pagina
            if modelo.source_run_id == run_id
            and modelo.name == "model"
            and str(modelo.status).upper().endswith("READY")
        )

        page_token = getattr(pagina, "token", None)

        if not page_token:
            break

    if len(encontrados) != 1:
        raise RuntimeError(
            f"Se esperaban exactamente un LoggedModel listo para "
            f"el run {run_id}; encontrados: {len(encontrados)}"
        )

    return encontrados[0]


def registrar_o_reutilizar(client, nombre, modelo_id, descripcion):
    """Evita duplicados y no reemplaza un champion de origen diferente."""
    existente = None

    try:
        existente = client.get_registered_model(nombre)
    except mlflow.exceptions.MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise

    if existente is None:
        version = mlflow.register_model(
            model_uri=f"models:/{modelo_id}",
            name=nombre,
        )
        numero = version.version
        print(f"Registrado: {nombre}, version {numero}")
    else:
        versiones = client.search_model_versions(f"name = '{nombre}'")
        coincidentes = [
            v for v in versiones
            if v.source and modelo_id in v.source
        ]

        if coincidentes:
            numero = sorted(
                coincidentes,
                key=lambda v: int(v.version),
            )[-1].version
            print(f"Version existente reutilizada: {nombre} v{numero}")
        else:
            # No cambiar automaticamente una version en produccion.
            raise RuntimeError(
                f"{nombre} ya existe pero no contiene el modelo "
                f"{modelo_id}. Revisar manualmente antes de registrar."
            )

    client.update_registered_model(
        name=nombre,
        description=descripcion,
    )
    client.update_model_version(
        name=nombre,
        version=numero,
        description=descripcion,
    )

    try:
        champion_actual = client.get_model_version_by_alias(
            nombre, "champion"
        )
    except mlflow.exceptions.MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        champion_actual = None

    if champion_actual is not None and champion_actual.version != numero:
        raise RuntimeError(
            f"{nombre} ya tiene champion v{champion_actual.version}. "
            "No se modificara automaticamente."
        )

    client.set_registered_model_alias(
        name=nombre,
        alias="champion",
        version=numero,
    )

    return numero


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise RuntimeError(
            "Configura MLFLOW_TRACKING_URI antes de ejecutar el script."
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experimento = client.get_experiment_by_name(EXPERIMENTO)

    if experimento is None:
        raise RuntimeError(f"No existe el experimento {EXPERIMENTO}")

    for pregunta, config in GANADORES.items():
        run = buscar_run(
            client,
            experimento.experiment_id,
            config["run_name"],
        )

        modelo = buscar_modelo_guardado(
            client,
            experimento.experiment_id,
            run.info.run_id,
        )

        numero = registrar_o_reutilizar(
            client,
            config["registry_name"],
            modelo.model_id,
            config["description"],
        )

        client.set_model_version_tag(
            config["registry_name"],
            numero,
            "source_run_id",
            run.info.run_id,
        )

        for etiqueta in ("git_commit", "data_version"):
            valor = run.data.tags.get(etiqueta) or run.data.params.get(
                etiqueta
            )
            if valor:
                client.set_model_version_tag(
                    config["registry_name"],
                    numero,
                    etiqueta,
                    str(valor),
                )

        uri = f"models:/{config['registry_name']}@champion"
        cargado = mlflow.pyfunc.load_model(uri)

        print(
            f"OK {pregunta.upper()}: {config['registry_name']} "
            f"v{numero} | {type(cargado).__name__}"
        )

    print("Registro y verificacion finalizados.")


if __name__ == "__main__":
    main()
