
import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from mibici.config import DATA_DIR, REPORTS_DIR, P1_SEMILLA
from mibici.features import COLUMNAS_P1_AMPLIADAS


def main():
    carpeta = DATA_DIR / "monitoring"

    referencia = pd.read_parquet(carpeta / "p1_referencia.parquet")
    actual = pd.read_parquet(carpeta / "p1_actual.parquet")

    # Conservar solo estaciones presentes en ambos periodos.
    estaciones_comunes = set(referencia["origen_id"]) & set(actual["origen_id"])

    referencia = referencia[
        referencia["origen_id"].isin(estaciones_comunes)
    ].copy()

    actual = actual[
        actual["origen_id"].isin(estaciones_comunes)
    ].copy()

    print(f"Estaciones comunes: {len(estaciones_comunes)}")

    # Muestreo reproducible.
    referencia = referencia.sample(
        n=min(50_000, len(referencia)), random_state=P1_SEMILLA
    )
    actual = actual.sample(
        n=min(50_000, len(actual)), random_state=P1_SEMILLA
    )

    referencia = referencia[COLUMNAS_P1_AMPLIADAS].copy()
    actual = actual[COLUMNAS_P1_AMPLIADAS].copy()

    categoricas = ["hora", "dia_semana", "fin_semana"]
    numericas = [
        columna for columna in COLUMNAS_P1_AMPLIADAS
        if columna not in categoricas
    ]

    for columna in categoricas:
        referencia[columna] = referencia[columna].astype("int32").astype(str)
        actual[columna] = actual[columna].astype("int32").astype(str)

    esquema = DataDefinition(
        numerical_columns=numericas,
        categorical_columns=categoricas,
    )

    reporte = Report([
        DataDriftPreset(method="psi"),
        DataSummaryPreset(),
    ])

    resultado = reporte.run(
        Dataset.from_pandas(actual, data_definition=esquema),
        Dataset.from_pandas(referencia, data_definition=esquema),
    )

    salida = REPORTS_DIR / "monitoring"
    salida.mkdir(parents=True, exist_ok=True)

    resultado.save_html(str(salida / "p1_drift_estaciones_comunes.html"))

    with open(salida / "p1_drift_estaciones_comunes.json", "w", encoding="utf-8") as f:
        f.write(resultado.json())

    print("Reporte de estaciones comunes generado correctamente.")


if __name__ == "__main__":
    main()
