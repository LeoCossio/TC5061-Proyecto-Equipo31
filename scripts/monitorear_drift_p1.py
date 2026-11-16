
import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from mibici.config import DATA_DIR, REPORTS_DIR, P1_SEMILLA
from mibici.features import COLUMNAS_P1_AMPLIADAS


def main():
    # Cargar los conjuntos preparados en el paso 1.
    carpeta = DATA_DIR / "monitoring"

    referencia = pd.read_parquet(carpeta / "p1_referencia.parquet")
    actual = pd.read_parquet(carpeta / "p1_actual.parquet")

    # Monitorear exclusivamente las entradas del modelo.
    referencia = referencia[COLUMNAS_P1_AMPLIADAS]
    actual = actual[COLUMNAS_P1_AMPLIADAS]

    # Muestreo reproducible para controlar tiempo y memoria.
    n = 50_000
    referencia = referencia.sample(
        n=min(n, len(referencia)), random_state=P1_SEMILLA
    )
    actual = actual.sample(
        n=min(n, len(actual)), random_state=P1_SEMILLA
    )

    # Definir explícitamente los tipos de variables.
    categoricas = ["hora", "dia_semana", "fin_semana"]
    numericas = [
        columna for columna in COLUMNAS_P1_AMPLIADAS
        if columna not in categoricas
    ]

    # Convertir las etiquetas categóricas a tipos compatibles con Evidently.
    for columna in categoricas:
        referencia[columna] = referencia[columna].astype("int32").astype(str)
        actual[columna] = actual[columna].astype("int32").astype(str)

    esquema = DataDefinition(
        numerical_columns=numericas,
        categorical_columns=categoricas,
    )

    datos_referencia = Dataset.from_pandas(
        referencia, data_definition=esquema
    )
    datos_actuales = Dataset.from_pandas(
        actual, data_definition=esquema
    )

    # Generar drift y calidad de datos.
    reporte = Report([
        DataDriftPreset(method="psi"),
        DataSummaryPreset(),
    ])

    resultado = reporte.run(datos_actuales, datos_referencia)

    # Guardar resultados.
    salida = REPORTS_DIR / "monitoring"
    salida.mkdir(parents=True, exist_ok=True)

    resultado.save_html(str(salida / "p1_drift_evidently.html"))

    with open(salida / "p1_drift_resultados.json", "w", encoding="utf-8") as f:
        f.write(resultado.json())

    print("Reporte Evidently generado correctamente.")
    print(f"Referencia: {len(referencia):,} observaciones")
    print(f"Actual: {len(actual):,} observaciones")
    print(f"Variables analizadas: {len(COLUMNAS_P1_AMPLIADAS)}")
    print(f"Resultados guardados en: {salida}")


if __name__ == "__main__":
    main()
