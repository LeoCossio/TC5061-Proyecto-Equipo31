
import pandas as pd

from mibici.config import (
    PROCESSED_DATA_DIR,
    DATA_DIR,
    P1_FECHA_INICIO_VALIDACION,
    P1_FECHA_INICIO_PRUEBA,
)
from mibici.features import (
    COLUMNAS_P1_AMPLIADAS,
    OBJETIVO_P1,
    construir_panel_horario,
    crear_features_p1,
    dividir_temporal_p1,
)


def main():
    # Cargar los mismos datos que utiliza P1.
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))

    if len(archivos) != 12:
        raise ValueError(f"Se esperaban 12 archivos Parquet, hay {len(archivos)}")

    print("Cargando datos...")
    viajes = pd.concat(
        [
            pd.read_parquet(archivo, columns=["origen_id", "inicio"])
            for archivo in archivos
        ],
        ignore_index=True,
    )

    # Reutilizar el procesamiento original de P1.
    print("Generando panel horario...")
    panel = construir_panel_horario(viajes)

    print("Calculando características...")
    features = crear_features_p1(panel)

    # Separación temporal.
    referencia, _, actual = dividir_temporal_p1(
        features,
        P1_FECHA_INICIO_VALIDACION,
        P1_FECHA_INICIO_PRUEBA,
    )

    # Conservar las variables del modelo y los identificadores.
    columnas = [
        "origen_id",
        "hora_inicio",
        *COLUMNAS_P1_AMPLIADAS,
        OBJETIVO_P1,
    ]

    referencia = referencia[columnas].copy()
    actual = actual[columnas].copy()

    # Guardar los conjuntos para Evidently.
    salida = DATA_DIR / "monitoring"
    salida.mkdir(parents=True, exist_ok=True)

    referencia.to_parquet(salida / "p1_referencia.parquet", index=False)
    actual.to_parquet(salida / "p1_actual.parquet", index=False)

    print("\nPreparación completada:")
    for nombre, datos in [
        ("Referencia", referencia),
        ("Actual", actual),
    ]:
        print(f"\n{nombre}:")
        print(f"  Observaciones: {len(datos):,}")
        print(f"  Desde: {datos['hora_inicio'].min()}")
        print(f"  Hasta: {datos['hora_inicio'].max()}")
        print(f"  Estaciones: {datos['origen_id'].nunique()}")
        print(f"  Valores nulos: {datos.isna().sum().sum():,}")

    print(f"\nCaracterísticas del modelo: {len(COLUMNAS_P1_AMPLIADAS)}")
    print(f"Archivos guardados en: {salida}")


if __name__ == "__main__":
    main()
