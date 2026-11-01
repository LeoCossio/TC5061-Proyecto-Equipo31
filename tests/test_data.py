
"""Pruebas unitarias de limpieza de datos de MiBici."""

import pandas as pd
import pytest

from mibici.dataset import REQUIRED_COLUMNS, clean_dataset


def crear_viajes():
    """Genera un conjunto pequeño de viajes sintéticos."""
    valores = {
        "viaje_id": [1, 2, 2, 3, 4],
        "origen_id": [10, 20, 20, 30, 40],
        "destino_id": [20, 30, 30, 40, 50],
        "inicio": [
            "2026-01-01 08:00",
            "2026-01-01 09:00",
            "2026-01-01 09:00",
            "2026-01-01 10:00",
            "2026-01-01 11:00",
        ],
        "fin": [
            "2026-01-01 08:15",
            "2026-01-01 09:20",
            "2026-01-01 09:20",
            "2026-01-01 09:50",
            "fecha_invalida",
        ],
    }
    return pd.DataFrame(valores)[REQUIRED_COLUMNS]


def test_limpieza_elimina_duplicados_y_registros_invalidos(tmp_path):
    datos = crear_viajes()
    entrada = tmp_path / "viajes.csv"
    salida = tmp_path / "procesados" / "viajes.parquet"

    datos.to_csv(entrada, index=False)

    reporte = clean_dataset(entrada, salida)
    resultado = pd.read_parquet(salida)

    assert reporte["registros_originales"] == 5
    assert reporte["duplicados_eliminados"] == 1
    assert reporte["registros_invalidos"] == 2
    assert reporte["registros_limpios"] == 2

    assert len(resultado) == 2
    assert salida.exists()
    assert resultado["inicio"].notna().all()
    assert resultado["fin"].notna().all()


def test_limpieza_conserva_registros_validos(tmp_path):
    datos = crear_viajes().iloc[:2]
    entrada = tmp_path / "validos.csv"
    salida = tmp_path / "validos.parquet"

    datos.to_csv(entrada, index=False)

    reporte = clean_dataset(entrada, salida)
    resultado = pd.read_parquet(salida)

    assert reporte["registros_limpios"] == 2
    assert reporte["duplicados_eliminados"] == 0
    assert reporte["registros_invalidos"] == 0
    assert list(resultado.columns) == REQUIRED_COLUMNS


def test_limpieza_rechaza_columnas_faltantes(tmp_path):
    datos = crear_viajes().drop(columns=["origen_id"])
    entrada = tmp_path / "incompleto.csv"
    salida = tmp_path / "incompleto.parquet"

    datos.to_csv(entrada, index=False)

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        clean_dataset(entrada, salida)

    assert not salida.exists()


def test_limpieza_acepta_dataset_sin_filas(tmp_path):
    datos = crear_viajes().iloc[:0]
    entrada = tmp_path / "vacio.csv"
    salida = tmp_path / "vacio.parquet"

    datos.to_csv(entrada, index=False)

    reporte = clean_dataset(entrada, salida)
    resultado = pd.read_parquet(salida)

    assert reporte["registros_originales"] == 0
    assert reporte["registros_limpios"] == 0
    assert resultado.empty
