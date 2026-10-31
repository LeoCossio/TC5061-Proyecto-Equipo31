
"""Prueba de reproducibilidad del entrenamiento de P1."""

import numpy as np
import pandas as pd

from mibici.feature_engineering.regresion import (
    COLUMNAS_P1,
    OBJETIVO_P1,
    entrenar_random_forest_p1,
)


def test_random_forest_misma_semilla_mismas_predicciones():
    """Dos entrenamientos con la misma semilla deben coincidir."""

    rng = np.random.default_rng(42)
    n_filas = 120

    # Datos sintéticos con las columnas reales del proyecto.
    datos = pd.DataFrame({
        columna: rng.integers(0, 10, size=n_filas).astype(float)
        for columna in COLUMNAS_P1
    })

    datos[OBJETIVO_P1] = (
        2 * datos[COLUMNAS_P1[0]]
        + datos[COLUMNAS_P1[1]]
        + rng.integers(0, 3, size=n_filas)
    )

    configuracion = {
        "semilla": 42,
        "n_estimators": 12,
        "min_samples_leaf": 2,
        "max_samples": 0.8,
    }

    modelo_1 = entrenar_random_forest_p1(
        entrenamiento=datos,
        **configuracion,
    )

    modelo_2 = entrenar_random_forest_p1(
        entrenamiento=datos,
        **configuracion,
    )

    predicciones_1 = modelo_1.predict(datos[COLUMNAS_P1])
    predicciones_2 = modelo_2.predict(datos[COLUMNAS_P1])

    np.testing.assert_allclose(
        predicciones_1,
        predicciones_2,
        rtol=0,
        atol=1e-12,
    )
