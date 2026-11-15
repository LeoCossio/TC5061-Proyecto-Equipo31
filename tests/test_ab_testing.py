"""Pruebas de la comparación A/B (mibici/ab_testing.py) con datos sintéticos."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici import ab_testing as ab
from mibici.features import COLUMNAS_P1_AMPLIADAS, OBJETIVO_P1


def conjunto(n_dias=30, filas_por_dia=48, semilla=0):
    """Filas sintéticas de varios días con las columnas que usa el A/B testing."""
    rng = np.random.default_rng(semilla)
    horas = pd.date_range("2026-09-01", periods=n_dias * 24, freq="h")
    tabla = pd.DataFrame({
        "hora_inicio": np.tile(horas, filas_por_dia // 24)[: n_dias * filas_por_dia],
        "origen_id": rng.integers(1, 20, n_dias * filas_por_dia),
    })
    tabla["hora"] = tabla["hora_inicio"].dt.hour
    tabla["fin_semana"] = (tabla["hora_inicio"].dt.dayofweek >= 5).astype(int)
    tabla[OBJETIVO_P1] = rng.poisson(3, len(tabla))
    return tabla


def test_metricas_con_valores_conocidos():
    datos = conjunto()
    real = datos[OBJETIVO_P1].to_numpy()

    perfecto = ab.metricas(datos, real)
    desplazado = ab.metricas(datos, real + 2.0)

    assert perfecto["mae"] == 0 and perfecto["rmse"] == 0
    assert np.isclose(desplazado["mae"], 2.0) and np.isclose(desplazado["rmse"], 2.0)


def test_comparacion_pareada_detecta_que_b_es_mejor():
    datos = conjunto()
    real = datos[OBJETIVO_P1].to_numpy(dtype=float)
    rng = np.random.default_rng(1)
    pred_a = real + rng.normal(0, 1.0, len(real))
    pred_b = real + rng.normal(0, 0.5, len(real))

    resultado = ab.comparar_pareado(datos, pred_a, pred_b)

    assert resultado["dias"] == 30
    assert resultado["diferencia_mae_diario"] > 0
    assert resultado["ic_inferior"] > 0
    assert resultado["p_wilcoxon"] < 0.01
    assert resultado["dias_en_que_gana_b"] >= 25


def test_comparacion_pareada_sin_diferencia():
    datos = conjunto()
    prediccion = datos[OBJETIVO_P1].to_numpy(dtype=float) + 0.5

    resultado = ab.comparar_pareado(datos, prediccion, prediccion)

    assert resultado["diferencia_mae_diario"] == 0
    assert resultado["p_wilcoxon"] == 1.0
    assert resultado["dias_en_que_gana_b"] == 0


def test_decision_elige_b_solo_si_mejora_y_no_empeora_otras_metricas():
    a = {"mae": 1.0, "rmse": 2.0, "mae_horas_pico": 1.5}
    significativo = {"ic_inferior": 0.001}
    no_significativo = {"ic_inferior": -0.001}

    mejor = {"mae": 0.9, "rmse": 1.9, "mae_horas_pico": 1.4}
    peor_rmse = {"mae": 0.9, "rmse": 2.2, "mae_horas_pico": 1.4}
    peor_pico = {"mae": 0.9, "rmse": 1.9, "mae_horas_pico": 1.6}

    assert ab.decidir(a, mejor, significativo)[0] == "B"
    assert ab.decidir(a, mejor, no_significativo)[0] == "A"
    assert ab.decidir(a, peor_rmse, significativo)[0] == "A"
    assert ab.decidir(a, peor_pico, significativo)[0] == "A"


def test_decision_tolera_un_empeoramiento_pequeno():
    a = {"mae": 1.0, "rmse": 2.0, "mae_horas_pico": 1.5}
    casi_igual = {"mae": 0.9, "rmse": 2.01, "mae_horas_pico": 1.5}  # RMSE 0.5 % peor

    assert ab.decidir(a, casi_igual, {"ic_inferior": 0.01})[0] == "B"


def test_error_por_grupo_de_estaciones():
    datos = conjunto()
    prediccion = datos[OBJETIVO_P1].to_numpy(dtype=float)
    prediccion[datos["origen_id"].to_numpy() >= 10] += 1.0  # solo las estaciones "nuevas" fallan

    grupos = ab.mae_por_grupo_de_estacion(datos, prediccion, estaciones_conocidas=set(range(1, 10)))

    assert np.isclose(grupos["conocidas"]["mae"], 0.0)
    assert np.isclose(grupos["nuevas"]["mae"], 1.0)
    assert grupos["conocidas"]["filas"] + grupos["nuevas"]["filas"] == len(datos)


def test_modelo_alternativo_conserva_hiperparametros_y_es_reproducible():
    rng = np.random.default_rng(0)
    entrenamiento = pd.DataFrame(rng.normal(size=(600, len(COLUMNAS_P1_AMPLIADAS))),
                                 columns=COLUMNAS_P1_AMPLIADAS)
    entrenamiento[OBJETIVO_P1] = rng.poisson(3, 600)
    modelo_a = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.1, max_iter=20,
                                             max_leaf_nodes=15, early_stopping=False, random_state=42)
    modelo_a.fit(entrenamiento[COLUMNAS_P1_AMPLIADAS], entrenamiento[OBJETIVO_P1])

    b1 = ab.entrenar_alternativa(modelo_a, entrenamiento)
    b2 = ab.entrenar_alternativa(modelo_a, entrenamiento)

    assert b1.get_params() == modelo_a.get_params()
    assert b1 is not modelo_a
    np.testing.assert_array_equal(b1.predict(entrenamiento[COLUMNAS_P1_AMPLIADAS]),
                                  b2.predict(entrenamiento[COLUMNAS_P1_AMPLIADAS]))


def test_mae_semanal_agrupa_por_semana():
    datos = conjunto(n_dias=14)
    real = datos[OBJETIVO_P1].to_numpy(dtype=float)

    semanal = ab.mae_semanal(datos, real + 1.0)

    assert len(semanal) >= 2 and np.allclose(semanal.values, 1.0)
