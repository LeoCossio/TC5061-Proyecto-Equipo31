import numpy as np
import pandas as pd

from mibici.features import (
    COLUMNAS_P1,
    OBJETIVO_P1,
    baseline_p1,
    construir_panel_horario,
    crear_features_p1,
)


def _viajes_sinteticos(semanas=3, semilla=0):
    """Viajes de dos estaciones durante varias semanas, con horas vacías."""
    rng = np.random.default_rng(semilla)
    horas = pd.date_range("2026-01-05", periods=semanas * 168, freq="h")
    filas = []
    for estacion in (1, 2):
        for hora in horas:
            for _ in range(rng.poisson(2)):
                filas.append((estacion, hora + pd.Timedelta(minutes=int(rng.integers(60)))))
    return pd.DataFrame(filas, columns=["origen_id", "inicio"])


def test_panel_incluye_horas_sin_viajes():
    viajes = pd.DataFrame({
        "origen_id": [1, 1],
        "inicio": pd.to_datetime(["2026-01-01 00:10", "2026-01-01 03:20"]),
    })
    panel = construir_panel_horario(viajes)

    assert len(panel) == 4
    assert panel[OBJETIVO_P1].tolist() == [1, 0, 0, 1]


def test_features_solo_usan_horas_pasadas():
    viajes = _viajes_sinteticos()
    panel = construir_panel_horario(viajes)
    features = crear_features_p1(panel)

    # lag_1 de cada fila debe ser el objetivo de la hora anterior.
    base = panel.set_index(["origen_id", "hora_inicio"])[OBJETIVO_P1]
    anterior = base.reindex(
        pd.MultiIndex.from_arrays([
            features["origen_id"],
            features["hora_inicio"] - pd.Timedelta(hours=1),
        ])
    ).to_numpy()
    assert np.array_equal(features["lag_1"].to_numpy(), anterior)

    # Cambiar el objetivo de una hora no modifica sus propias características
    # ni las de horas anteriores: solo puede afectar a horas posteriores.
    hora_corte = features["hora_inicio"].sort_values().iloc[len(features) // 2]
    modificado = panel.copy()
    seleccion = (modificado["origen_id"] == 1) & (modificado["hora_inicio"] == hora_corte)
    modificado.loc[seleccion, OBJETIVO_P1] += 50
    features_mod = crear_features_p1(modificado)

    hasta_corte = features["hora_inicio"] <= hora_corte
    pd.testing.assert_frame_equal(
        features.loc[hasta_corte, COLUMNAS_P1].reset_index(drop=True),
        features_mod.loc[features_mod["hora_inicio"] <= hora_corte, COLUMNAS_P1].reset_index(
            drop=True
        ),
    )


def test_baseline_no_tiene_valores_faltantes():
    features = crear_features_p1(construir_panel_horario(_viajes_sinteticos()))
    prediccion = baseline_p1(features, media_global=1.0)

    assert prediccion.notna().all()
    assert len(prediccion) == len(features)


def test_features_ampliadas_no_usan_la_hora_objetivo():
    from mibici.features import COLUMNAS_P1_AMPLIADAS

    panel = construir_panel_horario(_viajes_sinteticos(semanas=4))
    features = crear_features_p1(panel)

    hora_corte = features["hora_inicio"].sort_values().iloc[len(features) // 2]
    modificado = panel.copy()
    seleccion = (modificado["origen_id"] == 2) & (modificado["hora_inicio"] == hora_corte)
    modificado.loc[seleccion, OBJETIVO_P1] += 50
    features_mod = crear_features_p1(modificado)

    # Las filas hasta la hora modificada deben quedar idénticas.
    antes = features["hora_inicio"] <= hora_corte
    antes_mod = features_mod["hora_inicio"] <= hora_corte
    pd.testing.assert_frame_equal(
        features.loc[antes, COLUMNAS_P1_AMPLIADAS].reset_index(drop=True),
        features_mod.loc[antes_mod, COLUMNAS_P1_AMPLIADAS].reset_index(drop=True),
    )
    assert features[COLUMNAS_P1_AMPLIADAS].notna().all().all()
