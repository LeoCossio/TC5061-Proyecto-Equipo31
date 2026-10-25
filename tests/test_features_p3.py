import numpy as np
import pandas as pd

from mibici.feature_engineering.clustering import (
    COLUMNAS_P3_AMPLIADAS_EXTRA,
    crear_features_p3,
    crear_features_p3_ampliadas,
)


def _viajes(estaciones=4, dias=20, por_dia=12, semilla=0):
    rng = np.random.default_rng(semilla)
    filas = []
    for estacion in range(1, estaciones + 1):
        for dia in range(dias):
            for _ in range(por_dia):
                inicio = pd.Timestamp("2026-03-02") + pd.Timedelta(
                    days=dia, hours=int(rng.integers(24)), minutes=int(rng.integers(60))
                )
                filas.append((estacion, int(rng.integers(1, estaciones + 1)), inicio))
    return pd.DataFrame(filas, columns=["origen_id", "destino_id", "inicio"])


def test_features_ampliadas_conservan_las_estaciones_y_columnas_base():
    viajes = _viajes()
    base = crear_features_p3(viajes)
    ampliadas = crear_features_p3_ampliadas(viajes)

    assert ampliadas.index.equals(base.index)
    assert list(ampliadas.columns) == list(base.columns) + COLUMNAS_P3_AMPLIADAS_EXTRA
    pd.testing.assert_frame_equal(ampliadas[base.columns], base)
    assert ampliadas.notna().all().all()


def test_extras_tienen_rangos_validos():
    ampliadas = crear_features_p3_ampliadas(_viajes())

    # Entropía de 24 horas: entre 0 y log2(24) bits.
    assert ampliadas["entropia_horaria"].between(0, np.log2(24) + 1e-9).all()
    assert (ampliadas["cv_diario"] >= 0).all()
    assert ampliadas["prop_hora_pico"].between(0, 1).all()
