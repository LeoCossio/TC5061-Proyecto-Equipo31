"""Pruebas de la validación de datos con DeepChecks (mibici/validacion_datos.py).

Usan tablas sintéticas pequeñas para comprobar dos cosas: que los datos sanos
pasan los chequeos y que cada tipo de problema (tipos incorrectos, valores
faltantes, duplicados, columnas constantes, deriva y fuga temporal) se detecta.
La prueba con los datos reales solo corre si se define MIBICI_VALIDAR_DATOS_REALES=1.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mibici import validacion_datos as vd
from mibici.validacion_datos import Hallazgo

CATEGORICAS = ["hora"]


def tabla(n=2000, semilla=0, desplazamiento=0.0, inicio="2026-01-01"):
    """Tabla con una variable categórica, dos numéricas continuas, fecha y etiqueta."""
    rng = np.random.default_rng(semilla)
    return pd.DataFrame({
        "hora": rng.integers(0, 24, n),
        "lag_1": rng.normal(10 + desplazamiento, 2, n),
        "lag_24": rng.normal(5, 1, n),
        "fecha": pd.date_range(inicio, periods=n, freq="h"),
        "y": rng.normal(10 + desplazamiento, 2, n),
    })


def viajes(n=300, semilla=0):
    """Viajes sintéticos con el esquema que produce la ingesta."""
    rng = np.random.default_rng(semilla)
    inicio = pd.Timestamp("2025-10-01") + pd.to_timedelta(rng.integers(0, 200 * 24 * 60, n), unit="m")
    duracion = pd.to_timedelta(rng.integers(5, 60, n), unit="m")
    return pd.DataFrame({
        "viaje_id": np.arange(1, n + 1, dtype="int64"),
        "inicio": inicio,
        "fin": inicio + duracion,
        "origen_id": rng.integers(1, 20, n).astype("int64"),
        "destino_id": rng.integers(1, 20, n).astype("int64"),
    })


def estado(hallazgos, texto):
    """Estados de los hallazgos cuyo chequeo o condición contienen `texto`."""
    return [h.estado for h in hallazgos if texto in h.chequeo or texto in h.condicion]


# ---------------------------------------------------------------------------
# Esquema y dominio de la tabla de viajes
# ---------------------------------------------------------------------------


def test_esquema_de_viajes_limpios_no_tiene_errores():
    hallazgos = vd.validar_esquema(viajes(), vd.ESQUEMA_VIAJES, "viajes",
                                   periodo=("2025-10-01", "2026-09-30"), ids_catalogo=set(range(1, 20)))

    assert hallazgos and not any(h.estado != "PASS" for h in hallazgos)


def test_esquema_detecta_tipos_incorrectos():
    datos = viajes()
    datos["origen_id"] = datos["origen_id"].astype(str)
    datos["inicio"] = datos["inicio"].astype(str)

    hallazgos = vd.validar_esquema(datos, vd.ESQUEMA_VIAJES, "viajes")

    assert estado(hallazgos, "origen_id es de tipo int") == ["FAIL"]
    assert estado(hallazgos, "inicio es de tipo datetime") == ["FAIL"]
    assert estado(hallazgos, "destino_id es de tipo int") == ["PASS"]


def test_esquema_detecta_columna_faltante():
    hallazgos = vd.validar_esquema(viajes().drop(columns="fin"), vd.ESQUEMA_VIAJES, "viajes")

    assert estado(hallazgos, "todas las columnas") == ["FAIL"]


def test_esquema_detecta_ids_repetidos_y_fin_antes_de_inicio():
    datos = viajes()
    datos.loc[1, "viaje_id"] = datos.loc[0, "viaje_id"]
    datos.loc[2, "fin"] = datos.loc[2, "inicio"] - pd.Timedelta(minutes=5)

    hallazgos = vd.validar_esquema(datos, vd.ESQUEMA_VIAJES, "viajes")

    assert estado(hallazgos, "viaje_id es único") == ["FAIL"]
    assert estado(hallazgos, "termina antes de empezar") == ["FAIL"]


def test_esquema_marca_periodo_como_error_y_catalogo_como_advertencia():
    datos = viajes()
    datos.loc[0, ["inicio", "fin"]] = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 00:10")]

    hallazgos = vd.validar_esquema(datos, vd.ESQUEMA_VIAJES, "viajes",
                                   periodo=("2025-10-01", "2026-09-30"), ids_catalogo={1, 2, 3})

    periodo = [h for h in hallazgos if "inician entre" in h.condicion]
    catalogo = [h for h in hallazgos if h.chequeo == "Catálogo"]
    assert periodo[0].es_error
    assert catalogo and all(h.es_advertencia for h in catalogo)


def test_esquema_viaje_de_mas_de_un_dia_es_advertencia():
    datos = viajes()
    datos.loc[0, "fin"] = datos.loc[0, "inicio"] + pd.Timedelta(hours=30)

    hallazgos = vd.validar_esquema(datos, vd.ESQUEMA_VIAJES, "viajes")
    larga = [h for h in hallazgos if "más de" in h.condicion][0]

    assert larga.estado == "FAIL" and larga.es_advertencia and not larga.es_error


# ---------------------------------------------------------------------------
# Integridad con DeepChecks (un solo conjunto)
# ---------------------------------------------------------------------------


def test_integridad_de_datos_sanos_pasa():
    hallazgos, _ = vd.validar_integridad(tabla(), "sano", categoricas=CATEGORICAS, fecha="fecha",
                                         etiqueta="y", tipo_etiqueta="regression")

    assert hallazgos
    assert all(h.estado == "PASS" for h in hallazgos), [h for h in hallazgos if h.estado != "PASS"]


def test_integridad_detecta_valores_faltantes():
    datos = tabla()
    datos.loc[:99, "lag_1"] = np.nan

    hallazgos, _ = vd.validar_integridad(datos, "con nulos", categoricas=CATEGORICAS)

    assert estado(hallazgos, "Percent Of Nulls") == ["FAIL"]


def test_integridad_tolera_nulos_dentro_del_umbral():
    datos = tabla()
    datos.loc[:9, "lag_1"] = np.nan  # 0.5 %

    hallazgos, _ = vd.validar_integridad(datos, "pocos nulos", categoricas=CATEGORICAS, max_nulos=0.01)

    assert estado(hallazgos, "Percent Of Nulls") == ["PASS"]


def test_integridad_detecta_filas_duplicadas():
    datos = tabla()
    datos = pd.concat([datos, datos.iloc[:200]], ignore_index=True)

    hallazgos, _ = vd.validar_integridad(datos, "duplicados", categoricas=CATEGORICAS)

    assert estado(hallazgos, "Data Duplicates") == ["FAIL"]


def test_integridad_ignora_columna_al_buscar_duplicados():
    datos = tabla()
    datos = pd.concat([datos, datos.iloc[:200]], ignore_index=True)
    datos["secuencia"] = np.arange(len(datos))  # hace únicas todas las filas

    con_secuencia, _ = vd.validar_integridad(datos, "con id", categoricas=CATEGORICAS)
    sin_secuencia, _ = vd.validar_integridad(datos, "sin id", categoricas=CATEGORICAS,
                                             ignorar_duplicados=["secuencia"])

    assert estado(con_secuencia, "Data Duplicates") == ["PASS"]
    assert estado(sin_secuencia, "Data Duplicates") == ["FAIL"]


def test_integridad_detecta_columna_constante():
    datos = tabla()
    datos["constante"] = 7.0

    hallazgos, _ = vd.validar_integridad(datos, "constante", categoricas=CATEGORICAS)

    assert estado(hallazgos, "Is Single Value") == ["FAIL"]


def test_integridad_detecta_tipos_mezclados():
    datos = tabla()
    datos["lag_24"] = datos["lag_24"].astype(object)
    datos.loc[:99, "lag_24"] = "sin dato"  # 5 % de textos dentro de una columna numérica

    hallazgos, _ = vd.validar_integridad(datos, "mezcla", categoricas=CATEGORICAS)

    assert estado(hallazgos, "Mixed Data Types") == ["FAIL"]


# ---------------------------------------------------------------------------
# Entrenamiento contra prueba: deriva y fuga temporal
# ---------------------------------------------------------------------------


def test_train_test_sin_deriva_pasa():
    entrenamiento = tabla(semilla=1)
    prueba = tabla(semilla=2, inicio="2026-06-01")

    hallazgos, _ = vd.validar_train_test(entrenamiento, prueba, "estable", categoricas=CATEGORICAS,
                                         fecha="fecha", etiqueta="y", tipo_etiqueta="regression")

    assert all(h.estado == "PASS" for h in hallazgos), [h for h in hallazgos if h.estado != "PASS"]


def test_train_test_detecta_deriva_de_variables_y_etiqueta():
    entrenamiento = tabla(semilla=1)
    prueba = tabla(semilla=2, desplazamiento=6.0, inicio="2026-06-01")

    hallazgos, _ = vd.validar_train_test(entrenamiento, prueba, "deriva", categoricas=CATEGORICAS,
                                         fecha="fecha", etiqueta="y", tipo_etiqueta="regression")

    deriva = [h for h in hallazgos if h.chequeo in ("Feature Drift", "Label Drift")]
    assert len(deriva) == 2 and all(h.estado == "FAIL" and h.es_advertencia for h in deriva)


def test_train_test_deriva_de_variable_categorica():
    entrenamiento = tabla(semilla=1)
    prueba = tabla(semilla=2, inicio="2026-06-01")
    prueba["hora"] = np.random.default_rng(3).choice([7, 8, 9], len(prueba))  # solo horas de la mañana

    hallazgos, _ = vd.validar_train_test(entrenamiento, prueba, "categorica", categoricas=CATEGORICAS,
                                         fecha="fecha")

    assert estado(hallazgos, "Feature Drift") == ["FAIL"]


def test_train_test_detecta_fuga_temporal():
    entrenamiento = tabla(semilla=1)
    prueba = tabla(semilla=2, inicio="2026-03-01")  # empieza antes de que termine el entrenamiento

    hallazgos, _ = vd.validar_train_test(entrenamiento, prueba, "fuga", categoricas=CATEGORICAS,
                                         fecha="fecha")

    fuga = [h for h in hallazgos if "Leakage" in h.chequeo]
    assert fuga and any(h.estado == "FAIL" and h.es_error for h in fuga)


def test_train_test_con_particion_cronologica_no_tiene_fuga():
    entrenamiento = tabla(semilla=1)
    prueba = tabla(semilla=2, inicio="2026-06-01")

    hallazgos, _ = vd.validar_train_test(entrenamiento, prueba, "cronologico",
                                         categoricas=CATEGORICAS, fecha="fecha")

    assert estado(hallazgos, "Leakage") == ["PASS", "PASS"]


def test_train_test_detecta_filas_identicas_entre_conjuntos():
    entrenamiento = tabla(semilla=1)
    prueba = pd.concat([tabla(semilla=2, inicio="2026-06-01"), entrenamiento.iloc[:300]],
                       ignore_index=True)

    hallazgos, _ = vd.validar_train_test(entrenamiento, prueba, "mezcla", categoricas=CATEGORICAS)

    assert estado(hallazgos, "Train Test Samples Mix") == ["FAIL"]


# ---------------------------------------------------------------------------
# Integración con las funciones del proyecto y con la línea de comandos
# ---------------------------------------------------------------------------


def test_tabla_de_features_p1_pasa_la_integridad():
    from mibici.features import COLUMNAS_P1, OBJETIVO_P1, construir_panel_horario, crear_features_p1

    rng = np.random.default_rng(0)
    horas = pd.date_range("2026-01-05", periods=3 * 168, freq="h")
    filas = [(e, h + pd.Timedelta(minutes=int(rng.integers(60))))
             for e in (1, 2, 3) for h in horas for _ in range(rng.poisson(2))]
    panel = construir_panel_horario(pd.DataFrame(filas, columns=["origen_id", "inicio"]))
    features = crear_features_p1(panel)

    hallazgos, _ = vd.validar_integridad(
        features[COLUMNAS_P1 + [OBJETIVO_P1, "hora_inicio"]], "p1",
        categoricas=["hora", "dia_semana", "fin_semana"], fecha="hora_inicio",
        etiqueta=OBJETIVO_P1, tipo_etiqueta="regression",
        ignorar_duplicados=["hora_inicio"], max_duplicados=0.5)

    assert estado(hallazgos, "Percent Of Nulls") == ["PASS"]
    assert estado(hallazgos, "Mixed Nulls") == ["PASS"]


def test_el_comando_falla_con_errores_de_integridad(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "RESUMEN_MD", tmp_path / "resumen.md")
    monkeypatch.setitem(vd.VALIDADORES, "viajes", lambda _: [
        Hallazgo("viajes", "Esquema", "x", "integridad", "FAIL", "")])

    assert vd.principal(["--conjuntos", "viajes"]) == 1


def test_el_comando_tolera_advertencias_salvo_en_modo_estricto(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "RESUMEN_MD", tmp_path / "resumen.md")
    monkeypatch.setitem(vd.VALIDADORES, "viajes", lambda _: [
        Hallazgo("viajes", "Deriva", "x", "comparacion", "FAIL", "")])

    assert vd.principal(["--conjuntos", "viajes"]) == 0
    assert vd.principal(["--conjuntos", "viajes", "--estricto"]) == 1
    assert (tmp_path / "resumen.md").exists()


def test_el_comando_rechaza_conjuntos_desconocidos(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "RESUMEN_MD", tmp_path / "resumen.md")

    with pytest.raises(SystemExit):
        vd.principal(["--conjuntos", "p9"])


def test_el_comando_acepta_conjuntos_separados_por_comas(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "RESUMEN_MD", tmp_path / "resumen.md")
    llamados = []
    for nombre in vd.VALIDADORES:
        monkeypatch.setitem(vd.VALIDADORES, nombre, lambda _, n=nombre: llamados.append(n) or [])

    assert vd.principal(["--conjuntos", "viajes,p3"]) == 0
    assert llamados == ["viajes", "p3"]


def test_el_resumen_markdown_incluye_todas_las_condiciones():
    hallazgos = [Hallazgo("viajes", "A", "cond 1", "integridad", "PASS", "ok"),
                 Hallazgo("viajes", "B", "cond 2", "comparacion", "FAIL", "deriva")]

    texto = vd.resumen_markdown(hallazgos, {"viajes": 1.0})

    assert "cond 1" in texto and "cond 2" in texto and "1 advertencias" in texto


@pytest.mark.skipif(os.environ.get("MIBICI_VALIDAR_DATOS_REALES") != "1",
                    reason="Tarda más de un minuto; actívala con MIBICI_VALIDAR_DATOS_REALES=1")
def test_los_viajes_reales_no_tienen_errores_de_integridad():
    hallazgos = vd.validar_viajes({})

    assert not [h for h in hallazgos if h.es_error]
