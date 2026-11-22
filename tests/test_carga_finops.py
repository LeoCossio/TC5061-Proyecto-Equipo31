"""Pruebas de los scripts de análisis de la prueba de carga y del análisis FinOps."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]


def cargar_script(nombre):
    """Importa un archivo de scripts/ (que no es un paquete) como módulo."""
    especificacion = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


monitor = cargar_script("monitorear_recursos")
analisis = cargar_script("analizar_carga")
finops = cargar_script("finops")


@pytest.mark.parametrize("texto, esperado", [
    ("512MiB", 512.0), ("1GiB", 1024.0), ("2048KiB", 2.0), ("1.5GiB", 1536.0), ("0B", 0.0),
])
def test_conversion_de_memoria_de_docker_stats(texto, esperado):
    assert monitor.a_mib(texto) == pytest.approx(esperado)


def historia_sintetica():
    """Dos escalones de 60 s: 10 req/s sin errores y 20 req/s con un 10 % de errores."""
    filas = []
    for t in range(120):
        escalon = t // 60
        filas.append({
            "Timestamp": 1000 + t, "Name": "Aggregated", "t": t,
            "Requests/s": 10.0 if escalon == 0 else 20.0,
            "Failures/s": 0.0 if escalon == 0 else 2.0,
            "50%": 30.0, "95%": 50.0 if escalon == 0 else 700.0, "99%": 80.0,
        })
    return pd.DataFrame(filas)


def recursos_sinteticos():
    filas = []
    for t in range(0, 120, 2):
        for servicio, cpu, memoria in (("api", 100.0 + t, 200.0), ("mlflow", 2.0, 2000.0)):
            filas.append({"t": t, "servicio": servicio, "cpu_pct": cpu, "memoria_mib": memoria})
    return pd.DataFrame(filas)


def test_resumen_por_escalon_usa_el_regimen_estable():
    resumen = analisis.resumen_por_escalon(historia_sintetica(), recursos_sinteticos(), 60, [5, 10])

    assert list(resumen["usuarios"]) == [5, 10]
    assert resumen.loc[0, "rps"] == pytest.approx(10.0)
    assert resumen.loc[0, "tasa_error_pct"] == pytest.approx(0.0)
    assert resumen.loc[1, "tasa_error_pct"] == pytest.approx(10.0)
    assert resumen.loc[1, "p95_ms"] == pytest.approx(700.0)
    # CPU de la API del segundo escalón: promedio de las muestras de los últimos 40 s (t = 80 a 118).
    assert resumen.loc[1, "cpu_api_pct"] == pytest.approx(100.0 + 99.0)
    assert resumen.loc[1, "memoria_mlflow_mib"] == pytest.approx(2000.0)


def test_costo_de_fargate_por_hora():
    costo = finops.costo_hora_fargate(0.25, 0.5)

    assert costo == pytest.approx(0.25 * 0.04048 + 0.5 * 0.004445)


def test_costo_marginal_a_plena_carga():
    resumen = {"capacidad_sostenible": {"usuarios": 100, "rps": 100.0, "p95_ms": 300.0, "cpu_api_pct": 200.0,
                                        "memoria_api_mib": 200.0, "memoria_mlflow_mib": 2000.0}}

    marginal = finops.marginal_a_plena_carga(resumen)

    assert marginal["cpu_s_por_inferencia"] == pytest.approx(0.02)
    assert marginal["costo_por_inferencia"] == pytest.approx(0.02 * 0.0416 / 7200)


def test_escenarios_con_demanda_real_estan_ordenados_de_mayor_a_menor_costo():
    por_hora, por_mes, escenarios = finops.escenarios_demanda_real(10, 2.2, 0.6, 0.2)

    assert por_hora == 471 * 2
    assert por_mes == por_hora * 730
    costos = [e["costo_por_inferencia"] for e in escenarios]
    assert costos == sorted(costos, reverse=True)
    assert escenarios[0]["costo_mensual"] == pytest.approx(0.0416 * 730)


def test_el_lote_corto_se_factura_con_el_minimo_de_un_minuto():
    _, _, corto = finops.escenarios_demanda_real(5, 2.2, 0.6, 0.2)
    _, _, largo = finops.escenarios_demanda_real(60, 2.2, 0.6, 0.2)
    _, _, mas_largo = finops.escenarios_demanda_real(120, 2.2, 0.6, 0.2)

    assert corto[3]["costo_mensual"] == pytest.approx(largo[3]["costo_mensual"])
    assert mas_largo[3]["costo_mensual"] == pytest.approx(2 * largo[3]["costo_mensual"])
