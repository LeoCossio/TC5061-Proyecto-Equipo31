"""Análisis FinOps: costo por inferencia del sistema a partir de la prueba de carga.

Combina el rendimiento y el consumo de recursos medidos (reports/carga/**/resumen_carga.json)
con precios de referencia de una nube pública y calcula:

  1. El costo marginal de cómputo por inferencia con el sistema trabajando a su capacidad
     sostenible, para la configuración base y las optimizadas.
  2. El costo por inferencia con la demanda real de MiBici (una predicción de P1 y una de P2
     por estación y por hora) para cuatro formas de desplegar el servicio.

Los precios son de referencia (AWS, región us-east-1, Linux, consultados el 22 de noviembre de
2026) y deben confirmarse antes de usarlos para presupuestar. Cambiar `PRECIOS` basta para
recalcular todo.

Uso desde la raíz del repositorio:

    python scripts/finops.py --entrada reports/carga --salida reports/finops --segundos-lote 120
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

PRECIOS = {
    "ec2_t3_medium_hora": 0.0416,   # 2 vCPU, 4 GiB
    "ec2_t3_small_hora": 0.0208,    # 2 vCPU, 2 GiB
    "fargate_vcpu_hora": 0.04048,
    "fargate_gb_hora": 0.004445,
    "fargate_minimo_s": 60,         # Fargate factura por segundo, con un mínimo de 1 minuto
}
HORAS_POR_MES = 730
ESTACIONES = 471                    # estaciones activas en el periodo actual
MODELOS_POR_HORA = 2                # P1 (demanda) y P2 (riesgo) por estación y hora
TAREA_FARGATE = {"vcpu": 0.25, "gb": 0.5}  # configuración mínima de Fargate

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_SEC, REJILLA, SUPERFICIE = "#0b0b0b", "#52514e", "#e6e6e3", "#fcfcfb"


def costo_vcpu_segundo_ec2():
    """Costo de un vCPU-segundo en una t3.medium (2 vCPU)."""
    return PRECIOS["ec2_t3_medium_hora"] / (2 * 3600)


def costo_hora_fargate(vcpu, gb):
    return vcpu * PRECIOS["fargate_vcpu_hora"] + gb * PRECIOS["fargate_gb_hora"]


def marginal_a_plena_carga(resumen):
    """Costo de cómputo por inferencia cuando el sistema trabaja a su capacidad sostenible."""
    capacidad = resumen["capacidad_sostenible"]
    nucleos = capacidad["cpu_api_pct"] / 100
    cpu_s = nucleos / capacidad["rps"]
    return {
        "usuarios": capacidad["usuarios"],
        "rps": capacidad["rps"],
        "p95_ms": capacidad["p95_ms"],
        "nucleos_api": nucleos,
        "cpu_s_por_inferencia": cpu_s,
        "costo_por_inferencia": cpu_s * costo_vcpu_segundo_ec2(),
        "memoria_api_mib": capacidad["memoria_api_mib"],
        "memoria_mlflow_mib": capacidad["memoria_mlflow_mib"],
    }


def escenarios_demanda_real(segundos_lote, mlflow_gb_base, mlflow_gb_opt, api_gb):
    """Costo mensual y por inferencia de cuatro formas de desplegar con la demanda real."""
    por_hora = ESTACIONES * MODELOS_POR_HORA
    por_mes = por_hora * HORAS_POR_MES
    corrida_s = max(segundos_lote, PRECIOS["fargate_minimo_s"])
    costo_corrida = corrida_s * costo_hora_fargate(**TAREA_FARGATE) / 3600
    definiciones = [
        ("1. Sistema base 24/7 en una instancia t3.medium (4 GiB)",
         PRECIOS["ec2_t3_medium_hora"] * HORAS_POR_MES,
         f"API + MLflow base: {mlflow_gb_base + api_gb:.1f} GiB de memoria"),
        ("2. Sistema optimizado 24/7 en una instancia t3.small (2 GiB)",
         PRECIOS["ec2_t3_small_hora"] * HORAS_POR_MES,
         f"API + MLflow optimizado: {mlflow_gb_opt + api_gb:.1f} GiB de memoria"),
        ("3. Solo la API con modelos empaquetados, 24/7 en Fargate (0.25 vCPU, 0.5 GB)",
         costo_hora_fargate(**TAREA_FARGATE) * HORAS_POR_MES,
         "Sin MLflow en producción"),
        ("4. Predicción por lotes cada hora en Fargate (0.25 vCPU, 0.5 GB)",
         costo_corrida * HORAS_POR_MES,
         f"{HORAS_POR_MES} ejecuciones al mes de {corrida_s:g} s facturados"),
    ]
    return por_hora, por_mes, [
        {"escenario": nombre, "costo_mensual": costo, "nota": nota,
         "costo_por_inferencia": costo / por_mes, "costo_por_millon": 1e6 * costo / por_mes}
        for nombre, costo, nota in definiciones]


def grafica(escenarios, por_hora, ruta):
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_SEC, "xtick.color": TINTA_SEC,
        "ytick.color": TINTA_SEC, "text.color": TINTA, "axes.grid": True, "grid.color": REJILLA,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    etiquetas = ["1. Base 24/7 (t3.medium)", "2. Optimizado 24/7 (t3.small)", "3. API 24/7 (Fargate)",
                 "4. Por lotes cada hora (Fargate)"]
    valores = [e["costo_por_millon"] for e in escenarios]
    posiciones = list(range(len(valores)))[::-1]
    # Una escala logarítmica no admite barras (su longitud depende del origen): se usan puntos.
    a.hlines(posiciones, 1e-1, valores, color=REJILLA, linewidth=1.5)
    a.plot(valores, posiciones, "o", color=AZUL, markersize=9, markeredgecolor=SUPERFICIE, markeredgewidth=2)
    a.set_xscale("log")
    a.set_xlim(1e-1, 2e2)
    a.set_yticks(posiciones, etiquetas, fontsize=9)
    a.set_xlabel("USD por millón de inferencias (escala logarítmica)")
    a.set_title("Costo con la demanda real de MiBici", pad=10)
    for y, valor in zip(posiciones, valores):
        a.annotate(f"${valor:,.2f}", (valor, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9)
    a.grid(axis="y", visible=False)

    demanda = [10 ** (x / 4) for x in range(8, 21)]  # de 100 a 100,000 inferencias por hora (capacidad suficiente)
    fijo = {
        "Base 24/7 (t3.medium)": PRECIOS["ec2_t3_medium_hora"],
        "API 24/7 (Fargate)": costo_hora_fargate(**TAREA_FARGATE),
    }
    for (nombre, por_hora_fijo), color in zip(fijo.items(), (AZUL, NARANJA)):
        b.plot(demanda, [por_hora_fijo / d for d in demanda], color=color, linewidth=2, label=nombre)
    lote = escenarios[3]["costo_mensual"] / HORAS_POR_MES
    b.plot(demanda, [lote / d for d in demanda], color=AQUA, linewidth=2, label="Por lotes (1 ejecución/hora)")
    b.axvline(por_hora, color=TINTA_SEC, linestyle="--", linewidth=1)
    b.text(por_hora * 1.2, 2e-8, "demanda real\n(942 por hora)", fontsize=8, color=TINTA_SEC)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("Inferencias por hora")
    b.set_ylabel("USD por inferencia")
    b.set_title("Costo según el volumen", pad=10)
    b.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def principal():
    parser = argparse.ArgumentParser(description="Análisis FinOps del sistema MiBici")
    parser.add_argument("--entrada", default="reports/carga")
    parser.add_argument("--salida", default="reports/finops")
    parser.add_argument("--segundos-lote", type=float, default=120.0,
                        help="segundos facturados por ejecución por lotes. Localmente el trabajo (arranque del "
                             "contenedor, carga del modelo y 942 predicciones) tarda unos 2 s; en la nube se "
                             "suman la descarga de la imagen (1.3 GB) y el arranque de la tarea")
    args = parser.parse_args()
    entrada, salida = Path(args.entrada), Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    configuraciones = {
        "base": entrada / "resumen_carga.json",
        "optimizada_1_proceso": entrada / "optimizado_w1" / "resumen_carga.json",
        "optimizada_4_procesos": entrada / "optimizado_w4" / "resumen_carga.json",
    }
    resumenes = {k: json.loads(v.read_text()) for k, v in configuraciones.items() if v.exists()}
    plena = {k: marginal_a_plena_carga(v) for k, v in resumenes.items()}

    base = resumenes["base"]["memoria_maxima_mib"]
    optimizada = resumenes.get("optimizada_1_proceso", resumenes["base"])["memoria_maxima_mib"]
    por_hora, por_mes, escenarios = escenarios_demanda_real(
        args.segundos_lote, base["mlflow"] / 1024, optimizada["mlflow"] / 1024, base["api"] / 1024)

    resultado = {"precios_referencia": PRECIOS, "demanda_real": {"inferencias_por_hora": por_hora,
                 "inferencias_por_mes": por_mes}, "a_plena_carga": plena, "escenarios": escenarios}
    (salida / "finops_resumen.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    grafica(escenarios, por_hora, salida / "finops_costos.png")

    print("Costo marginal de cómputo a plena carga (por inferencia):")
    for k, v in plena.items():
        print(f"  {k:<24} {v['rps']:6.1f} req/s | {v['cpu_s_por_inferencia'] * 1000:6.2f} ms de CPU | "
              f"${v['costo_por_inferencia']:.2e} (${v['costo_por_inferencia'] * 1e6:.3f} por millón)")
    print(f"\nDemanda real: {por_hora:,} inferencias por hora, {por_mes:,} al mes")
    for e in escenarios:
        print(f"  {e['escenario']}\n     ${e['costo_mensual']:.2f}/mes -> ${e['costo_por_inferencia']:.2e} por inferencia "
              f"(${e['costo_por_millon']:,.2f} por millón)")


if __name__ == "__main__":
    principal()
