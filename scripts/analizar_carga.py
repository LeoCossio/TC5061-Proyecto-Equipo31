"""Genera las gráficas y el resumen de la prueba de carga (Locust + docker stats).

Lee los CSV que producen `carga/locustfile.py` y `scripts/monitorear_recursos.py` y escribe:

    carga_latencia.png        latencia p50, p95 y p99 a lo largo de la prueba
    carga_rps_errores.png     peticiones por segundo y tasa de error
    carga_capacidad.png       rendimiento y latencia p95 según el número de usuarios
    carga_recursos.png        CPU y memoria de cada contenedor
    resumen_escalones.csv     métricas de cada escalón de carga (régimen estable)
    resumen_carga.json        resumen global y capacidad sostenible

La capacidad sostenible es el mayor rendimiento observado en un escalón con tasa de error
menor a 1 % y latencia p95 por debajo del objetivo (500 ms por defecto).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Paleta categórica validada (ranuras 1 a 3) y tintas de texto.
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_SEC, REJILLA, SUPERFICIE = "#0b0b0b", "#52514e", "#e6e6e3", "#fcfcfb"
USUARIOS = [5, 10, 20, 40, 60, 100, 200]
COLORES_CONTENEDOR = {"api": AZUL, "mlflow": NARANJA}


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_SEC, "xtick.color": TINTA_SEC,
        "ytick.color": TINTA_SEC, "text.color": TINTA, "axes.grid": True, "grid.color": REJILLA,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2,
    })


def cargar_historia(ruta):
    h = pd.read_csv(ruta)
    h = h[h["Name"] == "Aggregated"].copy()
    h["t"] = h["Timestamp"] - h["Timestamp"].min()
    return h.reset_index(drop=True)


def cargar_recursos(ruta, origen):
    r = pd.read_csv(ruta)
    r["t"] = r["timestamp"] - origen
    r["servicio"] = r["contenedor"].str.extract(r"-(api|mlflow|registry-init)-\d+$")[0].fillna(r["contenedor"])
    return r[r["servicio"].isin(["api", "mlflow"])].copy()


def bandas(ax, duracion, usuarios, rotular=True):
    """Sombrea los escalones alternos y rotula cuántos usuarios tiene cada uno."""
    for i, u in enumerate(usuarios):
        if i % 2 == 0:
            ax.axvspan(i * duracion, (i + 1) * duracion, color="#f1f0ec", zorder=0)
        if rotular:
            ax.text((i + 0.5) * duracion, 1.01, f"{u}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TINTA_SEC)
    if rotular:
        ax.text(0, 1.085, "Usuarios simultáneos por escalón", transform=ax.transAxes, fontsize=8,
                color=TINTA_SEC, va="bottom")


def resumen_por_escalon(h, r, duracion, usuarios):
    """Métricas del régimen estable (últimos 40 s) de cada escalón."""
    filas = []
    for i, u in enumerate(usuarios):
        inicio, fin = i * duracion, (i + 1) * duracion
        ventana = h[(h["t"] >= inicio + duracion - 40) & (h["t"] < fin)]
        if ventana.empty:
            continue
        rps = ventana["Requests/s"].mean()
        fallas = ventana["Failures/s"].mean()
        fila = {
            "usuarios": u, "rps": rps, "fallas_por_s": fallas,
            "tasa_error_pct": 100 * fallas / rps if rps else 0.0,
            "p50_ms": ventana["50%"].median(), "p95_ms": ventana["95%"].median(),
            "p99_ms": ventana["99%"].median(),
        }
        recursos = r[(r["t"] >= inicio + duracion - 40) & (r["t"] < fin)]
        for servicio in ("api", "mlflow"):
            datos = recursos[recursos["servicio"] == servicio]
            fila[f"cpu_{servicio}_pct"] = datos["cpu_pct"].mean() if len(datos) else float("nan")
            fila[f"memoria_{servicio}_mib"] = datos["memoria_mib"].max() if len(datos) else float("nan")
        filas.append(fila)
    return pd.DataFrame(filas)


def grafica_latencia(h, duracion, usuarios, slo, ruta):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bandas(ax, duracion, usuarios)
    for col, color, etiqueta, desplazamiento in (("50%", AZUL, "p50", -12), ("95%", NARANJA, "p95", 0),
                                                 ("99%", AQUA, "p99", 12)):
        ax.plot(h["t"], h[col], color=color, label=etiqueta)
        ax.annotate(etiqueta, (h["t"].iloc[-1], h[col].iloc[-1]), xytext=(6, desplazamiento),
                    textcoords="offset points", color=TINTA_SEC, va="center", fontsize=9)
    ax.axhline(slo, color=TINTA_SEC, linestyle="--", linewidth=1)
    ax.text(2, slo * 1.08, f"objetivo p95 = {slo:g} ms", color=TINTA_SEC, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(0, h["t"].max() + 25)
    ax.set_xlabel("Tiempo de prueba (s)")
    ax.set_ylabel("Latencia (ms, escala logarítmica)")
    ax.set_title("Latencia de las peticiones a lo largo de la prueba", pad=30)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def grafica_rps_errores(h, duracion, usuarios, ruta):
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    bandas(a, duracion, usuarios)
    bandas(b, duracion, usuarios, rotular=False)
    a.plot(h["t"], h["Requests/s"], color=AZUL)
    a.set_ylabel("Peticiones por segundo")
    a.set_title("Rendimiento (peticiones por segundo)", pad=30)
    tasa = 100 * h["Failures/s"] / h["Requests/s"].where(h["Requests/s"] > 0)
    b.plot(h["t"], tasa.fillna(0), color=NARANJA)
    b.set_ylabel("Peticiones fallidas (%)")
    b.set_title("Tasa de error")
    b.set_xlabel("Tiempo de prueba (s)")
    b.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def grafica_capacidad(esc, slo, ruta):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.6))
    a.plot(esc["usuarios"], esc["rps"], marker="o", markersize=7, color=AZUL, markeredgecolor=SUPERFICIE,
           markeredgewidth=2)
    a.set_xlabel("Usuarios simultáneos")
    a.set_ylabel("Peticiones por segundo")
    a.set_title("Rendimiento según la carga")
    b.plot(esc["usuarios"], esc["p95_ms"], marker="o", markersize=7, color=NARANJA, markeredgecolor=SUPERFICIE,
           markeredgewidth=2)
    b.axhline(slo, color=TINTA_SEC, linestyle="--", linewidth=1)
    b.text(esc["usuarios"].min(), slo * 1.1, f"objetivo p95 = {slo:g} ms", color=TINTA_SEC, fontsize=8)
    b.set_yscale("log")
    b.set_xlabel("Usuarios simultáneos")
    b.set_ylabel("Latencia p95 (ms, escala logarítmica)")
    b.set_title("Latencia p95 según la carga")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def grafica_recursos(r, duracion, usuarios, ruta):
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    bandas(a, duracion, usuarios)
    bandas(b, duracion, usuarios, rotular=False)
    for servicio, nombre in (("api", "API (FastAPI)"), ("mlflow", "MLflow")):
        datos = r[r["servicio"] == servicio].sort_values("t")
        a.plot(datos["t"], datos["cpu_pct"], color=COLORES_CONTENEDOR[servicio], label=nombre)
        b.plot(datos["t"], datos["memoria_mib"], color=COLORES_CONTENEDOR[servicio], label=nombre)
        b.text(datos["t"].iloc[-1] + 3, datos["memoria_mib"].iloc[-1], nombre.split()[0], color=TINTA_SEC,
               va="center", fontsize=9)
    a.set_ylabel("CPU (% de un núcleo)")
    a.set_title("Uso de CPU de cada contenedor", pad=30)
    a.legend(loc="upper left")
    b.set_ylabel("Memoria (MiB)")
    b.set_title("Uso de memoria de cada contenedor")
    b.set_xlabel("Tiempo de prueba (s)")
    b.set_ylim(bottom=0)
    b.set_xlim(left=min(-10, r["t"].min()))
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def principal():
    parser = argparse.ArgumentParser(description="Análisis de la prueba de carga")
    parser.add_argument("--entrada", default="reportes_carga")
    parser.add_argument("--salida", default="reports/carga")
    parser.add_argument("--duracion", type=int, default=60, help="segundos por escalón")
    parser.add_argument("--usuarios", default=",".join(map(str, USUARIOS)))
    parser.add_argument("--slo-p95-ms", type=float, default=500.0)
    args = parser.parse_args()
    usuarios = [int(u) for u in args.usuarios.split(",")]

    entrada, salida = Path(args.entrada), Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    estilo()

    historia = cargar_historia(entrada / "locust_stats_history.csv")
    origen = pd.read_csv(entrada / "locust_stats_history.csv")["Timestamp"].min()
    recursos = cargar_recursos(entrada / "recursos.csv", origen)
    escalones = resumen_por_escalon(historia, recursos, args.duracion, usuarios)
    escalones.round(3).to_csv(salida / "resumen_escalones.csv", index=False)

    grafica_latencia(historia, args.duracion, usuarios, args.slo_p95_ms, salida / "carga_latencia.png")
    grafica_rps_errores(historia, args.duracion, usuarios, salida / "carga_rps_errores.png")
    grafica_capacidad(escalones, args.slo_p95_ms, salida / "carga_capacidad.png")
    grafica_recursos(recursos, args.duracion, usuarios, salida / "carga_recursos.png")

    totales = pd.read_csv(entrada / "locust_stats.csv")
    sostenibles = escalones[(escalones["tasa_error_pct"] < 1) & (escalones["p95_ms"] <= args.slo_p95_ms)]
    mejor = sostenibles.loc[sostenibles["rps"].idxmax()] if len(sostenibles) else None
    en_reposo = recursos[recursos["t"] < 0]
    resumen = {
        "duracion_escalon_s": args.duracion, "usuarios_por_escalon": usuarios,
        "objetivo_p95_ms": args.slo_p95_ms,
        "capacidad_sostenible": None if mejor is None else {k: float(v) for k, v in mejor.items()},
        "maximo_rps_observado": float(escalones["rps"].max()),
        "totales": totales.round(3).to_dict(orient="records"),
        "recursos_en_reposo": {
            s: {"cpu_pct": float(en_reposo[en_reposo["servicio"] == s]["cpu_pct"].mean()),
                "memoria_mib": float(en_reposo[en_reposo["servicio"] == s]["memoria_mib"].mean())}
            for s in ("api", "mlflow") if len(en_reposo[en_reposo["servicio"] == s])},
        "memoria_maxima_mib": {s: float(recursos[recursos["servicio"] == s]["memoria_mib"].max())
                               for s in ("api", "mlflow")},
    }
    (salida / "resumen_carga.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(escalones.round(1).to_string(index=False))
    print("\nCapacidad sostenible:", resumen["capacidad_sostenible"])


if __name__ == "__main__":
    principal()
