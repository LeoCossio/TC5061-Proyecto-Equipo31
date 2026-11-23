"""Registra el consumo de CPU y memoria de los contenedores con `docker stats`.

Muestrea cada pocos segundos los contenedores del proyecto de Docker Compose y guarda una
fila por contenedor y por muestra. Se detiene con Ctrl+C o con la señal SIGTERM.

Uso desde la raíz del repositorio (con el sistema levantado):

    python scripts/monitorear_recursos.py --salida reportes_carga/recursos.csv --intervalo 2

Nota: `docker stats` expresa la CPU como porcentaje de un núcleo, así que un contenedor que
usa dos núcleos completos aparece con 200 %.
"""

import argparse
import csv
import json
import signal
import subprocess
import time
from pathlib import Path

UNIDADES_MIB = {"B": 1 / 1024**2, "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "kB": 1 / 1024**2 * 1000,
                "MB": 1 / 1.048576, "GB": 1000 / 1.048576}
SEGUIR = True


def a_mib(texto):
    """Convierte '123.4MiB' o '1.2GiB' a mebibytes."""
    texto = texto.strip()
    for unidad in sorted(UNIDADES_MIB, key=len, reverse=True):
        if texto.endswith(unidad):
            return float(texto[: -len(unidad)]) * UNIDADES_MIB[unidad]
    return float(texto)


def contenedores_del_proyecto():
    """Identificadores de los contenedores en ejecución de `docker compose`."""
    salida = subprocess.run(["docker", "compose", "ps", "-q"], capture_output=True, text=True, check=True)
    return [i for i in salida.stdout.split() if i]


def muestrear(ids):
    """Una lectura de `docker stats` para los contenedores indicados."""
    salida = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids],
                            capture_output=True, text=True, check=True)
    filas = []
    for linea in salida.stdout.splitlines():
        dato = json.loads(linea)
        filas.append({
            "contenedor": dato["Name"],
            "cpu_pct": float(dato["CPUPerc"].rstrip("%")),
            "memoria_mib": a_mib(dato["MemUsage"].split("/")[0]),
            "memoria_limite_mib": a_mib(dato["MemUsage"].split("/")[1]),
            "memoria_pct": float(dato["MemPerc"].rstrip("%")),
            "procesos": int(dato["PIDs"]),
        })
    return filas


def principal():
    global SEGUIR
    parser = argparse.ArgumentParser(description="Monitor de recursos de los contenedores")
    parser.add_argument("--salida", default="reportes_carga/recursos.csv")
    parser.add_argument("--intervalo", type=float, default=2.0, help="segundos entre muestras")
    args = parser.parse_args()

    def detener(*_):
        global SEGUIR
        SEGUIR = False

    signal.signal(signal.SIGINT, detener)
    signal.signal(signal.SIGTERM, detener)

    ids = contenedores_del_proyecto()
    if not ids:
        raise SystemExit("No hay contenedores en ejecución. Levanta el sistema con: docker compose up --build --wait")
    ruta = Path(args.salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["timestamp", "contenedor", "cpu_pct", "memoria_mib", "memoria_limite_mib", "memoria_pct", "procesos"]
    with ruta.open("w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        while SEGUIR:
            inicio = time.time()
            for fila in muestrear(ids):
                escritor.writerow({"timestamp": f"{inicio:.1f}", **fila})
            archivo.flush()
            time.sleep(max(0.0, args.intervalo - (time.time() - inicio)))
    print(f"Monitoreo guardado en {ruta}")


if __name__ == "__main__":
    principal()
