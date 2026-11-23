"""Escenario de carga de Locust para la API de predicción de MiBici.

Simula usuarios que consultan los tres endpoints de predicción con la mezcla P1 60 %,
P2 20 % y P3 20 %, con una pausa aleatoria de 0.2 a 0.6 s entre peticiones. Cada petición
varía ligeramente los valores de entrada (dentro de los rangos que acepta el esquema) y
verifica que la respuesta sea HTTP 200 y contenga el campo esperado.

La carga crece por escalones (ver `EscalonesDeCarga`) hasta saturar el servicio, lo que
permite identificar el punto en que la latencia o los errores se disparan.

Uso (con el sistema levantado con `docker compose up --build --wait`):

    locust -f carga/locustfile.py --headless --host http://127.0.0.1:8000 \
        --csv reportes_carga/locust --csv-full-history --html reportes_carga/locust.html

Variables de entorno opcionales: MIBICI_ESCALON_S (duración de cada escalón en segundos, 60 por
defecto), MIBICI_USUARIOS (usuarios de cada escalón, separados por comas) y MIBICI_CALENTAMIENTO
(peticiones de calentamiento por modelo, 30 por defecto).
"""

import json
import os
import random
import urllib.request
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, events, task

EJEMPLOS = Path(__file__).resolve().parents[1] / "references" / "ejemplos_api"
CAMPOS = {"p1": "demanda_predicha", "p2": "clase_predicha", "p3": "cluster"}
BASE = {modelo: json.loads((EJEMPLOS / f"{modelo}.json").read_text()) for modelo in CAMPOS}


def variar(modelo, rng):
    """Devuelve una copia del ejemplo con variaciones pequeñas y válidas."""
    datos = dict(BASE[modelo])
    if modelo == "p1":
        datos["hora"] = rng.randint(0, 23)
        datos["dia_semana"] = rng.randint(0, 6)
        datos["fin_semana"] = int(datos["dia_semana"] >= 5)
        for campo in ("lag_1", "lag_2", "lag_3", "lag_24", "lag_168"):
            datos[campo] = float(rng.randint(0, 15))
    elif modelo == "p2":
        datos["hora"] = rng.randint(0, 23)
        datos["dia_semana"] = rng.randint(0, 6)
        datos["fin_semana"] = int(datos["dia_semana"] >= 5)
        datos["capacidad"] = float(rng.choice([15, 20, 25, 30]))
        datos["salidas_lag_1"] = float(rng.randint(0, 20))
        datos["llegadas_lag_1"] = float(rng.randint(0, 20))
        datos["flujo_neto_lag_1"] = datos["llegadas_lag_1"] - datos["salidas_lag_1"]
    else:
        datos["viajes_diarios"] = float(rng.randint(1, 300))
        datos["manana"] = round(rng.uniform(0.1, 0.5), 3)
        datos["mediodia"] = round(rng.uniform(0.1, 0.4), 3)
        datos["tarde"] = round(rng.uniform(0.1, 0.4), 3)
    return datos


@events.test_start.add_listener
def calentar_modelos(environment, **_):
    """Hace una petición a cada modelo antes de medir.

    La API carga P2 y P3 del Registry en la primera petición; sin este paso esa carga
    inicial aparecería como una latencia anómala en las estadísticas. Se envían varias peticiones
    por modelo para que, si la API tiene varios procesos, cada uno cargue sus modelos.
    """
    repeticiones = int(os.environ.get("MIBICI_CALENTAMIENTO", "30"))
    for modelo in CAMPOS:
        cuerpo = json.dumps(BASE[modelo]).encode()
        for _ in range(repeticiones):
            peticion = urllib.request.Request(
                f"{environment.host}/predict/{modelo}", data=cuerpo,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(peticion, timeout=120).read()


class UsuarioMiBici(HttpUser):
    wait_time = between(0.2, 0.6)

    def on_start(self):
        self.rng = random.Random()

    def predecir(self, modelo):
        with self.client.post(
            f"/predict/{modelo}", json=variar(modelo, self.rng),
            name=f"POST /predict/{modelo}", catch_response=True,
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}")
            elif CAMPOS[modelo] not in respuesta.json():
                respuesta.failure(f"falta el campo {CAMPOS[modelo]}")

    @task(6)
    def p1_demanda(self):
        self.predecir("p1")

    @task(2)
    def p2_riesgo(self):
        self.predecir("p2")

    @task(2)
    def p3_grupos(self):
        self.predecir("p3")


class EscalonesDeCarga(LoadTestShape):
    """Carga por escalones: (usuarios simultáneos) durante MIBICI_ESCALON_S segundos cada uno."""

    USUARIOS = [int(u) for u in os.environ.get("MIBICI_USUARIOS", "5,10,20,40,60,100,200").split(",")]
    DURACION = int(os.environ.get("MIBICI_ESCALON_S", "60"))

    def tick(self):
        escalon = int(self.get_run_time() // self.DURACION)
        if escalon >= len(self.USUARIOS):
            return None
        usuarios = self.USUARIOS[escalon]
        return usuarios, max(usuarios, 10)
