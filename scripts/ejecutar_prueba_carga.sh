#!/usr/bin/env bash
# Ejecuta la prueba de carga completa: monitorea los contenedores con docker stats mientras
# Locust aplica la carga por escalones, y luego genera las gráficas y el resumen.
#
# Uso desde la raíz del repositorio:
#   docker compose up --build --wait                       # sistema levantado
#   uv venv .venv-carga && uv pip install --python .venv-carga/bin/python -r carga/requirements-carga.txt
#   scripts/ejecutar_prueba_carga.sh
#
# Variables opcionales: MIBICI_ESCALON_S (segundos por escalón, 60 por defecto), MIBICI_USUARIOS
# (usuarios por escalón, separados por comas), HOST, SALIDA (CSV de Locust y de docker stats) y
# GRAFICAS (carpeta de gráficas; reports/carga por defecto). Si el sistema se levantó con varios
# archivos de Compose, exporta COMPOSE_FILE para que el monitor encuentre los contenedores.
set -euo pipefail

HOST="${HOST:-http://127.0.0.1:8000}"
SALIDA="${SALIDA:-reportes_carga}"
PY=".venv-carga/bin/python"
mkdir -p "$SALIDA"

curl -sf "$HOST/health" >/dev/null || { echo "La API no responde en $HOST. Ejecuta: docker compose up --build --wait"; exit 1; }

echo "== Iniciando el monitoreo de recursos (docker stats)"
"$PY" scripts/monitorear_recursos.py --salida "$SALIDA/recursos.csv" --intervalo 2 &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT
sleep 6   # líneas base de CPU y memoria en reposo antes de la carga

echo "== Aplicando la carga con Locust"
.venv-carga/bin/locust -f carga/locustfile.py --headless --host "$HOST" \
  --csv "$SALIDA/locust" --csv-full-history --html "$SALIDA/locust.html" --only-summary

sleep 6   # líneas de CPU y memoria en reposo después de la carga
kill $MONITOR 2>/dev/null || true
wait $MONITOR 2>/dev/null || true
trap - EXIT

echo "== Generando gráficas y resumen"
"$PY" scripts/analizar_carga.py --entrada "$SALIDA" --salida "${GRAFICAS:-reports/carga}" \
  --duracion "${MIBICI_ESCALON_S:-60}" --usuarios "${MIBICI_USUARIOS:-5,10,20,40,60,100,200}"
