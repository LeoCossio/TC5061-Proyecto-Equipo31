#!/usr/bin/env bash
# Construye la imagen, levanta el contenedor y comprueba los endpoints de la API.
#
# Uso desde la raíz del repositorio:
#   scripts/probar_contenedor.sh
#   MLFLOW_TRACKING_URI=http://host.docker.internal:5000 scripts/probar_contenedor.sh
#       # además prueba P2 y P3, que se cargan del Model Registry de MLflow
#
# Variables opcionales: IMAGEN, NOMBRE, PUERTO, SIN_CONSTRUIR=1,
# CONSERVAR=1 (no elimina el contenedor al terminar).
set -euo pipefail

IMAGEN="${IMAGEN:-mibici-api:1.0}"
NOMBRE="${NOMBRE:-mibici-api}"
PUERTO="${PUERTO:-8000}"
EJEMPLOS="references/ejemplos_api"
URL="http://127.0.0.1:${PUERTO}"
fallos=0

limpiar() {
  if [ "${CONSERVAR:-0}" != "1" ]; then
    docker rm -f "$NOMBRE" >/dev/null 2>&1 || true
  fi
}
trap limpiar EXIT

verificar() {  # descripción, método, ruta, código esperado, texto esperado, [archivo JSON]
  local descripcion="$1" metodo="$2" ruta="$3" esperado="$4" texto="$5" archivo="${6:-}"
  local cuerpo codigo respuesta
  cuerpo="$(mktemp)"
  if [ -n "$archivo" ]; then
    codigo="$(curl -s -m 180 -o "$cuerpo" -w '%{http_code}' -X "$metodo" "$URL$ruta" \
      -H 'Content-Type: application/json' -d @"$archivo")"
  else
    codigo="$(curl -s -m 30 -o "$cuerpo" -w '%{http_code}' -X "$metodo" "$URL$ruta")"
  fi
  respuesta="$(cat "$cuerpo")"; rm -f "$cuerpo"
  if [ "$codigo" = "$esperado" ] && printf '%s' "$respuesta" | grep -q "$texto"; then
    echo "[OK]    $descripcion -> HTTP $codigo  ${respuesta:0:110}"
  else
    echo "[FALLA] $descripcion -> HTTP $codigo (se esperaba $esperado)  ${respuesta:0:200}"
    fallos=$((fallos + 1))
  fi
}

if [ "${SIN_CONSTRUIR:-0}" != "1" ]; then
  echo "== Construyendo la imagen $IMAGEN con Docker"
  docker build -f Containerfile -t "$IMAGEN" .
fi

echo "== Levantando el contenedor $NOMBRE en el puerto $PUERTO"
docker rm -f "$NOMBRE" >/dev/null 2>&1 || true
entorno=()
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
  entorno=(-e "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI")
fi
# ${entorno[@]+...} evita el error de "variable sin definir" con arreglos vacíos en bash 3.2 (macOS)
docker run -d --name "$NOMBRE" -p "${PUERTO}:8000" ${entorno[@]+"${entorno[@]}"} "$IMAGEN" >/dev/null

echo "== Esperando a que la API responda"
for _ in $(seq 1 60); do
  if curl -s -m 2 -o /dev/null "$URL/health"; then break; fi
  sleep 1
done

echo "== Probando los endpoints"
verificar "GET  /health" GET /health 200 '"status":"ok"'
verificar "POST /predict/p1 (modelo empaquetado en la imagen)" POST /predict/p1 200 'demanda_predicha' "$EJEMPLOS/p1.json"
verificar "POST /predict/p1 con hora = 25 (debe rechazarse)" POST /predict/p1 422 'less_than_equal' "$EJEMPLOS/p1_hora_invalida.json"
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
  verificar "POST /predict/p2 (modelo del Registry)" POST /predict/p2 200 'clase_predicha' "$EJEMPLOS/p2.json"
  verificar "POST /predict/p3 (modelo del Registry)" POST /predict/p3 200 'cluster' "$EJEMPLOS/p3.json"
else
  echo "[OMITIDO] P2 y P3: definir MLFLOW_TRACKING_URI para probarlos desde el Registry"
fi

if [ "$fallos" -eq 0 ]; then
  echo "== Todas las pruebas pasaron"
else
  echo "== $fallos prueba(s) fallaron"
  docker logs --tail 20 "$NOMBRE" || true
  exit 1
fi
