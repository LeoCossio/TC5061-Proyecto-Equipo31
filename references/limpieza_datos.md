# Limpieza y normalización de datos — MiBici (Equipo 31)

**Periodo de estudio:** octubre de 2025 a septiembre de 2026  
**Componente de ingesta:** `mibici/dataset.py`  
**Alcance:** preparación compartida de datos para P1 (regresión), P2 (clasificación) y P3 (clustering).

## 1. Objetivo

Se implementó un proceso automatizado de descarga, limpieza y normalización de registros históricos de viajes de MiBici. Su propósito es entregar a los tres modelos una base común de datos operativos, evitando repetir la ingesta o aplicar criterios de depuración diferentes en cada pipeline.

Los archivos originales y los datos derivados se almacenan por separado. El conjunto depurado se guarda en Parquet mensual para facilitar la lectura y agregación de millones de registros.

## 2. Fuentes, periodo y volumen

**Fuente oficial:** [MiBici — Datos abiertos](https://www.mibici.net/es/datos-abiertos/).

Se procesaron **12 CSV mensuales de viajes** —de octubre de 2025 a septiembre de 2026— y la **nomenclatura de estaciones de septiembre de 2026**. El periodo comprende un ciclo anual de actividad y permite estudiar variaciones horarias, semanales y entre meses.

| Indicador | Resultado |
|---|---:|
| Archivos mensuales de viajes | 12 |
| Registros originales | 4,497,877 |
| Registros procesados | 4,497,119 |
| Registros duplicados eliminados | 758 |
| Estaciones identificadas en los viajes | 471 |
| Estaciones registradas en el catálogo | 484 |
| Archivos Parquet generados | 12 |

Las cifras corresponden a la ejecución de ingesta reportada y probada por el equipo para el periodo indicado.

## 3. Proceso de ingesta y limpieza

La implementación de `mibici/dataset.py` contempla las siguientes etapas:

1. **Descarga:** obtiene los archivos mensuales de viajes y el catálogo de estaciones desde las fuentes oficiales.
2. **Codificación:** intenta la lectura en UTF-8 y utiliza CP1252 cuando se requiere.
3. **Normalización:** unifica los nombres de columnas entre meses.
4. **Conversión de tipos:** convierte fechas a `datetime` e identificadores de estaciones a tipos numéricos.
5. **Eliminación de duplicados:** retira filas completamente duplicadas.
6. **Validación:** descarta filas con nulos en campos operativos esenciales o con una fecha de finalización anterior a la de inicio.
7. **Selección de atributos:** conserva `viaje_id`, `inicio`, `fin`, `origen_id` y `destino_id`, sin incorporar identificadores de usuario ni variables demográficas a los datos procesados.
8. **Persistencia:** escribe un archivo Parquet por mes mediante PyArrow y compresión Snappy.

El proceso de limpieza es común; la construcción de características específicas de P1, P2 y P3 ocurre después, en sus respectivos módulos.

## 4. Resultados de los controles de calidad

| Comprobación | Resultado reportado |
|---|---:|
| Duplicados exactos eliminados | 758 |
| Registros excluidos por fechas inválidas o nulos esenciales | 0 |
| Identificadores de viaje duplicados restantes | 0 |
| Fechas de inicio fuera del mes correspondiente | 0 |
| Archivos Parquet mensuales generados | 12 |

Los **758 duplicados** se localizaron en septiembre de 2026. Después de su eliminación se conservaron **4,497,119 viajes**. Estos resultados corresponden a la ejecución documentada y deberán comprobarse nuevamente si cambian las fuentes descargadas.

## 5. Hallazgos y limitaciones de calidad

**Codificación.** El archivo de enero de 2026 requirió CP1252; los otros once pudieron procesarse con UTF-8. La alternativa de lectura evita errores de decodificación.

**Estaciones ausentes del catálogo.** Los identificadores **405 y 407** aparecen en los viajes pero no en la nomenclatura de septiembre de 2026. Sus registros se conservaron para no descartar actividad válida sin justificación. El universo de estaciones activas a lo largo del año no tiene por qué coincidir con el catálogo de un solo mes.

**Capacidad histórica.** La nomenclatura es una fotografía de septiembre de 2026; no garantiza que la capacidad de cada estación haya sido constante durante el periodo. Esta limitación es relevante para P2, cuyo indicador de presión relativa utiliza capacidad de estación.

**Privacidad.** Los archivos originales pueden incluir identificadores pseudónimos y variables demográficas. Los Parquet destinados al modelado conservan únicamente las cinco columnas operativas; ni los datos originales ni los procesados se publican en GitHub.

## 6. Ejecución y almacenamiento reproducible

Desde la raíz del repositorio, con las dependencias de `requirements.txt` instaladas:

```bash
python -m mibici.dataset
```

| Ruta | Contenido |
|---|---|
| `data/raw/` | CSV originales de viajes |
| `data/external/` | Catálogo o nomenclatura de estaciones |
| `data/processed/` | Parquet mensuales normalizados |

El archivo `.gitignore` excluye los datos descargados y procesados, las credenciales, los modelos serializados y los archivos locales de MLflow.

**Trazabilidad de los datos.** Los scripts de experimentación registran identificadores de versión de datos basados en SHA-256, junto con parámetros y etiquetas en MLflow. Este mecanismo permite identificar las entradas de una corrida, pero **no sustituye el manifiesto versionado en Git** que incluya nombre, cantidad de registros y hash individual de cada archivo. El responsable de la ingesta está completando dicho manifiesto; su incorporación y verificación deben confirmarse antes de la entrega.

## 7. Integración con los tres modelos

- **P1 — Regresión:** agrega salidas por estación y hora, incorpora intervalos sin viajes y construye variables de calendario, rezagos y promedios históricos causales. Los modelos se evalúan con partición cronológica.
- **P2 — Clasificación:** agrega **salidas y llegadas por estación-hora** para estimar un riesgo *proxy* de desbalance. El indicador de presión se calcula como `abs(llegadas - salidas) / capacidad`, con una etiqueta de riesgo definida por un **umbral equivalente al percentil 90 calculado únicamente sobre entrenamiento**. La clasificación utiliza 13 características y compara un baseline de persistencia con `RandomForestClassifier` optimizado. **Este indicador no equivale a inventario real de bicicletas.**
- **P3 — Clustering:** reúne los viajes por estación y construye perfiles de demanda, balance relativo, uso de fin de semana y distribución horaria. Se excluyen estaciones con menos de 100 salidas, lo que deja **470 estaciones** para el análisis mediante terciles de demanda y K-Means.

La separación entre datos procesados comunes y transformaciones específicas reduce duplicidad de código y contribuye a la consistencia de los experimentos.

## 8. Referencias

- MiBici. (s. f.). *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/
- MiBici. (s. f.). *Aviso de privacidad*. https://www.mibici.net/es/aviso-de-privacidad/
- pandas development team. (s. f.). *Parquet*. https://pandas.pydata.org/docs/user_guide/io.html#parquet
- Python Software Foundation. (s. f.). *hashlib — Secure hashes and message digests*. https://docs.python.org/3/library/hashlib.html
