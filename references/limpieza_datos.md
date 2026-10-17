# Limpieza y normalización de datos — MiBici

## 1. Objetivo

Implementar un proceso automatizado y reproducible para descargar, limpiar, normalizar y almacenar los datos históricos de viajes del sistema MiBici, utilizando Python y el formato Parquet.

Este proceso proporciona una base de datos común para los modelos de regresión (P1), clasificación (P2) y clustering (P3) del Equipo 31.

## 2. Datos utilizados

- **Fuente:** Portal oficial de datos abiertos de MiBici.
- **Periodo:** Octubre de 2025 a septiembre de 2026.
- **Archivos:** 12 CSV mensuales de viajes y una nomenclatura de estaciones de septiembre de 2026.
- **Registros originales:** 4,497,877.
- **Registros procesados:** 4,497,119.
- **Estaciones distintas en los viajes originales:** 471.
- **Estaciones en el catálogo:** 484.

## 3. Procedimiento de limpieza

El procesamiento se implementó en `mibici/dataset.py` y contempla:

1. Descarga automatizada de los archivos mensuales desde el portal oficial.
2. Lectura de archivos CSV utilizando UTF-8 y, cuando se presenta un error de decodificación, CP1252.
3. Normalización de nombres de columnas.
4. Conversión de fechas a tipos `datetime` e identificadores a `int64`.
5. Eliminación de filas completamente duplicadas.
6. Exclusión de registros con valores faltantes en campos esenciales o con fecha de finalización anterior al inicio.
7. Selección de columnas necesarias para el análisis, sin conservar identificadores de usuarios ni datos demográficos en los archivos procesados.
8. Almacenamiento de los archivos mensuales en formato Parquet, utilizando PyArrow y compresión Snappy.

## 4. Resultados

| Indicador | Resultado |
|---|---:|
| Registros originales | 4,497,877 |
| Registros procesados | 4,497,119 |
| Filas duplicadas eliminadas | 758 |
| Registros excluidos por fechas o nulos esenciales | 0 |
| Identificadores duplicados restantes | 0 |
| Fechas de inicio fuera del mes correspondiente | 0 |
| Archivos Parquet generados | 12 |

Los 758 registros duplicados se identificaron en septiembre de 2026.

## 5. Hallazgos y limitaciones

**Codificación:** el archivo de enero de 2026 requirió lectura mediante CP1252. Los otros once archivos se pudieron leer con UTF-8.

**Estaciones sin catálogo:** se identificaron viajes asociados con las estaciones 405 y 407, ausentes de la nomenclatura de septiembre de 2026. Estos registros se conservaron en los datos procesados para evitar pérdida injustificada de información.

**Capacidad histórica:** el catálogo corresponde a septiembre de 2026, por lo que no se puede garantizar que la capacidad registrada represente la capacidad de cada estación durante todos los meses estudiados.

**Integridad de descargas:** la versión actual comprueba la existencia y tamaño no nulo de los archivos descargados, pero todavía no valida su integridad mediante hashes.

## 6. Reproducibilidad

Desde la raíz del repositorio, con las dependencias instaladas, ejecutar:

```bash
python -m mibici.dataset main
```

Los CSV se almacenan en `data/raw/`, la nomenclatura en `data/external/` y los archivos procesados en `data/processed/`.

Los datos originales y procesados se excluyen del repositorio mediante `.gitignore`.

**Pendiente:** implementar un manifiesto versionado con hashes SHA-256 y cantidades de registros para completar el mecanismo de versionado solicitado en la actividad.