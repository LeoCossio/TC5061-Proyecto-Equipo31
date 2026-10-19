# Limpieza y normalización de datos — MiBici (Equipo 31)

**Periodo de estudio:** octubre de 2025 a septiembre de 2026  
**Componente:** `mibici/dataset.py`  
**Alcance:** ingesta y preparación compartida de datos para P1 (regresión), P2 (clasificación) y P3 (clustering).

## 1. Objetivo

Se desarrolló un proceso automatizado de descarga, limpieza y normalización de los registros históricos de viajes del sistema MiBici. Su propósito es ofrecer una fuente común y consistente para los tres problemas de aprendizaje automático del equipo, evitando repetir la preparación de datos en cada modelo.

Los archivos originales se mantienen separados de las versiones procesadas, almacenadas mensualmente en formato Parquet para facilitar su lectura y análisis.

## 2. Fuentes, periodo y volumen

**Fuente oficial:** [Portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).

Se utilizaron **12 archivos CSV mensuales**, correspondientes a octubre de 2025–septiembre de 2026, y la **nomenclatura de estaciones de septiembre de 2026**. La elección de doce meses consecutivos permite estudiar variaciones horarias, semanales y dentro de un ciclo anual.

| Indicador | Resultado |
|---|---:|
| Archivos mensuales de viajes | 12 |
| Registros originales | 4,497,877 |
| Registros procesados | 4,497,119 |
| Registros duplicados eliminados | 758 |
| Estaciones distintas en los viajes | 471 |
| Estaciones registradas en el catálogo | 484 |
| Archivos Parquet generados | 12 |

Las cifras corresponden al reporte de ejecución de ingesta del equipo para el periodo indicado.

## 3. Proceso de ingesta y limpieza

El procesamiento está centralizado en `mibici/dataset.py` y comprende las siguientes etapas:

1. **Descarga de datos:** recuperación de los CSV mensuales y del catálogo de estaciones desde la fuente oficial.
2. **Lectura y codificación:** lectura de archivos en UTF-8 y uso de CP1252 cuando se presentan errores de decodificación.
3. **Normalización de columnas:** unificación de nombres para procesar archivos de distintos meses con una estructura común.
4. **Conversión de tipos:** transformación de fechas a `datetime` y de identificadores de estaciones a tipos numéricos adecuados.
5. **Control de duplicados:** eliminación de filas completamente duplicadas.
6. **Validación de registros:** exclusión de filas con valores faltantes en campos esenciales o con una fecha de finalización anterior al inicio.
7. **Selección de atributos operativos:** conservación de `viaje_id`, `inicio`, `fin`, `origen_id` y `destino_id`, sin trasladar identificadores de usuarios ni atributos demográficos a los datos procesados.
8. **Almacenamiento:** escritura de un archivo Parquet por mes mediante PyArrow y compresión Snappy.

Esta preparación común permite que los modelos construyan sus propias agregaciones sin modificar los datos originales.

## 4. Resultados de las validaciones

| Comprobación | Resultado reportado |
|---|---:|
| Duplicados exactos eliminados | 758 |
| Registros excluidos por fechas inválidas o nulos esenciales | 0 |
| Identificadores de viaje duplicados restantes | 0 |
| Fechas de inicio fuera del mes del archivo | 0 |
| Archivos Parquet mensuales obtenidos | 12 |

Los **758 duplicados exactos** se localizaron en el archivo correspondiente a septiembre de 2026. Tras la limpieza, el conjunto quedó integrado por **4,497,119 viajes**.

Estos controles describen los resultados de la ejecución documentada por el equipo; no sustituyen una nueva verificación cuando se regeneran los archivos de entrada.

## 5. Hallazgos y consideraciones de calidad

**Codificación de archivos.** El CSV de enero de 2026 requirió decodificación CP1252; los once archivos restantes pudieron leerse utilizando UTF-8. El proceso contempla esta diferencia para evitar pérdida o lectura incorrecta de caracteres.

**Estaciones sin correspondencia en el catálogo.** Los identificadores **405 y 407** aparecen en los viajes, pero no en la nomenclatura de septiembre de 2026. Los registros se conservaron para evitar una eliminación injustificada. La cantidad de estaciones identificadas en viajes y la cantidad de estaciones catalogadas no tienen por qué coincidir.

**Capacidad de estaciones.** La nomenclatura disponible corresponde a septiembre de 2026. Por tanto, la capacidad incluida en ese catálogo no debe interpretarse automáticamente como capacidad histórica válida durante los doce meses. Esta precaución es particularmente relevante para el indicador de riesgo de P2.

**Privacidad.** Los CSV originales pueden contener identificadores pseudónimos o información demográfica. Los archivos Parquet destinados al modelado contienen únicamente las cinco columnas operativas seleccionadas; no se publican datos individuales en el repositorio.

## 6. Reproducibilidad y almacenamiento

Desde la raíz del repositorio, después de instalar las dependencias indicadas en `requirements.txt`, se ejecuta:

```bash
python -m mibici.dataset
```

Los archivos se organizan en las siguientes carpetas:

| Ruta | Contenido |
|---|---|
| `data/raw/` | Archivos de viajes descargados |
| `data/external/` | Nomenclatura de estaciones |
| `data/processed/` | Archivos mensuales normalizados en Parquet |

Estas carpetas de datos están excluidas del seguimiento de Git mediante `.gitignore`, al igual que las credenciales, modelos serializados y archivos locales de MLflow.

El entrenamiento de P3 registra en MLflow una **huella SHA-256 conjunta de los archivos Parquet utilizados**, que permite identificar las entradas de esa corrida. Este registro no sustituye el **manifiesto de datos versionado por archivo**, con nombres, cantidades de registros y hashes individuales, requerido para una trazabilidad integral. La generación y verificación de ese manifiesto debe confirmarse en la versión final de la ingesta.

## 7. Integración con los modelos

- **P1 — Regresión:** utiliza los registros de salida para construir un panel estación-hora, variables temporales e históricos de demanda, con separación cronológica entre entrenamiento, validación y prueba.
- **P2 — Clasificación:** emplea los flujos de salidas y llegadas para definir un indicador *proxy* de riesgo de desbalance; la etiqueta y sus condiciones deben corresponder a la especificación final del modelo.
- **P3 — Clustering:** agrega salidas y llegadas por estación y construye perfiles de intensidad, balance y distribución temporal de uso.

La separación entre la ingesta común y las transformaciones específicas reduce duplicaciones y ayuda a mantener criterios consistentes de preparación de datos entre los integrantes.

## Referencias

- MiBici. (s. f.). *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/
- MiBici. (s. f.). *Aviso de privacidad*. https://www.mibici.net/es/aviso-de-privacidad/
- pandas development team. (s. f.). *Parquet*. https://pandas.pydata.org/docs/user_guide/io.html#parquet
- Python Software Foundation. (s. f.). *hashlib — Secure hashes and message digests*. https://docs.python.org/3/library/hashlib.html
