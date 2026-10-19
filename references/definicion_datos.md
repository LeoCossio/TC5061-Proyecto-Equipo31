# Definición de datos — Proyecto MiBici (Equipo 31)

**Versión:** documentación actualizada de P1, P2 y P3 para la entrega  
**Periodo de análisis:** 1 de octubre de 2025 al 30 de septiembre de 2026  
**Alcance:** datos compartidos para P1 (regresión), P2 (clasificación) y P3 (clustering).

## 1. Periodo seleccionado y justificación

Se seleccionaron **12 meses consecutivos**, de octubre de 2025 a septiembre de 2026. Un ciclo anual permite examinar diferencias por hora, día de la semana y mes, así como describir variaciones estacionales dentro del periodo. La selección evita combinar registros de épocas demasiado distantes en las que pudieran variar la operación de estaciones o los formatos de publicación.

El periodo constituye una **ventana histórica de observación**, no evidencia de que los patrones estacionales se repitan entre años. La disponibilidad, capacidad y fecha de incorporación de cada estación pueden cambiar durante estos doce meses.

## 2. Fuentes y tamaño real del conjunto

**Fuente primaria:** [Portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).

Se utilizaron los siguientes archivos oficiales:

- **Historial de viajes:** 12 CSV mensuales denominados `datos_abiertos_AAAA_MM.csv`, desde `datos_abiertos_2025_10.csv` hasta `datos_abiertos_2026_09.csv`. El módulo `mibici/dataset.py` descubre automáticamente los enlaces de descarga vigentes desde el portal, verifica que estén disponibles los doce meses y descarga los archivos correspondientes.
- **Catálogo de estaciones:** [nomenclatura de septiembre de 2026](https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv), descargada en `data/external/`.

### 2.1 Resultados de la ingesta

| Indicador | Valor |
|---|---:|
| Archivos mensuales | 12 |
| Registros de viaje originales | 4,497,877 |
| Registros de viaje procesados | 4,497,119 |
| Duplicados eliminados | 758 |
| Estaciones identificadas en los viajes | 471 |
| Estaciones en el catálogo de septiembre de 2026 | 484 |
| Archivos Parquet procesados | 12 |
| Estaciones incluidas en P3 después del filtro de actividad | 470 |

Estas cifras corresponden a los resultados reportados durante la ingesta y exploración del equipo. El conjunto de estaciones registradas en viajes no coincide necesariamente con el catálogo de septiembre: puede incluir estaciones retiradas o ausentes de esa versión y estaciones catalogadas sin actividad en el periodo. Entre los identificadores presentes en viajes pero ausentes en la nomenclatura consultada se señalaron **405 y 407**.

### 2.2 Preparación de los archivos

La ingesta implementada en `mibici/dataset.py` realiza las siguientes operaciones:

1. Descubre y descarga los CSV oficiales del periodo y el catálogo de estaciones.
2. Lee los CSV en UTF-8 o, cuando es necesario, con codificación CP1252.
3. Normaliza nombres de columnas, identificadores numéricos y campos de fecha.
4. Elimina duplicados exactos y registros inválidos (campos operativos obligatorios ausentes o fin anterior al inicio).
5. Conserva únicamente las columnas operativas y genera Parquet mensuales comprimidos con Snappy mediante PyArrow en `data/processed/`.

El proceso se invoca desde la raíz del proyecto mediante:

```powershell
python -m mibici.dataset
```

Los datos crudos se mantienen en `data/raw/`, la nomenclatura en `data/external/` y los archivos derivados en `data/processed/`. Estas carpetas de datos se excluyen del control de versiones de Git.

## 3. Variables originales y construcción de resultados

### 3.1 Variables de origen y tratamiento

Los CSV incluyen campos operativos y atributos de usuarios; el proceso de limpieza retiene exclusivamente las cinco columnas necesarias para los modelos:

| Columna procesada | Procedencia | Uso |
|---|---|---|
| `viaje_id` | `Viaje_Id` | Identificación del registro y controles de calidad |
| `inicio` | `Inicio_del_viaje` | Fecha/hora de salida y agregaciones temporales |
| `fin` | `Fin_del_viaje` | Fecha/hora de finalización y análisis de llegadas |
| `origen_id` | `Origen_Id` | Estación de origen |
| `destino_id` | `Destino_Id` | Estación de destino |

Los archivos originales también pueden incluir `Usuario_Id`, `Genero` y `Año_de_nacimiento`, que **no se conservan en los Parquet procesados ni se utilizan como variables de modelado**.

El catálogo de estaciones proporciona metadatos complementarios, como localización y, cuando están disponibles, atributos de capacidad. **La capacidad del catálogo de septiembre de 2026 no debe interpretarse automáticamente como capacidad histórica válida para meses anteriores.**

### 3.2 P1 — Regresión de demanda

- **Unidad de análisis:** estación-hora.
- **Objetivo:** número de viajes iniciados durante la hora que se desea predecir (horizonte operativo: una hora).
- **Construcción:** agrupar los registros por `origen_id` y hora de `inicio`. Se crea un panel horario que incluye las horas sin salidas, con objetivo igual a cero, desde la primera actividad observada de cada estación; no se generan horas anteriores a esa primera observación.
- **Variables predictoras implementadas:** hora, día de la semana, indicador de fin de semana, rezagos de 1, 2, 3, 24 y 168 horas, promedios móviles de 24 y 168 horas, media histórica de la estación y media histórica para la misma estación, día de la semana y hora.

Estas variables se encuentran implementadas en `mibici/feature_engineering/regresion.py`. Sus rezagos y promedios históricos utilizan observaciones anteriores a la hora objetivo. Se excluyen las primeras 168 horas de cada estación para disponer del historial requerido.

### 3.3 P2 — Clasificación de riesgo estimado de desbalance

- **Unidad de análisis:** estación-hora. Cada fila contiene las variables disponibles antes de la hora evaluada y una etiqueta para esa hora; el horizonte operativo es de **una hora**.
- **Fuentes:** viajes procesados y nomenclatura de septiembre de 2026, de la cual `cargar_capacidades()` obtiene `id` y `dpcapacity` (identificador y capacidad). Se excluyen estaciones sin capacidad válida, y el panel comienza en la primera hora con actividad registrada por estación; las horas sin movimientos se completan con ceros.
- **Presión de desbalance:** para cada estación-hora se calcula `flujo_neto = llegadas - salidas` y `presion_desbalance = abs(flujo_neto) / capacidad`. El valor captura la intensidad relativa del desbalance de flujos, **no** inventario ni ocupación reales.
- **Etiqueta binaria:** `riesgo_siguiente_hora = 1` cuando la presión de la hora objetivo es mayor o igual al **percentil 90 de la presión calculado exclusivamente con el conjunto de entrenamiento**; en otro caso vale 0. El umbral resultante se aplica sin reajustar a validación y prueba. No se usa un umbral numérico arbitrario.
- **Características predictoras:** 13 variables: `hora`, `dia_semana`, `fin_semana`, `capacidad`, `salidas_lag_1`, `llegadas_lag_1`, `flujo_neto_lag_1`, `presion_lag_1`, `presion_lag_2`, `presion_lag_24`, `presion_lag_168`, `presion_media_24` y `presion_media_168`. Los rezagos y las medias se calculan solo con horas anteriores a la hora objetivo.
- **Preparación:** se descartan las primeras 168 horas de cada estación para disponer del historial requerido. Las variables de P2 se implementaron en `mibici/feature_engineering/classification.py` y el entrenamiento en `mibici/train_p2.py`.
- **Modelos:** baseline de **persistencia** (alerta si la presión de la hora anterior supera el umbral de riesgo) y `RandomForestClassifier` optimizado. La evaluación registra precision, recall, F1 y PR-AUC para validación y prueba, además de matrices de confusión como artefactos de MLflow.

**Limitación:** se utiliza la capacidad del catálogo de septiembre de 2026 como referencia para el periodo completo. No se comprobó que permaneciera constante en cada estación durante los doce meses. Además, el indicador mide presión de flujos: no permite afirmar que una estación estuviera vacía o llena. La presión de la hora evaluada solo se usa para etiquetar, nunca como característica predictora de esa misma hora.

### 3.4 P3 — Clustering de estaciones

- **Unidad de análisis:** una estación, caracterizada por su comportamiento histórico durante los doce meses de estudio.
- **Objetivo:** agrupación no supervisada; no existe etiqueta verdadera ni horizonte predictivo.
- **Características construidas:** demanda promedio diaria (`viajes_diarios`), balance relativo entre llegadas y salidas, proporción de salidas en fin de semana y proporción de salidas en franjas de mañana, mediodía, tarde y noche.
- **Preprocesamiento:** se excluyen estaciones con menos de 100 salidas durante el periodo; para modelar se descarta la proporción nocturna por redundancia con las otras franjas, se aplica `log1p` a la demanda y se estandarizan las seis variables restantes con `StandardScaler`.
- **Métodos utilizados:** baseline por terciles de demanda y `KMeans` de tres grupos.

La implementación se encuentra en `mibici/feature_engineering/clustering.py` y se ejecuta con `mibici/train_p3.py`. Se incluyeron **470 estaciones**. El promedio de viajes diarios emplea los días observados en el conjunto global; para estaciones con cobertura parcial, esa cifra puede subestimar la intensidad durante sus días efectivamente activos. Por tanto, los resultados deben interpretarse considerando los posibles cambios de cobertura.

## 4. Condiciones de uso y privacidad

La información procede del [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). En la documentación de trabajo no se identificó una licencia específica explícita para los CSV (por ejemplo, CC BY o CC0); por ello, se cita la fuente original y no se atribuye una licencia de redistribución no verificada. También se considera el [aviso de privacidad de MiBici](https://www.mibici.net/es/aviso-de-privacidad/).

Las medidas aplicadas son:

- Excluir de los datos procesados `Usuario_Id`, `Genero` y `Año_de_nacimiento`.
- Mantener los CSV originales y los Parquet localmente, fuera del repositorio público.
- Compartir resultados agregados por estación o periodo y no divulgar secuencias de trayectos individuales ni identificadores de usuario.
- No registrar en MLflow credenciales ni información personal como parámetros, etiquetas o artefactos.

## 5. Estrategia temporal y prevención de fuga de información

### 5.1 P1 — Partición implementada

P1 utiliza una separación cronológica implementada mediante `dividir_temporal_p1()` y fechas centralizadas en `mibici/config.py`:

| Conjunto | Periodo | Función |
|---|---|---|
| Entrenamiento | Octubre de 2025 a mayo de 2026 | Ajustar el modelo |
| Validación | Junio a julio de 2026 | Comparar alternativas y evaluar decisiones de modelado |
| Prueba | Agosto a septiembre de 2026 | Evaluar sobre observaciones posteriores |

La construcción de rezagos, promedios móviles y estadísticas históricas de P1 excluye la hora objetivo y las horas posteriores. Los hiperparámetros y la semilla del experimento se centralizan en `config.py`. El baseline utiliza medias históricas causales, con alternativas de respaldo para observaciones sin suficiente historia.

### 5.2 P2 — Partición implementada y umbral de clasificación

P2 utiliza una partición cronológica implementada mediante `dividir_temporal_p2()` en `classification.py`, con fechas de corte **1 de junio de 2026** y **1 de agosto de 2026**, definidas en `train_p2.py`. Por tanto, los conjuntos comprenden entrenamiento (octubre de 2025 a mayo de 2026), validación (junio y julio de 2026) y prueba (agosto y septiembre de 2026), después de retirar las horas iniciales necesarias para los rezagos.

El umbral que determina el riesgo se calcula usando **únicamente la presión observada en entrenamiento** (`cuantíl = 0.90`) y se reutiliza para etiquetar validación y prueba. Los predictores históricos se desplazan respecto de la hora objetivo, evitando introducir sus salidas y llegadas futuras como entrada. El baseline utiliza `presion_lag_1`, mientras que el Random Forest utiliza las 13 características históricas y de calendario.

La configuración del experimento P2 (fechas, semilla, cuantil y parámetros del modelo) está declarada actualmente en `mibici/train_p2.py`; para cumplir estrictamente la centralización solicitada en la rúbrica, estos valores deberán trasladarse a `mibici/config.py`.

### 5.3 P3 — Evaluación de agrupaciones históricas

P3 construye perfiles de estaciones con **el periodo completo de octubre de 2025 a septiembre de 2026** y evalúa la separación de los grupos sobre las mismas características agregadas. Esta metodología describe patrones históricos; **no equivale a una validación de generalización temporal**. Como extensión metodológica se contempla formar clusters con un periodo anterior y analizar su estabilidad en meses posteriores, especialmente para estaciones nuevas o con actividad parcial.

## 6. Versionado y trazabilidad de datos

Los scripts de P1, P2 y P3 pueden identificar los doce Parquet utilizados mediante una **huella SHA-256 combinada**. P2 reutiliza `version_datos()` desde `mibici/train_p3.py`; los runs de MLflow registran la huella como `data_version`, el periodo, los parámetros del experimento y el commit de Git, junto con métricas y artefactos de cada tarea.

El manifiesto versionado en Git con **nombre, conteo de registros y SHA-256 por archivo**, así como su verificación de integridad, es un requisito adicional al hash combinado. Su implementación se está completando por separado y **debe comprobarse antes de cerrar la entrega**. No se presenta aquí como ejecutada o verificada.

## 7. Síntesis

El conjunto de datos se preparó con una ingesta compartida y doce archivos Parquet de estructura uniforme. P1 utiliza un panel cronológico estación-hora para predicción de demanda; P2 clasifica el riesgo proxy de desbalance mediante la presión neta de flujos normalizada por capacidad y un umbral aprendido del entrenamiento; y P3 analiza perfiles históricos agregados por estación. La separación entre datos crudos, procesados, modelos y artefactos, junto con las etiquetas de MLflow y el historial de Git, aporta trazabilidad al proyecto.

## Referencias

- MiBici. (s. f.). *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/
- MiBici. (s. f.). *Aviso de privacidad*. https://www.mibici.net/es/aviso-de-privacidad/
- Documentación técnica del proyecto: `mibici/dataset.py`, `mibici/feature_engineering/regresion.py`, `mibici/feature_engineering/clustering.py`, `mibici/feature_engineering/classification.py`, `mibici/train_p1.py`, `mibici/train_p2.py`, `mibici/train_p3.py` y `mibici/config.py`.