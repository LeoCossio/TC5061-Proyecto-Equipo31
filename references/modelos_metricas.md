# Modelos y métricas — Proyecto MiBici (Equipo 31)

**Periodo de análisis:** octubre de 2025 a septiembre de 2026  
**Objetivo:** comparar métodos de referencia (*baselines*) con modelos candidatos para tres preguntas operativas, utilizando métricas acordes con cada tarea y experimentos trazables en MLflow.  
**Estado de verificación:** P1 y P3 cuentan con implementación en el repositorio revisado; los resultados numéricos de P3 se comprobaron en MLflow. P2 se especifica conforme al ML Canvas, sin atribuirle resultados ni ejecuciones no verificados.

## 1. Resumen de modelos

| Pregunta | Unidad de análisis / horizonte | Baseline | Modelo candidato | Métricas |
|---|---|---|---|---|
| **P1 — Regresión de demanda** | Estación-hora; demanda de la hora objetivo siguiente | Media histórica por estación, día de semana y hora | `RandomForestRegressor` | MAE, RMSE y MAE en horas pico |
| **P2 — Clasificación de riesgo de desbalance** | Estación-hora; próxima hora | Frecuencia histórica de riesgo por estación y franja horaria | `RandomForestClassifier` | Recall, precision y F1-score de la clase riesgo |
| **P3 — Clustering de estaciones** | Estación; perfiles históricos, sin horizonte predictivo | Terciles de demanda diaria | `KMeans` (`k=3`) | Silhouette Score, tamaños de grupos e interpretación |

Los tres enfoques son complementarios: P1 apoya la anticipación de demanda, P2 el análisis preventivo de riesgo operativo y P3 la caracterización de patrones de uso de las estaciones.

## 2. P1 — Regresión de demanda

### 2.1 Problema y variables

Se estima el número de viajes que se iniciarán en una estación durante una hora objetivo. El código construye un **panel estación-hora**, con periodos sin viajes representados mediante ceros a partir de la primera salida observada de cada estación.

Las características implementadas en `mibici/feature_engineering/regresion.py` incluyen:

- Calendario: hora del día, día de la semana e indicador de fin de semana.
- Demanda rezagada: 1, 2, 3, 24 y 168 horas.
- Promedios móviles de las 24 y 168 horas anteriores.
- Promedio histórico de cada estación y promedio histórico para la combinación estación–día de semana–hora.

Las características históricas se construyen a partir de observaciones **anteriores a la hora objetivo**; se descartan las primeras 168 horas de cada estación para disponer de historial suficiente para los rezagos.

### 2.2 Baseline y candidato

**Baseline — Media histórica.** Predice el promedio previo de la estación en el mismo día de la semana y hora. Si falta ese historial, utiliza la media histórica general de la estación y, como último recurso, la media global de entrenamiento.

**Candidato — Random Forest Regressor.** Entrena un ensamble de árboles con las doce características descritas. La configuración fijada en `mibici/config.py` es:

| Hiperparámetro | Valor |
|---|---:|
| `n_estimators` | 100 |
| `min_samples_leaf` | 20 |
| `max_samples` | 0.3 |
| Semilla (`random_state`) | 42 |

El script ejecutable `mibici/train_p1.py` produce las predicciones, calcula las métricas y registra el baseline y el candidato en MLflow.

### 2.3 Evaluación y justificación

La separación de P1 es cronológica:

| Conjunto | Periodo | Uso |
|---|---|---|
| Entrenamiento | Octubre de 2025 – mayo de 2026 | Entrenamiento y medias de referencia |
| Validación | Junio – julio de 2026 | Comparación y selección |
| Prueba | Agosto – septiembre de 2026 | Evaluación sobre el periodo más reciente |

- **MAE:** error absoluto promedio en viajes por estación-hora; ofrece una medida directamente interpretable para operación.
- **RMSE:** penaliza más los errores grandes y permite detectar dificultades ante picos de demanda.
- **MAE en horas pico:** evalúa específicamente las horas 07:00–09:00 y 17:00–20:00 entre semana, cuando los errores pueden tener mayor impacto operativo.

El script registra las tres métricas en validación y prueba, con nombres como `val_mae`, `test_rmse` y `test_mae_horas_pico`, y guarda figuras de demanda real frente a predicha e importancia de variables.

**Resultados numéricos:** el código y la generación de artefactos constan en el repositorio, pero las cifras definitivas de MAE/RMSE deben transcribirse de una ejecución verificada de MLflow. No se asignan valores supuestos.

## 3. P2 — Clasificación de riesgo *proxy* de desbalance

### 3.1 Problema y construcción de la etiqueta

Se plantea una clasificación binaria del **riesgo indirecto de desbalance** por estación y hora, con horizonte de una hora. Los archivos públicos de viajes no incluyen inventarios históricos de bicicletas ni de espacios libres; por tanto, el objetivo **no equivale a predecir directamente estaciones vacías o llenas**.

La etiqueta se define conceptualmente a partir del desbalance neto futuro entre salidas y llegadas, normalizado por capacidad cuando esta sea válida. La implementación deberá establecer de manera explícita la fórmula y el umbral de riesgo, y documentar el tratamiento de capacidades faltantes o no históricas. Si se analizan por separado exceso de salidas y exceso de llegadas, deberán aclararse las etiquetas correspondientes.

### 3.2 Baseline y candidato establecidos en el ML Canvas

**Baseline — Frecuencia histórica.** Estima la probabilidad de riesgo para cada estación y franja horaria a partir de periodos anteriores. Se utiliza una frecuencia más general cuando el historial de una estación es insuficiente.

**Candidato — Random Forest Classifier.** Utiliza información disponible hasta el momento de predicción: hora, día, salidas y llegadas históricas, rezagos, balances previos y capacidad únicamente cuando sea aplicable.

### 3.3 Métricas y criterio de selección

- **Recall de la clase riesgo:** proporción de episodios etiquetados como riesgo que son detectados. Importa porque omitir alertas puede retrasar una revisión operativa.
- **Precision:** proporción de alertas emitidas que corresponden al indicador de riesgo; permite controlar falsas alertas.
- **F1-score:** resume el equilibrio entre recall y precision. Se recomienda complementar con matriz de confusión y prevalencia de la clase positiva.

No se debe seleccionar un modelo únicamente por recall si clasifica indiscriminadamente todas las horas como riesgosas. La selección requiere analizar recall y precision conjuntamente, con umbral definido mediante validación cronológica.

La división temporal de referencia acordada para los modelos predictivos es octubre 2025–mayo 2026 (entrenamiento), junio–julio 2026 (validación) y agosto–septiembre 2026 (prueba). **Esta sección describe la solución definida para P2; la implementación exacta, el umbral y los resultados todavía deben contrastarse con su script definitivo.** Los flujos de la hora objetivo pueden formar la etiqueta, pero nunca alimentar las características predictivas de esa misma hora.

## 4. P3 — Clustering de estaciones

### 4.1 Problema y características

Se agrupan estaciones según su comportamiento histórico, sin variable objetivo supervisada ni horizonte de predicción. En la implementación revisada se utilizaron **12 meses de viajes**, de octubre de 2025 a septiembre de 2026, y **470 estaciones** tras aplicar un mínimo de **100 salidas** por estación durante el periodo.

El módulo `mibici/feature_engineering/clustering.py` construye siete indicadores iniciales: viajes diarios, balance relativo de llegadas frente a salidas, proporción de salidas en fin de semana y proporciones por mañana, mediodía, tarde y noche.

Para el entrenamiento se elimina la franja **noche**, redundante con las otras proporciones horarias; se aplica `log1p` a los viajes diarios y `StandardScaler` a las **seis características** restantes.

### 4.2 Baseline y candidato

**Baseline — Terciles de demanda.** Divide las estaciones en tres grupos de baja, media y alta demanda según sus viajes diarios.

**Candidato — K-Means.** Agrupa estaciones sobre las seis características escaladas, con:

| Hiperparámetro | Valor |
|---|---:|
| `n_clusters` | 3 |
| `random_state` | 42 |
| `n_init` | 10 |

Ambos métodos se evalúan mediante Silhouette Score **sobre la misma matriz estandarizada**, para que la comparación de separación de grupos sea consistente.

### 4.3 Resultados verificados

| Modelo | Estaciones | Grupos | Silhouette Score |
|---|---:|---:|---:|
| Baseline por terciles | 470 | 3 | 0.0732 |
| K-Means | 470 | 3 | **0.1992** |

K-Means mostró una separación relativa superior al baseline, aunque un Silhouette Score de 0.1992 todavía sugiere agrupaciones con superposición considerable.

| Cluster | Estaciones | Perfil predominante |
|---|---:|---|
| 0 | 154 | Menor demanda, mayor proporción matutina y balance prácticamente neutro |
| 1 | 198 | Mayor demanda, uso destacado al mediodía y por la tarde |
| 2 | 118 | Demanda intermedia-baja, concentración vespertina y balance relativo negativo |

Para apoyar la interpretación, se generó una visualización mediante **PCA de dos componentes**, que conserva aproximadamente **56.3 %** de la varianza de las características estandarizadas. PCA se utiliza únicamente para visualización: K-Means se entrena con las seis características.

Las figuras `p3_baseline.png` y `p3_kmeans_pca.png` se generan desde `mibici/train_p3.py` y se registran como artefactos en el experimento `mibici-equipo31`.

### 4.4 Interpretación y limitaciones

Los grupos aportan información exploratoria para detectar estaciones con distintos niveles y horarios de actividad y posibles desbalances de flujo. **Un balance negativo no prueba falta de bicicletas**, ya que no se dispone del inventario real.

El promedio diario de P3 se calculó considerando los días observados en el periodo completo; esto puede subestimar la intensidad de estaciones con incorporación tardía. Tampoco se ha demostrado todavía estabilidad de las asignaciones entre periodos. Por ello, los clusters no deben interpretarse como categorías operativas definitivas sin validación temporal y geográfica.

## 5. Registro y reproducibilidad de los experimentos

La estructura común conserva los datos en Parquet, las transformaciones reutilizables en el módulo `mibici` y el código de entrenamiento separado de los notebooks. Las ejecuciones se realizan desde la raíz del repositorio:

```bash
python -m mibici.dataset
python -m mibici.train_p1
python -m mibici.train_p3
```

Los scripts de P1 y P3 registran dos runs descriptivos por pregunta en **`mibici-equipo31`**, con parámetros, fechas de estudio, huella SHA-256 conjunta de los Parquet utilizados, métricas con prefijos de conjunto cuando corresponde, etiquetas `equipo`, `pregunta` y `git_commit`, y figuras como artefactos. El historial se consulta mediante MLflow Tracking con backend SQLite local.

Para P2 se mantiene el mismo esquema de registro, que deberá comprobarse sobre el código y los runs definitivos. El archivo `run_all.py` coordina los módulos habilitados; su ejecución conjunta con las tres preguntas queda sujeta a integración y prueba.

**Distinción de reproducibilidad:** una huella calculada durante el entrenamiento identifica el contenido de los archivos usados, pero no sustituye un manifiesto por archivo, con hash y conteo de registros, versionado en Git. La reproducibilidad completa requiere también verificar la instalación de dependencias y la regeneración de datos desde un entorno limpio.

## 6. Criterios de evaluación y decisiones operativas

| Pregunta | Qué indica una mejora | Decisión que puede apoyar | Precaución |
|---|---|---|---|
| P1 | Menores MAE/RMSE y menor error en horas pico frente al baseline | Anticipar demanda para planear redistribución | No usar información de horas futuras como predictor |
| P2 | Mejor detección de la clase riesgo, con precision aceptable | Priorizar revisión de estaciones | El riesgo es un *proxy*, no disponibilidad observada |
| P3 | Grupos más cohesionados, separados e interpretables | Analizar perfiles horarios y necesidades de infraestructura | Silhouette no demuestra estabilidad ni utilidad operativa por sí solo |

## 7. Referencias

- MiBici. (s. f.). *Datos abiertos*. https://mibici.net/es/datos-abiertos/
- Scikit-learn developers. (s. f.). *Metrics and scoring: Quantifying the quality of predictions*. https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn developers. (s. f.). *Clustering*. https://scikit-learn.org/stable/modules/clustering.html
- Scikit-learn developers. (s. f.). *Selecting the number of clusters with silhouette analysis on KMeans clustering*. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
- MLflow. (s. f.). *MLflow Tracking*. https://mlflow.org/docs/latest/ml/tracking/

**Documentos relacionados:** `references/definicion_datos.md`, `references/pipelines_ml.md` y `references/arquitectura_sistema.md`.
