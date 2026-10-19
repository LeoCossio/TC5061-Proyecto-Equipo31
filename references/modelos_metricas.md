# Modelos y métricas — Proyecto MiBici (Equipo 31)

**Periodo de análisis:** octubre de 2025 a septiembre de 2026  
**Objetivo:** comparar modelos de referencia (*baselines*) y candidatos para tres problemas operativos, con evaluación específica por tarea y trazabilidad mediante MLflow.

## 1. Resumen de modelos

| Pregunta | Unidad / horizonte | Baseline implementado | Modelo candidato implementado | Métricas |
|---|---|---|---|---|
| **P1 — Regresión de demanda** | Estación-hora; siguiente hora | Media histórica causal por estación y calendario | `RandomForestRegressor` | MAE, RMSE, MAE en horas pico |
| **P2 — Clasificación de riesgo de desbalance** | Estación-hora; siguiente hora | Persistencia de la presión de desbalance de la hora anterior | `RandomForestClassifier` optimizado | Precision, recall, F1 y PR-AUC |
| **P3 — Clustering de estaciones** | Estación; perfil histórico sin horizonte predictivo | Terciles de demanda | `KMeans` (`k=3`) | Silhouette Score, tamaño y perfil de los grupos |

Los tres enfoques son complementarios: P1 estima demanda futura, P2 señala periodos de presión operativa indirecta y P3 caracteriza patrones de uso. **P2 no observa inventario real de bicicletas ni de espacios disponibles**: su etiqueta se construye a partir de los flujos de viajes y la capacidad catalogada.

## 2. P1 — Regresión de demanda

### 2.1 Problema y características

El objetivo consiste en estimar las salidas en la próxima hora para cada estación. `mibici/feature_engineering/regresion.py` construye un panel estación-hora con horas sin viajes representadas como cero desde la primera actividad observada de cada estación. Las variables incluyen:

- Calendario: hora, día de la semana y fin de semana.
- Rezagos de demanda de **1, 2, 3, 24 y 168 horas**.
- Promedios móviles de las **24 y 168 horas anteriores**.
- Promedios históricos por estación y por combinación estación–día de semana–hora.

El cálculo histórico es causal: no utiliza la hora objetivo ni periodos posteriores. Se descartan las primeras 168 horas de cada estación para disponer del historial requerido.

### 2.2 Baseline y modelo candidato

**Baseline — Media histórica.** Predice la media previa para una estación, día de la semana y hora. En ausencia de historial suficiente utiliza referencias históricas más generales.

**Candidato — Random Forest Regressor.** Modelo basado en ensamble de árboles, con los hiperparámetros establecidos para P1:

| Parámetro | Valor |
|---|---:|
| `n_estimators` | 100 |
| `min_samples_leaf` | 20 |
| `max_samples` | 0.3 |
| `random_state` | 42 |

La ejecución reside en `mibici/train_p1.py` y registra el baseline y el candidato en MLflow.

### 2.3 Partición y métricas

| Conjunto | Periodo | Propósito |
|---|---|---|
| Entrenamiento | Octubre 2025 – mayo 2026 | Ajustar el modelo y referencias históricas |
| Validación | Junio – julio 2026 | Comparar métodos |
| Prueba | Agosto – septiembre 2026 | Evaluación en periodo posterior |

- **MAE:** error medio absoluto en viajes por estación-hora; permite interpretar el desvío promedio de demanda.
- **RMSE:** otorga mayor peso a errores grandes, relevantes durante picos.
- **MAE en horas pico:** analiza específicamente periodos operativamente relevantes (07:00–09:00 y 17:00–20:00 en días laborables, según la implementación de P1).

Las métricas se registran separando validación y prueba, por ejemplo `val_mae`, `test_rmse` y `test_mae_horas_pico`. **Los valores numéricos finales deben tomarse de los runs ejecutados**, no deducirse del código.

## 3. P2 — Clasificación de riesgo *proxy* de desbalance

### 3.1 Problema y definición de la etiqueta

P2 estima si una estación presentará **presión de desbalance elevada** durante la hora objetivo. `mibici/feature_engineering/classification.py` integra salidas por hora de inicio y llegadas por hora de fin, construye un panel estación-hora con horas sin viajes y utiliza la capacidad positiva de la nomenclatura de septiembre de 2026. Las estaciones sin capacidad válida en ese catálogo quedan fuera de este panel.

Se calcula:

\[
\text{flujo neto}_{s,t}=\text{llegadas}_{s,t}-\text{salidas}_{s,t}
\]

\[
\text{presión}_{s,t}=\frac{|\text{flujo neto}_{s,t}|}{\text{capacidad}_{s}}
\]

La etiqueta **`riesgo_siguiente_hora`** vale 1 si la presión observada en la hora objetivo alcanza o supera el **percentil 90 de la presión del conjunto de entrenamiento** y 0 en caso contrario. Dicho percentil se calcula *solo* con entrenamiento y después se aplica sin recalcularlo a validación y prueba. Este riesgo indica presión relativa de flujos, **no que una estación esté realmente vacía o llena**. Además, la capacidad se obtiene de un catálogo de septiembre de 2026, cuya validez histórica para todos los meses no está garantizada.

### 3.2 Características, baseline y candidato

Se utilizan **13 características**:

- Calendario: `hora`, `dia_semana`, `fin_semana`.
- Capacidad catalogada: `capacidad`.
- Flujos de la hora anterior: `salidas_lag_1`, `llegadas_lag_1`, `flujo_neto_lag_1`.
- Presión rezagada: `presion_lag_1`, `presion_lag_2`, `presion_lag_24`, `presion_lag_168`.
- Promedios históricos de presión: `presion_media_24`, `presion_media_168`.

Los rezagos y ventanas utilizan observaciones anteriores a la hora objetivo. Se descartan las primeras **168 horas** por estación para disponer de historial.

**Baseline — Persistencia.** Predice riesgo alto cuando la **presión de la hora anterior** es mayor o igual al umbral aprendido del entrenamiento. A diferencia del baseline contemplado originalmente en el ML Canvas, este método no usa frecuencias históricas por estación y franja horaria.

**Candidato — Random Forest Classifier optimizado.** El script `mibici/train_p2.py` utiliza la siguiente configuración:

| Parámetro | Valor |
|---|---:|
| `n_estimators` | 150 |
| `max_depth` | 30 |
| `min_samples_leaf` | 10 |
| `max_samples` | 0.3 |
| `max_features` | 0.8 |
| `class_weight` | `balanced_subsample` |
| `random_state` | 42 |
| Umbral de probabilidad para clase positiva | 0.5 |
| Máximo de filas para entrenar el bosque | 250,000 |

El código documenta una búsqueda de 72 combinaciones y evaluación de cinco finalistas. Cuando hay más de 250,000 observaciones de entrenamiento, selecciona una muestra aleatoria reproducible para ajustar el bosque.

### 3.3 Partición, métricas y artefactos

Se aplica la partición cronológica: **octubre 2025–mayo 2026** para entrenamiento, **junio–julio 2026** para validación y **agosto–septiembre 2026** para prueba.

- **Recall de riesgo:** proporción de instancias positivas correctamente detectadas; importa para no omitir posibles alertas.
- **Precision:** proporción de predicciones de riesgo que corresponden a la etiqueta positiva; controla alertas falsas.
- **F1-score:** equilibrio entre precision y recall.
- **PR-AUC (average precision):** resume la relación precision–recall a distintos umbrales y complementa las métricas con corte fijo.

Los modelos registran métricas con prefijos `val_` y `test_` y generan **matrices de confusión del conjunto de prueba** como artefactos: `p2_baseline_confusion.png` y `p2_rf_optimizado_confusion.png`. Los runs se denominan `P2-baseline-persistencia` y `P2-random-forest-optimizado`.

**Criterio de comparación:** evaluar mejora en detección de la clase de riesgo sin aumentar desproporcionadamente las falsas alertas. Deben reportarse los valores exactos observados en MLflow, especialmente recall junto con precision y F1; no se atribuyen aquí cifras que no se verificaron en los registros proporcionados.

**Precaución metodológica:** la presión de la hora objetivo se usa para construir la etiqueta, mientras que las variables predictoras contienen únicamente información anterior. El catálogo de capacidades es una referencia contemporánea del final del periodo, por lo que el resultado debe interpretarse como indicador aproximado y no como predicción de disponibilidad histórica real.

## 4. P3 — Clustering de estaciones

### 4.1 Problema y preparación

P3 agrupa estaciones a partir de los doce meses de viajes, sin etiquetas supervisadas ni horizonte predictivo. Se conservan **470 estaciones**, después de excluir aquellas con menos de **100 salidas** durante el periodo.

El módulo `mibici/feature_engineering/clustering.py` construye siete indicadores: viajes diarios, balance relativo, proporción de salidas en fin de semana y fracciones por mañana, mediodía, tarde y noche. Para entrenar se elimina `noche` por redundancia, se transforma la demanda con `log1p` y se estandarizan las **seis características restantes** mediante `StandardScaler`.

### 4.2 Baseline y candidato

- **Baseline — Terciles de demanda:** tres grupos de demanda diaria baja, media y alta.
- **K-Means:** `n_clusters=3`, `random_state=42` y `n_init=10`, ajustado sobre las seis características escaladas.

Ambos métodos se evalúan mediante Silhouette Score **sobre la misma matriz de características estandarizadas**.

### 4.3 Resultados observados

| Modelo | Estaciones | Grupos | Silhouette Score |
|---|---:|---:|---:|
| Baseline por terciles | 470 | 3 | 0.0732 |
| K-Means | 470 | 3 | **0.1992** |

| Cluster | Estaciones | Interpretación descriptiva |
|---|---:|---|
| 0 | 154 | Demanda más baja, predominio relativo matutino y balance cercano a cero |
| 1 | 198 | Mayor demanda, actividad destacada al mediodía y por la tarde |
| 2 | 118 | Demanda intermedia-baja, mayor peso vespertino y balance relativo negativo |

K-Means consigue una separación mayor que el baseline, aunque un valor de **0.1992** revela superposición considerable. La proyección PCA a dos componentes explica aproximadamente el **56.3 %** de la varianza estandarizada; se utiliza para visualización, no para entrenar K-Means. El script `mibici/train_p3.py` registra las figuras `p3_baseline.png` y `p3_kmeans_pca.png` en MLflow.

**Limitaciones:** la demanda diaria se calcula con los días observados del periodo global, lo cual puede afectar estaciones con cobertura parcial. La estabilidad temporal y geográfica de los grupos no se ha demostrado; la segmentación se considera exploratoria.

## 5. Seguimiento de experimentos y reproducibilidad

Los tres scripts (`mibici/train_p1.py`, `mibici/train_p2.py` y `mibici/train_p3.py`) registran sus corridas en el experimento **`mibici-equipo31`**, con MLflow Tracking y backend SQLite local. `mibici/run_all.py` coordina la ejecución conjunta, que el equipo ha comprobado que funciona.

Los registros contienen parámetros como periodo, semilla, hiperparámetros y huella de los datos; métricas de la tarea y artefactos; además de etiquetas `equipo`, `pregunta` y `git_commit` para identificar la versión del código. El comando general es:

```powershell
python -m mibici.dataset
python -m mibici.run_all
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

La huella SHA-256 calculada sobre los Parquet en el entrenamiento **no reemplaza un manifiesto por archivo versionado en Git**, con nombre, número de registros y hash individual. La comprobación de este manifiesto corresponde al cierre de la ingesta reproducible.

## 6. Criterios de evaluación y utilidad operativa

| Pregunta | Criterio de mejora | Aplicación posible | Limitación |
|---|---|---|---|
| P1 | Menor MAE/RMSE y menor MAE en horas pico frente al baseline | Prever demanda y apoyar redistribución | Exige evitar fuga de horas futuras |
| P2 | Mejor recall y F1 con precision operativamente aceptable | Priorizar revisión de estaciones con presión de flujos | El riesgo es indirecto; capacidad no necesariamente histórica |
| P3 | Mayor separación, interpretación coherente y estabilidad futura | Describir tipos de estaciones y patrones horarios | Silhouette no garantiza utilidad ni estabilidad |

## 7. Referencias

- MiBici. (s. f.). *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/
- MLflow. (s. f.). *MLflow Tracking*. https://mlflow.org/docs/latest/ml/tracking/
- scikit-learn developers. (s. f.). *Metrics and scoring: Quantifying the quality of predictions*. https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn developers. (s. f.). *Clustering*. https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn developers. (s. f.). *Precision-Recall*. https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

**Documentos relacionados:** `references/definicion_datos.md`, `references/pipelines_ml.md`, `references/arquitectura_sistema.md` y `references/limpieza_datos.md`.
