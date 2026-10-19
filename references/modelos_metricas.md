# Propuesta de modelos y métricas — MiBici (Equipo 31)

**Estado:** Propuesta, pendiente de validación con el equipo y resultados experimentales.  
**Periodo de datos:** Octubre de 2025 a septiembre de 2026.  
**Documentos relacionados:** `references/definicion_datos.md` y `references/pipelines_ml.md`.

## 1. Objetivo

Definir, para cada pregunta de negocio, un método de referencia (*baseline*), al menos un modelo candidato y criterios de evaluación adecuados. El propósito de esta entrega es establecer comparaciones reproducibles; **no se reportan resultados ni umbrales de mejora inventados**. Los valores se obtendrán en los experimentos y se registrarán en MLflow.

| Pregunta | Unidad / horizonte | Baseline | Modelo candidato | Métrica principal |
|---|---|---|---|---|
| P1: demanda | Estación-hora; próxima hora | Media histórica por estación, hora y día de la semana | `RandomForestRegressor` | MAE |
| P2: desbalance estimado | Estación-hora; próxima hora | Frecuencia histórica de la clase de riesgo por estación y franja horaria | `RandomForestClassifier` | Recall de la clase de riesgo, acompañado de precision |
| P3: perfiles de estaciones | Estación; sin horizonte predictivo | Grupos de demanda baja/media/alta | `KMeans` | Silhouette Score e interpretabilidad |

## 2. P1 — Regresión de demanda

**Objetivo:** Predecir cuántos viajes se iniciarán en una estación durante la hora siguiente.

- **Variable objetivo:** número entero de salidas de la estación en la hora futura, incluyendo las horas sin viajes (cero).
- **Baseline:** promedio histórico de salidas de la misma estación para la misma hora y día de la semana, calculado solo con observaciones anteriores. Si no hay suficiente historial, utilizar una media histórica más general definida de antemano (por estación o global).
- **Modelo candidato:** `RandomForestRegressor`, utilizando hora, día de semana, estación, conteos rezagados y promedios históricos causales. La codificación de estación y cualquier transformación ajustable se aprende solo en entrenamiento.
- **Métrica principal — MAE:** error absoluto medio expresado en **viajes por estación-hora**. Es fácil de interpretar para estimar el error de planificación: menor MAE indica mejor precisión promedio.
- **Métrica complementaria — RMSE:** penaliza más los errores grandes; es útil porque subestimar de manera considerable las horas de mayor demanda puede afectar la distribución de bicicletas.
- **Decisión operativa asociada:** planificar disponibilidad y redistribución con base en las horas y estaciones con mayor demanda estimada.
- **Criterio de comparación:** el candidato debe obtener un `val_mae` inferior al baseline para considerarse una mejora inicial; reportar `test_mae` y `test_rmse` una sola vez después de seleccionar el diseño en validación. Analizar también el error en horas pico.

**Precaución:** los promedios y rezagos no pueden incorporar observaciones de la hora que se intenta predecir ni horas posteriores. La demanda observada de la hora objetivo se usa solo como etiqueta.

## 3. P2 — Clasificación de riesgo *proxy* de desbalance

**Objetivo:** Identificar estaciones que podrían experimentar un desequilibrio operativo durante la hora siguiente. **No se predice inventario real**, porque MiBici no lo publica en estos CSV.

- **Variable objetivo (pendiente de aprobación):** etiqueta binaria de *riesgo estimado* construida a partir del **balance neto futuro** de llegadas y salidas, relativo a la capacidad de la estación, mediante una fórmula y un umbral definidos previamente. Se documentarán ambos y el manejo de capacidades ausentes o no históricas.
- **Baseline:** probabilidad histórica de riesgo para una estación y franja horaria, calculada únicamente con etiquetas de periodos anteriores. Clasificar como riesgo si la probabilidad supera un umbral **elegido en validación**. En casos de poco historial, utilizar una frecuencia histórica general.
- **Modelo candidato:** `RandomForestClassifier`, con entradas y salidas rezagadas, balances históricos, capacidad cuando sea válida y variables temporales.
- **Métrica principal — Recall (clase riesgo):** proporción de episodios etiquetados como riesgo que el modelo detecta. Es pertinente porque pasar por alto un episodio podría retrasar una intervención.
- **Métricas complementarias — Precision y F1:** precision indica qué proporción de alertas corresponde a la etiqueta proxy y ayuda a limitar falsas alarmas; F1 resume el equilibrio entre precision y recall. Reportar además matriz de confusión y proporción de clase positiva.
- **Decisión operativa asociada:** priorizar estaciones para revisar y, eventualmente, rebalancear; la cantidad de alertas accionables depende de recursos disponibles que aún no conocemos.
- **Criterio de comparación:** no elegir un modelo solo porque alcance recall alto alertando siempre. Comparar baseline y candidato con **recall junto con precision**, y seleccionar umbral en validación antes de informar resultados de prueba.

**Precaución crítica:** la etiqueta se calcula con observaciones de la **hora futura**, pero ninguna variable predictora puede usar ese balance futuro. Además, el baseline no debe repetir la regla de creación de etiquetas usando los mismos valores futuros: sería una evaluación circular. Las métricas miden detección del *proxy*, **no episodios reales comprobados de estaciones vacías o llenas**.

## 4. P3 — Clustering de estaciones

**Objetivo:** Agrupar estaciones con patrones históricos similares de uso para apoyar la interpretación de demanda y la planeación de infraestructura.

- **Unidad de análisis:** una estación representada por características agregadas de un periodo histórico; no existe horizonte predictivo.
- **Features propuestas:** salidas promedio por día activo, perfil horario de salidas, porcentaje de viajes en fines de semana, proporción de actividad en horas pico y balance relativo entre llegadas y salidas. Evaluar redundancia entre variables, ausencia de estaciones y meses de actividad. Escalar con `StandardScaler` ajustado sobre los datos de desarrollo.
- **Baseline:** tres grupos definidos mediante demanda histórica baja/media/alta, utilizando puntos de corte (por ejemplo, terciles) calculados únicamente con el periodo de desarrollo. Es una segmentación sencilla, transparente y fácil de explicar.
- **Modelo candidato:** `KMeans` aplicado a las variables estandarizadas. Probar valores pequeños y razonables de `k` (por ejemplo, de 2 a 6), usando semilla fija y varias inicializaciones.
- **Métrica principal — Silhouette Score:** mide si las estaciones están relativamente cercanas a otras de su grupo y separadas de otros grupos; valores mayores suelen indicar grupos más definidos, aunque no garantizan utilidad operativa.
- **Criterios complementarios:** tamaño y equilibrio de los grupos, interpretación de sus perfiles (horarios y demanda), estabilidad de asignaciones en meses posteriores y visualización sobre el mapa de Guadalajara.
- **Decisión asociada:** describir tipos de estaciones y localizar zonas con patrones de movilidad diferentes que ameriten análisis de capacidad o expansión.
- **Comparación justa:** calcular Silhouette sobre **la misma matriz de características escaladas** para el baseline y para K-Means. Si una solución produce un solo grupo o un grupo por estación, Silhouette no aplica y debe reportarse como no calculable.

**Evaluación temporal propuesta:** construir perfiles usando octubre de 2025–mayo de 2026; utilizar junio–julio para seleccionar características y `k`; evaluar estabilidad con agosto–septiembre sin redefinir retrospectivamente los grupos. Comparar estaciones con cobertura histórica suficientemente consistente. El clustering carece de etiquetas verdaderas; un buen Silhouette por sí solo no demuestra que las agrupaciones sean útiles.

## 5. Protocolo común de evaluación y trazabilidad

1. Usar una ingesta compartida y versionada. Conservar identificadores del periodo y versión de datos en cada experimento.
2. Para P1 y P2: partición cronológica propuesta **octubre 2025–mayo 2026 (entrenamiento), junio–julio 2026 (validación), agosto–septiembre 2026 (prueba)**. No mezclar aleatoriamente horas del futuro con el entrenamiento.
3. Elegir variables, hiperparámetros y umbrales usando entrenamiento/validación. Reservar prueba para la evaluación final; no ajustar después en función del resultado de prueba.
4. Mantener idénticas las definiciones de datos y conjuntos de evaluación para baseline y candidato de cada pregunta.
5. Ejecutar los baselines desde scripts del repositorio, no únicamente desde notebooks.
6. Registrar al menos una corrida MLflow por pregunta con nombre descriptivo, fecha inicial/final, versión de datos, semilla, parámetros, métricas con prefijos `val_` o `test_`, etiquetas de equipo/P1-P2-P3/commit y al menos un artefacto (gráfica de errores, matriz de confusión o mapa de clusters).
7. No registrar identificadores de usuarios, credenciales ni archivos brutos con datos personales.

## 6. Decisiones pendientes para la reunión del equipo

- Aprobar periodo de doce meses y cortes temporales.
- **P2:** fijar fórmula, umbral y tratamiento de capacidades históricas/faltantes para la variable *proxy* de riesgo.
- **P3:** fijar cobertura mínima de actividad por estación y el conjunto final de features, una vez explorados los datos.
- Confirmar quién implementará y ejecutará cada baseline, así como la integración común en MLflow.

## 7. Referencias

- scikit-learn developers. (s. f.). *Metrics and scoring: Quantifying the quality of predictions*. https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn developers. (s. f.). *Clustering*. https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn developers. (s. f.). *Selecting the number of clusters with silhouette analysis on KMeans clustering*. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
- MiBici. (s. f.). *Datos abiertos*. https://mibici.net/es/datos-abiertos/
