# Diseño de pipelines de Machine Learning — MiBici (Equipo 31)

**Estado:** Propuesta técnica para revisión y aprobación del equipo  
**Periodo de estudio:** Octubre de 2025 a septiembre de 2026  
**Preguntas:** P1 (regresión), P2 (clasificación) y P3 (clustering)

## 1. Objetivo y flujo compartido

Diseñar un flujo reproducible desde los registros originales de viajes de MiBici hasta la evaluación de los tres modelos. Se reutilizará una sola ingesta y limpieza; cada problema tendrá sus propias agregaciones, características, entrenamiento y evaluación.

```text
Portal MiBici (12 CSV de viajes + catálogo de estaciones)
       ↓
Descarga y limpieza compartida — mibici/dataset.py
       ↓
Archivos Parquet mensuales — data/processed/
       ↓
Construcción de agregaciones y variables — mibici/features.py
       ├── P1: tabla estación-hora → regresión → evaluación
       ├── P2: tabla estación-hora → clasificación → evaluación
       └── P3: tabla estación → clustering → evaluación e interpretación
       ↓
Registro de parámetros, métricas y artefactos en MLflow
```

### 1.1 Limpieza compartida (implementada parcialmente)

El código disponible en `mibici/dataset.py` descarga los archivos publicados, lee UTF-8 o CP1252, normaliza columnas y tipos, elimina duplicados exactos y registros inválidos (campos esenciales faltantes o finalización anterior al inicio), y produce archivos Parquet comprimidos. Conserva `viaje_id`, `inicio`, `fin`, `origen_id` y `destino_id`. Los identificadores de usuario y atributos demográficos no se incorporan a los Parquet procesados.

El equipo reportó **4,497,119 viajes procesados** a partir de 4,497,877 originales. Estas cifras deberán comprobarse con una ejecución reproducible. Queda pendiente el manifiesto versionado con hashes SHA-256 y la verificación de integridad de las descargas.

**Controles comunes adicionales propuestos:** verificar identificadores de estación contra el catálogo, documentar estaciones no presentes (405 y 407), identificar estaciones incorporadas durante el año, y mantener los intervalos sin viajes como ceros cuando corresponda. El catálogo de septiembre de 2026 no garantiza capacidades históricas constantes.

## 2. P1 — Predicción de demanda (regresión)

**Pregunta:** ¿Cuántos viajes se iniciarán en una estación durante la próxima hora?  
**Unidad de análisis:** estación-hora. **Horizonte:** una hora.

| Etapa | Diseño propuesto |
|---|---|
| Limpieza | Utilizar Parquet procesados y verificar fechas, estaciones y duplicados. |
| Agregación | Contar salidas por `origen_id` y hora de `inicio`. Construir una secuencia horaria por estación activa e incluir horas con 0 salidas. |
| Feature engineering | Hora del día, día de la semana, indicador de fin de semana, salidas en horas anteriores y promedios históricos calculados solo con datos anteriores a la predicción. |
| Objetivo | Conteo de salidas durante la hora siguiente (`viajes_siguiente_hora`). |
| Baseline | Promedio de salidas histórico de la estación para esa hora y día de la semana, calculado únicamente sobre datos pasados. |
| Modelo candidato | `RandomForestRegressor` de scikit-learn. |
| Evaluación | MAE como métrica principal y RMSE complementaria; comparar contra baseline en validación y prueba cronológicas. Registrar resultados por estación o nivel de demanda si aporta interpretación. |

**Justificación de variables:** la hora y el día capturan ciclos de movilidad; los rezagos y promedios históricos reflejan la intensidad reciente y habitual de la demanda. El identificador de estación permite distinguir comportamientos locales, con codificación apropiada para el modelo.

**Precaución temporal:** para predecir la hora `[t, t+1)`, ninguna característica puede incluir salidas iniciadas en esa hora ni periodos posteriores. Los promedios deben calcularse de forma causal, nunca sobre todo el año antes de dividir los datos.

## 3. P2 — Riesgo estimado de desbalance (clasificación)

**Pregunta:** ¿Una estación presentará un flujo neto suficientemente desbalanceado durante la próxima hora como para ameritar atención operativa?  
**Unidad de análisis:** estación-hora. **Horizonte:** una hora.

| Etapa | Diseño propuesto |
|---|---|
| Limpieza | Utilizar viajes procesados y combinar con capacidad del catálogo cuando exista; identificar faltantes y advertir que la capacidad histórica puede cambiar. |
| Agregación | Contar salidas por hora de `inicio` y llegadas por hora de `fin`, agrupadas por estación; crear intervalos sin movimientos. |
| Feature engineering | Entradas y salidas previas, balances netos en ventanas históricas, hora y día de la semana, y capacidad de estación si es válida. |
| Objetivo | Etiqueta **proxy** de riesgo futuro basada en magnitud y dirección del balance neto de la próxima hora en relación con la capacidad (fórmula y umbral pendientes de acuerdo). |
| Baseline | Predecir riesgo a partir de la frecuencia histórica de riesgo de cada estación/franja horaria, usando exclusivamente periodos anteriores; ajustar el umbral con validación. |
| Modelo candidato | `RandomForestClassifier` de scikit-learn. |
| Evaluación | Recall de la clase de riesgo, precision y F1; examinar matriz de confusión y distribución de clases frente al baseline. |

**Justificación de variables:** entradas, salidas y balance resumen presiones recientes sobre las estaciones; la capacidad permite expresar un flujo relativo al tamaño de la estación; hora y día recogen recurrencias temporales.

**Limitación crítica:** MiBici no publica inventarios reales por estación. Un flujo neto alto **no demuestra** que la estación esté vacía o llena. El objetivo solo mide un **indicador indirecto de desbalance**. Deben documentarse fórmula, umbral, tratamiento de capacidades faltantes y reglas para distinguir exceso de salidas vs. llegadas antes de implementar P2.

**Precaución temporal:** el balance futuro se utiliza exclusivamente para construir la etiqueta de evaluación, jamás como característica de entrada. La regla del baseline tampoco debe repetir literalmente la fórmula que define las etiquetas usando datos futuros.

## 4. P3 — Segmentación de estaciones (clustering)

**Pregunta:** ¿Qué estaciones tienen patrones históricos de uso semejantes y dónde se encuentran?  
**Unidad de análisis:** estación. **Horizonte predictivo:** no aplica; se usan perfiles históricos.

| Etapa | Diseño propuesto |
|---|---|
| Limpieza | Usar Parquet procesados; identificar estaciones con meses incompletos de actividad o ausencia en el catálogo. |
| Agregación | Construir una fila por estación a partir de conteos de salidas y llegadas por hora, día y fin de semana. |
| Feature engineering | Viajes iniciados diarios promedio; porcentaje de salidas en horas pico; porcentaje de salidas en fin de semana; balance medio de entradas vs. salidas; perfil horario normalizado. |
| Preparación | Tratar valores faltantes, evaluar estaciones con poca actividad y escalar características numéricas con `StandardScaler` antes de K-Means. Evitar variables redundantes o altamente correlacionadas. |
| Referencia (*baseline*) | Segmentación sencilla por niveles de demanda (baja/media/alta) usando umbrales definidos con datos históricos. |
| Modelo candidato | `KMeans` sobre las características estandarizadas; explorar varios valores pequeños de `k` con semilla fija. |
| Evaluación | Silhouette Score, tamaño de grupos, diferencias entre perfiles, estabilidad entre periodos y visualización geográfica de clusters. Comparar con el baseline sobre el mismo espacio de características escaladas. |

**Justificación de variables:** la demanda diaria distingue intensidad de uso; los perfiles horarios y de fin de semana identifican estilos de movilidad; el balance de entradas/salidas refleja el papel de cada estación en los flujos. Las coordenadas del catálogo se utilizarán principalmente para **mapear e interpretar**, no como variables iniciales del clustering, para evitar agrupar solo por proximidad.

**Evaluación temporal:** proponer aprender perfiles y grupos con octubre de 2025–mayo de 2026; utilizar junio–julio para ajustar diseño de características/`k`, y agosto–septiembre como evaluación final de estabilidad. Comparar solamente estaciones con cobertura adecuada y documentar altas/bajas. No se trata de un test supervisado ni existe una etiqueta verdadera de cluster.

## 5. Partición y prevención transversal de fuga de información

Se propone, sujeto a validación del equipo:

- **Entrenamiento:** octubre de 2025 – mayo de 2026.
- **Validación:** junio – julio de 2026.
- **Prueba:** agosto – septiembre de 2026.

Para P1 y P2, separar cronológicamente los ejemplos por hora objetivo; calcular rezagos y estadísticas utilizando únicamente datos conocidos hasta cada instante de predicción. Si se aprovecha historia previa a la frontera del conjunto para construir rezagos, no debe utilizarse ninguna etiqueta futura. Ajustar codificadores, imputadores y escaladores con entrenamiento y aplicarlos sin reajustar a los demás conjuntos.

Para P3, esta división se interpreta como evaluación de **estabilidad temporal de perfiles**, no como división supervisada. Evitar calcular perfiles anuales completos antes de la evaluación temporal y luego tratarlos como resultados independientes de los meses posteriores.

## 6. Ejecución reproducible y trazabilidad previstas

- Mantener las rutas, fechas y semillas compartidas en `mibici/config.py`.
- Trasladar agregaciones y construcción de variables desde notebooks exploratorios a funciones reutilizables con docstrings en `mibici/features.py`.
- Implementar scripts ejecutables desde la terminal para cada baseline/modelo, separados de los notebooks.
- Registrar en **MLflow** por P1, P2 y P3: fechas, versión/hash de datos, semilla, hiperparámetros, métricas con identificación del conjunto evaluado (`test_mae`, `test_recall`, etc.), etiquetas de equipo/pregunta/commit, y al menos un artefacto por run.
- Mantener la definición de las transformaciones compartidas consistente entre modelos y registrar las decisiones sobre los cambios de esquema.

## 7. Estado de implementación y decisiones pendientes

**Implementado/revisado a partir del ZIP recibido:** descarga, normalización básica, filtrado y escritura de Parquet en `mibici/dataset.py`. El `mibici/features.py` disponible todavía contiene el ejemplo inicial de CCDS, no las transformaciones aquí propuestas. No se verificó un entrenamiento de P1/P2/P3 ni sus corridas en MLflow.

**Pendientes antes de cerrar este diseño:**

1. Aprobar formalmente las fechas y partición temporal.
2. Precisar fórmula y umbral del proxy de P2; decidir cómo tratar estaciones sin capacidad válida.
3. Elegir horas pico y política de inclusión de estaciones con actividad parcial para P3.
4. Implementar el versionado con hashes y asegurar que los tres modelos registren la misma versión de datos.
5. Validar los baselines y candidatos mediante experimentos reproducibles, sin presentar métricas hipotéticas como obtenidas.

Este documento describe **el pipeline propuesto**, no afirma que todas sus etapas estén ya implementadas.
