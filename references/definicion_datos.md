# Definición de datos — Proyecto MiBici (Equipo 31)

**Estado:** Borrador para validación del equipo  
**Periodo de estudio:** 1 de octubre de 2025 al 30 de septiembre de 2026

## 1. Periodo seleccionado y justificación

Se utilizarán **12 meses consecutivos**, de octubre de 2025 a septiembre de 2026. Este intervalo cubre un ciclo anual completo y permite explorar variaciones por hora, día de la semana y mes, así como posibles efectos estacionales. Es un periodo reciente y evita combinar registros de años muy distantes, en los que podrían cambiar las estaciones o el formato de publicación.

**Limitación:** contar con un único ciclo anual permite analizar variaciones dentro del año, pero no confirma que los patrones estacionales se repitan de un año a otro. La actividad de cada estación y su capacidad pueden cambiar durante el periodo.

## 2. Fuentes y tamaño real

**Fuente oficial:** [Portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).

- **Viajes:** 12 archivos mensuales, `datos_abiertos_AAAA_MM.csv`, para octubre–diciembre de 2025 y enero–septiembre de 2026. Los enlaces directos se descubren automáticamente desde el portal en `mibici/dataset.py`; debe conservarse un listado de URLs efectivamente descargadas para la entrega.
- **Estaciones:** [nomenclatura de septiembre de 2026](https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv), con identificadores y atributos de estaciones.

**Resultados reportados por la ejecución de ingesta del equipo (pendientes de verificación independiente):**

| Indicador | Valor |
|---|---:|
| Meses | 12 |
| Registros originales | 4,497,877 |
| Registros procesados | 4,497,119 |
| Filas duplicadas eliminadas | 758 |
| Estaciones distintas presentes en viajes originales | 471 |
| Estaciones en el catálogo de septiembre de 2026 | 484 |
| Archivos Parquet mensuales generados | 12 |

Las estaciones distintas en viajes y las estaciones del catálogo son universos diferentes: algunas estaciones catalogadas pueden carecer de viajes durante el periodo y otras pueden no figurar en el catálogo de septiembre. Se reportaron las estaciones 405 y 407 como ausentes de dicha nomenclatura, aunque presentes en registros de viajes.

## 3. Variables originales y construcción de objetivos

### 3.1 Variables originales

Los CSV originales contienen `Viaje_Id`, `Usuario_Id`, `Genero`, `Año_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id` y `Destino_Id` (los nombres pueden variar por archivo). El proceso implementado normaliza y conserva en Parquet las cinco variables operativas siguientes:

| Campo procesado | Descripción y uso |
|---|---|
| `viaje_id` | Identificador del viaje; control de calidad |
| `inicio` | Fecha y hora de inicio; agregaciones de demanda |
| `fin` | Fecha y hora de terminación; agregaciones de llegadas |
| `origen_id` | Estación de salida |
| `destino_id` | Estación de llegada |

El catálogo agrega `id`, ubicación y capacidad según su estructura real; los nombres exactos de columnas se confirmarán al leer ese archivo. La capacidad del catálogo de septiembre de 2026 **no debe suponerse válida para todos los meses anteriores**.

### 3.2 Variables objetivo / resultados

**P1 — Regresión (estación-hora):** número de viajes cuya fecha de **inicio** cae en la siguiente hora para una estación. Se obtendrá contando `viaje_id` agrupado por `origen_id` y hora de `inicio`. Se incluirán intervalos con **cero viajes** para estaciones activas, de modo que los periodos sin salidas no desaparezcan del análisis.

**P2 — Clasificación (estación-hora):** etiqueta binaria de **riesgo estimado de desbalance**, no de disponibilidad real. La propuesta es construirla a partir del balance neto futuro `salidas - llegadas` durante la hora siguiente, normalizado por la capacidad de la estación, y compararlo con un umbral a definir. El indicador deberá identificar por separado riesgo por exceso de salidas y por exceso de llegadas, si es relevante. **Pendiente de acuerdo del equipo:** fórmula exacta, umbral, manejo de estaciones sin capacidad y limitaciones. No puede inferirse el inventario real solo con flujos de viajes.

**P3 — Clustering (estación):** no existe etiqueta objetivo supervisada. Se construirá una fila por estación con atributos históricos como promedio diario de viajes, distribución horaria, proporción de viajes en fines de semana y relación entre llegadas y salidas. Los grupos resultantes se interpretarán con ayuda de la ubicación geográfica y, cuando sea apropiado, la capacidad.

## 4. Condiciones de uso y tratamiento de datos personales

El [portal oficial](https://www.mibici.net/es/datos-abiertos/) publica archivos de uso y alienta a explorarlos y compartirlos. **No se identificó en esa página una licencia específica explícita para los CSV** (por ejemplo, CC BY o CC0); por lo tanto, no se afirmará una licencia de reutilización concreta sin verificación adicional. Se citará la fuente y se revisarán las condiciones aplicables antes de redistribuir datos derivados.

Se consultó también el [aviso de privacidad publicado por MiBici](https://www.mibici.net/es/aviso-de-privacidad/). Para este proyecto:

- No se utilizarán `Usuario_Id`, `Genero` ni `Año_de_nacimiento` para entrenar los modelos.
- El código de limpieza conserva solamente los cinco campos operativos enumerados arriba en los Parquet procesados.
- Los CSV originales se mantendrán localmente y excluidos del repositorio con `.gitignore`; pueden contener identificadores pseudónimos y secuencias de trayectos que requieren precaución.
- Los resultados compartidos serán, de preferencia, estadísticas agregadas por estación o periodo. No se publicarán trayectorias individuales ni identificadores de usuario.

## 5. Partición temporal y prevención de fuga de información

**Propuesta para P1 y P2 (sujeta a aprobación del equipo):**

| Conjunto | Meses | Objetivo |
|---|---|---|
| Entrenamiento | Oct 2025 – May 2026 (8) | Ajustar modelos y transformaciones |
| Validación | Jun – Jul 2026 (2) | Elegir hiperparámetros y umbrales |
| Prueba | Ago – Sep 2026 (2) | Evaluación final sin reajustes |

La separación será cronológica, sin mezclar aleatoriamente viajes de meses futuros con los de entrenamiento. Las variables de demanda histórica, balances y promedios para predecir la hora siguiente se calcularán **solo a partir de eventos disponibles hasta el instante de predicción**. El escalado y otras transformaciones aprendidas se ajustarán únicamente con entrenamiento, y se aplicarán sin volver a ajustarlas a validación y prueba.

Para **P2**, los flujos futuros sirven para construir la etiqueta *después* del periodo objetivo, pero jamás como predictores de esa misma hora. Se evitará usar como variable el mismo indicador futuro que define el objetivo.

Para **P3**, no se requiere partición supervisada, pero se propone calcular perfiles y agrupaciones con los datos del periodo de entrenamiento y examinar su estabilidad en meses posteriores. Al comparar periodos deberán tratarse por separado las estaciones nuevas o con presencia parcial para evitar interpretaciones equivocadas.

## 6. Estado y pendientes de validación

- [ ] Adjuntar el listado exacto de enlaces mensuales utilizados por el proceso.
- [ ] Verificar los conteos y la unicidad directamente sobre los Parquet generados.
- [ ] Confirmar nombres y cobertura histórica de la nomenclatura de estaciones.
- [ ] Acordar la fórmula y el umbral de P2.
- [ ] Aprobar la partición temporal por el equipo.
- [ ] Implementar y versionar un manifiesto de integridad con hashes SHA-256; registrar su versión en MLflow.
- [ ] Probar la ingesta desde un clon limpio.

## Referencias de consulta

- MiBici. (s. f.). *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/
- MiBici. (s. f.). *Aviso de privacidad*. https://www.mibici.net/es/aviso-de-privacidad/
