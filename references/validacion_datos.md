# Validación de calidad de datos con DeepChecks — Equipo 31

**Responsable de esta parte:** Rafael Posadas Ruíz (A01747128)
**Código:** [`mibici/validacion_datos.py`](../mibici/validacion_datos.py) · **Pruebas:** [`tests/test_validacion_datos.py`](../tests/test_validacion_datos.py)
**Resultado de la última ejecución:** [`reports/validacion_datos.md`](../reports/validacion_datos.md)

## 1. Qué se valida y por qué

DeepChecks se ejecuta sobre cuatro conjuntos, uno por cada etapa donde un problema de datos llegaría a los modelos:

| Conjunto | Contenido | Por qué se valida |
|---|---|---|
| `viajes` | Los 12 Parquet limpios de MiBici (4.5 millones de viajes) | Es la entrada de P1, P2 y P3; un error aquí afecta a las tres preguntas |
| `p1` | Tabla estación-hora de demanda con 12 variables y la etiqueta `viajes_siguiente_hora` | Datos de entrada de la regresión |
| `p2` | Tabla estación-hora con 13 variables y la etiqueta binaria `riesgo_siguiente_hora` | Datos de entrada de la clasificación |
| `p3` | Perfiles por estación calculados con el periodo de entrenamiento y con el de prueba | Entrada del clustering; permite ver si los perfiles se mantienen entre periodos (369 estaciones cumplen el mínimo de 100 salidas en entrenamiento y 468 en prueba) |

La separación entrenamiento/prueba es la misma de los modelos: entrenamiento antes del 1 de junio de 2026 y prueba desde el 1 de agosto (los meses de junio y julio son validación y no entran en esta comparación).

## 2. Dos niveles de chequeo

| Nivel | Qué ocurre si falla | Chequeos |
|---|---|---|
| **Integridad** | Error: el comando termina con código 1 y los datos no deben usarse | Tipos de datos, valores faltantes, tipos mezclados, filas duplicadas, columnas constantes, identificadores repetidos, fin anterior al inicio, fechas fuera del periodo, fuga temporal entre entrenamiento y prueba |
| **Comparación** | Advertencia: se reporta pero no bloquea (con `--estricto` sí falla) | Deriva de variables y de la etiqueta, filas idénticas entre entrenamiento y prueba, tamaño relativo de los conjuntos, estaciones ausentes del catálogo, viajes de más de 24 h |

La deriva se trata como advertencia porque puede ser legítima: entrenar con octubre–mayo y probar con agosto–septiembre mezcla estaciones del año distintas. Lo importante es que quede registrada y se revise.

## 3. Chequeos de DeepChecks utilizados

| Chequeo | Condición | Umbral |
|---|---|---|
| `MixedNulls` | Máximo un tipo de nulo por columna | 1 |
| `PercentOfNulls` | Porcentaje de nulos por columna | 0 % (los datos limpios no deben tener faltantes) |
| `DataDuplicates` | Proporción de filas repetidas | 0 % en `viajes` y `p3`; hasta 20 % en `p1` y `p2` (ver nota) |
| `IsSingleValue` | Ninguna columna constante | — |
| `MixedDataTypes` | Sin proporción intermedia de tipos mezclados | 1 %–20 % |
| `FeatureDrift` | Deriva por variable entre entrenamiento y prueba | < 0.2 (valor recomendado por DeepChecks) |
| `LabelDrift` | Deriva de la etiqueta (P1 y P2) | < 0.2 |
| `TrainTestSamplesMix` | Filas de prueba idénticas a filas de entrenamiento | ≤ 5 % |
| `DatasetsSizeComparison` | Tamaño de prueba frente al de entrenamiento | > 5 % |
| `DateTrainTestLeakageOverlap` y `DateTrainTestLeakageDuplicates` | Ninguna fecha de prueba anterior al fin del entrenamiento | 0 % |

Además, `validar_esquema` comprueba con pandas lo que DeepChecks no cubre: tipo esperado de cada columna (`viaje_id`, `origen_id` y `destino_id` enteros; `inicio` y `fin` fechas), unicidad de `viaje_id`, `fin >= inicio`, fechas dentro del periodo del proyecto y estaciones presentes en el catálogo de nomenclatura.

**Nota sobre duplicados en `p1` y `p2`:** en una tabla de variables por estación-hora puede haber filas idénticas legítimas (por ejemplo, horas de madrugada sin viajes y con el mismo historial). Por eso el umbral es del 20 % y la columna de fecha se excluye al comparar filas.

## 4. Cómo ejecutarlo

Desde la raíz del repositorio, con el entorno instalado y los 12 Parquet en `data/processed/`:

```bash
# Los cuatro conjuntos (aprox. 5 minutos; escribe reports/validacion_datos.md)
python -m mibici.validacion_datos

# Uno o varios conjuntos
python -m mibici.validacion_datos --conjuntos viajes p3

# Con los reportes interactivos de DeepChecks en reports/validacion/*.html
python -m mibici.validacion_datos --html

# Las advertencias de comparación también hacen fallar el comando
python -m mibici.validacion_datos --estricto

# Mediante el MLflow Project
mlflow run . -e validate -P conjuntos=todos --env-manager uv
```

Las pruebas unitarias (tablas sintéticas, se ejecutan en unos segundos):

```bash
python -m pytest tests/test_validacion_datos.py
```

La prueba contra los datos reales está desactivada por defecto porque tarda más de un minuto; se activa con `MIBICI_VALIDAR_DATOS_REALES=1`.

## 5. Compatibilidad con NumPy 2

DeepChecks 0.19.1 (la versión más reciente) usa `np.Inf`, `np.NaN` y `np.float_`, que NumPy 2 eliminó, y necesita `pkg_resources` (de `setuptools`). El proyecto fija `numpy==2.3.3`, así que el módulo restablece esos tres alias antes de importar DeepChecks y `requirements.txt` fija `setuptools==80.9.0`. Con este ajuste se probaron los chequeos anteriores con las versiones del proyecto (pandas 2.3.2, scikit-learn 1.7.2). Si DeepChecks publica una versión compatible con NumPy 2, la función `_compatibilidad_numpy2` puede eliminarse.
