# Validación de datos con DeepChecks

Generado con `python -m mibici.validacion_datos`.

## viajes

22 condiciones: 0 errores de integridad, 3 advertencias de comparación (96 s).

| Nivel | Chequeo | Condición | Resultado | Detalle |
|---|---|---|---|---|
| integridad | Esquema | Están todas las columnas esperadas | ✅ PASS | 5 columnas presentes |
| integridad | Esquema | viaje_id es de tipo int | ✅ PASS | dtype real: int64 |
| integridad | Esquema | inicio es de tipo datetime | ✅ PASS | dtype real: datetime64[ns] |
| integridad | Esquema | fin es de tipo datetime | ✅ PASS | dtype real: datetime64[ns] |
| integridad | Esquema | origen_id es de tipo int | ✅ PASS | dtype real: int64 |
| integridad | Esquema | destino_id es de tipo int | ✅ PASS | dtype real: int64 |
| integridad | Dominio | viaje_id es único | ✅ PASS | 0 identificadores repetidos |
| integridad | Dominio | Ningún viaje termina antes de empezar | ✅ PASS | 0 viajes con duración negativa |
| comparacion | Dominio | Ningún viaje dura más de 24 h | ❌ FAIL | 13 viajes más largos (0.000 %) |
| integridad | Dominio | Los viajes inician entre 2025-10-01 y 2026-09-30 | ✅ PASS | 0 viajes fuera del periodo |
| comparacion | Catálogo | origen_id existe en la nomenclatura de estaciones | ❌ FAIL | Estaciones ausentes del catálogo: [405, 407] |
| comparacion | Catálogo | destino_id existe en la nomenclatura de estaciones | ❌ FAIL | Estaciones ausentes del catálogo: [405, 407] |
| integridad | Mixed Nulls | Number of different null types is less or equal to 1 | ✅ PASS | Passed for 7 relevant columns |
| integridad | Percent Of Nulls | Percent of null values in each column is not greater than 0% | ✅ PASS | Passed for 5 relevant columns |
| integridad | Data Duplicates | Duplicate data ratio is less or equal to 0% | ✅ PASS | Found 0% duplicate data |
| integridad | Is Single Value | Does not contain only a single value | ✅ PASS | Passed for 7 relevant columns |
| integridad | Mixed Data Types | Rare data types in column are either more than 20% or less than 1% of the data | ✅ PASS | 7 columns passed: found 0 columns with negligible types mix, and 7 columns without any types mix |
| comparacion | Datasets Size Comparison | Test-Train size ratio is greater than 0.05 | ✅ PASS | Test-Train size ratio is 0.31 |
| comparacion | Feature Drift | categorical drift score < 0.2 and numerical drift score < 0.2 | ✅ PASS | Passed for 5 columns out of 5 columns. Found column "dia_semana" has the highest categorical drift score: 0.04 Found column "duracion_min" has the highest numer |
| comparacion | Train Test Samples Mix | Percentage of test data samples that appear in train data is less or equal to 5% | ✅ PASS | Percent of test data samples that appear in train data: 1.37% |
| integridad | Date Train Test Leakage Overlap | Date leakage ratio is less or equal to 0% | ✅ PASS | No leaked dates found |
| integridad | Date Train Test Leakage Duplicates | Date leakage ratio is less or equal to 0% | ✅ PASS | No leaked dates found |

## p1

11 condiciones: 0 errores de integridad, 0 advertencias de comparación (87 s).

| Nivel | Chequeo | Condición | Resultado | Detalle |
|---|---|---|---|---|
| integridad | Mixed Nulls | Number of different null types is less or equal to 1 | ✅ PASS | Passed for 14 relevant columns |
| integridad | Percent Of Nulls | Percent of null values in each column is not greater than 0% | ✅ PASS | Passed for 12 relevant columns |
| integridad | Data Duplicates | Duplicate data ratio is less or equal to 20% | ✅ PASS | Found 0.0006% duplicate data |
| integridad | Is Single Value | Does not contain only a single value | ✅ PASS | Passed for 14 relevant columns |
| integridad | Mixed Data Types | Rare data types in column are either more than 20% or less than 1% of the data | ✅ PASS | 14 columns passed: found 0 columns with negligible types mix, and 14 columns without any types mix |
| comparacion | Datasets Size Comparison | Test-Train size ratio is greater than 0.05 | ✅ PASS | Test-Train size ratio is 0.33 |
| comparacion | Feature Drift | categorical drift score < 0.2 and numerical drift score < 0.2 | ✅ PASS | Passed for 12 columns out of 12 columns. Found column "dia_semana" has the highest categorical drift score: 0.03 Found column "media_hist_estacion" has the high |
| comparacion | Train Test Samples Mix | Percentage of test data samples that appear in train data is less or equal to 5% | ✅ PASS | Percent of test data samples that appear in train data: 0.0001% |
| comparacion | Label Drift | Label drift score < 0.2 | ✅ PASS | Label's drift score Kolmogorov-Smirnov is 0.02 |
| integridad | Date Train Test Leakage Overlap | Date leakage ratio is less or equal to 0% | ✅ PASS | No leaked dates found |
| integridad | Date Train Test Leakage Duplicates | Date leakage ratio is less or equal to 0% | ✅ PASS | No leaked dates found |

## p2

11 condiciones: 0 errores de integridad, 1 advertencias de comparación (82 s).

| Nivel | Chequeo | Condición | Resultado | Detalle |
|---|---|---|---|---|
| integridad | Mixed Nulls | Number of different null types is less or equal to 1 | ✅ PASS | Passed for 15 relevant columns |
| integridad | Percent Of Nulls | Percent of null values in each column is not greater than 0% | ✅ PASS | Passed for 13 relevant columns |
| integridad | Data Duplicates | Duplicate data ratio is less or equal to 20% | ✅ PASS | Found 8.98% duplicate data |
| integridad | Is Single Value | Does not contain only a single value | ✅ PASS | Passed for 15 relevant columns |
| integridad | Mixed Data Types | Rare data types in column are either more than 20% or less than 1% of the data | ✅ PASS | 15 columns passed: found 0 columns with negligible types mix, and 15 columns without any types mix |
| comparacion | Train Test Samples Mix | Percentage of test data samples that appear in train data is less or equal to 5% | ❌ FAIL | Percent of test data samples that appear in train data: 12.77% |
| comparacion | Datasets Size Comparison | Test-Train size ratio is greater than 0.05 | ✅ PASS | Test-Train size ratio is 0.33 |
| comparacion | Feature Drift | categorical drift score < 0.2 and numerical drift score < 0.2 | ✅ PASS | Passed for 13 columns out of 13 columns. Found column "dia_semana" has the highest categorical drift score: 0.03 Found column "presion_media_168" has the highes |
| comparacion | Label Drift | Label drift score < 0.2 | ✅ PASS | Label's drift score Cramer's V is 0.02 |
| integridad | Date Train Test Leakage Overlap | Date leakage ratio is less or equal to 0% | ✅ PASS | No leaked dates found |
| integridad | Date Train Test Leakage Duplicates | Date leakage ratio is less or equal to 0% | ✅ PASS | No leaked dates found |

## p3

8 condiciones: 0 errores de integridad, 0 advertencias de comparación (1 s).

| Nivel | Chequeo | Condición | Resultado | Detalle |
|---|---|---|---|---|
| integridad | Mixed Nulls | Number of different null types is less or equal to 1 | ✅ PASS | Passed for 7 relevant columns |
| integridad | Percent Of Nulls | Percent of null values in each column is not greater than 0% | ✅ PASS | Passed for 7 relevant columns |
| integridad | Data Duplicates | Duplicate data ratio is less or equal to 0% | ✅ PASS | Found 0% duplicate data |
| integridad | Is Single Value | Does not contain only a single value | ✅ PASS | Passed for 7 relevant columns |
| integridad | Mixed Data Types | Rare data types in column are either more than 20% or less than 1% of the data | ✅ PASS | 7 columns passed: found 0 columns with negligible types mix, and 7 columns without any types mix |
| comparacion | Datasets Size Comparison | Test-Train size ratio is greater than 0.05 | ✅ PASS | Test-Train size ratio is 1.27 |
| comparacion | Feature Drift | categorical drift score < 0.2 and numerical drift score < 0.2 | ✅ PASS | Passed for 7 columns out of 7 columns. Found column "balance_relativo" has the highest numerical drift score: 0.16 |
| comparacion | Train Test Samples Mix | Percentage of test data samples that appear in train data is less or equal to 5% | ✅ PASS | No samples mix found |
