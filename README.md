# MiBici — Análisis de demanda y patrones de uso

**Proyecto integrador de MLOps | Equipo 31**

**Integrantes:**
- Leonardo Cossío Dinorín — A01753898
- Santiago Espinosa Domínguez — A01747478
- Rafael Posadas Ruíz — A01747128

**Repositorio:** https://github.com/LeoCossio/TC5061-Proyecto-Equipo31

## Descripción

Este proyecto analiza los datos abiertos del sistema de bicicletas públicas **MiBici** mediante tres problemas complementarios de aprendizaje automático: predicción de demanda, clasificación de riesgo indirecto de desbalance y segmentación de estaciones. La solución utiliza una ingesta común, código de entrenamiento reutilizable y registro de experimentos con MLflow.

**Estado:** los pipelines de **P1, P2 y P3 están implementados**, y el equipo ha comprobado la ejecución conjunta con `python -m mibici.run_all`. La verificación del manifiesto versionado de integridad de datos y la reproducción desde un clon limpio deben documentarse por separado cuando estén concluidas.

# Parte 1

## 1. Preguntas y modelos

| Pregunta | Tipo | Unidad / horizonte | Baseline | Modelo candidato | Evaluación |
|---|---|---|---|---|---|
| **P1 — Demanda** | Regresión | Estación-hora; próxima hora | Media histórica | `RandomForestRegressor` | MAE, RMSE y MAE en horas pico |
| **P2 — Riesgo de desbalance** | Clasificación | Estación-hora; próxima hora | Persistencia del riesgo de la hora anterior | `RandomForestClassifier` | Precision, recall, F1 y PR-AUC |
| **P3 — Perfiles de estaciones** | Clustering | Estación; sin horizonte predictivo | Terciles de demanda | `KMeans` (`k=3`) | Silhouette Score y análisis de perfiles |

**Alcance de P2:** el objetivo es un **indicador indirecto (*proxy*)** construido con flujos de viajes, no la ocupación real de bicicletas o anclajes. Se calcula la presión de desbalance como `abs(llegadas - salidas) / capacidad`, y el umbral de riesgo se establece mediante el percentil 90 de los datos de entrenamiento. La capacidad procede del catálogo disponible y no garantiza reflejar la capacidad histórica de cada estación durante todo el periodo.

## 2. Datos y preparación

- **Fuente:** [Portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).
- **Periodo:** octubre de 2025 a septiembre de 2026, con 12 CSV mensuales y catálogo de estaciones de septiembre de 2026.
- **Registros originales:** 4,497,877.
- **Registros procesados:** 4,497,119, tras eliminar 758 duplicados exactos.
- **Estaciones:** 471 identificadas en viajes; 484 en el catálogo de septiembre de 2026.
- **P3:** 470 estaciones después de exigir al menos 100 salidas durante el periodo.
- **Campos procesados:** `viaje_id`, `inicio`, `fin`, `origen_id`, `destino_id`.

`mibici/dataset.py` descarga los insumos, normaliza campos y codificaciones, valida fechas y genera los 12 archivos Parquet. Los CSV y Parquet permanecen en `data/` y **no se suben a Git**. Los identificadores de usuario y los atributos demográficos no se conservan en los Parquet para modelado.

### Trazabilidad de datos

Los entrenamientos registran en MLflow una huella SHA-256 de los datos utilizados, junto con parámetros y la versión de código. El equipo está incorporando un **manifiesto versionado por archivo** que permita verificar nombres, cantidades de registros y hashes SHA-256. El manifiesto debe añadirse a Git y ejecutarse su verificación antes de dar este requisito por completado; no debe confundirse con el hash agregado de un experimento.

Detalles: [`references/definicion_datos.md`](references/definicion_datos.md) y [`references/limpieza_datos.md`](references/limpieza_datos.md).

## 3. Estructura del repositorio

```text
.
├── data/                         # Insumos y Parquet locales; ignorados por Git
│   ├── raw/
│   ├── external/
│   └── processed/
├── mibici/
│   ├── config.py                 # Configuración compartida
│   ├── dataset.py                # Ingesta y limpieza
│   ├── features.py               # Acceso a ingeniería de características
│   ├── feature_engineering/
│   │   ├── regresion.py          # Variables de P1
│   │   ├── classification.py     # Variables y etiquetas de P2
│   │   └── clustering.py         # Variables de P3
│   ├── train_p1.py               # Regresión y MLflow
│   ├── train_p2.py               # Clasificación y MLflow
│   ├── train_p3.py               # Clustering y MLflow
│   └── run_all.py                # Ejecución conjunta de P1, P2 y P3
├── notebooks/                    # Exploración
├── models/                       # Artefactos locales no versionados
├── references/                   # Documentación técnica
├── reports/figures/              # Figuras del análisis
├── tests/                        # Pruebas
├── requirements.txt              # Dependencias
├── pyproject.toml
└── README.md
```

La organización se basa en **Cookiecutter Data Science** y separa la exploración en notebooks de la ejecución de modelos mediante módulos de Python. Los scripts o archivos adicionales del manifiesto se incorporarán a este esquema al integrarse su PR.

## 4. Instalación

El equipo desarrolló el proyecto con **Python 3.13**. Se necesita Git y conexión a Internet para la primera descarga de datos. Desde una terminal de PowerShell:

```powershell
git clone https://github.com/LeoCossio/TC5061-Proyecto-Equipo31.git
cd TC5061-Proyecto-Equipo31
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ejecutar siempre los comandos siguientes desde la **raíz del repositorio**. La reproducción en otro entorno depende de utilizar versiones compatibles de Python y las dependencias especificadas.

## 5. Ejecución

### 5.1 Descargar y procesar los datos

```powershell
python -m mibici.dataset
```

Al finalizar deben existir 12 Parquet mensuales en `data/processed/` y el catálogo en `data/external/`.

### 5.2 Entrenar cada pregunta por separado

```powershell
python -m mibici.train_p1
python -m mibici.train_p2
python -m mibici.train_p3
```

Cada script registra el baseline y el modelo candidato en el experimento **`mibici-equipo31`**, con métricas, parámetros, etiquetas y artefactos pertinentes a su pregunta.

### 5.3 Ejecutar los tres pipelines

```powershell
python -m mibici.run_all
```

Este comando coordina P1, P2 y P3 utilizando los Parquet ya preparados; **no sustituye la descarga de datos**. Su ejecución conjunta fue comprobada por el equipo.

### 5.4 Visualizar los experimentos

```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Abrir http://127.0.0.1:5000 y seleccionar `mibici-equipo31`. Es posible revisar los runs individuales, descargar sus artefactos y comparar el baseline con el modelo candidato mediante **Compare**. La base de datos local de MLflow no se versiona en Git; para reproducir las corridas hay que ejecutar los scripts con las entradas y dependencias correspondientes.

## 6. Resultados y evaluación

**P1.** Compara la media histórica con `RandomForestRegressor` usando una separación cronológica: entrenamiento de octubre de 2025 a mayo de 2026, validación en junio–julio y prueba en agosto–septiembre. Consulta los valores de MAE, RMSE y error en horas pico en MLflow; las cifras dependen de la corrida documentada.

**P2.** Construye una etiqueta binaria de riesgo *proxy* de desbalance. Compara un baseline de persistencia con un `RandomForestClassifier` ajustado, y evalúa precision, recall, F1, PR-AUC y matriz de confusión en los periodos cronológicos correspondientes. Las variables futuras usadas para definir la etiqueta no deben emplearse como predictores.

**P3.** Compara terciles de demanda frente a K-Means sobre las mismas seis características estandarizadas:

| Modelo | Estaciones | Grupos | Silhouette Score |
|---|---:|---:|---:|
| Baseline por terciles de demanda | 470 | 3 | 0.0732 |
| K-Means (`k=3`) | 470 | 3 | **0.1992** |

K-Means produjo grupos de 154, 198 y 118 estaciones. Una proyección PCA de dos componentes representa aproximadamente el **56.3 % de la varianza**, únicamente con fines de visualización. La separación relativa mejora frente al baseline, aunque los grupos tienen superposición y no se ha demostrado todavía estabilidad temporal.

Detalles metodológicos: [`references/modelos_metricas.md`](references/modelos_metricas.md) y [`references/pipelines_ml.md`](references/pipelines_ml.md).

## 7. Flujo de trabajo con Git y GitHub

El equipo utiliza **GitHub Flow**, con los siguientes acuerdos:

1. **`main` protegida:** los cambios no se incorporan mediante pushes directos, sino mediante **Pull Requests (PR)** dirigidos a `main`.
2. **Ramas por propósito:** `feature/p1-regression`, `feature/p2-clasificacion`, `feature/p3-clustering`, `feature/data-manifest`, `fix/descripcion` o `docs/descripcion`. Cada rama debe partir de la versión actualizada de `main` siempre que sea posible.
3. **Conventional Commits:** mensajes con tipo, alcance opcional y descripción breve, como `feat(p2): implement risk classifier`, `fix(data): handle csv encoding` o `docs: update project references`.
4. **Revisión antes de fusionar:** se publica una rama, se abre un PR hacia `main`, se solicita revisión y se atienden observaciones antes de integrarlo, respetando las reglas de protección configuradas.
5. **Participación individual:** **cada integrante debe abrir al menos un PR y realizar al menos una revisión del PR de otra persona**. La evidencia verificable son los PR y las revisiones registradas en GitHub; la presencia de commits por sí sola no acredita una revisión.

Enlaces: [Repositorio](https://github.com/LeoCossio/TC5061-Proyecto-Equipo31) · [Pull Requests](https://github.com/LeoCossio/TC5061-Proyecto-Equipo31/pulls) · [PR cerrados](https://github.com/LeoCossio/TC5061-Proyecto-Equipo31/pulls?q=is%3Apr+is%3Aclosed).

> **Verificación final de Git:** antes de entregar, comprobar en GitHub que los tres integrantes cumplieron individualmente con un PR abierto y una revisión ajena. No se atribuyen revisiones o aprobaciones que no estén comprobadas en la plataforma.

## 8. Integrantes y responsabilidades

| Integrante | Responsabilidad principal |
|---|---|
| Leonardo Cossío Dinorín | P3: exploración, segmentación por estaciones, K-Means y MLflow |
| Santiago Espinosa Domínguez | P2: clasificación de riesgo y contribuciones a la ingesta compartida |
| Rafael Posadas Ruíz | P1: predicción de demanda mediante regresión |

Las contribuciones definitivas y las revisiones se acreditan mediante el historial de commits y Pull Requests de GitHub.

## 9. Documentación y verificaciones finales

- [`references/arquitectura_sistema.md`](references/arquitectura_sistema.md): componentes, decisiones técnicas y diagrama.
- [`references/definicion_datos.md`](references/definicion_datos.md): fuentes, periodo, variables, privacidad y particiones.
- [`references/limpieza_datos.md`](references/limpieza_datos.md): controles de calidad e ingesta.
- [`references/modelos_metricas.md`](references/modelos_metricas.md): baselines, modelos y métricas.
- [`references/pipelines_ml.md`](references/pipelines_ml.md): flujos reproducibles por pregunta.

**Comprobaciones pendientes de cierre documental:**

- [ ] Integrar y verificar el manifiesto Git de datos con SHA-256 y conteos por archivo.
- [ ] Comprobar desde un clon limpio la instalación, ingesta y ejecución conjunta con el conjunto de dependencias fijado.
- [ ] Confirmar en GitHub un PR y una revisión ajena por cada integrante.
- [ ] Confirmar que las capturas finales de MLflow muestran lista de runs, parámetros/métricas/etiquetas, artefacto y comparación de modelos con el commit correspondiente.
- [ ] Incorporar al PDF los resultados verificados de P1/P2 y las evidencias de los tres pipelines.

## 10. Fuentes y herramientas

- [MiBici — Datos abiertos](https://www.mibici.net/es/datos-abiertos/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [scikit-learn — Evaluación de modelos](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [GitHub — Ramas protegidas](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/)


# Parte 2

### MLflow Tracking Server

El proyecto utiliza un servidor local de MLflow con SQLite como backend para el seguimiento de experimentos y la gestión de modelos mediante Model Registry.

**1. Iniciar el servidor desde la raíz del repositorio:**

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

**2. Configurar la conexión en una segunda terminal PowerShell:**

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

**3. Acceder a la interfaz web:**

http://127.0.0.1:5000

Los experimentos se registrarán bajo el nombre `mibici-equipo31`.

La base de datos `mlflow.db` y los directorios de artefactos locales están excluidos del control de versiones mediante `.gitignore`.

**Nota:** El servidor debe permanecer activo durante las ejecuciones. La configuración de `MLFLOW_TRACKING_URI` debe realizarse en cada nueva sesión de PowerShell.