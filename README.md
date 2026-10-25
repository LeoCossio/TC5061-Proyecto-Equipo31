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

## 4. Instalación y reproducción del entorno

El proyecto está desarrollado en **Python 3.13.7** y utiliza MLflow Projects para ejecutar los pipelines de manera reproducible.

Las dependencias se encuentran declaradas en `requirements.txt` y `python_env.yaml`. Para la creación del entorno aislado se recomienda utilizar **uv**, ya que su funcionamiento fue comprobado en Windows durante las pruebas de reproducibilidad.

### 4.1. Requisitos previos

- Python 3.13.
- Git.
- MLflow y uv.
- Conexión a Internet para descargar dependencias y datos de MiBici.

Instalar las herramientas necesarias:

```powershell
python -m pip install mlflow uv
```

Verificar la instalación:

```powershell
mlflow --version
uv --version
```

### 4.2. Clonar el repositorio

```powershell
git clone https://github.com/LeoCossio/TC5061-Proyecto-Equipo31.git
cd TC5061-Proyecto-Equipo31
```

Los comandos siguientes deben ejecutarse desde la raíz del repositorio.

### 4.3. Gestión del entorno con uv

El archivo `MLproject` declara el entorno definido en `python_env.yaml`, que especifica la versión de Python y las dependencias necesarias para ejecutar los modelos.

Para utilizar dicho entorno, los comandos de MLflow incluyen:

```text
--env-manager uv
```

MLflow utiliza uv para preparar un entorno aislado e instalar las dependencias. No es necesario instalar manualmente `requirements.txt` para los entrenamientos ejecutados mediante este mecanismo.

Se recomienda utilizar `uv` en Windows porque la ejecución con `--env-manager virtualenv` presentó dificultades con la detección de `pyenv`.

También es posible utilizar `--env-manager local` si ya existe un entorno configurado, aunque esta opción no proporciona el mismo aislamiento.

## 5. Ejecución reproducible mediante MLflow

### 5.1. Iniciar el servidor de Tracking

El proyecto utiliza MLflow Tracking con una base de datos SQLite para almacenar experimentos y gestionar el Model Registry.

Desde una primera terminal PowerShell, ubicada en la raíz del repositorio, ejecutar:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

El servidor debe permanecer activo durante la ejecución de los experimentos.

La interfaz web estará disponible en:

http://127.0.0.1:5000

La base de datos `mlflow.db` se crea localmente y está excluida del control de versiones.

### 5.2. Configurar la conexión al servidor

Abrir una segunda terminal PowerShell en la raíz del repositorio.

Ejecutar:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

Esta variable permite que los entrenamientos registren sus resultados en el servidor local.

La configuración debe repetirse en cada nueva sesión de PowerShell.

### 5.3. Ingesta y preparación de datos

Los datos originales de MiBici no se incluyen en el repositorio.

Por tanto, antes de entrenar es necesario ejecutar la ingesta:

```powershell
mlflow run . -e ingest --env-manager uv --experiment-name mibici-equipo31
```

Este proceso:

1. Descarga los archivos históricos de MiBici.
2. Realiza limpieza y normalización de los registros.
3. Genera los 12 archivos Parquet en `data/processed/`.
4. Genera el manifiesto `references/manifest_datos.json` con información de integridad y hashes SHA-256.

El conjunto procesado de referencia contiene **4,497,119 viajes** correspondientes al periodo octubre de 2025 a septiembre de 2026.

### 5.4. Entrenar P1 — Regresión de demanda

Random Forest Regressor:

```powershell
mlflow run . -e train -P question=p1 -P model=rfr --env-manager uv --experiment-name mibici-equipo31
```

Baseline de media histórica:

```powershell
mlflow run . -e train -P question=p1 -P model=baseline --env-manager uv --experiment-name mibici-equipo31
```

Para probar una configuración personalizada:

```powershell
mlflow run . -e train -P question=p1 -P model=rfr -P n_estimators=100 -P min_samples_leaf=20 -P max_samples=0.3 --env-manager uv --experiment-name mibici-equipo31
```

### 5.5. Entrenar P2 — Clasificación de riesgo

Random Forest Classifier:

```powershell
mlflow run . -e train -P question=p2 -P model=rf --env-manager uv --experiment-name mibici-equipo31
```

Baseline de persistencia:

```powershell
mlflow run . -e train -P question=p2 -P model=baseline --env-manager uv --experiment-name mibici-equipo31
```

Los parámetros configurables incluyen `p2_n_estimators`, `max_depth`, `p2_min_samples_leaf`, `p2_max_samples` y `max_features`.

### 5.6. Entrenar P3 — Clustering de estaciones

K-Means:

```powershell
mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=3 --env-manager uv --experiment-name mibici-equipo31
```

Baseline de terciles de demanda:

```powershell
mlflow run . -e train -P question=p3 -P model=baseline --env-manager uv --experiment-name mibici-equipo31
```

Para evaluar otra cantidad de grupos, puede modificarse el parámetro `n_clusters`.

### 5.7. Consultar los experimentos

Acceder a http://127.0.0.1:5000 y seleccionar el experimento `mibici-equipo31`.

Desde la interfaz de MLflow es posible consultar las corridas, métricas, parámetros y artefactos registrados.

Los modelos y baselines se ejecutan de manera independiente, permitiendo comparaciones controladas entre configuraciones.

## 6. Validación de reproducibilidad

Se realizó una prueba desde un clon limpio del repositorio en Windows, utilizando el gestor de entornos `uv`.

Las verificaciones exitosas incluyeron:

- Clonación del repositorio desde GitHub.
- Creación automática de un entorno aislado mediante uv.
- Instalación de las dependencias declaradas.
- Conexión al servidor de MLflow Tracking.
- Descarga y procesamiento de los datos originales.
- Generación de los 12 archivos Parquet y del manifiesto de datos.
- Ejecución de P1, P2 y P3 mediante `mlflow run`.

Los resultados reprodujeron las métricas de referencia de la primera entrega.

**Limitaciones:** la prueba se realizó utilizando un servidor de Tracking local previamente configurado. La reproducción independiente del Model Registry y la gestión del ciclo de vida de modelos se abordarán en la siguiente etapa.

Los archivos de datos, bases de datos de MLflow, directorios de artefactos locales y modelos serializados no deben incorporarse al repositorio.

## 7. Experimentación y comparación de modelos (Parte B)

Todas las corridas de esta sección se lanzan con `mlflow run` desde el MLflow Project, con el servidor de Tracking de la sección 5.1 activo y `MLFLOW_TRACKING_URI` apuntando a él.

### 7.1 Cómo ejecutarlas

Primero los baselines (una corrida por pregunta), igual que en la sección 5:

```powershell
mlflow run . -e train -P question=p1 -P model=baseline --env-manager uv --experiment-name mibici-equipo31
mlflow run . -e train -P question=p2 -P model=baseline --env-manager uv --experiment-name mibici-equipo31
mlflow run . -e train -P question=p3 -P model=baseline --env-manager uv --experiment-name mibici-equipo31
```

Después las búsquedas de hiperparámetros y los candidatos de cada pregunta (el punto de entrada `experiments` está definido en `MLproject` y ejecuta `mibici/experimentos/`):

```powershell
mlflow run . -e experiments -P question=p1 --env-manager uv --experiment-name mibici-equipo31
mlflow run . -e experiments -P question=p2 --env-manager uv --experiment-name mibici-equipo31
mlflow run . -e experiments -P question=p3 --env-manager uv --experiment-name mibici-equipo31
```

Los valores por defecto (espacios de búsqueda, semillas, fechas y particiones) están en `mibici/config.py`; no se repiten en el código. P1 y P2 tardan varios minutos porque construyen la serie horaria completa (más de 3 millones de filas por pregunta).

### 7.2 Qué se registra

Cada comando crea una corrida principal (`P1-experimentos`, `P2-experimentos`, `P3-experimentos`) y debajo, como *nested runs*:

| Corrida | Etiqueta `stage_of_work` | Contenido |
|---|---|---|
| `P?-busqueda-...` (padre) | `search` | Espacio de búsqueda, criterio de selección y mejor combinación |
| `P?-...-comb-NN` / `P3-kmeans-kN` (hijas) | `search` | Una por combinación; solo métricas de validación |
| `P?-...` (candidatos) | `candidate` | Modelo, firma, ejemplo de entrada, métricas de validación y prueba, gráficas, datos y configuración |

Las etiquetas para filtrar en la interfaz son `question` (`p1`, `p2`, `p3`), `model_family`, `stage_of_work` (`baseline`, `search`, `candidate`), `equipo` y `git_commit`. El parámetro `data_version` conserva la huella de los datos.

Cada candidato guarda sus artefactos en subcarpetas:

```text
plots/    gráficas de desempeño (P1: perfil horario, real vs. predicho, residuales;
          P2: matriz de confusión y curva precisión-recall;
          P3: PCA, perfil de los grupos y mapa de estaciones)
data/     manifiesto con hashes SHA-256 de los Parquet (references/manifest_datos.json)
config/   config.py y los parámetros con que se ejecutó la corrida
model/    modelo con firma (signature) y ejemplo de entrada
```

Además, el conjunto de entrenamiento se asocia a la corrida con `mlflow.log_input` (esquema, resumen y huella; no se copian las filas). No se registran identificadores de usuario: los Parquet de trabajo solo conservan viaje, fechas y estaciones.

La partición temporal (entrenamiento hasta mayo de 2026, validación en junio–julio y prueba en agosto–septiembre) y las métricas son las mismas de los baselines, de modo que la comparación es válida.

### 7.3 Candidatos por pregunta

| Pregunta | Búsqueda (nested runs) | Candidatos |
|---|---|---|
| P1 | `HistGradientBoostingRegressor`: función de pérdida, tasa de aprendizaje y hojas máximas (8 combinaciones, criterio `val_mae`) | Random Forest con características base y ampliadas; Gradient Boosting con la mejor combinación, con características base y ampliadas |
| P2 | `HistGradientBoostingClassifier`: tasa de aprendizaje, hojas máximas y regularización L2 (8 combinaciones, criterio `val_pr_auc`) | Random Forest de la primera entrega; regresión logística; Gradient Boosting con umbral 0.5; el mismo con el umbral que maximiza F1 en validación |
| P3 | K-Means con k de 2 a 8 (criterio: silhouette con k ≥ 3) | K-Means con el mejor k; mezcla gaussiana; K-Means con características ampliadas |

Las características ampliadas de P1 agregan el promedio histórico por hora, la tendencia reciente, la hora como seno y coseno y la proporción de horas sin salidas. Las de P3 agregan la entropía del perfil horario, la variabilidad de las salidas diarias y la proporción de uso en horas pico.

### 7.4 Comparar y elegir

```powershell
python scripts/comparar_experimentos.py            # genera reports/comparacion_experimentos.md
python scripts/comparar_experimentos.py --marcar   # además etiqueta al elegido: selected_for_production=true
```

El script lee las corridas del servidor de Tracking y arma, por pregunta, la tabla de baseline contra candidatos con la mejora respecto al baseline y el costo de entrenamiento. Los criterios de selección usan solo métricas de validación y están documentados en el propio script. En la interfaz de MLflow se puede filtrar con `tags.stage_of_work = "candidate"` y seleccionar corridas para **Compare**.

**Limitaciones conocidas**

- En P2 el modelo registrado con umbral de validación conserva el umbral como parámetro (`umbral_prediccion`); `predict()` aplica 0.5, por lo que `predict_proba` debe combinarse con ese umbral al desplegarlo.
- Los modelos de P3 se guardan como `Pipeline` (log1p, estandarización y algoritmo). Para cargarlos solo se necesita scikit-learn.
- Las corridas del baseline y de los candidatos deben ejecutarse con el mismo commit y los mismos datos (`data_version`) para que la comparación sea válida.
