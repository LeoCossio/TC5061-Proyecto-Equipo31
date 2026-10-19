# MiBici — Análisis de demanda y patrones de uso

**Proyecto integrador de MLOps | Equipo 31**

- Leonardo Cossío Dinorín | A01753898
- Santiago Espinosa Domínguez | A01747478
- Rafael Posadas Ruíz | A01747128

Este proyecto utiliza los datos abiertos del sistema de bicicletas públicas **MiBici** para estudiar la demanda de viajes, explorar posibles desequilibrios operativos y segmentar estaciones de acuerdo con sus patrones históricos de uso. El objetivo académico es construir un flujo **reproducible y trazable** de ingesta, preparación de datos, entrenamiento, evaluación y registro de experimentos con MLflow.

> **Estado:** proyecto en desarrollo. P3 (clustering) y P1 (regresión) ya funcionan; P2 y algunos elementos de integración se encuentran pendientes de validación por el equipo.

## 1. Preguntas de análisis

| Pregunta | Tipo de problema | Unidad de análisis | Objetivo | Modelo de referencia y candidato |
|---|---|---|---|---|
| **P1 — Demanda** | Regresión | Estación-hora | Estimar el número de viajes que iniciarán en la siguiente hora. | Media histórica / RandomForestRegressor (implementados). |
| **P2 — Riesgo de desbalance** | Clasificación | Estación-hora | Estimar un **indicador proxy** de riesgo operativo en la siguiente hora a partir de flujos de viajes. | Frecuencia histórica / RandomForestClassifier (propuestos). |
| **P3 — Perfiles de estaciones** | Clustering | Estación | Agrupar estaciones según demanda, balance y distribución temporal de salidas. Sin horizonte predictivo. | Terciles de demanda / K-Means (implementados). |

**Importante:** los CSV de viajes no contienen inventarios históricos de bicicletas o anclajes libres. Por tanto, P2 **no predice directamente si una estación está vacía o llena**; su etiqueta es un indicador indirecto que debe definirse y documentarse con precisión.

## 2. Fuentes y datos

- **Fuente:** [MiBici — Datos abiertos](https://www.mibici.net/es/datos-abiertos/).
- **Periodo:** octubre de 2025 a septiembre de 2026 (**12 meses**).
- **Viajes originales reportados:** 4,497,877.
- **Viajes procesados reportados:** 4,497,119, después de retirar 758 registros duplicados.
- **Estaciones identificadas en viajes:** 471; **estaciones del catálogo de septiembre de 2026:** 484.
- **Estaciones utilizadas en P3:** 470, tras excluir estaciones con menos de 100 salidas en el periodo.
- **Campos operativos procesados:** `viaje_id`, `inicio`, `fin`, `origen_id`, `destino_id`.
- **Catálogo de estaciones:** nomenclatura de septiembre de 2026, obtenida por el proceso de ingesta.

Los CSV mensuales se descargan y procesan desde `mibici/dataset.py`. Los datos generados se conservan localmente en `data/`, sin subirse al repositorio. No se utilizan identificadores de usuario, género ni año de nacimiento como variables de modelado.

**Versionado de datos:** el script de P3 registra en MLflow una huella SHA-256 calculada sobre los Parquet utilizados. **Pendiente:** crear y versionar en Git un manifiesto por archivo con conteos y hashes, y comprobar su integridad. La huella registrada en un run no sustituye ese manifiesto.

Para más detalles, consultar [`references/definicion_datos.md`](references/definicion_datos.md) y [`references/limpieza_datos.md`](references/limpieza_datos.md).

## 3. Organización del repositorio

```text
.
├── data/                         # Datos locales: raw, external, interim, processed (ignorados por Git)
├── mibici/
│   ├── config.py                 # Rutas y configuración compartida; fechas y semilla de P3
│   ├── dataset.py                # Descarga, limpieza y creación de Parquet
│   ├── features.py               # Acceso a funciones de ingeniería de características
│   ├── feature_engineering/
│   │   ├── clustering.py         # Características, baseline y K-Means de P3
│   │   └── regresion.py          # Panel estación-hora, características, baseline y RandomForest de P1
│   ├── train_p1.py               # Entrenamiento y registro de P1 en MLflow
│   ├── train_p3.py               # Entrenamiento y registro de P3 en MLflow
│   └── run_all.py                # Coordinador provisional de P1, P2 y P3
├── notebooks/                    # Exploración y análisis
├── models/                       # Modelos generados localmente (no versionados)
├── references/                   # Definición de datos, modelos, pipelines y arquitectura
├── reports/
│   └── figures/                 # Gráficas para el reporte
├── tests/                        # Pruebas automatizadas
├── requirements.txt              # Dependencias de Python
├── pyproject.toml                # Configuración del proyecto
└── README.md
```

La estructura parte de **Cookiecutter Data Science (CCDS)** con `include_code_scaffold=Yes`. Los scripts de P1 y P2 se incorporarán conforme estén disponibles.

## 4. Instalación

Requisitos: **Python 3.13** utilizado durante el desarrollo, Git y acceso a Internet para descargar los archivos públicos. La compatibilidad en otras versiones de Python deberá verificarse.

Desde la raíz del repositorio:

```powershell
# Clonar (si aún no lo tienes)
git clone https://github.com/LeoCossio/TC5061-Proyecto-Equipo31.git
cd TC5061-Proyecto-Equipo31

# Crear y activar entorno virtual (Windows / PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
python -m pip install -r requirements.txt
```

> **Pendiente:** verificar que `requirements.txt` incluya todas las dependencias de P1, P2 y P3 y que sus versiones estén fijadas. La creación del entorno no garantiza reproducibilidad completa mientras esto siga pendiente.

## 5. Ejecución

Todos los comandos siguientes se ejecutan desde la **raíz del repositorio**.

### 5.1 Descargar y preparar datos

```powershell
python -m mibici.dataset
```

Este comando corresponde a la interfaz Typer utilizada y probada durante el desarrollo. Al terminar deben existir **12 archivos Parquet** en `data/processed/`. La nomenclatura de estaciones también se descarga durante la ingesta.

### 5.2 Ejecutar P3 (baseline y K-Means)

```powershell
python -m mibici.train_p3
```

El script registra dos runs en el experimento MLflow `mibici-equipo31`:

- `P3-baseline-demanda`: agrupación en terciles por demanda histórica.
- `P3-kmeans-k3`: K-Means con tres grupos, seis características escaladas, `random_state=42` y `n_init=10`.

Se registran parámetros, métricas, etiquetas (`equipo`, `pregunta`, `git_commit`) y gráficas. El backend utilizado es una base de datos SQLite local, `mlflow.db`.

### 5.3 Ejecutar P1 (baseline y RandomForest)

```powershell
python -m mibici.train_p1
```

El script registra dos runs en el experimento MLflow `mibici-equipo31`:

- `P1-baseline-media-historica`: media histórica de la estación para el mismo día de la semana y hora, calculada solo con semanas anteriores.
- `P1-random-forest`: `RandomForestRegressor` con hora, día de la semana, rezagos de 1, 2, 3, 24 y 168 horas, promedios móviles y promedios históricos de la estación.

La partición es cronológica: entrenamiento hasta mayo de 2026, validación en junio–julio y prueba en agosto–septiembre. Se registran `val_` y `test_` de `mae`, `rmse` y `mae_horas_pico`, además de gráficas de perfil horario e importancia de variables.

### 5.4 Ejecutar los modelos habilitados

```powershell
python -m mibici.run_all
```

**Estado actual:** el coordinador comprueba los 12 Parquet y ejecuta **P1 y P3**. P2 permanece deshabilitado hasta integrar y probar su script. Este comando **no realiza la descarga automáticamente**; ejecutar primero la ingesta si faltan los datos.

### 5.5 Consultar experimentos en MLflow

```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Abrir <http://127.0.0.1:5000> y seleccionar el experimento `mibici-equipo31`. Para comparar modelos, seleccionar sus runs y utilizar **Compare**.

**Nota:** la base de datos y los artefactos locales de MLflow no se almacenan en Git. Para reproducir los runs en otra máquina, se deben ejecutar nuevamente los scripts con la misma versión de código, datos y dependencias. Para preservar el historial original, debe respaldarse por separado el almacenamiento de MLflow.

## 6. Resultados preliminares de P3

| Modelo | Estaciones | Grupos | Silhouette Score |
|---|---:|---:|---:|
| Terciles de demanda (baseline) | 470 | 3 | 0.0732 |
| K-Means | 470 | 3 | **0.1992** |

Los grupos de K-Means contienen **154, 198 y 118 estaciones**. La visualización con PCA de dos dimensiones conserva alrededor del **56.3 %** de la varianza de las características escaladas.

La mejora de Silhouette indica mayor separación relativa frente al baseline, **no precisión predictiva ni validación operativa**. Aún deben evaluarse la estabilidad temporal, la cobertura histórica de cada estación y la utilidad geográfica de la segmentación.

Las figuras se encuentran en `reports/figures/` y el desarrollo exploratorio en `notebooks/1.0-lc-exploracion-estaciones.ipynb`.

## 7. Flujo de trabajo con Git y GitHub

Se adopta **GitHub Flow**:

1. La rama **`main` está protegida**. Los cambios se desarrollan en ramas independientes y se integran mediante **Pull Requests**.
2. Crear una rama desde una base actualizada. Convención: `feature/p1-regression`, `feature/p2-clasificacion`, `feature/p3-clustering` o `docs/descripcion`.
3. Realizar cambios y registrar commits descriptivos siguiendo **Conventional Commits**, por ejemplo:
   - `feat(p3): implement station clustering with KMeans and MLflow`
   - `fix(data): correct duplicated command decorator`
   - `docs: complete reproducibility instructions`
4. Publicar la rama (`git push -u origin nombre-de-rama`) y abrir un Pull Request dirigido a `main`.
5. Solicitar revisión a otro integrante; atender comentarios y **fusionar solo cuando se cumplan las reglas de protección**.
6. Cada integrante debe **abrir al menos un PR y revisar al menos un PR ajeno**. Conservar los enlaces de evidencia para el reporte final.

Enlaces: [Repositorio](https://github.com/LeoCossio/TC5061-Proyecto-Equipo31) · [Pull Requests](https://github.com/LeoCossio/TC5061-Proyecto-Equipo31/pulls).

## 8. Integrantes y responsabilidades

| Integrante | Responsabilidad principal | Estado |
|---|---|---|
| **[Nombre del integrante P1]** | Regresión: predicción de demanda por estación-hora | En integración |
| **[Nombre del integrante P2]** | Clasificación: riesgo proxy de desbalance e ingesta compartida (confirmar roles) | En integración |
| **[Nombre del integrante P3]** | Clustering: análisis exploratorio, ingeniería de características, baseline, K-Means y MLflow | Implementado; PR sujeto a revisión |

> **Pendiente del equipo:** sustituir los nombres entre corchetes, confirmar los roles y agregar los enlaces a los PR revisados. Si una persona realizó más de una tarea, reflejarlo en la tabla.

## 9. Documentación y pendientes

Se incluyen los siguientes documentos de diseño en `references/`:

- [`definicion_datos.md`](references/definicion_datos.md): fuentes, tamaño y variables.
- [`pipelines_ml.md`](references/pipelines_ml.md): etapas de preparación, modelado y evaluación.
- [`modelos_metricas.md`](references/modelos_metricas.md): baselines, candidatos y métricas.
- [`arquitectura_sistema.md`](references/arquitectura_sistema.md): componentes, decisiones y diagrama de arquitectura.

**Antes de entregar:**

- [ ] Integrar y ejecutar baselines de P1 y P2 desde scripts del repositorio.
- [ ] Fijar y verificar todas las versiones de `requirements.txt`.
- [ ] Crear el manifiesto Git de datos (conteos y hashes SHA-256 por archivo) y validar integridad.
- [ ] Actualizar la documentación de diseño para reflejar la implementación final, sin dejar estados desactualizados.
- [ ] Confirmar integrantes, roles y evidencias de PR/revisiones.
- [ ] Probar la instalación, ingesta y ejecución en un clon limpio.
- [ ] Reejecutar los experimentos tras los commits definitivos para registrar etiquetas `git_commit` correctas.
- [ ] Preparar las capturas de MLflow: lista de runs, detalle, artefacto y comparación.

## 10. Referencias

- [MiBici — Datos abiertos](https://www.mibici.net/es/datos-abiertos/)
- [MLflow — Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Scikit-learn — Evaluación de modelos](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/)
