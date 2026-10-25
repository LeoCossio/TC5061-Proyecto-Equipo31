from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


# Configuración del experimento P3
P3_FECHA_INICIO = "2025-10-01"
P3_FECHA_FIN = "2026-09-30"
P3_SEMILLA = 42
P3_MIN_SALIDAS = 100

P3_N_CLUSTERS = 3

MLFLOW_EXPERIMENT = "mibici-equipo31"

# Configuración del experimento P1 (regresión de demanda por estación-hora)
P1_FECHA_INICIO = "2025-10-01"
P1_FECHA_FIN = "2026-09-30"
P1_FECHA_INICIO_VALIDACION = "2026-06-01"
P1_FECHA_INICIO_PRUEBA = "2026-08-01"
P1_SEMILLA = 42
P1_RF_N_ESTIMADORES = 100
P1_RF_MIN_SAMPLES_LEAF = 20
P1_RF_MAX_SAMPLES = 0.3

# Configuración de P2 - Clasificación
P2_RF_N_ESTIMATORS = 150
P2_RF_MAX_DEPTH = 30
P2_RF_MIN_SAMPLES_LEAF = 10
P2_RF_MAX_SAMPLES = 0.3
P2_RF_MAX_FEATURES = 0.8

# ---------------------------------------------------------
# Experimentación (entrega 4): búsquedas y candidatos
# ---------------------------------------------------------
EQUIPO_ID = "31"

# P1: búsqueda de HistGradientBoostingRegressor (criterio: val_mae).
P1_HGB_ESPACIO = {
    "loss": ["poisson", "absolute_error"],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [31, 127],
}
P1_HGB_MAX_ITER = 200
P1_HGB_MIN_SAMPLES_LEAF = 100

# P2: búsqueda de HistGradientBoostingClassifier (criterio: val_pr_auc).
P2_HGB_ESPACIO = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [31, 63],
    "l2_regularization": [0.0, 1.0],
}
P2_HGB_MAX_ITER = 200
P2_HGB_MIN_SAMPLES_LEAF = 100
P2_LR_C = 1.0

# P3: búsqueda del número de grupos (criterio: silhouette, con k >= 3).
P3_K_BUSQUEDA = [2, 3, 4, 5, 6, 7, 8]
P3_K_MINIMO_SELECCION = 3
