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
