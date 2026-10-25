from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mibici.feature_engineering.clustering import (
    crear_features_p3,
    preprocesar_features_p3,
    baseline_p3,
    entrenar_kmeans_p3,
)
from mibici.feature_engineering.regresion import (
    COLUMNAS_P1,
    COLUMNAS_P1_AMPLIADAS,
    OBJETIVO_P1,
    construir_panel_horario,
    crear_features_p1,
    dividir_temporal_p1,
    baseline_p1,
    entrenar_random_forest_p1,
    es_hora_pico,
)

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
