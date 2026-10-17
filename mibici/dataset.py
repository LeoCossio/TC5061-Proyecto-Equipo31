from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd
import requests
import typer

from mibici.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT; WindowsPowerShell/5.1)",
    "Accept": "*/*",
}

PORTAL_URL = "https://www.mibici.net/es/datos-abiertos/"

MESES = [
    (2025, 10), (2025, 11), (2025, 12),
    (2026, 1), (2026, 2), (2026, 3),
    (2026, 4), (2026, 5), (2026, 6),
    (2026, 7), (2026, 8), (2026, 9),
]


def discover_files() -> dict[str, str]:
    """Encuentra los CSV oficiales correspondientes al periodo seleccionado."""
    logger.info("Buscando archivos en el portal de MiBici...")

    response = requests.get(
        PORTAL_URL,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    expected = {
        f"datos_abiertos_{year}_{month:02d}.csv"
        for year, month in MESES
    }

    discovered = {}

    for link in soup.find_all("a", href=True):
        url = urljoin(PORTAL_URL, link["href"])
        filename = url.split("/")[-1]

        if filename in expected:
            discovered[filename] = url

    missing = expected - discovered.keys()

    if missing:
        raise RuntimeError(
            f"No se encontraron los siguientes archivos: {sorted(missing)}"
        )

    logger.success(f"Se encontraron {len(discovered)} archivos mensuales.")

    return discovered

def download_file(url: str, destination: Path) -> Path:
    """Descarga un archivo si no existe localmente."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        logger.info(f"Archivo existente: {destination.name}")
        return destination

    logger.info(f"Descargando {destination.name}...")

    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        with requests.get(
            url,
            headers=HEADERS,
            stream=True,
            timeout=(15, 180),
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise ValueError("El servidor devolvió HTML en lugar de CSV")

            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        temporary.replace(destination)
        logger.success(f"Descargado: {destination.name}")
        return destination

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

COLUMN_MAP = {
    "Viaje_Id": "viaje_id",
    "Usuario_Id": "usuario_id",
    "Genero": "genero",
    "Año_de_nacimiento": "anio_nacimiento",
    "Inicio_del_viaje": "inicio",
    "Fin_del_viaje": "fin",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
}

REQUIRED_COLUMNS = [
    "viaje_id",
    "inicio",
    "fin",
    "origen_id",
    "destino_id",
]


def clean_dataset(input_path: Path, output_path: Path) -> dict:
    """Limpia, normaliza y guarda un mes de viajes en Parquet."""

    logger.info(f"Procesando {input_path.name}")

    # 1. Leer el archivo respetando su codificación
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="cp1252")
        encoding = "cp1252"

    initial_rows = len(df)

    # 2. Normalizar nombres de columnas
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAP)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing)}"
        )

    # 3. Normalizar fechas
    for column in ["inicio", "fin"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    # 4. Normalizar identificadores numéricos
    for column in ["viaje_id", "origen_id", "destino_id"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 5. Eliminar duplicados exactos
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 6. Detectar y excluir registros inválidos
    invalid = (
        df[REQUIRED_COLUMNS].isna().any(axis=1)
        | (df["fin"] < df["inicio"])
    )

    invalid_rows = int(invalid.sum())
    df = df.loc[~invalid].copy()

    # 7. Aplicar tipos consistentes
    for column in ["viaje_id", "origen_id", "destino_id"]:
        df[column] = df[column].astype("int64")

    # No conservamos datos personales en los Parquet de trabajo
    df = df[REQUIRED_COLUMNS].copy()

    # 8. Guardar en formato Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

    report = {
        "archivo": input_path.name,
        "encoding": encoding,
        "registros_originales": initial_rows,
        "duplicados_eliminados": duplicates,
        "registros_invalidos": invalid_rows,
        "registros_limpios": len(df),
    }

    logger.success(
        f"{input_path.name}: "
        f"{initial_rows:,} originales -> "
        f"{len(df):,} limpios"
    )

    return report


@app.command()
@app.command()




def main():
    """Descarga los doce meses de viajes de MiBici."""
    files = discover_files()

    for filename, url in sorted(files.items()):
        destination = RAW_DATA_DIR / filename
        download_file(url, destination)
        nomenclature_url = (
        "https://www.mibici.net/site/assets/files/1118/"
        "nomenclatura_2026_09.csv"
    )

    download_file(
        nomenclature_url,
        EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv",
    )
    logger.success("Descarga de los 12 meses completada.")

    
    logger.info("Iniciando limpieza y normalización...")

    for year, month in MESES:
        filename = f"datos_abiertos_{year}_{month:02d}.csv"

        input_path = RAW_DATA_DIR / filename
        output_path = (
            PROCESSED_DATA_DIR
            / f"datos_abiertos_{year}_{month:02d}.parquet"
        )

        report = clean_dataset(input_path, output_path)
        logger.info(f"Reporte: {report}")

    logger.success("Limpieza de los 12 meses completada.")

if __name__ == "__main__":
    app()