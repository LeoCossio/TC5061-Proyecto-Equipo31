"""Ejecuta los experimentos del equipo 31 desde un único comando.

Uso (desde la raíz del repositorio):
    python -m mibici.run_all

ANTES DE USARLO:
1. Guarda este archivo en mibici/run_all.py (no en la raíz).
2. Instala las dependencias del proyecto.
3. Ejecuta la ingesta: python -m mibici.dataset
4. Comprueba que existan los 12 Parquet en data/processed/.

POR COMPLETAR CON EL EQUIPO:
- Confirmar el nombre real del script de P1 y P2.
- Cambiar enabled=True cuando cada script funcione independientemente.
- Verificar que todos registren runs en el mismo experimento y la misma BD:
  experimento 'mibici-equipo31', backend sqlite:///mlflow.db de PROJ_ROOT.
- Documentar este comando y el comando de MLflow UI en README.md.

Este archivo solo coordina los scripts: no duplica entrenamiento ni features.
"""

import subprocess
import sys
from pathlib import Path

# Raíz del repositorio: carpeta superior a mibici/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# TODO EQUIPO: sustituir módulos si tus compañeros usan otros nombres.
# Habilitar P1 y P2 SOLO después de probarlos por separado.
EXPERIMENTS = [
    ("P1 - Regresión", "mibici.train_p1", False),  # TODO: activar
    ("P2 - Clasificación", "mibici.train_p2", False),  # TODO: activar
    ("P3 - Clustering", "mibici.train_p3", True),
]


def main() -> None:
    """Verifica los datos y ejecuta cada experimento habilitado."""
    parquet_files = sorted(PROCESSED_DIR.glob("*.parquet"))
    if len(parquet_files) != 12:
        raise SystemExit(
            f"Se esperaban 12 archivos Parquet en {PROCESSED_DIR}; "
            f"se encontraron {len(parquet_files)}. "
            "Ejecuta primero: python -m mibici.dataset"
        )

    print(f"Datos listos: {len(parquet_files)} Parquet")

    for title, module, enabled in EXPERIMENTS:
        if not enabled:
            print(f"[PENDIENTE] {title} ({module})")
            continue

        print(f"\n[INICIANDO] {title}", flush=True)
        # Ejecutar como módulo asegura imports consistentes.
        # sys.executable usa el mismo entorno de Python que run_all.
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_ROOT,
            check=True,
        )
        print(f"[COMPLETADO] {title}", flush=True)

    print("\nListo. Para revisar los runs, desde la raíz ejecuta:")
    print('mlflow ui --backend-store-uri "sqlite:///mlflow.db"')
    print("Abre http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
