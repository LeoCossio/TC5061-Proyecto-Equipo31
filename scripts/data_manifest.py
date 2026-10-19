"""Genera o verifica el manifiesto versionado de los 12 Parquet de MiBici.

Uso desde la raíz del repositorio:
  python scripts/data_manifest.py generate
  python scripts/data_manifest.py verify
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MANIFEST = ROOT / "references" / "manifest_datos.json"
MONTHS = [(2025, m) for m in range(10, 13)] + [(2026, m) for m in range(1, 10)]
EXPECTED = [f"datos_abiertos_{year}_{month:02d}.parquet" for year, month in MONTHS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_files() -> list[dict]:
    actual = {p.name for p in DATA_DIR.glob("*.parquet")}
    missing = sorted(set(EXPECTED) - actual)
    extra = sorted(actual - set(EXPECTED))
    if missing or extra:
        raise ValueError(f"Se requieren exactamente 12 Parquet. Faltantes: {missing}; adicionales: {extra}")

    entries = []
    for name in EXPECTED:
        path = DATA_DIR / name
        item = {
            "archivo": f"data/processed/{name}",
            "filas": pq.ParquetFile(path).metadata.num_rows,
            "tamano_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        entries.append(item)
        print(f"  {name}: {item['filas']:,} filas | SHA-256 {item['sha256'][:12]}...")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "verify"])
    args = parser.parse_args()

    try:
        print("Inspeccionando archivos Parquet...")
        entries = inspect_files()
        result = {
            "schema_version": 1,
            "fuente": "MiBici - datos abiertos",
            "periodo": "2025-10/2026-09",
            "total_archivos": len(entries),
            "total_filas": sum(e["filas"] for e in entries),
            "archivos": entries,
        }
        if args.action == "generate":
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"\nManifiesto generado: {MANIFEST.relative_to(ROOT)}")
            print(f"Total de filas: {result['total_filas']:,}")
        else:
            if not MANIFEST.is_file():
                raise FileNotFoundError(f"No existe {MANIFEST.relative_to(ROOT)}. Ejecuta primero 'generate'.")
            saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if saved != result:
                for saved_entry, actual_entry in zip(saved.get("archivos", []), entries):
                    if saved_entry != actual_entry:
                        print(f"Diferencia detectada: {actual_entry['archivo']}", file=sys.stderr)
                raise ValueError("El contenido local no coincide con el manifiesto versionado.")
            print("\nVERIFICACIÓN CORRECTA: 12 archivos, conteos, tamaños y hashes coinciden.")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
