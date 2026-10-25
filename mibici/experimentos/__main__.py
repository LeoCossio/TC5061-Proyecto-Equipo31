"""Punto de entrada: python -m mibici.experimentos --question p1|p2|p3|all"""

import argparse

from mibici.experimentos import comun


def main():
    parser = argparse.ArgumentParser(
        description="Búsquedas de hiperparámetros y modelos candidatos de MiBici"
    )
    parser.add_argument("--question", choices=["p1", "p2", "p3", "all"], default="all")
    args = parser.parse_args()

    comun.conectar_mlflow()

    if args.question in ("p1", "all"):
        from mibici.experimentos import p1
        p1.ejecutar()
    if args.question in ("p2", "all"):
        from mibici.experimentos import p2
        p2.ejecutar()
    if args.question in ("p3", "all"):
        from mibici.experimentos import p3
        p3.ejecutar()


if __name__ == "__main__":
    main()
