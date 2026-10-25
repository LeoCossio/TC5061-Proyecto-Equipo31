
import argparse

from mibici import train_p1, train_p2, train_p3
from mibici.config import (
    P1_RF_N_ESTIMADORES,
    P1_RF_MIN_SAMPLES_LEAF,
    P1_RF_MAX_SAMPLES,
    P2_RF_N_ESTIMATORS,
    P2_RF_MAX_DEPTH,
    P2_RF_MIN_SAMPLES_LEAF,
    P2_RF_MAX_SAMPLES,
    P2_RF_MAX_FEATURES,
    P3_N_CLUSTERS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos MiBici"
    )

    parser.add_argument(
        "--question",
        choices=["p1", "p2", "p3"],
        required=True,
    )

    parser.add_argument(
        "--n-estimators",
        "--n_estimators",
        type=int,
        default=P1_RF_N_ESTIMADORES,
    )

    parser.add_argument(
        "--min-samples-leaf",
        "--min_samples_leaf",
        type=int,
        default=P1_RF_MIN_SAMPLES_LEAF,
    )

    parser.add_argument(
        "--max-samples",
        "--max_samples",
        type=float,
        default=P1_RF_MAX_SAMPLES,
    )

    parser.add_argument(
        "--p2-n-estimators",
        "--p2_n_estimators",
        type=int,
        default=P2_RF_N_ESTIMATORS,
    )

    parser.add_argument(
        "--max-depth",
        "--max_depth",
        type=int,
        default=P2_RF_MAX_DEPTH,
    )

    parser.add_argument(
        "--p2-min-samples-leaf",
        "--p2_min_samples_leaf",
        type=int,
        default=P2_RF_MIN_SAMPLES_LEAF,
    )

    parser.add_argument(
        "--p2-max-samples",
        "--p2_max_samples",
        type=float,
        default=P2_RF_MAX_SAMPLES,
    )

    parser.add_argument(
        "--max-features",
        "--max_features",
        type=float,
        default=P2_RF_MAX_FEATURES,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--n-clusters",
        "--n_clusters",
        type=int,
        default=P3_N_CLUSTERS,
    )
    args = parser.parse_args()


    if args.question == "p1":
        if args.model not in ["baseline", "rfr"]:
            parser.error("P1 admite baseline y rfr")

        if args.model == "rfr":
            if args.n_estimators < 1:
                parser.error("--n-estimators debe ser positivo")
            if args.min_samples_leaf < 1:
                parser.error("--min-samples-leaf debe ser positivo")
            if not 0 < args.max_samples <= 1:
                parser.error("--max-samples debe estar entre 0 y 1")

        train_p1.main(
            model=args.model,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_samples=args.max_samples,
        )
    elif args.question == "p2":
        if args.model not in ["baseline", "rf"]:
            parser.error("P2 admite baseline y rf")

        if args.p2_n_estimators < 1:
            parser.error("--p2-n-estimators debe ser positivo")

        if args.max_depth < 1:
            parser.error("--max-depth debe ser positivo")

        if args.p2_min_samples_leaf < 1:
            parser.error("--p2-min-samples-leaf debe ser positivo")

        if not 0 < args.p2_max_samples <= 1:
            parser.error("--p2-max-samples debe estar entre 0 y 1")

        if not 0 < args.max_features <= 1:
            parser.error("--max-features debe estar entre 0 y 1")

        train_p2.main(
            model=args.model,
            n_estimators=args.p2_n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.p2_min_samples_leaf,
            max_samples=args.p2_max_samples,
            max_features=args.max_features,
        )
        
    else:
        if args.model not in ["baseline", "kmeans"]:
            parser.error("P3 admite los modelos baseline y kmeans")

        if args.model == "kmeans" and args.n_clusters < 2:
            parser.error("--n-clusters debe ser al menos 2")

        train_p3.main(
            model=args.model,
            n_clusters=args.n_clusters,
        )


if __name__ == "__main__":
    main()
