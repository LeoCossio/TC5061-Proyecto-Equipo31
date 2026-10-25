"""Experimentos de P3 (perfiles de estaciones, clustering).

Cada ejecución registra, dentro de la corrida activa:
  1. Una búsqueda del número de grupos de K-Means como nested runs: una
     corrida padre y una hija por valor de k.
  2. Tres candidatos: K-Means con el mejor k, Mezcla Gaussiana (GMM) con el
     mismo número de grupos y K-Means con características ampliadas.

Los candidatos se guardan como Pipeline (transformación + escalado + modelo):
reciben directamente las características por estación sin escalar.

Silhouette se calcula siempre sobre el mismo espacio escalado del baseline
(`silhouette_espacio_base`) para que la comparación sea justa; los candidatos
con características ampliadas reportan además su silhouette en su propio espacio.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mibici.config import (
    EXTERNAL_DATA_DIR,
    P3_FECHA_FIN,
    P3_FECHA_INICIO,
    P3_K_BUSQUEDA,
    P3_K_MINIMO_SELECCION,
    P3_MIN_SALIDAS,
    P3_SEMILLA,
    PROCESSED_DATA_DIR,
)
from mibici.experimentos.comun import (
    archivos_y_version,
    commit_actual,
    etiquetas,
    guardar_figura,
    medir,
    registrar_candidato,
    resumen_json,
)
from mibici.feature_engineering.clustering import (
    COLUMNAS_P3_AMPLIADAS_EXTRA,
    baseline_p3,
    crear_features_p3_ampliadas,
    preprocesar_features_p3,
)

PREGUNTA = "p3"
COLUMNAS_BASE = [
    "viajes_diarios", "balance_relativo", "proporcion_fin_semana",
    "manana", "mediodia", "tarde",
]


@dataclass
class DatosP3:
    features: pd.DataFrame
    ampliadas: pd.DataFrame
    escaladas: pd.DataFrame
    parametros: dict
    silhouette_baseline: float


def preparar_datos(version):
    """Construye las características por estación y el espacio escalado base."""
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    viajes = pd.concat([pd.read_parquet(a) for a in archivos], ignore_index=True)
    ampliadas = crear_features_p3_ampliadas(viajes)
    features = ampliadas.drop(columns=COLUMNAS_P3_AMPLIADAS_EXTRA)
    del viajes
    escaladas, _ = preprocesar_features_p3(features)

    parametros = {
        "fecha_inicio": P3_FECHA_INICIO,
        "fecha_fin": P3_FECHA_FIN,
        "data_version": version,
        "min_salidas": P3_MIN_SALIDAS,
        "semilla": P3_SEMILLA,
        "scaler": "StandardScaler",
        "transformacion": "log1p",
        "n_estaciones": len(features),
    }
    silhouette_baseline = float(silhouette_score(escaladas, baseline_p3(features)))
    return DatosP3(features, ampliadas, escaladas, parametros, silhouette_baseline)


def crear_pipeline(estimador, ampliada=False):
    """Pipeline: log1p a la demanda, escalado estándar y el algoritmo de grupos.

    Equivale a `preprocesar_features_p3`, pero queda guardado junto al modelo
    para poder agrupar estaciones nuevas a partir de sus características crudas.
    """
    resto = COLUMNAS_BASE[1:] + (COLUMNAS_P3_AMPLIADAS_EXTRA if ampliada else [])
    transformador = ColumnTransformer([
        ("demanda", FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
         ["viajes_diarios"]),
        ("resto", "passthrough", resto),
    ])
    return Pipeline([
        ("prep", transformador),
        ("escala", StandardScaler()),
        ("modelo", estimador),
    ])


def metricas_grupos(matriz, etiquetas_grupo, prefijo="train"):
    """Silhouette, Davies-Bouldin, Calinski-Harabasz y equilibrio de los grupos."""
    tamanos = pd.Series(etiquetas_grupo).value_counts()
    return {
        f"{prefijo}_silhouette_score": float(silhouette_score(matriz, etiquetas_grupo)),
        "davies_bouldin": float(davies_bouldin_score(matriz, etiquetas_grupo)),
        "calinski_harabasz": float(calinski_harabasz_score(matriz, etiquetas_grupo)),
        "grupo_mas_pequeno_pct": float(100 * tamanos.min() / tamanos.sum()),
        "grupo_mas_grande_pct": float(100 * tamanos.max() / tamanos.sum()),
        "n_estaciones": int(tamanos.sum()),
    }


def buscar_kmeans(datos, commit):
    """Evalúa varios k con nested runs y elige el de mayor silhouette (k >= 3).

    Con k = 2 casi cualquier conjunto se separa en dos bloques y el silhouette
    sube sin aportar interpretación; el baseline usa tres niveles, así que se
    exige al menos tres grupos.
    """
    resultados = []

    with mlflow.start_run(run_name="P3-busqueda-kmeans", nested=True):
        mlflow.set_tags(etiquetas(PREGUNTA, "kmeans", "search", commit))
        mlflow.log_params({
            **datos.parametros,
            "algoritmo": "KMeans",
            "espacio_busqueda": resumen_json({"n_clusters": P3_K_BUSQUEDA}),
            "n_combinaciones": len(P3_K_BUSQUEDA),
            "criterio_seleccion": f"silhouette con k >= {P3_K_MINIMO_SELECCION}",
        })

        for k in P3_K_BUSQUEDA:
            modelo = KMeans(n_clusters=k, random_state=P3_SEMILLA, n_init=10)
            grupos, t = medir(modelo.fit_predict, datos.escaladas)
            metricas = metricas_grupos(datos.escaladas, grupos)
            with mlflow.start_run(run_name=f"P3-kmeans-k{k}", nested=True):
                mlflow.log_params({"n_grupos": k, "n_init": 10})
                mlflow.log_metrics({**metricas, "inertia": float(modelo.inertia_),
                                    "tiempo_entrenamiento_s": t})
                mlflow.set_tags(etiquetas(PREGUNTA, "kmeans", "search", commit))
            resultados.append({"k": k, "inertia": float(modelo.inertia_), **metricas})
            print(f"  [busqueda] k={k} silhouette={metricas['train_silhouette_score']:.4f} "
                  f"min_grupo={metricas['grupo_mas_pequeno_pct']:.1f}%", flush=True)

        tabla = pd.DataFrame(resultados)
        elegibles = tabla[tabla["k"] >= P3_K_MINIMO_SELECCION]
        mejor_k = int(elegibles.loc[elegibles["train_silhouette_score"].idxmax(), "k"])
        mlflow.log_metric("mejor_silhouette",
                          float(elegibles["train_silhouette_score"].max()))
        mlflow.log_param("mejor_n_clusters", mejor_k)
        mlflow.log_dict({"mejor_n_clusters": mejor_k}, "config/mejores_hiperparametros.json")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(tabla["k"], tabla["train_silhouette_score"], marker="o")
        ax1.axvline(mejor_k, color="red", linestyle="--", label=f"k elegido = {mejor_k}")
        ax1.set(title="Silhouette por número de grupos", xlabel="k", ylabel="Silhouette")
        ax1.legend()
        ax2.plot(tabla["k"], tabla["inertia"], marker="o", color="#F58518")
        ax2.set(title="Inercia por número de grupos (codo)", xlabel="k", ylabel="Inercia")
        guardar_figura(fig, "silhouette_e_inercia_por_k.png")

    return mejor_k


def figuras_p3(titulo, features_escaladas, grupos, indice):
    """PCA, perfiles de los grupos y mapa de las estaciones."""
    grupos = pd.Series(np.asarray(grupos), index=indice, name="grupo")
    figuras = []

    coordenadas = PCA(n_components=2, random_state=P3_SEMILLA).fit_transform(features_escaladas)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    puntos = ax.scatter(coordenadas[:, 0], coordenadas[:, 1], c=grupos.to_numpy(),
                        cmap="viridis", s=24, alpha=0.8)
    ax.set(title=f"{titulo}: grupos en el plano PCA", xlabel="Componente 1",
           ylabel="Componente 2")
    fig.colorbar(puntos, ax=ax, label="Grupo")
    figuras.append((fig, "pca_grupos.png"))

    perfiles = features_escaladas.groupby(grupos.to_numpy()).mean()
    fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(perfiles)))
    imagen = ax.imshow(perfiles.to_numpy(), cmap="coolwarm", aspect="auto",
                       vmin=-2, vmax=2)
    ax.set_xticks(range(perfiles.shape[1]), perfiles.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(perfiles)), [f"Grupo {g}" for g in perfiles.index])
    for i in range(perfiles.shape[0]):
        for j in range(perfiles.shape[1]):
            ax.text(j, i, f"{perfiles.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{titulo}: perfil medio de cada grupo (valores estandarizados)")
    fig.colorbar(imagen, ax=ax)
    figuras.append((fig, "perfiles_grupos.png"))

    catalogo = pd.read_csv(EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv",
                           usecols=["id", "latitude", "longitude"]).set_index("id")
    mapa = catalogo.join(grupos, how="inner")
    fig, ax = plt.subplots(figsize=(7, 6))
    puntos = ax.scatter(mapa["longitude"], mapa["latitude"], c=mapa["grupo"],
                        cmap="viridis", s=22, alpha=0.85)
    ax.set(title=f"{titulo}: ubicación de las estaciones por grupo",
           xlabel="Longitud", ylabel="Latitud")
    ax.set_aspect(1 / np.cos(np.deg2rad(mapa["latitude"].mean())))
    fig.colorbar(puntos, ax=ax, label="Grupo")
    figuras.append((fig, "mapa_estaciones.png"))

    return figuras


def registrar(nombre, familia, parametros, entrada, escaladas_base, pipeline, datos,
              commit, ampliada=False):
    """Ajusta el pipeline, evalúa los grupos y registra el candidato."""
    _, t = medir(pipeline.fit, entrada)
    grupos = pipeline.predict(entrada)
    espacio_propio = pipeline[:-1].transform(entrada)

    metricas = metricas_grupos(espacio_propio, grupos)
    metricas["silhouette_espacio_base"] = float(silhouette_score(escaladas_base, grupos))
    metricas["delta_silhouette_vs_baseline"] = (
        metricas["silhouette_espacio_base"] - datos.silhouette_baseline
    )
    modelo = pipeline.named_steps["modelo"]
    if hasattr(modelo, "inertia_"):
        metricas["inertia"] = float(modelo.inertia_)

    # El pipeline deja la demanda primero y el resto en el mismo orden de entrada.
    perfiles = pd.DataFrame(espacio_propio, index=entrada.index, columns=entrada.columns)
    registrar_candidato(
        nombre=nombre,
        pregunta=PREGUNTA,
        familia=familia,
        parametros={**datos.parametros, **parametros,
                    "caracteristicas": "ampliadas" if ampliada else "base",
                    "n_features": entrada.shape[1]},
        metricas=metricas,
        commit=commit,
        modelo=pipeline,
        ejemplo=entrada,
        salida=grupos,
        entrenamiento=entrada.reset_index(drop=True),
        objetivo=None,
        figuras=figuras_p3(nombre, perfiles, grupos, entrada.index),
        tiempos={"tiempo_entrenamiento_s": t},
    )
    print(f"    silhouette(propio)={metricas['train_silhouette_score']:.4f} "
          f"silhouette(base)={metricas['silhouette_espacio_base']:.4f} "
          f"(baseline={datos.silhouette_baseline:.4f})", flush=True)


def ejecutar():
    """Corre la búsqueda de k y los tres candidatos de P3."""
    _, version = archivos_y_version()
    commit = commit_actual()
    print("P3: preparando datos...", flush=True)
    datos = preparar_datos(version)
    print(f"P3: {len(datos.features)} estaciones; "
          f"silhouette baseline={datos.silhouette_baseline:.4f}", flush=True)

    entrada_base = datos.features[COLUMNAS_BASE + ["noche"]].drop(columns="noche")
    entrada_ampliada = datos.ampliadas[COLUMNAS_BASE + COLUMNAS_P3_AMPLIADAS_EXTRA]

    with mlflow.start_run(run_name="P3-experimentos") as umbrella:
        mlflow.set_tag("mlflow.runName", "P3-experimentos")
        mlflow.set_tags(etiquetas(PREGUNTA, "experimentos", "search", commit))
        mlflow.log_params(datos.parametros)

        print("P3: búsqueda del número de grupos (K-Means)...", flush=True)
        k = buscar_kmeans(datos, commit)

        print("P3: candidatos...", flush=True)
        registrar("P3-kmeans-mejor-k", "kmeans",
                  {"algoritmo": "KMeans", "n_grupos": k, "n_init": 10},
                  entrada_base, datos.escaladas,
                  crear_pipeline(KMeans(n_clusters=k, random_state=P3_SEMILLA, n_init=10)),
                  datos, commit)
        registrar("P3-gmm-mejor-k", "gaussian_mixture",
                  {"algoritmo": "GaussianMixture", "n_grupos": k,
                   "covariance_type": "full", "n_init": 5},
                  entrada_base, datos.escaladas,
                  crear_pipeline(GaussianMixture(n_components=k, covariance_type="full",
                                                 n_init=5, random_state=P3_SEMILLA)),
                  datos, commit)
        registrar("P3-kmeans-caracteristicas-ampliadas", "kmeans",
                  {"algoritmo": "KMeans", "n_grupos": k, "n_init": 10},
                  entrada_ampliada, datos.escaladas,
                  crear_pipeline(KMeans(n_clusters=k, random_state=P3_SEMILLA, n_init=10),
                                 ampliada=True),
                  datos, commit, ampliada=True)

    print(f"P3 listo. Corrida principal: {umbrella.info.run_id}", flush=True)
