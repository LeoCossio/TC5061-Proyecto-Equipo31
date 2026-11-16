"""Pasos 4 y 5 de la entrega 8: versión alternativa de P1, registro y A/B testing.

1. Toma como modelo A la versión vigente del Model Registry (alias `champion`, o la versión 1).
   Si el Registry no tiene el modelo, registra como versión 1 la copia empaquetada en
   `modelos_serving/` (la misma que usa el contenedor).
2. Entrena el modelo B con los mismos hiperparámetros pero con todos los datos anteriores
   al mes de evaluación, y lo registra como una nueva versión del mismo modelo.
3. Evalúa A y B sobre las mismas filas del mes de evaluación, compara sus errores día por
   día, aplica la regla de decisión de `mibici.ab_testing.decidir` y lo registra en MLflow.

Uso desde la raíz del repositorio (con el servidor de MLflow activo):

    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python scripts/ab_test_p1.py
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python scripts/ab_test_p1.py --promover

Sin `--promover` solo se asigna el alias `challenger` a la versión nueva y no se modifica el
alias `champion` (la versión en producción). Con `--promover` el alias `champion` pasa a la
versión que gane el A/B testing.
"""

import argparse
import json
import sys
from importlib import metadata
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from mibici import ab_testing as ab  # noqa: E402
from mibici.config import (  # noqa: E402
    AB_FECHA_INICIO_EVALUACION,
    AB_NIVEL_CONFIANZA,
    AB_NOMBRE_MODELO,
    AB_TOLERANCIA_EMPEORAMIENTO,
    FIGURES_DIR,
    P1_FECHA_INICIO_VALIDACION,
    PROJ_ROOT,
    REPORTS_DIR,
)
from mibici.experimentos.comun import (  # noqa: E402
    archivos_y_version,
    commit_actual,
    conectar_mlflow,
    registrar_artefactos_base,
)
from mibici.features import COLUMNAS_P1_AMPLIADAS, OBJETIVO_P1  # noqa: E402

COLOR_A, COLOR_B, COLOR_BASE = "#4C78A8", "#F58518", "#7F7F7F"
RUTA_MODELO_EMPAQUETADO = PROJ_ROOT / "modelos_serving"
SALIDA = REPORTS_DIR / "ab_testing"


def requisitos_modelo():
    """Requisitos mínimos del modelo, con las versiones del entorno que lo cargó.

    Se pasan de forma explícita porque, si no, MLflow infiere los requisitos a partir del
    `requirements.txt` del repositorio, cuyo `scikit-learn==1.7.2` choca con la 1.9.1 que
    usa el modelo vigente.
    """
    paquetes = ["scikit-learn", "numpy", "pandas", "cloudpickle"]
    return [f"{p}=={metadata.version(p)}" for p in paquetes]


def obtener_modelo_a(cliente):
    """Devuelve (modelo A, versión del Registry, ¿se registró ahora?)."""
    versiones = [v for v in cliente.search_model_versions(f"name='{AB_NOMBRE_MODELO}'")]
    if versiones:
        try:
            version = cliente.get_model_version_by_alias(AB_NOMBRE_MODELO, "champion").version
        except Exception:
            version = min(versiones, key=lambda v: int(v.version)).version
        return mlflow.sklearn.load_model(f"models:/{AB_NOMBRE_MODELO}/{version}"), version, False
    modelo = mlflow.sklearn.load_model(str(RUTA_MODELO_EMPAQUETADO))
    return modelo, None, True


def figuras(datos, modelo_a, pred_a, pred_b, resultado, pareado, directorio):
    """Genera las gráficas del A/B testing y devuelve {archivo: ruta}."""
    rutas = {}
    ev = datos.evaluacion

    # 1. Métricas de A y B
    fig, ejes = plt.subplots(1, 3, figsize=(11, 4))
    for eje, (clave, titulo) in zip(ejes, [("mae", "MAE"), ("rmse", "RMSE"),
                                            ("mae_horas_pico", "MAE en horas pico")]):
        a, b = resultado["A"][clave], resultado["B"][clave]
        barras = eje.bar(["A (vigente)", "B (reentrenado)"], [a, b], color=[COLOR_A, COLOR_B])
        eje.set_title(titulo)
        eje.set_ylabel("viajes por estación-hora")
        eje.set_ylim(0, max(a, b) * 1.18)
        for barra, valor in zip(barras, (a, b)):
            eje.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.4f}",
                     ha="center", va="bottom", fontsize=9)
        eje.text(0.5, 0.04, f"cambio de B: {100 * (b - a) / a:+.2f} %", transform=eje.transAxes,
                 ha="center", fontsize=9, color="#333333")
    fig.suptitle("A/B testing de P1 sobre septiembre de 2026 (mismas filas para A y B)")
    fig.tight_layout()
    rutas["ab_metricas.png"] = directorio / "ab_metricas.png"
    fig.savefig(rutas["ab_metricas.png"], dpi=140)
    plt.close(fig)

    # 2. MAE diario de A y B
    diario = pareado["diario"]
    fig, (eje1, eje2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2]})
    eje1.plot(diario.index, diario["error_a"], marker="o", ms=3, color=COLOR_A, label="A (vigente)")
    eje1.plot(diario.index, diario["error_b"], marker="o", ms=3, color=COLOR_B, label="B (reentrenado)")
    eje1.set_ylabel("MAE diario")
    eje1.set_title("Error absoluto medio por día de septiembre")
    eje1.legend()
    eje1.grid(alpha=0.25)
    colores = np.where(diario["diferencia"] > 0, COLOR_B, COLOR_A)
    eje2.bar(diario.index, diario["diferencia"], color=colores, width=0.8)
    eje2.axhline(0, color="black", linewidth=0.8)
    eje2.set_ylabel("MAE de A menos MAE de B")
    eje2.set_title(f"Diferencia diaria (positiva = B mejor); B gana {pareado['dias_en_que_gana_b']} "
                   f"de {pareado['dias']} días")
    eje2.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    rutas["ab_mae_diario.png"] = directorio / "ab_mae_diario.png"
    fig.savefig(rutas["ab_mae_diario.png"], dpi=140)
    plt.close(fig)

    # 3. MAE semanal del modelo vigente desde la validación hasta el final
    posterior = datos.todo_posterior
    pred_posterior = modelo_a.predict(posterior[COLUMNAS_P1_AMPLIADAS])
    semanal_a = ab.mae_semanal(posterior, pred_posterior)
    media_global = float(datos.historia_a[OBJETIVO_P1].mean())
    semanal_base = ab.mae_baseline_semanal(posterior, media_global)
    fig, eje = plt.subplots(figsize=(9, 4.5))
    eje.plot(semanal_a.index, semanal_a.values, marker="o", ms=3, color=COLOR_A,
             label="Modelo vigente (A)")
    eje.plot(semanal_base.index, semanal_base.values, marker="s", ms=3, color=COLOR_BASE,
             linestyle="--", label="Baseline (media histórica)")
    eje.axvspan(pd.Timestamp(AB_FECHA_INICIO_EVALUACION), semanal_a.index.max() + pd.Timedelta(days=7),
                color="#FDE7C8", alpha=0.6, label="Mes de evaluación A/B")
    eje.axvline(pd.Timestamp(P1_FECHA_INICIO_VALIDACION), color="black", linewidth=0.8)
    eje.text(pd.Timestamp(P1_FECHA_INICIO_VALIDACION), eje.get_ylim()[1] * 0.98,
             " fin del entrenamiento de A", va="top", fontsize=8)
    eje.set_ylabel("MAE semanal")
    eje.set_title("Error del modelo vigente con el paso del tiempo")
    eje.grid(alpha=0.25)
    eje.legend(loc="lower left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    rutas["ab_mae_semanal_modelo_vigente.png"] = directorio / "ab_mae_semanal_modelo_vigente.png"
    fig.savefig(rutas["ab_mae_semanal_modelo_vigente.png"], dpi=140)
    plt.close(fig)

    # 4. MAE por hora del día
    por_hora_a = ab.mae_por_hora(ev, pred_a)
    por_hora_b = ab.mae_por_hora(ev, pred_b)
    fig, eje = plt.subplots(figsize=(9, 4.2))
    eje.plot(por_hora_a.index, por_hora_a.values, marker="o", ms=3, color=COLOR_A, label="A (vigente)")
    eje.plot(por_hora_b.index, por_hora_b.values, marker="o", ms=3, color=COLOR_B, label="B (reentrenado)")
    eje.set_xlabel("Hora del día")
    eje.set_ylabel("MAE")
    eje.set_xticks(range(0, 24, 2))
    eje.set_title("MAE por hora del día en septiembre")
    eje.grid(alpha=0.25)
    eje.legend()
    fig.tight_layout()
    rutas["ab_mae_por_hora.png"] = directorio / "ab_mae_por_hora.png"
    fig.savefig(rutas["ab_mae_por_hora.png"], dpi=140)
    plt.close(fig)

    return rutas, semanal_a, semanal_base, pred_posterior


def descripcion_version(rol, entrenamiento, resultado):
    return (f"Modelo {rol} de demanda por estación-hora (HistGradientBoostingRegressor, pérdida Poisson). "
            f"Entrenado con {entrenamiento}. MAE en septiembre de 2026: {resultado['mae']:.4f}.")


def principal(promover=False):
    conectar_mlflow()
    cliente = MlflowClient()
    commit = commit_actual()
    _, version_datos = archivos_y_version()

    print("Cargando el modelo A (versión vigente)...", flush=True)
    modelo_a, version_a, a_nuevo = obtener_modelo_a(cliente)

    print("Preparando los datos...", flush=True)
    datos = ab.preparar_datos_ab()
    ev = datos.evaluacion
    columnas = COLUMNAS_P1_AMPLIADAS
    print(f"  Entrenamiento de B: {len(datos.entrenamiento_b):,} filas "
          f"({datos.entrenamiento_b['hora_inicio'].min().date()} a "
          f"{datos.entrenamiento_b['hora_inicio'].max().date()})", flush=True)
    print(f"  Evaluación común:   {len(ev):,} filas "
          f"({ev['hora_inicio'].min().date()} a {ev['hora_inicio'].max().date()})", flush=True)

    print("Entrenando el modelo B...", flush=True)
    inicio = time.perf_counter()
    modelo_b = ab.entrenar_alternativa(modelo_a, datos.entrenamiento_b)
    segundos_b = time.perf_counter() - inicio

    pred_a = modelo_a.predict(ev[columnas])
    pred_b = modelo_b.predict(ev[columnas])
    resultado = {"A": ab.metricas(ev, pred_a), "B": ab.metricas(ev, pred_b)}
    pareado = ab.comparar_pareado(ev, pred_a, pred_b)
    ganador, condiciones = ab.decidir(resultado["A"], resultado["B"], pareado)
    conocidas = set(datos.historia_a["origen_id"].unique())
    grupos = {"A": ab.mae_por_grupo_de_estacion(ev, pred_a, conocidas),
              "B": ab.mae_por_grupo_de_estacion(ev, pred_b, conocidas)}

    directorio_tmp = Path(tempfile.mkdtemp())
    rutas, semanal_a, semanal_base, pred_posterior = figuras(
        datos, modelo_a, pred_a, pred_b, resultado, pareado, directorio_tmp)
    mensual = ab.resumen_mensual(datos.todo_posterior, pred_posterior,
                                 float(datos.historia_a[OBJETIVO_P1].mean()))

    resumen = {
        "modelo": AB_NOMBRE_MODELO,
        "evaluacion": {"desde": str(ev["hora_inicio"].min()), "hasta": str(ev["hora_inicio"].max()),
                       "filas": len(ev)},
        "entrenamiento_a": "2025-10-01 a 2026-05-31",
        "entrenamiento_b": f"2025-10-01 a {datos.entrenamiento_b['hora_inicio'].max().date()}",
        "filas_entrenamiento_b": len(datos.entrenamiento_b),
        "metricas": resultado,
        "mejora_b_pct": {k: 100 * (resultado["A"][k] - resultado["B"][k]) / resultado["A"][k]
                         for k in resultado["A"]},
        "comparacion_diaria": {k: v for k, v in pareado.items() if k != "diario"},
        "grupos_de_estaciones": grupos,
        "regla": {"nivel_confianza": AB_NIVEL_CONFIANZA, "tolerancia": AB_TOLERANCIA_EMPEORAMIENTO,
                  "condiciones": condiciones},
        "version_vigente_recomendada": ganador,
        "tiempo_entrenamiento_b_s": segundos_b,
        "data_version": version_datos,
        "git_commit": commit,
        "resumen_mensual_modelo_vigente": mensual.round(4).to_dict(orient="index"),
        "mae_semanal_a": {str(k.date()): float(v) for k, v in semanal_a.items()},
        "mae_semanal_baseline": {str(k.date()): float(v) for k, v in semanal_base.items()},
    }

    etiquetas_base = {"equipo": "31", "pregunta": "P1", "question": "p1",
                      "model_family": "hist_gradient_boosting", "stage_of_work": "ab_test",
                      "git_commit": commit}
    parametros_comunes = {
        "fecha_inicio_evaluacion": AB_FECHA_INICIO_EVALUACION,
        "filas_evaluacion": len(ev), "data_version": version_datos,
        "granularidad": "estacion-hora", "horizonte": "1 hora", "n_features": len(columnas),
    }

    print("Registrando en MLflow...", flush=True)
    with mlflow.start_run(run_name="P1-AB-test-septiembre"):
        mlflow.set_tags({**etiquetas_base, "rol": "comparacion"})
        mlflow.log_params({**parametros_comunes, "regla_decision": "ver mibici.ab_testing.decidir",
                           "tolerancia_empeoramiento": AB_TOLERANCIA_EMPEORAMIENTO})

        # --- Modelo A -----------------------------------------------------------------
        with mlflow.start_run(run_name="P1-A-vigente", nested=True):
            mlflow.set_tags({**etiquetas_base, "rol": "A"})
            mlflow.log_params({**parametros_comunes, "entrenamiento": resumen["entrenamiento_a"],
                               "origen": "Registry" if version_a else "modelos_serving (copia empaquetada)",
                               **{k: v for k, v in modelo_a.get_params().items()
                                  if k in ("loss", "learning_rate", "max_iter", "max_leaf_nodes",
                                           "min_samples_leaf", "random_state")}})
            mlflow.log_metrics({f"ab_{k}": v for k, v in resultado["A"].items()})
            if a_nuevo:
                ejemplo = ev[columnas].head(5).astype("float64")
                info_a = mlflow.sklearn.log_model(
                    modelo_a, name="model", signature=infer_signature(ejemplo, modelo_a.predict(ejemplo)),
                    input_example=ejemplo, serialization_format="cloudpickle",
                    pip_requirements=requisitos_modelo())
                version_a = mlflow.register_model(f"models:/{info_a.model_id}", AB_NOMBRE_MODELO).version
                cliente.update_model_version(
                    AB_NOMBRE_MODELO, version_a,
                    descripcion_version("A (vigente)", "octubre de 2025 a mayo de 2026", resultado["A"]))
                cliente.set_model_version_tag(AB_NOMBRE_MODELO, version_a, "rol", "A")
                cliente.set_model_version_tag(AB_NOMBRE_MODELO, version_a, "origen",
                                              "copia de modelos_serving/")
                cliente.set_registered_model_alias(AB_NOMBRE_MODELO, "champion", version_a)

        # --- Modelo B -----------------------------------------------------------------
        with mlflow.start_run(run_name="P1-B-reentrenado-hasta-agosto", nested=True) as corrida_b:
            mlflow.set_tags({**etiquetas_base, "rol": "B"})
            mlflow.log_params({**parametros_comunes, "entrenamiento": resumen["entrenamiento_b"],
                               "filas_entrenamiento": len(datos.entrenamiento_b),
                               **{k: v for k, v in modelo_b.get_params().items()
                                  if k in ("loss", "learning_rate", "max_iter", "max_leaf_nodes",
                                           "min_samples_leaf", "random_state")}})
            mlflow.log_metrics({f"ab_{k}": v for k, v in resultado["B"].items()})
            mlflow.log_metric("tiempo_entrenamiento_s", segundos_b)
            registrar_artefactos_base({**parametros_comunes, "rol": "B"})
            ejemplo = ev[columnas].head(5).astype("float64")
            info_b = mlflow.sklearn.log_model(
                modelo_b, name="model", signature=infer_signature(ejemplo, modelo_b.predict(ejemplo)),
                input_example=ejemplo, serialization_format="cloudpickle",
                    pip_requirements=requisitos_modelo())
            version_b = mlflow.register_model(f"models:/{info_b.model_id}", AB_NOMBRE_MODELO).version
            for nombre_archivo in ("ab_mae_diario.png", "ab_mae_por_hora.png"):
                mlflow.log_artifact(str(rutas[nombre_archivo]), artifact_path="plots")
            id_corrida_b = corrida_b.info.run_id

        cliente.update_model_version(
            AB_NOMBRE_MODELO, version_b,
            descripcion_version("B (reentrenado)", f"octubre de 2025 a {resumen['entrenamiento_b'][-10:]}",
                                resultado["B"]))
        for clave, valor in {"rol": "B", "run_id_entrenamiento": id_corrida_b,
                             "ventana_entrenamiento": resumen["entrenamiento_b"],
                             "evaluacion": "septiembre de 2026", "data_version": version_datos,
                             "git_commit": commit, "version_comparada": str(version_a)}.items():
            cliente.set_model_version_tag(AB_NOMBRE_MODELO, version_b, clave, valor)

        # --- Alias según la decisión ----------------------------------------------------
        # `champion` es la versión en producción y `challenger` la siguiente mejor.
        if promover:
            campeon, retador = (version_b, version_a) if ganador == "B" else (version_a, version_b)
            cliente.set_registered_model_alias(AB_NOMBRE_MODELO, "champion", campeon)
            cliente.set_registered_model_alias(AB_NOMBRE_MODELO, "challenger", retador)
        else:
            cliente.set_registered_model_alias(AB_NOMBRE_MODELO, "challenger", version_b)
        resumen["versiones"] = {"A": str(version_a), "B": str(version_b)}
        resumen["alias_champion_modificado"] = bool(promover)

        # --- Resultado de la comparación ------------------------------------------------
        mlflow.log_metrics({
            "ab_mejora_mae_pct": resumen["mejora_b_pct"]["mae"],
            "ab_mejora_rmse_pct": resumen["mejora_b_pct"]["rmse"],
            "ab_mejora_mae_pico_pct": resumen["mejora_b_pct"]["mae_horas_pico"],
            "ab_diferencia_mae_diario": pareado["diferencia_mae_diario"],
            "ab_ic_inferior": pareado["ic_inferior"], "ab_ic_superior": pareado["ic_superior"],
            "ab_p_wilcoxon": pareado["p_wilcoxon"], "ab_dias_gana_b": pareado["dias_en_que_gana_b"],
        })
        mlflow.set_tag("version_vigente_recomendada", ganador)
        for archivo in rutas.values():
            mlflow.log_artifact(str(archivo), artifact_path="plots")
        mlflow.log_dict(resumen, "resultado_ab.json")
        pareado["diario"].to_csv(directorio_tmp / "mae_diario.csv")
        mlflow.log_artifact(str(directorio_tmp / "mae_diario.csv"), artifact_path="data")

    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "resultado_p1.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for nombre_archivo, ruta in rutas.items():
        (FIGURES_DIR / nombre_archivo).write_bytes(ruta.read_bytes())

    print("\n=== RESULTADO DEL A/B TESTING (septiembre de 2026) ===")
    for clave in ("mae", "rmse", "mae_horas_pico"):
        print(f"  {clave:<16} A = {resultado['A'][clave]:.4f}   B = {resultado['B'][clave]:.4f}   "
              f"B mejora {resumen['mejora_b_pct'][clave]:+.2f} %")
    print(f"  Diferencia diaria de MAE (A - B): {pareado['diferencia_mae_diario']:+.4f} "
          f"IC95 [{pareado['ic_inferior']:+.4f}, {pareado['ic_superior']:+.4f}]  "
          f"Wilcoxon p = {pareado['p_wilcoxon']:.2e}  B gana {pareado['dias_en_que_gana_b']}/{pareado['dias']} días")
    print(f"  Condiciones: {condiciones}")
    print(f"  Versión recomendada como vigente: {ganador} "
          f"(A = v{version_a}, B = v{version_b}; champion {'actualizado' if promover else 'sin cambios'})")
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B testing del modelo P1")
    parser.add_argument("--promover", action="store_true",
                        help="Mueve el alias champion a la versión que gane el A/B testing")
    principal(parser.parse_args().promover)
