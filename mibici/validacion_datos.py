"""Validación de la calidad de los datos con DeepChecks.

Revisa cuatro conjuntos del proyecto antes de entrenar:

    viajes   Parquet limpios de MiBici (entrada común de P1, P2 y P3).
    p1       Tabla estación-hora de demanda (regresión).
    p2       Tabla estación-hora de riesgo estimado de desbalance (clasificación).
    p3       Perfiles de estación de un periodo de entrenamiento y otro de prueba.

Cada conjunto pasa por dos tipos de chequeo:

    Integridad (error)       tipos de datos, valores faltantes, duplicados,
                             columnas constantes, valores fuera de dominio y fuga
                             temporal entre entrenamiento y prueba. Si falla uno,
                             los datos no deben usarse.
    Comparación (advertencia) deriva (drift) de variables y de la etiqueta, mezcla
                             de filas idénticas y tamaño relativo de los
                             conjuntos. Una deriva puede ser legítima (por ejemplo,
                             estacionalidad entre meses de invierno y de verano),
                             así que se reporta pero no bloquea.

Uso desde la raíz del repositorio:

    python -m mibici.validacion_datos                      # los cuatro conjuntos
    python -m mibici.validacion_datos --conjuntos viajes p3
    python -m mibici.validacion_datos --html               # reportes de DeepChecks
    python -m mibici.validacion_datos --estricto           # advertencias también fallan

Termina con código 1 si falla algún chequeo de integridad.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _compatibilidad_numpy2():
    """DeepChecks 0.19 usa alias que NumPy 2 eliminó (np.Inf, np.NaN, np.float_).

    Se restablecen antes de importar DeepChecks para poder usar la versión
    publicada con las dependencias fijadas del proyecto (numpy 2.3).
    """
    for viejo, nuevo in {"Inf": "inf", "NaN": "nan", "float_": "float64"}.items():
        if not hasattr(np, viejo):
            setattr(np, viejo, getattr(np, nuevo))


_compatibilidad_numpy2()

# DeepChecks 0.19 emite avisos de funciones obsoletas de pandas y setuptools que no
# dependen de nuestro código y ocultan el resultado de los chequeos.
warnings.filterwarnings("ignore", module="deepchecks")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*applymap.*")

from deepchecks.core.check_result import CheckFailure, CheckResult  # noqa: E402
from deepchecks.tabular import Dataset, Suite  # noqa: E402
from deepchecks.tabular.checks import (  # noqa: E402
    DataDuplicates,
    DatasetsSizeComparison,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    IsSingleValue,
    MixedDataTypes,
    FeatureDrift,
    LabelDrift,
    MixedNulls,
    PercentOfNulls,
    TrainTestSamplesMix,
)

from mibici.config import (  # noqa: E402
    EXTERNAL_DATA_DIR,
    P1_FECHA_FIN,
    P1_FECHA_INICIO,
    P1_FECHA_INICIO_PRUEBA,
    P1_FECHA_INICIO_VALIDACION,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)

DIRECTORIO_REPORTES = REPORTS_DIR / "validacion"
RESUMEN_MD = REPORTS_DIR / "validacion_datos.md"

SEMILLA = 42
# Los gráficos de DeepChecks solo se calculan cuando se pide el reporte HTML.
CON_GRAFICOS = False
UMBRAL_DERIVA = 0.2  # valor por defecto de DeepChecks (Cramér's V / Earth Mover's)
MAX_DURACION_HORAS = 24
# DeepChecks revisa por defecto solo 100 000 filas en PercentOfNulls; se amplía para
# que el conteo de nulos cubra todos los viajes (4.5 millones).
N_MUESTRAS_INTEGRIDAD = 10_000_000

ESQUEMA_VIAJES = {
    "viaje_id": "int",
    "inicio": "datetime",
    "fin": "datetime",
    "origen_id": "int",
    "destino_id": "int",
}


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------


@dataclass
class Hallazgo:
    """Resultado de una condición de calidad."""

    conjunto: str
    chequeo: str
    condicion: str
    nivel: str  # "integridad" o "comparacion"
    estado: str  # "PASS", "FAIL" o "ERROR"
    detalle: str

    @property
    def es_error(self) -> bool:
        return self.estado != "PASS" and self.nivel == "integridad"

    @property
    def es_advertencia(self) -> bool:
        return self.estado != "PASS" and self.nivel == "comparacion"


def _hallazgos_de_suite(resultado, conjunto, nivel_por_chequeo):
    """Convierte el resultado de una Suite de DeepChecks en una lista de Hallazgo."""
    hallazgos = []
    for r in resultado.results:
        clase = type(r.check).__name__
        nombre = r.check.name()
        nivel = nivel_por_chequeo.get(clase, "integridad")
        if isinstance(r, CheckFailure):
            hallazgos.append(Hallazgo(
                conjunto, nombre, "El chequeo se ejecutó sin errores", nivel, "ERROR",
                str(r.exception)[:200],
            ))
        elif isinstance(r, CheckResult):
            for c in r.conditions_results:
                estado = "PASS" if c.category.name == "PASS" else (
                    "ERROR" if c.category.name == "ERROR" else "FAIL")
                hallazgos.append(Hallazgo(conjunto, nombre, c.name, nivel, estado, c.details))
    return hallazgos


# ---------------------------------------------------------------------------
# Construcción de los chequeos
# ---------------------------------------------------------------------------


def construir_suite_integridad(nombre, ignorar_duplicados=None, max_nulos=0.0,
                               max_duplicados=0.0):
    """Suite de un solo conjunto: valores faltantes, duplicados, constantes y tipos.

    `ignorar_duplicados` excluye columnas (por ejemplo, un identificador único)
    al buscar filas repetidas.
    """
    chequeos = [
        MixedNulls().add_condition_different_nulls_less_equal_to(1),
        PercentOfNulls(n_samples=N_MUESTRAS_INTEGRIDAD).add_condition_percent_of_nulls_not_greater_than(
            max_nulos),
        DataDuplicates(ignore_columns=ignorar_duplicados).add_condition_ratio_less_or_equal(
            max_duplicados),
        IsSingleValue().add_condition_not_single_value(),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0.01, 0.2)),
    ]
    return Suite(f"Integridad de {nombre}", *chequeos)


def construir_suite_train_test(nombre, con_etiqueta=False, con_fecha=True,
                               umbral_deriva=UMBRAL_DERIVA, max_filas_repetidas=0.05):
    """Suite de entrenamiento contra prueba: deriva, mezcla de filas y fuga temporal."""
    chequeos = [
        DatasetsSizeComparison().add_condition_test_train_size_ratio_greater_than(0.05),
        FeatureDrift(n_samples=100_000).add_condition_drift_score_less_than(
            max_allowed_categorical_score=umbral_deriva, max_allowed_numeric_score=umbral_deriva),
        TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(max_filas_repetidas),
    ]
    if con_etiqueta:
        chequeos.append(LabelDrift().add_condition_drift_score_less_than(
            max_allowed_drift_score=umbral_deriva))
    if con_fecha:
        chequeos += [
            DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0.0),
            DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_less_or_equal(0.0),
        ]
    return Suite(f"Entrenamiento contra prueba: {nombre}", *chequeos)


# Chequeos cuyo fallo es una advertencia (el resto son errores de integridad).
CHEQUEOS_DE_COMPARACION = {
    "DatasetsSizeComparison", "FeatureDrift", "LabelDrift", "TrainTestSamplesMix",
}
NIVELES = {c: "comparacion" for c in CHEQUEOS_DE_COMPARACION}


def crear_dataset(df, nombre, categoricas=(), fecha=None, etiqueta=None, tipo_etiqueta=None,
                  identificador=None):
    """Envuelve un DataFrame en un Dataset de DeepChecks.

    Las columnas `fecha`, `etiqueta` e `identificador` no se tratan como variables.
    """
    return Dataset(
        df,
        label=etiqueta,
        label_type=tipo_etiqueta,
        cat_features=list(categoricas),
        datetime_name=fecha,
        index_name=identificador,
        dataset_name=nombre,
    )


# ---------------------------------------------------------------------------
# Validaciones por conjunto
# ---------------------------------------------------------------------------


def validar_esquema(df, esquema, conjunto, periodo=None, ids_catalogo=None):
    """Valida tipos y reglas de dominio de la tabla de viajes con pandas.

    Complementa a DeepChecks: comprueba que cada columna tenga el tipo esperado,
    que los identificadores de viaje sean únicos, que no haya viajes con fin
    anterior al inicio ni duraciones mayores a un día, que las fechas caigan en
    el periodo del proyecto y que las estaciones existan en el catálogo.
    """
    hallazgos = []

    def agregar(chequeo, condicion, ok, detalle, nivel="integridad"):
        hallazgos.append(Hallazgo(conjunto, chequeo, condicion, nivel,
                                  "PASS" if ok else "FAIL", detalle))

    faltantes = [c for c in esquema if c not in df.columns]
    agregar("Esquema", "Están todas las columnas esperadas", not faltantes,
            "Faltan: " + ", ".join(faltantes) if faltantes else f"{len(esquema)} columnas presentes")
    verificadores = {
        "int": pd.api.types.is_integer_dtype,
        "datetime": pd.api.types.is_datetime64_any_dtype,
        "float": pd.api.types.is_float_dtype,
    }
    for columna, tipo in esquema.items():
        if columna in df.columns:
            ok = verificadores[tipo](df[columna])
            agregar("Esquema", f"{columna} es de tipo {tipo}", ok, f"dtype real: {df[columna].dtype}")

    if "viaje_id" in df.columns:
        repetidos = int(df["viaje_id"].duplicated().sum())
        agregar("Dominio", "viaje_id es único", repetidos == 0, f"{repetidos:,} identificadores repetidos")

    if {"inicio", "fin"} <= set(df.columns) and all(
            pd.api.types.is_datetime64_any_dtype(df[c]) for c in ("inicio", "fin")):
        duracion = (df["fin"] - df["inicio"]).dt.total_seconds() / 3600
        negativas = int((duracion < 0).sum())
        agregar("Dominio", "Ningún viaje termina antes de empezar", negativas == 0,
                f"{negativas:,} viajes con duración negativa")
        largas = int((duracion > MAX_DURACION_HORAS).sum())
        agregar("Dominio", f"Ningún viaje dura más de {MAX_DURACION_HORAS} h", largas == 0,
                f"{largas:,} viajes más largos ({100 * largas / max(len(df), 1):.3f} %)",
                nivel="comparacion")
        if periodo is not None:
            inicio, fin = (pd.Timestamp(periodo[0]), pd.Timestamp(periodo[1]) + pd.Timedelta(days=1))
            fuera = int(((df["inicio"] < inicio) | (df["inicio"] >= fin)).sum())
            agregar("Dominio", f"Los viajes inician entre {periodo[0]} y {periodo[1]}", fuera == 0,
                    f"{fuera:,} viajes fuera del periodo")

    if ids_catalogo is not None:
        for columna in ("origen_id", "destino_id"):
            if columna in df.columns:
                desconocidas = sorted(set(df[columna].unique()) - set(ids_catalogo))
                agregar("Catálogo", f"{columna} existe en la nomenclatura de estaciones",
                        not desconocidas,
                        f"Estaciones ausentes del catálogo: {[int(x) for x in desconocidas[:10]]}"
                        if desconocidas else "todas las estaciones están en el catálogo",
                        nivel="comparacion")
    return hallazgos


def validar_integridad(df, conjunto, categoricas=(), fecha=None, etiqueta=None,
                       tipo_etiqueta=None, identificador=None, ignorar_duplicados=None,
                       max_nulos=0.0, max_duplicados=0.0):
    """Corre los chequeos de integridad de DeepChecks sobre un solo conjunto."""
    dataset = crear_dataset(df, conjunto, categoricas, fecha, etiqueta, tipo_etiqueta, identificador)
    suite = construir_suite_integridad(conjunto, ignorar_duplicados, max_nulos, max_duplicados)
    resultado = suite.run(train_dataset=dataset, with_display=CON_GRAFICOS)
    return _hallazgos_de_suite(resultado, conjunto, {}), resultado


def validar_train_test(entrenamiento, prueba, conjunto, categoricas=(), fecha=None,
                       etiqueta=None, tipo_etiqueta=None, identificador=None,
                       umbral_deriva=UMBRAL_DERIVA, max_filas_repetidas=0.05):
    """Compara entrenamiento y prueba: deriva, filas repetidas, tamaño y fuga temporal."""
    comunes = dict(categoricas=categoricas, fecha=fecha, etiqueta=etiqueta,
                   tipo_etiqueta=tipo_etiqueta, identificador=identificador)
    ds_tr = crear_dataset(entrenamiento, "entrenamiento", **comunes)
    ds_te = crear_dataset(prueba, "prueba", **comunes)
    suite = construir_suite_train_test(conjunto, con_etiqueta=etiqueta is not None,
                                       con_fecha=fecha is not None, umbral_deriva=umbral_deriva,
                                       max_filas_repetidas=max_filas_repetidas)
    resultado = suite.run(train_dataset=ds_tr, test_dataset=ds_te, with_display=CON_GRAFICOS)
    return _hallazgos_de_suite(resultado, conjunto, NIVELES), resultado


# ---------------------------------------------------------------------------
# Carga de datos reales
# ---------------------------------------------------------------------------


def cargar_viajes():
    """Lee los 12 Parquet limpios de MiBici."""
    archivos = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    if len(archivos) != 12:
        raise FileNotFoundError(
            f"Se esperaban 12 Parquet en {PROCESSED_DATA_DIR} y hay {len(archivos)}. "
            "Ejecuta primero: python -m mibici.dataset")
    return pd.concat([pd.read_parquet(a) for a in archivos], ignore_index=True)


def ids_de_catalogo():
    """Identificadores de estación de la nomenclatura descargada por la ingesta."""
    ruta = EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"
    if not ruta.exists():
        return None
    return set(pd.read_csv(ruta, usecols=["id"])["id"].astype(int))


def preparar_viajes(viajes):
    """Añade las variables derivadas que se comparan entre entrenamiento y prueba."""
    datos = viajes.copy()
    datos["hora"] = datos["inicio"].dt.hour.astype("int8")
    datos["dia_semana"] = datos["inicio"].dt.dayofweek.astype("int8")
    datos["duracion_min"] = ((datos["fin"] - datos["inicio"]).dt.total_seconds() / 60).astype("float32")
    return datos


def _partir_por_fecha(df, columna, inicio_validacion=P1_FECHA_INICIO_VALIDACION,
                      inicio_prueba=P1_FECHA_INICIO_PRUEBA):
    """Entrenamiento antes de la validación y prueba desde `inicio_prueba` (como P1 y P2)."""
    fechas = df[columna]
    return (df[fechas < pd.Timestamp(inicio_validacion)],
            df[fechas >= pd.Timestamp(inicio_prueba)])


def validar_viajes(resultados_dc):
    """Integridad de la tabla de viajes y deriva entre entrenamiento y prueba."""
    viajes = cargar_viajes()
    hallazgos = validar_esquema(
        viajes, ESQUEMA_VIAJES, "viajes", periodo=(P1_FECHA_INICIO, P1_FECHA_FIN),
        ids_catalogo=ids_de_catalogo())

    datos = preparar_viajes(viajes)
    del viajes
    columnas = ["origen_id", "destino_id", "hora", "dia_semana", "duracion_min", "inicio"]
    h, r = validar_integridad(datos[columnas + ["viaje_id"]], "viajes", fecha="inicio",
                              categoricas=["origen_id", "destino_id", "hora", "dia_semana"],
                              identificador="viaje_id")
    hallazgos += h
    resultados_dc["viajes (integridad)"] = r

    entrenamiento, prueba = _partir_por_fecha(datos[columnas], "inicio")
    h, r = validar_train_test(entrenamiento, prueba, "viajes", fecha="inicio",
                              categoricas=["origen_id", "destino_id", "hora", "dia_semana"])
    hallazgos += h
    resultados_dc["viajes (entrenamiento contra prueba)"] = r
    return hallazgos


def validar_p1(resultados_dc):
    """Tabla estación-hora de P1: integridad y comparación de entrenamiento con prueba."""
    from mibici.features import (
        COLUMNAS_P1,
        OBJETIVO_P1,
        construir_panel_horario,
        crear_features_p1,
        dividir_temporal_p1,
    )

    viajes = cargar_viajes()[["origen_id", "inicio"]]
    features = crear_features_p1(construir_panel_horario(viajes))
    del viajes
    categoricas = ["hora", "dia_semana", "fin_semana"]
    columnas = COLUMNAS_P1 + [OBJETIVO_P1, "hora_inicio"]

    hallazgos = []
    h, r = validar_integridad(features[columnas], "p1", categoricas=categoricas,
                              fecha="hora_inicio", etiqueta=OBJETIVO_P1,
                              tipo_etiqueta="regression", ignorar_duplicados=["hora_inicio"],
                              max_duplicados=0.2)
    hallazgos += h
    resultados_dc["p1 (integridad)"] = r

    entrenamiento, _, prueba = dividir_temporal_p1(
        features, P1_FECHA_INICIO_VALIDACION, P1_FECHA_INICIO_PRUEBA)
    h, r = validar_train_test(entrenamiento[columnas], prueba[columnas], "p1",
                              categoricas=categoricas, fecha="hora_inicio",
                              etiqueta=OBJETIVO_P1, tipo_etiqueta="regression")
    hallazgos += h
    resultados_dc["p1 (entrenamiento contra prueba)"] = r
    return hallazgos


def validar_p2(resultados_dc):
    """Tabla estación-hora de P2 con la etiqueta de riesgo calculada solo con entrenamiento."""
    from mibici.feature_engineering.classification import (
        COLUMNAS_P2,
        OBJETIVO_P2,
        asignar_etiqueta_p2,
        cargar_capacidades,
        construir_panel_p2,
        crear_features_p2,
        definir_umbral_p2,
        dividir_temporal_p2,
    )
    from mibici.train_p2 import CUANTIL_RIESGO

    viajes = cargar_viajes()[["inicio", "fin", "origen_id", "destino_id"]]
    catalogo = cargar_capacidades(EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv")
    features = crear_features_p2(construir_panel_p2(viajes, catalogo))
    del viajes
    entrenamiento, _, prueba = dividir_temporal_p2(
        features, P1_FECHA_INICIO_VALIDACION, P1_FECHA_INICIO_PRUEBA)
    umbral = definir_umbral_p2(entrenamiento, CUANTIL_RIESGO)
    entrenamiento = asignar_etiqueta_p2(entrenamiento, umbral)
    prueba = asignar_etiqueta_p2(prueba, umbral)
    todo = pd.concat([entrenamiento, prueba], ignore_index=True)

    categoricas = ["hora", "dia_semana", "fin_semana"]
    columnas = COLUMNAS_P2 + [OBJETIVO_P2, "fecha_hora"]

    hallazgos = []
    h, r = validar_integridad(todo[columnas], "p2", categoricas=categoricas, fecha="fecha_hora",
                              etiqueta=OBJETIVO_P2, tipo_etiqueta="binary",
                              ignorar_duplicados=["fecha_hora"], max_duplicados=0.2)
    hallazgos += h
    resultados_dc["p2 (integridad)"] = r

    h, r = validar_train_test(entrenamiento[columnas], prueba[columnas], "p2",
                              categoricas=categoricas, fecha="fecha_hora",
                              etiqueta=OBJETIVO_P2, tipo_etiqueta="binary")
    hallazgos += h
    resultados_dc["p2 (entrenamiento contra prueba)"] = r
    return hallazgos


def validar_p3(resultados_dc):
    """Perfiles de estación de P3 calculados con el periodo de entrenamiento y el de prueba."""
    from mibici.feature_engineering.clustering import crear_features_p3

    viajes = cargar_viajes()
    entrenamiento, prueba = _partir_por_fecha(viajes, "inicio")
    perfiles_tr = crear_features_p3(entrenamiento)
    perfiles_te = crear_features_p3(prueba)
    del viajes, entrenamiento, prueba

    hallazgos = []
    h, r = validar_integridad(perfiles_tr.reset_index(drop=True), "p3", max_duplicados=0.0)
    hallazgos += h
    resultados_dc["p3 (integridad)"] = r

    h, r = validar_train_test(perfiles_tr.reset_index(drop=True), perfiles_te.reset_index(drop=True), "p3")
    hallazgos += h
    resultados_dc["p3 (perfiles de entrenamiento contra prueba)"] = r
    return hallazgos


VALIDADORES = {"viajes": validar_viajes, "p1": validar_p1, "p2": validar_p2, "p3": validar_p3}


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def resumen_markdown(hallazgos, duraciones):
    """Tabla en Markdown con el resultado de cada condición."""
    iconos = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️"}
    lineas = ["# Validación de datos con DeepChecks", "",
              "Generado con `python -m mibici.validacion_datos`.", ""]
    for conjunto in dict.fromkeys(h.conjunto for h in hallazgos):
        propios = [h for h in hallazgos if h.conjunto == conjunto]
        errores = sum(h.es_error for h in propios)
        avisos = sum(h.es_advertencia for h in propios)
        lineas += [f"## {conjunto}", "",
                   f"{len(propios)} condiciones: {errores} errores de integridad, "
                   f"{avisos} advertencias de comparación ({duraciones.get(conjunto, 0):.0f} s).", "",
                   "| Nivel | Chequeo | Condición | Resultado | Detalle |", "|---|---|---|---|---|"]
        for h in propios:
            detalle = h.detalle.replace("|", "/").replace("\n", " ")[:160]
            lineas.append(f"| {h.nivel} | {h.chequeo} | {h.condicion} | {iconos[h.estado]} {h.estado} | {detalle} |")
        lineas.append("")
    return "\n".join(lineas)


def imprimir_resumen(hallazgos):
    """Imprime un resumen legible en la terminal."""
    for conjunto in dict.fromkeys(h.conjunto for h in hallazgos):
        propios = [h for h in hallazgos if h.conjunto == conjunto]
        print(f"\n=== {conjunto.upper()} ===")
        for h in propios:
            marca = {"PASS": "OK   ", "FAIL": "FALLA", "ERROR": "ERROR"}[h.estado]
            nota = " (advertencia)" if h.es_advertencia else ""
            print(f"  [{marca}] {h.nivel:<11} {h.chequeo}: {h.condicion}{nota}")
            if h.estado != "PASS":
                print(f"           {h.detalle[:150]}")
    errores = sum(h.es_error for h in hallazgos)
    avisos = sum(h.es_advertencia for h in hallazgos)
    print(f"\nTotal: {len(hallazgos)} condiciones | {errores} errores de integridad | "
          f"{avisos} advertencias de comparación")


def _ruta_relativa(ruta):
    """Ruta relativa a la raíz del proyecto, o la ruta completa si está fuera de ella."""
    try:
        return Path(ruta).relative_to(PROJ_ROOT)
    except ValueError:
        return Path(ruta)


def principal(argv=None):
    parser = argparse.ArgumentParser(description="Validación de datos de MiBici con DeepChecks")
    parser.add_argument("--conjuntos", nargs="+", default=["todos"],
                        help="viajes, p1, p2, p3 o todos (también separados por comas)")
    parser.add_argument("--html", action="store_true",
                        help=f"Guarda los reportes de DeepChecks en {_ruta_relativa(DIRECTORIO_REPORTES)}")
    parser.add_argument("--estricto", action="store_true",
                        help="Las advertencias de comparación también hacen fallar el comando")
    args = parser.parse_args(argv)
    global CON_GRAFICOS
    CON_GRAFICOS = args.html

    pedidos = [c for texto in args.conjuntos for c in texto.replace(",", " ").split()]
    if "todos" in pedidos:
        pedidos = list(VALIDADORES)
    desconocidos = [c for c in pedidos if c not in VALIDADORES]
    if desconocidos:
        parser.error(f"Conjuntos desconocidos: {desconocidos}. Opciones: {list(VALIDADORES)} o todos")

    hallazgos, resultados_dc, duraciones = [], {}, {}
    for conjunto in pedidos:
        print(f"Validando {conjunto}...", flush=True)
        inicio = time.perf_counter()
        hallazgos += VALIDADORES[conjunto](resultados_dc)
        duraciones[conjunto] = time.perf_counter() - inicio

    imprimir_resumen(hallazgos)
    RESUMEN_MD.parent.mkdir(parents=True, exist_ok=True)
    RESUMEN_MD.write_text(resumen_markdown(hallazgos, duraciones), encoding="utf-8")
    print(f"Resumen escrito en {_ruta_relativa(RESUMEN_MD)}")

    if args.html:
        DIRECTORIO_REPORTES.mkdir(parents=True, exist_ok=True)
        for nombre, resultado in resultados_dc.items():
            archivo = DIRECTORIO_REPORTES / (nombre.replace(" ", "_").replace("(", "").replace(")", "") + ".html")
            resultado.save_as_html(str(archivo), as_widget=False, connected=True)
        print(f"Reportes HTML en {_ruta_relativa(DIRECTORIO_REPORTES)}")
        (DIRECTORIO_REPORTES / "hallazgos.json").write_text(
            json.dumps([asdict(h) for h in hallazgos], ensure_ascii=False, indent=2), encoding="utf-8")

    errores = sum(h.es_error for h in hallazgos)
    avisos = sum(h.es_advertencia for h in hallazgos)
    return 1 if errores or (args.estricto and avisos) else 0


if __name__ == "__main__":
    raise SystemExit(principal())
