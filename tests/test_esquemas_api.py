
"""Pruebas unitarias para los esquemas de la API MiBici."""

import pytest
from pydantic import ValidationError

from mibici.esquemas_api import P1Input, P2Input, P3Input


def crear_datos_validos(esquema):
    datos = {campo: 1 for campo in esquema.model_fields}

    if "hora" in datos:
        datos["hora"] = 12

    if "dia_semana" in datos:
        datos["dia_semana"] = 2

    if "fin_semana" in datos:
        datos["fin_semana"] = 0

    return datos


@pytest.mark.parametrize("esquema", [P1Input, P2Input, P3Input])
def test_acepta_entradas_validas(esquema):
    datos = crear_datos_validos(esquema)
    resultado = esquema(**datos)

    assert resultado is not None


@pytest.mark.parametrize("esquema", [P1Input, P2Input])
def test_rechaza_hora_invalida(esquema):
    datos = crear_datos_validos(esquema)
    datos["hora"] = 30

    with pytest.raises(ValidationError):
        esquema(**datos)


@pytest.mark.parametrize("esquema", [P1Input, P2Input, P3Input])
def test_rechaza_campos_extra(esquema):
    datos = crear_datos_validos(esquema)
    datos["campo_desconocido"] = 10

    with pytest.raises(ValidationError):
        esquema(**datos)


@pytest.mark.parametrize("esquema", [P1Input, P2Input, P3Input])
def test_rechaza_valores_infinitos(esquema):
    datos = crear_datos_validos(esquema)
    campo_numerico = next(
        nombre for nombre in datos if nombre not in
        {"hora", "dia_semana", "fin_semana"}
    )
    datos[campo_numerico] = float("inf")

    with pytest.raises(ValidationError):
        esquema(**datos)
