
"""Esquemas de entrada y salida para la API REST de MiBici."""

from pydantic import BaseModel, ConfigDict, Field


class EntradaBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )


class P1Input(EntradaBase):
    hora: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6)
    fin_semana: int = Field(ge=0, le=1)
    lag_1: float = Field(ge=0)
    lag_2: float = Field(ge=0)
    lag_3: float = Field(ge=0)
    lag_24: float = Field(ge=0)
    lag_168: float = Field(ge=0)
    media_movil_24: float = Field(ge=0)
    media_movil_168: float = Field(ge=0)
    media_hist_estacion_dow_hora: float = Field(ge=0)
    media_hist_estacion: float = Field(ge=0)
    media_hist_estacion_hora: float = Field(ge=0)
    tendencia_24_168: float
    hora_sin: float = Field(ge=-1, le=1)
    hora_cos: float = Field(ge=-1, le=1)
    prop_horas_cero_24: float = Field(ge=0, le=1)


class P2Input(EntradaBase):
    hora: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6)
    fin_semana: int = Field(ge=0, le=1)
    capacidad: float = Field(gt=0)
    salidas_lag_1: float = Field(ge=0)
    llegadas_lag_1: float = Field(ge=0)
    flujo_neto_lag_1: float
    presion_lag_1: float
    presion_lag_2: float
    presion_lag_24: float
    presion_lag_168: float
    presion_media_24: float
    presion_media_168: float


class P3Input(EntradaBase):
    viajes_diarios: float = Field(ge=0)
    balance_relativo: float
    proporcion_fin_semana: float = Field(ge=0, le=1)
    manana: float = Field(ge=0)
    mediodia: float = Field(ge=0)
    tarde: float = Field(ge=0)


class P1Output(BaseModel):
    demanda_predicha: float


class P2Output(BaseModel):
    clase_predicha: int


class P3Output(BaseModel):
    cluster: int
