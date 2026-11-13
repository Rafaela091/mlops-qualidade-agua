"""API de inferência para dados sintéticos de água."""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mlops_demo import config
from mlops_demo.data import validar_dataframe

app = FastAPI(title="API de alerta de qualidade da água")


class EntradaAgua(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ph: float = Field(alias="pH", ge=0, le=14)
    turbidez: float = Field(alias="Turbidity (NTU)", ge=0, le=100)
    temperatura: float = Field(alias="Temperature (°C)", ge=0, le=60)
    oxigenio: float = Field(alias="DO (mg/L)", ge=0, le=25)
    bod: float = Field(alias="BOD (mg/L)", ge=0, le=50)
    chumbo: float = Field(alias="Lead (mg/L)", ge=0, le=1)
    mercurio: float = Field(alias="Mercury (mg/L)", ge=0, le=0.1)
    arsenio: float = Field(alias="Arsenic (mg/L)", ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def raiz():
    return {"status": "ok", "projeto": "Alerta de qualidade da água sintética"}


@app.post("/predict")
def predict(entrada: EntradaAgua):
    if not config.MODELO_CAMPEAO.exists():
        raise HTTPException(status_code=503, detail="Modelo campeão indisponível")
    df = pd.DataFrame([entrada.model_dump(by_alias=True)])[config.FEATURES]
    if not validar_dataframe(df, exigir_alvo=False):
        raise HTTPException(status_code=422, detail="Medições de água inválidas")
    modelo = joblib.load(config.MODELO_CAMPEAO)
    predicao = int(modelo.predict(df)[0])
    return {"predicao": predicao, "alerta": predicao == config.CLASSE_POSITIVA}
