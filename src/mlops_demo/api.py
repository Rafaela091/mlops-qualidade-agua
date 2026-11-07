import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import FEATURES

app = FastAPI(title="API Qualidade da Água")


# Definimos o formato exato que a API deve receber (os 8 parâmetros químicos)
# Usamos 'alias' para lidar com os espaços e parênteses dos nomes originais das colunas
class EntradaAgua(BaseModel):
    pH: float = Field(alias="pH")
    Turbidity: float = Field(alias="Turbidity (NTU)")
    Temperature: float = Field(alias="Temperature (°C)")
    DO: float = Field(alias="DO (mg/L)")
    BOD: float = Field(alias="BOD (mg/L)")
    Lead: float = Field(alias="Lead (mg/L)")
    Mercury: float = Field(alias="Mercury (mg/L)")
    Arsenic: float = Field(alias="Arsenic (mg/L)")


@app.get("/")
def health_check():
    return {"status": "ok", "projeto": "Monitorização de Água"}


@app.post("/predict")
def predict(entrada: EntradaAgua):
    caminho_modelo = "models/champion/model.joblib"
    if not os.path.exists(caminho_modelo):
        raise HTTPException(status_code=503, detail="Modelo campeão não encontrado.")

    modelo = joblib.load(caminho_modelo)

    # Converte os dados recebidos para o formato de tabela do Pandas
    dados = entrada.dict(by_alias=True)
    df = pd.DataFrame([dados])

    # Garante que a ordem das colunas é exatamente a esperada pelo modelo
    df = df[FEATURES]

    # Faz a previsão
    predicao = int(modelo.predict(df)[0])

    return {"predicao": predicao, "alerta": predicao == 1}
