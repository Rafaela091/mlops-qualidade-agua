"""Treina candidato via DVC ou campeão inicial com --campeao."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from mlops_demo import config
from mlops_demo.evaluate import avaliar
from mlops_demo.features import construir_pipeline, separar_xy


def treinar(df_treino, params):
    X, y = separar_xy(df_treino)
    modelo = construir_pipeline(params)
    modelo.fit(X, y)
    return modelo


def salvar(modelo, metricas, caminho_modelo, caminho_metricas):
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    caminho_metricas.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho_modelo)
    caminho_metricas.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def treinar_candidato():
    modelo = treinar(pd.read_csv(config.TREINO_PROCESSADO), config.carregar_params())
    metricas = avaliar(modelo, pd.read_csv(config.AVALIACAO_PROCESSADA))
    salvar(modelo, metricas, config.MODELO_CANDIDATO, config.METRICAS_CANDIDATO)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/candidate_metrics.json").write_text(
        json.dumps(metricas, indent=2), encoding="utf-8"
    )
    print("candidato:", json.dumps(metricas))
    return metricas


def treinar_campeao_inicial():
    modelo = treinar(pd.read_csv(config.BASE_TREINO), config.carregar_params())
    metricas = avaliar(modelo, pd.read_csv(config.BASE_HOLDOUT))
    metricas["origem"] = "base_treino.csv"
    salvar(modelo, metricas, config.MODELO_CAMPEAO, config.METRICAS_CAMPEAO)
    print("campeão inicial:", json.dumps(metricas))
    return metricas


if __name__ == "__main__":
    treinar_campeao_inicial() if "--campeao" in sys.argv else treinar_candidato()
