import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.evaluate import avaliar
from mlops_demo.gate import decidir

pytestmark = pytest.mark.modelo


def test_modelo_atual_cumpre_contrato(modelo_atual, base_holdout):
    m = avaliar(modelo_atual, base_holdout)
    assert m["recall"] >= config.RECALL_MINIMO
    assert m["precision"] >= config.PRECISAO_MINIMA


def test_campeao_salvo_cumpre_contrato(campeao, base_holdout):
    m = avaliar(campeao, base_holdout)
    assert m["recall"] >= config.RECALL_MINIMO
    assert m["precision"] >= config.PRECISAO_MINIMA


def test_sentido_do_alerta_com_maior_turbidez_e_chumbo(campeao):
    base = pd.read_csv(config.BASE_HOLDOUT).head(1)[config.FEATURES].copy()
    base["Turbidity (NTU)"] = 4
    base["Lead (mg/L)"] = 0.005
    alterada = base.copy()
    alterada["Turbidity (NTU)"] = 25
    alterada["Lead (mg/L)"] = 0.06
    i = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    assert campeao.predict_proba(alterada)[0, i] > campeao.predict_proba(base)[0, i]


def test_portao_reprova_precisao_baixa_com_recall_alto():
    candidato = {"recall": 1.0, "precision": 0.5, "n_amostras": 100}
    campeao = {"recall": 0.9, "precision": 0.9, "n_amostras": 100}
    assert not decidir(candidato, campeao)["aprovado"]
