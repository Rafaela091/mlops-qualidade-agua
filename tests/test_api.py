import pytest
from demo.cenarios import CENARIO_AGUA_BOA
from fastapi.testclient import TestClient

from mlops_demo.api import app

pytestmark = pytest.mark.api
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_contrato():
    resposta = client.post("/predict", json=CENARIO_AGUA_BOA)
    assert resposta.status_code == 200
    assert resposta.json()["predicao"] in (0, 1)
    assert resposta.json()["alerta"] == (resposta.json()["predicao"] == 1)


def test_falta_medicao_obrigatoria():
    amostra = CENARIO_AGUA_BOA.copy()
    del amostra["pH"]
    assert client.post("/predict", json=amostra).status_code == 422


def test_rejeita_chumbo_negativo_e_atributo_extra():
    amostra = CENARIO_AGUA_BOA.copy()
    amostra["Lead (mg/L)"] = -0.1
    assert client.post("/predict", json=amostra).status_code == 422
    amostra = CENARIO_AGUA_BOA.copy()
    amostra["Location"] = "L1"
    assert client.post("/predict", json=amostra).status_code == 422
