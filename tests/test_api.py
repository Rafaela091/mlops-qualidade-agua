from demo.cenarios import CENARIO_AGUA_BOA
from fastapi.testclient import TestClient

from mlops_demo.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_endpoint_falta_parametro_especifico():
    # Teste NOVO: Tentar prever sem enviar o pH (dado químico obrigatório)
    cenario_incompleto = CENARIO_AGUA_BOA.copy()
    del cenario_incompleto["pH"]

    response = client.post("/predict", json=cenario_incompleto)
    assert response.status_code == 422  # Pydantic deve barrar e dar erro 422 (Unprocessable Entity)
