import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.data import criar_base, dividir_dataset
from mlops_demo.features import separar_xy

pytestmark = pytest.mark.codigo


def test_contrato_tem_oito_medicoes():
    assert len(config.FEATURES) == 8
    assert config.ALVO == "alerta_contaminacao"
    assert config.CLASSE_POSITIVA == 1


def test_separacao_preserva_ordem_das_colunas(base_treino):
    X, y = separar_xy(base_treino)
    assert list(X.columns) == config.FEATURES
    assert len(X) == len(y)


def test_geracao_reproduzivel_e_estratificada():
    base = criar_base(600, 7)
    pd.testing.assert_frame_equal(base, criar_base(600, 7))
    treino, holdout = dividir_dataset(base, 7)
    assert len(treino) == 480 and len(holdout) == 120
    assert abs(treino[config.ALVO].mean() - holdout[config.ALVO].mean()) < 0.02
