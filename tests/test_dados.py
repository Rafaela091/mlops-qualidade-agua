import numpy as np
import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.data import aplicar_drift, validar_dataframe

pytestmark = pytest.mark.dados


def test_bases_validas_sem_sobreposicao(base_treino, base_holdout):
    assert validar_dataframe(base_treino)
    assert validar_dataframe(base_holdout)
    assert set(map(tuple, base_treino[config.FEATURES].to_numpy())).isdisjoint(
        set(map(tuple, base_holdout[config.FEATURES].to_numpy()))
    )


@pytest.mark.parametrize(
    "coluna,valor", [("pH", -1), ("Lead (mg/L)", -0.01), ("Mercury (mg/L)", np.inf)]
)
def test_medicao_fisicamente_invalida(base_treino, coluna, valor):
    amostra = base_treino.head(1).copy()
    amostra.loc[amostra.index[0], coluna] = valor
    assert not validar_dataframe(amostra)


def test_esquema_e_alvo_invalidos(base_treino):
    assert not validar_dataframe(base_treino.drop(columns=["pH"]))
    amostra = base_treino.head(1).copy()
    amostra[config.ALVO] = 2
    assert not validar_dataframe(amostra)


def test_drift_aumenta_turbidez_sem_alterar_oxigenio(base_treino):
    amostra = base_treino.head(1)
    alterada = aplicar_drift(amostra, fator=1.5)
    assert alterada["Turbidity (NTU)"].iloc[0] > amostra["Turbidity (NTU)"].iloc[0]
    pd.testing.assert_series_equal(alterada["DO (mg/L)"], amostra["DO (mg/L)"])
