import pandas as pd
from demo.cenarios import CENARIO_AGUA_BOA

from mlops_demo.data import validar_dataframe


def test_validar_dataframe_correto():
    df = pd.DataFrame([CENARIO_AGUA_BOA])
    df["Pollution Level"] = 0
    assert validar_dataframe(df) is True


def test_validar_dataframe_ph_invalido_especifico():
    # Teste NOVO: Garante que águas com pH impossível falhem na validação
    cenario_ruim = CENARIO_AGUA_BOA.copy()
    cenario_ruim["pH"] = -2.5  # pH negativo não existe no mundo real
    df = pd.DataFrame([cenario_ruim])
    df["Pollution Level"] = 1
    assert validar_dataframe(df) is False
