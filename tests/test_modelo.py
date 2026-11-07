import joblib
import pandas as pd
import os
from mlops_demo.config import FEATURES
from demo.cenarios import CENARIO_AGUA_BOA


def test_modelo_direcional_chumbo_especifico():
    if not os.path.exists("models/champion/model.joblib"):
        return

    modelo = joblib.load("models/champion/model.joblib")

    df_base = pd.DataFrame([CENARIO_AGUA_BOA])[FEATURES]
    prob_base = modelo.predict_proba(df_base)[0][1]

    cenario_toxico = CENARIO_AGUA_BOA.copy()
    cenario_toxico["Lead (mg/L)"] = 5.0
    df_toxico = pd.DataFrame([cenario_toxico])[FEATURES]
    prob_toxica = modelo.predict_proba(df_toxico)[0][1]

    assert prob_toxica >= prob_base
