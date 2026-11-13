"""Métricas da classe positiva: alerta de contaminação."""

from sklearn.metrics import f1_score, precision_score, recall_score

from mlops_demo import config
from mlops_demo.features import separar_xy


def avaliar(modelo, df):
    X, y = separar_xy(df)
    previsoes = modelo.predict(X)
    return {
        "recall": float(
            recall_score(y, previsoes, pos_label=config.CLASSE_POSITIVA, zero_division=0)
        ),
        "precision": float(
            precision_score(y, previsoes, pos_label=config.CLASSE_POSITIVA, zero_division=0)
        ),
        "f1": float(f1_score(y, previsoes, pos_label=config.CLASSE_POSITIVA, zero_division=0)),
        "n_amostras": int(len(df)),
    }
