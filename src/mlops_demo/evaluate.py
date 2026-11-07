import json

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import ALVO, CLASSE_POSITIVA, FEATURES


def avaliar(modelo, df, pos_label=CLASSE_POSITIVA):
    """Recebe o modelo e o dataframe completo, separa os dados e calcula as métricas."""
    X = df[FEATURES]
    y = df[ALVO]
    y_pred = modelo.predict(X)

    metricas = {
        "precision": float(precision_score(y, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y, y_pred, pos_label=pos_label, zero_division=0)),
    }
    return metricas


def main():
    modelo = joblib.load("models/champion/model.joblib")
    df = pd.read_csv("data/raw/base_holdout.csv")

    metricas = avaliar(modelo, df)

    with open("metrics.json", "w") as f:
        json.dump(metricas, f, indent=4)


if __name__ == "__main__":
    main()
