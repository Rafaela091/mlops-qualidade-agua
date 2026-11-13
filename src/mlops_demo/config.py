"""Contrato do pipeline de alerta de qualidade da água."""

from pathlib import Path

import yaml

BASE_TREINO = Path("data/raw/base_treino.csv")
BASE_HOLDOUT = Path("data/raw/base_holdout.csv")
PASTA_LOTES = Path("data/raw/lotes")
TREINO_PROCESSADO = Path("data/processed/treino.csv")
AVALIACAO_PROCESSADA = Path("data/processed/avaliacao.csv")
MODELO_CAMPEAO = Path("models/champion/model.joblib")
METRICAS_CAMPEAO = Path("models/champion/metrics.json")
MODELO_CANDIDATO = Path("models/candidate/model.joblib")
METRICAS_CANDIDATO = Path("models/candidate/metrics.json")
RELATORIO_DRIFT = Path("reports/drift.md")
RELATORIO_GATE = Path("reports/gate.json")
RESUMO_GATE = Path("reports/gate.md")
FEATURES = [
    "pH",
    "Turbidity (NTU)",
    "Temperature (°C)",
    "DO (mg/L)",
    "BOD (mg/L)",
    "Lead (mg/L)",
    "Mercury (mg/L)",
    "Arsenic (mg/L)",
]
ALVO = "alerta_contaminacao"
CLASSE_POSITIVA = 1
COLUNAS_DRIFT = ["Turbidity (NTU)", "BOD (mg/L)", "Lead (mg/L)"]
RECALL_MINIMO = 0.85
PRECISAO_MINIMA = 0.80
TOLERANCIA_CAMPEAO = 0.02
LIMITE_DRIFT_P = 0.01
MIN_FEATURES_DRIFT = 2
LIMITES = {
    "pH": (0, 14),
    "Turbidity (NTU)": (0, 100),
    "Temperature (°C)": (0, 60),
    "DO (mg/L)": (0, 25),
    "BOD (mg/L)": (0, 50),
    "Lead (mg/L)": (0, 1),
    "Mercury (mg/L)": (0, 0.1),
    "Arsenic (mg/L)": (0, 1),
}


def carregar_params(caminho="params.yaml"):
    with open(caminho, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
