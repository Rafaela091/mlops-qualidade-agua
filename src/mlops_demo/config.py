import yaml

# Caminhos dos dados
BASE_TREINO = "data/raw/base_treino.csv"
BASE_HOLDOUT = "data/raw/base_holdout.csv"
PASTA_LOTES = "data/raw/lotes/"

# Caminhos dos modelos e métricas
MODELO_CAMPEAO = "models/champion/model.joblib"
METRICAS_CAMPEAO = "models/champion/metrics.json"
MODELO_DESAFIANTE = "models/challenger/model.joblib"
METRICAS_DESAFIANTE = "models/challenger/metrics.json"

# Contrato de Dados e Modelo para Qualidade da Água
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

ALVO = "Pollution Level"
CLASSE_POSITIVA = 1
COLUNAS_TESTE = ["pH", "Lead (mg/L)"]

CONTRATO_LIMIARES = {"recall": 0.85, "precision": 0.70}


def carregar_params(caminho="params.yaml"):
    with open(caminho) as f:
        return yaml.safe_load(f)
