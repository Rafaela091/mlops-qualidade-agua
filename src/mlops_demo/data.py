"""Geração, validação e preparo dos dados sintéticos."""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_demo import config


def criar_base(n=1200, semente=42):
    rng = np.random.default_rng(semente)
    risco = rng.normal(size=n)
    dados = {
        "pH": np.clip(7.2 - 0.45 * risco + rng.normal(0, 0.6, n), 0, 14),
        "Turbidity (NTU)": np.clip(9 + 3.5 * risco + rng.normal(0, 2, n), 0, 100),
        "Temperature (°C)": np.clip(rng.normal(25, 5, n), 0, 60),
        "DO (mg/L)": np.clip(7 - 1.2 * risco + rng.normal(0, 0.9, n), 0, 25),
        "BOD (mg/L)": np.clip(4 + 1.5 * risco + rng.normal(0, 1, n), 0, 50),
        "Lead (mg/L)": np.clip(0.015 + 0.004 * risco + rng.normal(0, 0.003, n), 0, 1),
        "Mercury (mg/L)": np.clip(0.001 + 0.0003 * risco + rng.normal(0, 0.0002, n), 0, 0.1),
        "Arsenic (mg/L)": np.clip(0.012 + 0.003 * risco + rng.normal(0, 0.002, n), 0, 1),
    }
    df = pd.DataFrame(dados)[config.FEATURES]
    pontuacao = (
        (df["Turbidity (NTU)"] - 9) / 3.5
        + (df["BOD (mg/L)"] - 4) / 1.5
        + (7 - df["DO (mg/L)"]) / 1.2
        + (df["Lead (mg/L)"] - 0.015) / 0.004
        + rng.normal(0, 0.45, n)
    )
    df[config.ALVO] = (pontuacao > 1.0).astype(int)
    return df


def dividir_dataset(df, semente=42):
    return train_test_split(df, test_size=0.2, random_state=semente, stratify=df[config.ALVO])


def validar_dataframe(df, exigir_alvo=True):
    colunas = config.FEATURES + ([config.ALVO] if exigir_alvo else [])
    if list(df.columns) != colunas or df.empty or df[colunas].isna().any().any():
        return False
    for coluna in config.FEATURES:
        numeros = pd.to_numeric(df[coluna], errors="coerce")
        minimo, maximo = config.LIMITES[coluna]
        if (
            numeros.isna().any()
            or not np.isfinite(numeros).all()
            or not numeros.between(minimo, maximo).all()
        ):
            return False
    return not exigir_alvo or set(df[config.ALVO].unique()).issubset({0, 1})


def listar_lotes():
    return sorted(config.PASTA_LOTES.glob("*.csv"))


def aplicar_drift(df, colunas=None, fator=1.5):
    if fator <= 0:
        raise ValueError("O fator de drift deve ser positivo")
    resultado = df.copy()
    for coluna in colunas or config.COLUNAS_DRIFT:
        if coluna not in config.COLUNAS_DRIFT:
            raise ValueError(f"Coluna sem simulação de drift: {coluna}")
        resultado[coluna] = (resultado[coluna] * fator).clip(upper=config.LIMITES[coluna][1])
    return resultado


def gerar_lote(nome, n=120, drift=1.0, semente=1):
    if n < 1 or not nome.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Nome ou tamanho de lote inválido")
    df = criar_base(n, semente)
    if drift != 1:
        df = aplicar_drift(df, fator=drift)
    config.PASTA_LOTES.mkdir(parents=True, exist_ok=True)
    caminho = config.PASTA_LOTES / f"{nome}.csv"
    df.to_csv(caminho, index=False)
    return caminho


def preparar():
    treino = pd.read_csv(config.BASE_TREINO)
    holdout = pd.read_csv(config.BASE_HOLDOUT)
    lotes = [pd.read_csv(c) for c in listar_lotes()]
    if not all(validar_dataframe(df) for df in [treino, holdout, *lotes]):
        raise ValueError("Dados brutos inválidos")
    config.TREINO_PROCESSADO.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([treino, *lotes], ignore_index=True).to_csv(config.TREINO_PROCESSADO, index=False)
    holdout.to_csv(config.AVALIACAO_PROCESSADA, index=False)


def validar_arquivos():
    bases = [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]
    for caminho in bases:
        if not validar_dataframe(pd.read_csv(caminho)):
            raise ValueError(f"Dados inválidos: {caminho}")
    treino = pd.read_csv(config.BASE_TREINO)
    holdout = pd.read_csv(config.BASE_HOLDOUT)
    if len(treino) + len(holdout) < 500:
        raise ValueError("O dataset deve ter ao menos 500 linhas")
    if set(map(tuple, treino[config.FEATURES].to_numpy())) & set(
        map(tuple, holdout[config.FEATURES].to_numpy())
    ):
        raise ValueError("Treino e holdout possuem linhas sobrepostas")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"validar", "preparar"}:
        raise SystemExit("Uso: python -m mlops_demo.data [validar|preparar]")
    {"validar": validar_arquivos, "preparar": preparar}[sys.argv[1]]()
