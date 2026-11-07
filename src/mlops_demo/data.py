import pandas as pd

from .config import ALVO, FEATURES


def validar_dataframe(df: pd.DataFrame) -> bool:
    """Valida o esquema e as regras de negócio dos dados da qualidade da água."""
    # 1. Verifica se todas as features químicas esperadas estão presentes
    for col in FEATURES:
        if col not in df.columns:
            return False

    # 2. Verifica se o alvo (nível de poluição) é binário (0 ou 1)
    if ALVO in df.columns:
        if not set(df[ALVO].dropna().unique()).issubset({0, 1}):
            return False

    # 3. Regra de negócio do domínio: o pH tem de estar estritamente entre 0 e 14
    if "pH" in df.columns:
        if df["pH"].min() < 0.0 or df["pH"].max() > 14.0:
            return False

    return True


def aplicar_drift(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """Simula um drift (desvio) multiplicando os valores químicos para forçar uma anomalia."""
    df_drift = df.copy()
    for col in colunas:
        if col in df_drift.columns:
            # Aumenta os valores em 80% para simular um pico de poluição ou erro no sensor
            df_drift[col] = df_drift[col] * 1.8
    return df_drift
