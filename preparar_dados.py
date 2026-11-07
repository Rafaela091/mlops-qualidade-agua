import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Importamos os nomes exatos que o pipeline espera!
from mlops_demo.config import ALVO, FEATURES

df = pd.read_csv("Water_Quality_Dataset.csv")

# Removemos as duas primeiras colunas (Data e Local)
df = df.iloc[:, 2:]

# Forçamos as colunas do dataset a terem exatamente os nomes do nosso config.py
df.columns = FEATURES + [ALVO]

# Binarizamos o alvo
df[ALVO] = df[ALVO].apply(lambda x: 0 if int(x) == 0 else 1)
df = df.dropna()

treino, holdout = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs("data/raw", exist_ok=True)
treino.to_csv("data/raw/base_treino.csv", index=False)
holdout.to_csv("data/raw/base_holdout.csv", index=False)

os.makedirs("data/raw/lotes", exist_ok=True)
for f in os.listdir("data/raw/lotes"):
    os.remove(os.path.join("data/raw/lotes", f))

print("Sucesso! Os dados foram corrigidos com os nomes exatos das colunas.")
