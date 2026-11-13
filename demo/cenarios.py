"""Cenários controlados de CI e CT. Execute em branch descartável."""

import argparse
from pathlib import Path

CENARIO_AGUA_BOA = {
    "pH": 7.2,
    "Turbidity (NTU)": 4.0,
    "Temperature (°C)": 25.0,
    "DO (mg/L)": 9.0,
    "BOD (mg/L)": 1.5,
    "Lead (mg/L)": 0.004,
    "Mercury (mg/L)": 0.0002,
    "Arsenic (mg/L)": 0.003,
}
CENARIO_AGUA_POLUIDA = dict(
    CENARIO_AGUA_BOA, **{"Turbidity (NTU)": 24.0, "BOD (mg/L)": 11.0, "Lead (mg/L)": 0.05}
)
DESCRICOES = {
    1: "Remove um atributo na separação X/y: falha no teste de código",
    2: "Insere chumbo negativo na base: falha na validação de dados",
    3: "Aumenta o peso da classe 0: recall do alerta cai no teste de modelo",
    4: "Cria lote válido com drift em turbidez, BOD e chumbo: CT detecta drift",
    5: "Peso extremo da classe de alerta: recall alto e precisão baixa no portão",
}


def aplicar(numero):
    if numero == 1:
        p = Path("src/mlops_demo/features.py")
        texto = p.read_text()
        p.write_text(texto.replace("X = df[FEATURES].copy()", "X = df[FEATURES[:-1]].copy()"))
    elif numero == 2:
        p = Path("data/raw/base_treino.csv")
        import pandas as pd

        df = pd.read_csv(p)
        df.loc[0, "Lead (mg/L)"] = -1
        df.to_csv(p, index=False)
    elif numero == 3:
        p = Path("params.yaml")
        texto = p.read_text()
        p.write_text(texto.replace("class_weight: balanced", "class_weight: {0: 20, 1: 1}"))
    elif numero == 4:
        from mlops_demo.data import gerar_lote

        print(gerar_lote("lote_drift", n=120, drift=1.5, semente=91))
    elif numero == 5:
        p = Path("params.yaml")
        texto = p.read_text()
        p.write_text(texto.replace("class_weight: balanced", "class_weight: {0: 1, 1: 1000}"))
    else:
        raise ValueError("Cenário deve estar entre 1 e 5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["listar", "aplicar"])
    ap.add_argument("numero", type=int, nargs="?")
    args = ap.parse_args()
    if args.acao == "listar":
        for numero, descricao in DESCRICOES.items():
            print(f"{numero}: {descricao}")
    else:
        aplicar(args.numero)
