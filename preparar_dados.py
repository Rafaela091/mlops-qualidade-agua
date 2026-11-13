"""Recria as bases sintéticas com divisão estratificada e semente fixa."""

from mlops_demo import config
from mlops_demo.data import criar_base, dividir_dataset


def main():
    treino, holdout = dividir_dataset(criar_base())
    config.BASE_TREINO.parent.mkdir(parents=True, exist_ok=True)
    config.PASTA_LOTES.mkdir(parents=True, exist_ok=True)
    treino.to_csv(config.BASE_TREINO, index=False)
    holdout.to_csv(config.BASE_HOLDOUT, index=False)
    print(f"Gerados {len(treino)} exemplos de treino e {len(holdout)} de holdout")


if __name__ == "__main__":
    main()
