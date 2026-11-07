import sys

import requests

from cenarios import CENARIO_AGUA_BOA


def run_smoke_test(url="http://localhost:8000"):
    print(f"Testando API em {url}/predict...")
    try:
        response = requests.post(f"{url}/predict", json=CENARIO_AGUA_BOA)
        response.raise_for_status()
        resultado = response.json()
        print("Teste passou! Resposta:", resultado)
        assert "predicao" in resultado
        return 0
    except Exception as e:
        print(f"Teste falhou: {e}")
        return 1


if __name__ == "__main__":
    url_alvo = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    sys.exit(run_smoke_test(url_alvo))
