"""Teste de fumaça da API de qualidade da água, apenas com stdlib."""

import json
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

from cenarios import CENARIO_AGUA_BOA


def run_smoke_test(url="http://localhost:8000"):
    try:
        base = url.rstrip("/")
        with urlopen(f"{base}/health", timeout=10) as resposta:
            assert json.load(resposta) == {"status": "ok"}
        requisicao = Request(
            f"{base}/predict",
            data=json.dumps(CENARIO_AGUA_BOA).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(requisicao, timeout=10) as resposta:
            resultado = json.load(resposta)
        assert resultado["predicao"] in (0, 1)
        assert resultado["alerta"] == (resultado["predicao"] == 1)
        print("Teste de fumaça passou:", resultado)
        return 0
    except (URLError, AssertionError, KeyError, ValueError) as erro:
        print("Teste de fumaça falhou:", erro)
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
