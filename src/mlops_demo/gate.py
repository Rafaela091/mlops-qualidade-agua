"""Portão de qualidade que bloqueia a promoção do candidato reprovado."""

import json
import shutil
import sys

import joblib
import pandas as pd

from mlops_demo import config
from mlops_demo.evaluate import avaliar


def decidir(candidato, campeao):
    verificacoes = [
        {
            "regra": f"recall de alerta >= {config.RECALL_MINIMO:.2f}",
            "ok": candidato["recall"] >= config.RECALL_MINIMO,
            "valor": candidato["recall"],
        },
        {
            "regra": f"precisão de alerta >= {config.PRECISAO_MINIMA:.2f}",
            "ok": candidato["precision"] >= config.PRECISAO_MINIMA,
            "valor": candidato["precision"],
        },
        {
            "regra": "recall do candidato >= recall do campeão (mesmo holdout, com tolerância)",
            "ok": candidato["recall"] >= campeao["recall"] - config.TOLERANCIA_CAMPEAO,
            "valor": candidato["recall"],
        },
    ]
    return {
        "aprovado": all(item["ok"] for item in verificacoes),
        "verificacoes": verificacoes,
        "candidato": candidato,
        "campeao": campeao,
    }


def resumo_markdown(resultado):
    selo = "APROVADO" if resultado["aprovado"] else "REPROVADO"
    linhas = [
        f"## Portão de qualidade: {selo}",
        "",
        "| Modelo | Recall alerta | Precisão alerta | Amostras |",
        "|---|---:|---:|---:|",
    ]
    for nome in ("campeao", "candidato"):
        m = resultado[nome]
        linhas.append(f"| {nome} | {m['recall']:.3f} | {m['precision']:.3f} | {m['n_amostras']} |")
    linhas.extend(["", "| Regra | Resultado | Valor |", "|---|---|---:|"])
    for item in resultado["verificacoes"]:
        linhas.append(
            f"| {item['regra']} | {'passou' if item['ok'] else 'FALHOU'} | {item['valor']:.3f} |"
        )
    return "\n".join(linhas) + "\n"


def executar():
    avaliacao = pd.read_csv(config.AVALIACAO_PROCESSADA)
    candidato = avaliar(joblib.load(config.MODELO_CANDIDATO), avaliacao)
    campeao = avaliar(joblib.load(config.MODELO_CAMPEAO), avaliacao)
    resultado = decidir(candidato, campeao)
    config.RELATORIO_GATE.parent.mkdir(parents=True, exist_ok=True)
    config.RELATORIO_GATE.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    texto = resumo_markdown(resultado)
    config.RESUMO_GATE.write_text(texto, encoding="utf-8")
    print(texto)
    return 0 if resultado["aprovado"] else 1


def promover():
    resultado = json.loads(config.RELATORIO_GATE.read_text(encoding="utf-8"))
    if not resultado["aprovado"]:
        raise SystemExit("O candidato foi reprovado; promoção bloqueada.")
    shutil.copy2(config.MODELO_CANDIDATO, config.MODELO_CAMPEAO)
    metricas = dict(resultado["candidato"], origem="treinamento contínuo (CT)")
    config.METRICAS_CAMPEAO.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(promover() if "--promover" in sys.argv else executar())
