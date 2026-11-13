# CI/CD/CT para alerta de qualidade da água

Projeto de classificação binária com **dados sintéticos** de medições de água. A classe positiva (`1`) representa um alerta fictício de contaminação que pede revisão humana. O modelo não determina potabilidade e não deve orientar decisões sanitárias reais.

A base é gerada por [preparar_dados.py](preparar_dados.py), com semente 42: 1.200 linhas, oito atributos numéricos, 960 linhas de treino e 240 de holdout estratificado sem sobreposição. `Timestamp` e `Location` não são usados. Os dados são originais e disponibilizados sob [CC0 1.0](DATA_LICENSE.md). Nenhum dado pessoal ou sensível é usado.

## Contrato

| Item | Valor |
|---|---|
| Classe positiva | `alerta_contaminacao = 1` |
| Métrica principal | Recall da classe positiva >= 0,85 |
| Proteção contra falsos alertas | Precisão da classe positiva >= 0,80 |
| Comparação no CT | Recall do candidato >= recall do campeão menos 0,02, no mesmo holdout |
| Drift | KS com p < 0,01 em pelo menos 2 dos 8 atributos |

Um falso negativo deixaria passar um alerta simulado, por isso recall tem prioridade. A precisão mínima evita que alertar quase tudo seja aceito. Os limiares são critérios de exercício e devem ser recalibrados com dados validados antes de qualquer aplicação real.

## Execução local

Use Python 3.11 e rode na raiz:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ct]"
python preparar_dados.py
python -m mlops_demo.train --campeao
python demo/local.py ci
python demo/local.py ct
uvicorn mlops_demo.api:app
```

Em outro terminal: `python demo/smoke_test.py http://localhost:8000`. O Dockerfile inclui o modelo campeão, e o CD testa `/health` e `/predict` antes de publicar.

## Pipelines

- **CI** (`.github/workflows/ci.yml`): lint, formato, validação de dados e testes de código, dados, modelo e API em Python 3.11 e 3.12.
- **CT** (`.github/workflows/ct.yml`): monitora lotes em `data/raw/lotes/`, reproduz treino com DVC, compara candidato e campeão e bloqueia promoção se o portão reprovar. Agenda e execução manual sempre retreinam. A variável `PROMOCAO_AUTOMATICA=false` mantém revisão por Pull Request.
- **CD** (`.github/workflows/cd.yml`): em tag `v*.*.*`, roda CI, constrói/testa Docker e publica no GHCR.

`python demo/cenarios.py listar` mostra cinco cenários. Aplique cada um em uma branch descartável. O cenário 4 gera um lote com drift válido; o 5 usa peso extremo para produzir alto recall e baixa precisão. `python demo/gerar_lote.py lote_02 --drift 1.8` cria outros lotes. Os lotes não são incluídos na base inicial.

## Preparação do GitHub

O repositório da equipe é [Rafaela091/mlops-qualidade-agua](https://github.com/Rafaela091/mlops-qualidade-agua), público e com histórico próprio. Envie a adaptação em uma branch e abra um Pull Request para `main`. Após o merge, confira os três jobs do CI. Em Settings > Actions > General, ative `Read and write permissions` e `Allow GitHub Actions to create and approve pull requests`; em Settings > Secrets and variables > Actions, crie `PROMOCAO_AUTOMATICA=false`. Cada integrante deve enviar seus próprios commits. Nenhum token deve ser versionado.

Para registrar E1–E5, use branches e PRs para os cenários, acompanhe os steps e copie as URLs das execuções para [ADAPTACAO.md](ADAPTACAO.md). Só depois de CI verde na `main`, crie e envie uma tag `v1.0.0`. Torne o pacote GHCR público em Packages e confirme o download e o smoke test. A promoção automática é opcional; requer `PROMOCAO_AUTOMATICA=true` e `main` sem proteção.
