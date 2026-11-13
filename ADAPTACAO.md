# Relatório de adaptação — qualidade da água

## Problema e dados

Classificação binária de um **alerta fictício de contaminação** a partir de oito medições numéricas de água. Um operador de monitoramento usaria a saída para priorizar revisão humana; a saída não certifica potabilidade. A base é sintética, gerada localmente por `preparar_dados.py` com semente 42, 1.200 amostras (960 treino, 240 holdout, sem sobreposição). Fonte: gerador original deste repositório. Licença dos CSVs: [CC0 1.0](DATA_LICENSE.md). Não há dados pessoais ou sensíveis. O CSV de origem incerta que estava na pasta inicial foi removido.

## Contrato de qualidade

Classe positiva: `alerta_contaminacao=1`. Métrica principal: recall >= **0,85**, pois o falso negativo deixa passar um alerta simulado. Precisão >= **0,80**, para limitar falsos alertas. O candidato não pode perder mais que 0,02 de recall contra o campeão no mesmo holdout. O campeão inicial obteve recall **0,959** e precisão **0,895** em 240 amostras. Esses limiares são didáticos, não padrões sanitários.

## Alterações

| Arquivos | Adaptação |
|---|---|
| `data/raw/base_*.csv`, `preparar_dados.py` | Gerador reproduzível, alvo binário, divisão estratificada e sem sobreposição. `data/raw/lotes/` inicia vazia. |
| `config.py`, `data.py` | Oito atributos, caminhos e limiares, limites físicos, validação, preparo DVC e drift. |
| `features.py`, `train.py`, `params.yaml` | Regressão logística padronizada; parâmetros no YAML; treino do candidato e campeão. |
| `evaluate.py`, `gate.py`, `monitor.py` | Métricas da classe 1, decisão de portão e teste KS de drift. |
| `api.py`, `demo/smoke_test.py` | Schema de oito medições, rejeição de inválidos e rotas `/health` e `/predict`. |
| `tests/` | Testes novos específicos por camada, com marcadores do CI. |
| `demo/cenarios.py` | Cinco cenários para falhas e drift de água. |
| `models/champion/` | Campeão inicial regenerado com a nova base. |
| `README.md`, `DATA_LICENSE.md` | Domínio, execução, pipelines e licença. |

## Evidências do GitHub Actions

| Evidência | URL da execução |
|---|---|
| E1 CI verde na `main` | [CI verde na main](https://github.com/Rafaela091/mlops-qualidade-agua/actions/runs/43461605982) |
| E2 CI vermelho controlado em PR | [CI vermelho no step Testes de código](https://github.com/Rafaela091/mlops-qualidade-agua/actions/runs/43461817592) |
| E3 CT com drift e portão aprovado | [CT com drift e portão aprovado](https://github.com/Rafaela091/mlops-qualidade-agua/actions/runs/43461836435) |
| E4 CT com portão reprovado | [CT com portão reprovado](https://github.com/Rafaela091/mlops-qualidade-agua/actions/runs/43461853441) |
| E5 CD verde em tag e pacote GHCR | [CD da tag v1.0.0](https://github.com/Rafaela091/mlops-qualidade-agua/actions/runs/43461886851) |

[Pacote publicado no GHCR](https://github.com/Rafaela091/mlops-qualidade-agua/pkgs/container/mlops-qualidade-agua).

## Limites, riscos e contribuição

O portão usa apenas um holdout sintético e pode falhar com dados reais, mudança de distribuição, rótulos errados ou vazamento. Em produção, monitoraríamos prevalência, qualidade das medições, drift, recall/precisão com rótulos posteriores, latência e erros da API. O modelo exige validação de domínio antes de qualquer uso real.

**Contribuições:** Rafaela Barreiros: criação do repositório e commits iniciais; o commit de adaptação consta em sua conta. Integrantes informados: [Luan-bs](https://github.com/Luan-bs), [Borgessz23](https://github.com/Borgessz23), [TainaDr](https://github.com/TainaDr) e [cauasauer](https://github.com/cauasauer). Suas contribuições específicas e commits ainda precisam ser registrados pela equipe; não constam no histórico atual. **IA:** assistente de IA usado para adaptar código, gerar dados sintéticos, escrever testes e documentação, executar os pipelines e registrar evidências.
