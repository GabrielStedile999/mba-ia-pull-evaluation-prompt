# Evidências da avaliação no LangSmith

Capturas do experimento `gabrielstedile-bug_to_user_story_v2-71879cb7` (28/09/2026), referenciadas no `README.md` principal.

| Arquivo | O que mostra |
|---------|--------------|
| `01-terminal-evaluate.txt` | Saída completa de `python src/evaluate.py` (STATUS: APROVADO) e de `python src/share_dataset.py` |
| `02-dataset-15-exemplos.jpg` | Dataset público `mba-ia-pull-evaluation-prompt-eval` com os 15 exemplos |
| `03-experimento-v2-metricas.jpg` | Experimento v2 com as médias das 5 métricas (clarity 0.83, correctness 0.90, f1_score 0.88, helpfulness 0.87, precision 0.91) |
| `03b-experimento-v2-por-exemplo.jpg` | Tabela do experimento com as 5 notas por exemplo (15/15 runs) |
| `04-tracing-exemplo-simples.jpg` | Tracing do exemplo #11 (bug simples): feedback, input, output e referência |
| `05-tracing-exemplo-medio.jpg` | Tracing do exemplo #7 (bug médio) |
| `06-tracing-exemplo-complexo.jpg` | Tracing do exemplo #13 (bug complexo) |
| `06-tracing-exemplo-complexo-llm.jpg` | Run `ChatOpenAI` do exemplo #13: system/user prompt renderizados, tokens e custo |
| `07-prompt-hub-v2.jpg` | Prompt `gabrielstedile/bug_to_user_story_v2` público no Hub, com tags, técnicas e commit |

Links públicos (abrem sem login):

- Dataset + experimentos: https://smith.langchain.com/public/b4fd0458-bbd6-4114-bb78-2fad70037ab2/d
- Experimento v2: https://smith.langchain.com/public/b4fd0458-bbd6-4114-bb78-2fad70037ab2/d/compare?selectedSessions=8a5f1681-43e6-42b9-945f-9f1c2cb59396
- Prompt no Hub: https://smith.langchain.com/hub/gabrielstedile/bug_to_user_story_v2
