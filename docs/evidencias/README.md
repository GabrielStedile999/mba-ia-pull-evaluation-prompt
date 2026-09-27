# Evidências da avaliação no LangSmith

Salve aqui os screenshots referenciados no `README.md` principal:

| Arquivo | O que deve mostrar |
|---------|--------------------|
| `01-terminal-evaluate.png` | Saída do `python src/evaluate.py` com STATUS: APROVADO e todas as métricas >= 0.8 |
| `02-dataset-15-exemplos.png` | Dataset `mba-ia-pull-evaluation-prompt-eval` no LangSmith com os 15 exemplos |
| `03-experimento-v2-metricas.png` | Experimento do prompt v2 com as colunas helpfulness, correctness, f1_score, clarity e precision |
| `04-tracing-exemplo-1.png` | Tracing detalhado de um exemplo (simples) |
| `05-tracing-exemplo-2.png` | Tracing detalhado de um exemplo (médio) |
| `06-tracing-exemplo-3.png` | Tracing detalhado de um exemplo (complexo) |
| `07-prompt-hub-v2.png` | Prompt `{handle}/bug_to_user_story_v2` publicado (público) no Prompt Hub |

Obs.: a pasta `screenshots/` está no `.gitignore` do template, por isso as evidências
ficam em `docs/evidencias/`.
