# Desafio MBA Engenharia de Software com IA - Full Cycle

## Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

Software capaz de fazer **pull** de um prompt de baixa qualidade do LangSmith Prompt Hub, **refatorá-lo** com técnicas avançadas de Prompt Engineering, fazer **push** da versão otimizada de volta ao Hub e **avaliá-la** com 5 métricas customizadas (Helpfulness, Correctness, F1-Score, Clarity e Precision), atingindo **>= 0.8 em todas**.

Tarefa do prompt: converter relatos de bugs em User Stories ágeis.

## Sumário

- [Estrutura do projeto](#estrutura-do-projeto)
- [Técnicas Aplicadas (Fase 2)](#técnicas-aplicadas-fase-2)
- [Processo de otimização (iterações)](#processo-de-otimização-iterações)
- [Resultados Finais](#resultados-finais)
- [Como Executar](#como-executar)
- [Testes de validação](#testes-de-validação)

## Estrutura do projeto

```
mba-ia-pull-evaluation-prompt/
├── .env.example                  # Template das variáveis de ambiente
├── requirements.txt              # Dependências Python
├── README.md                     # Esta documentação
├── prompts/
│   ├── bug_to_user_story_v1.yml  # Prompt inicial (baixa qualidade) - obtido via pull
│   └── bug_to_user_story_v2.yml  # Prompt otimizado
├── datasets/
│   └── bug_to_user_story.jsonl   # 15 bugs (5 simples, 7 médios, 3 complexos)
├── docs/evidencias/              # Screenshots das avaliações no LangSmith
├── src/
│   ├── pull_prompts.py           # Pull do LangSmith Hub (implementado)
│   ├── push_prompts.py           # Push ao LangSmith Hub (implementado)
│   ├── share_dataset.py          # Gera o link público do dataset (extra)
│   ├── evaluate.py               # Avaliação automática (fornecido)
│   ├── metrics.py                # 5 métricas LLM-as-Judge (fornecido)
│   └── utils.py                  # Funções auxiliares (fornecido)
└── tests/
    └── test_prompts.py           # Testes de validação do prompt (implementado)
```

## Tecnologias

- Python 3.10+ e LangChain (`langchain-core`)
- LangSmith (Prompt Hub, datasets, experimentos e tracing)
- OpenAI: `gpt-4.1-mini` para gerar as respostas e para avaliar (LLM-as-Judge). Ambos aceitam `temperature=0`, garantindo avaliação reproduzível. `gpt-4.1` pode ser usado como juiz mais rigoroso via `EVAL_MODEL`.
- Prompts versionados em YAML

## Diagnóstico do prompt original (v1)

O prompt `leonanluppi/bug_to_user_story_v1` tem os seguintes problemas:

| Problema | Consequência nas métricas |
|----------|---------------------------|
| Variável `{bug_report}` duplicada no system e no user prompt | Relato enviado 2x, ruído no contexto |
| Persona genérica ("assistente que ajuda") | Saídas sem visão de produto, persona da user story vaga |
| Nenhuma definição de formato de saída | Sem "Como um / eu quero / para que", sem critérios Dado/Quando/Então → Clarity e F1 baixos |
| Nenhum exemplo (zero-shot) | Formato e nível de detalhe inconsistentes entre execuções |
| Nenhuma regra de comportamento | Modelo inventa contexto e omite dados do relato → Precision baixa |
| Nenhum tratamento de edge cases | Bugs complexos recebem o mesmo tratamento de bugs simples |

## Técnicas Aplicadas (Fase 2)

O prompt otimizado está em [`prompts/bug_to_user_story_v2.yml`](prompts/bug_to_user_story_v2.yml). Foram aplicadas 4 técnicas: **Few-shot Learning** (obrigatória), **Role Prompting**, **Chain of Thought** e **Skeleton of Thought**.

### 1. Role Prompting (persona e contexto detalhado)

**Por quê:** a métrica de Clarity avalia organização e linguagem, e o formato de user story exige visão de produto (persona, valor de negócio). Uma persona de Product Manager sênior ancora o vocabulário, o tom profissional e a prioridade por valor ao usuário, em vez de uma descrição técnica do defeito.

**Como foi aplicado** (trecho do `system_prompt`):

```
# PAPEL
Você é um Product Manager sênior, com mais de 10 anos de experiência em times ágeis
(Scrum/Kanban) de produtos SaaS, e-commerce, mobile e ERP. Você é reconhecido por
transformar relatos de bugs confusos em User Stories claras, completas e prontas para
entrar na sprint, sem inventar informações e sem perder nenhum detalhe relevante.
```

### 2. Few-shot Learning (exemplos de entrada/saída)

**Por quê:** o dataset tem 3 níveis de complexidade e as respostas de referência têm formatos diferentes para cada um (simples: story + critérios; médio: + contexto técnico; complexo: seções `=== ... ===` com critérios técnicos, contexto e tasks). Exemplos demonstram esse formato de forma muito mais precisa que qualquer descrição em prosa, o que impacta diretamente F1-Score (recall do que a referência espera) e Clarity.

**Como foi aplicado:** 3 exemplos completos, um por nível de complexidade, todos criados fora do dataset de avaliação (para não contaminar a métrica):

| Exemplo | Entrada | Saída demonstrada |
|---------|---------|-------------------|
| 1 - Simples | "Botão 'Esqueci minha senha' não envia o e-mail de redefinição." | Story + 5 critérios Dado/Quando/Então, nada mais |
| 2 - Médio | Upload de foto falha (endpoint, HTTP 413, limite Nginx x limite documentado) | Story + critérios + `Contexto Técnico:` preservando endpoint, erro e causa |
| 3 - Complexo | Sistema de agendamento com 2 problemas numerados, contexto e impacto | Story + `USER STORY PRINCIPAL`, `CRITÉRIOS DE ACEITAÇÃO` (A, B), `CRITÉRIOS TÉCNICOS`, `CONTEXTO DO BUG`, `TASKS TÉCNICAS SUGERIDAS` |

### 3. Chain of Thought (raciocínio passo a passo)

**Por quê:** converter um bug em user story exige decisões intermediárias (quem é afetado, o que a pessoa quer, qual a complexidade, quais dados do relato precisam ser preservados). Sem um roteiro, o modelo pula etapas e omite dados, o que derruba recall (F1) e Precision. O raciocínio é feito **internamente**: expor o passo a passo na resposta prejudicaria Clarity e a métrica de "foco na pergunta" da Precision.

**Como foi aplicado:**

```
# RACIOCÍNIO (Chain of Thought - faça internamente, NÃO inclua na resposta)
Antes de escrever, pense passo a passo:
1. PERSONA: quem é afetado pelo bug? ...
2. AÇÃO: o que essa persona QUER conseguir fazer ...
3. BENEFÍCIO: qual valor real ela obtém com isso?
4. COMPLEXIDADE: classifique o relato em NÍVEL 1, 2 ou 3 ...
5. FATOS: liste todos os dados concretos do relato (IDs, endpoints, valores, números...)
   Cada um deles precisa aparecer na saída, de forma resumida.
6. ESCRITA: preencha o esqueleto do nível escolhido e revise: nada inventado, nada omitido.
```

### 4. Skeleton of Thought (esqueleto de resposta por complexidade)

**Por quê:** a maior fonte de erro em prompts zero-shot é o tamanho/estrutura errados: bugs simples com seções demais (Precision penaliza "informações não solicitadas") e bugs complexos com seções de menos (F1 penaliza omissões). O esqueleto fixa a estrutura exata de cada nível, tornando a saída previsível e comparável com a referência.

**Como foi aplicado:** o prompt define critérios objetivos de classificação (NÍVEL 1/2/3) e um esqueleto para cada nível, por exemplo:

```
## NÍVEL 1 - SIMPLES (use EXATAMENTE esta estrutura, sem seções extras)
Como um [persona específica], eu quero [ação/objetivo], para que [benefício].

Critérios de Aceitação:
- Dado que [contexto/pré-condição]
- Quando [ação do usuário ou evento]
- Então [resultado esperado principal]
- E [resultado complementar]
```

### Requisitos adicionais atendidos

- **Instruções claras e específicas:** tarefa definida em uma frase, com idioma, formato e estrutura.
- **Regras explícitas de comportamento:** 9 regras obrigatórias (persona específica, linguagem positiva, Gherkin, fidelidade ao relato, nunca inventar, proporcionalidade, formato Markdown simples, responder só com a story, português).
- **Tratamento de edge cases:** relato vago, vários problemas curtos, logs/stack traces, relato que já traz a causa, bugs de segurança, pedido de melhoria disfarçado e texto que não é um bug.
- **System vs User prompt:** o `system_prompt` concentra papel, raciocínio, esqueletos, regras e exemplos; o `user_prompt` contém apenas o relato (`{bug_report}`) e a instrução de execução. A duplicação da variável do v1 foi eliminada.

## Processo de otimização (iterações)

<!-- PREENCHER após rodar `python src/evaluate.py`: copie as notas de cada execução. -->

| Iteração | Mudança principal | Helpfulness | Correctness | F1 | Clarity | Precision | Status |
|----------|-------------------|-------------|-------------|----|---------|-----------|--------|
| v1 (baseline) | Prompt original do Hub | - | - | - | - | - | - |
| v2.0 | Role + Few-shot + CoT + Skeleton, regras e edge cases | - | - | - | - | - | - |

Como analisar uma iteração reprovada: abra o experimento no LangSmith, ordene pela métrica mais baixa, leia o `comment` do juiz (campo *reasoning*) e o tracing do exemplo. Ajuste o prompt em `prompts/bug_to_user_story_v2.yml`, rode `python src/push_prompts.py` e depois `python src/evaluate.py` novamente.

## Resultados Finais

### Link público do dataset de avaliação (com os experimentos)

<!-- PREENCHER com a saída de `python src/share_dataset.py` -->

Dataset `mba-ia-pull-evaluation-prompt-eval` (15 exemplos): **[COLE AQUI O LINK PÚBLICO]**

Prompt otimizado no Hub: `https://smith.langchain.com/hub/{USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2`

### Screenshots das avaliações

<!-- PREENCHER: salve as imagens em docs/evidencias/ (ver docs/evidencias/README.md) -->

| Evidência | Imagem |
|-----------|--------|
| Terminal com STATUS: APROVADO (todas >= 0.8) | ![evaluate](docs/evidencias/01-terminal-evaluate.png) |
| Dataset com 15 exemplos | ![dataset](docs/evidencias/02-dataset-15-exemplos.png) |
| Experimento v2 com as 5 métricas | ![experimento](docs/evidencias/03-experimento-v2-metricas.png) |
| Tracing detalhado - exemplo simples | ![tracing1](docs/evidencias/04-tracing-exemplo-1.png) |
| Tracing detalhado - exemplo médio | ![tracing2](docs/evidencias/05-tracing-exemplo-2.png) |
| Tracing detalhado - exemplo complexo | ![tracing3](docs/evidencias/06-tracing-exemplo-3.png) |
| Prompt v2 publicado no Hub | ![hub](docs/evidencias/07-prompt-hub-v2.png) |

### Comparação v1 x v2: o que mudou e por quê

| Aspecto | v1 (original) | v2 (otimizado) | Por quê |
|---------|---------------|----------------|---------|
| Persona | "assistente que ajuda a transformar relatos" | Product Manager sênior em times ágeis | Tom profissional, foco em valor e na persona da story |
| Variável `{bug_report}` | No system **e** no user prompt | Somente no user prompt | Elimina duplicação e separa instruções (system) de dados (user) |
| Formato de saída | Não definido ("crie uma user story") | "Como um / eu quero / para que" + critérios Dado/Quando/Então, Markdown simples | Saída comparável à referência; Clarity e F1 sobem |
| Exemplos | Nenhum | 3 exemplos (simples, médio, complexo) | Modelo copia o formato e o nível de detalhe correto |
| Raciocínio | Nenhum | 6 passos internos (persona, ação, benefício, complexidade, fatos, escrita) | Menos omissões e menos invenções |
| Estrutura por complexidade | Única | 3 esqueletos (NÍVEL 1/2/3) | Bugs simples ficam enxutos; complexos ganham contexto, critérios técnicos e tasks |
| Regras | Nenhuma | 9 regras obrigatórias | Persona específica, fidelidade aos dados, nunca inventar, responder só a story |
| Edge cases | Nenhum | 7 casos tratados | Comportamento previsível em relatos vagos, logs, segurança, não-bug |
| Metadados | version, tags | + techniques_applied, changelog, author | Rastreabilidade no Hub e nos testes |

## Como Executar

### Pré-requisitos

- Python 3.10+ (recomendado 3.12; versões muito novas podem não ter wheels para todas as dependências)
- Conta no [LangSmith](https://smith.langchain.com) com API Key
- API Key da [OpenAI](https://platform.openai.com/api-keys) (custo estimado: < US$ 1 por avaliação com `gpt-4.1-mini`)
- Handle público do LangSmith Hub (ver passo 3)

### 1. Clonar e preparar o ambiente

```bash
git clone https://github.com/GabrielStedile999/mba-ia-pull-evaluation-prompt.git
cd mba-ia-pull-evaluation-prompt

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Preencha no `.env`:

```
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=mba-ia-pull-evaluation-prompt
USERNAME_LANGSMITH_HUB=seu-handle
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
EVAL_MODEL=gpt-4.1-mini
```

### 3. Criar o handle do LangSmith Hub (uma única vez)

O handle só existe depois que você torna algum prompt público:

1. LangSmith → **Prompts** → crie um prompt qualquer (pode ser de teste)
2. Clique nos três pontinhos (ao lado de *Playground*) → **Make Public**
3. Em *Choose your public handle*, defina o handle (é definitivo)
4. Coloque o handle em `USERNAME_LANGSMITH_HUB` no `.env`

### 4. Fase 1 - Pull do prompt original

```bash
python src/pull_prompts.py
```

Faz pull de `leonanluppi/bug_to_user_story_v1` e salva em `prompts/bug_to_user_story_v1.yml`.

### 5. Fase 2 - Prompt otimizado

O prompt otimizado já está em `prompts/bug_to_user_story_v2.yml`. Para iterar, edite esse arquivo e valide com os testes (passo 8).

### 6. Fase 3 - Push do prompt otimizado

```bash
python src/push_prompts.py
```

Valida o YAML, monta o `ChatPromptTemplate` (system + user), publica como **público** em `{USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2` com descrição, tags e técnicas nos metadados, e imprime a URL do prompt.

### 7. Fase 4 - Avaliação

```bash
python src/evaluate.py
```

Cria (ou reutiliza) o dataset `mba-ia-pull-evaluation-prompt-eval` com os 15 exemplos, puxa o prompt v2 do Hub, roda o experimento no LangSmith, grava as 5 notas como feedback e imprime o resumo com o link do experimento. Repita os passos 5 → 6 → 7 até todas as métricas ficarem >= 0.8.

### 8. Testes de validação do prompt

```bash
pytest tests/test_prompts.py -v
```

### 9. Link público do dataset (evidência)

```bash
python src/share_dataset.py
```

Imprime o link público do dataset, que expõe também os experimentos. Rode uma vez e guarde o endereço.

## Testes de validação

`tests/test_prompts.py` implementa os 6 testes obrigatórios e 4 extras:

| Teste | O que verifica |
|-------|----------------|
| `test_prompt_has_system_prompt` | Campo existe e não está vazio |
| `test_prompt_has_role_definition` | Persona definida ("Você é um Product Manager...") |
| `test_prompt_mentions_format` | Exige Markdown / formato padrão de User Story e critérios Dado/Quando/Então |
| `test_prompt_has_few_shot_examples` | Ao menos 2 pares de Entrada/Saída |
| `test_prompt_no_todos` | Nenhum `[TODO]` ou `TODO` esquecido |
| `test_minimum_techniques` | `techniques_applied` com >= 2 técnicas, incluindo Few-shot |
| `test_prompt_structure_is_valid` (extra) | `validate_prompt_structure` de `utils.py` |
| `test_bug_report_variable_only_in_user_prompt` (extra) | `{bug_report}` só no user prompt |
| `test_prompt_template_compiles` (extra) | `ChatPromptTemplate` compila com apenas `bug_report` |
| `test_prompt_handles_edge_cases` (extra) | Seção de edge cases presente |

## Solução de problemas

| Erro | Causa / solução |
|------|-----------------|
| `Variáveis de ambiente faltando` | Confira o `.env` na raiz do projeto e rode os scripts a partir da raiz |
| Push retorna 403/404 ou erro de handle | Crie o handle (passo 3) e confira `USERNAME_LANGSMITH_HUB` |
| `Nothing to commit` no push | O conteúdo não mudou desde o último push; apenas os metadados foram atualizados |
| Erro 400 de `temperature` | O modelo escolhido não aceita `temperature=0`; use `gpt-4.1-mini` / `gpt-4.1` |
| Prompt não encontrado no evaluate | Rode `python src/push_prompts.py` antes de avaliar |
