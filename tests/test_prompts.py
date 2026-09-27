"""
Testes automatizados para validação do prompt otimizado (v2).

Executar: pytest tests/test_prompts.py -v
"""
import re
import sys
from pathlib import Path

import pytest
import yaml

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure  # noqa: E402

PROMPTS_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompts():
    data = load_prompts(str(PROMPTS_FILE))
    assert data, f"Arquivo vazio ou inválido: {PROMPTS_FILE}"
    return data


@pytest.fixture(scope="module")
def prompt(prompts):
    assert PROMPT_KEY in prompts, f"Chave '{PROMPT_KEY}' não encontrada no YAML"
    return prompts[PROMPT_KEY]


@pytest.fixture(scope="module")
def full_text(prompt):
    """system_prompt + user_prompt concatenados (texto completo enviado ao modelo)."""
    return f"{prompt.get('system_prompt', '')}\n{prompt.get('user_prompt', '')}"


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt, "Campo 'system_prompt' não existe"
        system_prompt = prompt["system_prompt"]
        assert isinstance(system_prompt, str), "'system_prompt' deve ser uma string"
        assert system_prompt.strip(), "'system_prompt' está vazio"
        assert len(system_prompt.strip()) > 200, "'system_prompt' é curto demais para um prompt otimizado"

    def test_prompt_has_role_definition(self, prompt):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt["system_prompt"]
        role_pattern = re.compile(
            r"(você é (um|uma)|you are (a|an)|atue como|act as)\s+[^\n]{3,}",
            re.IGNORECASE,
        )
        assert role_pattern.search(system_prompt), "Prompt não define uma persona (ex.: 'Você é um Product Manager')"
        assert re.search(r"product manager|product owner|analista de produto", system_prompt, re.IGNORECASE), (
            "A persona deve estar relacionada a produto (ex.: Product Manager)"
        )

    def test_prompt_mentions_format(self, prompt):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = prompt["system_prompt"]
        mentions_markdown = "markdown" in system_prompt.lower()
        mentions_user_story = re.search(
            r"como um.*eu quero.*para que", system_prompt, re.IGNORECASE | re.DOTALL
        )
        assert mentions_markdown or mentions_user_story, (
            "Prompt deve exigir formato Markdown ou o formato padrão de User Story"
        )
        assert re.search(r"crit[ée]rios de aceita[çc][ãa]o", system_prompt, re.IGNORECASE), (
            "Prompt deve exigir Critérios de Aceitação"
        )
        assert re.search(r"dado que.*quando.*então", system_prompt, re.IGNORECASE | re.DOTALL), (
            "Prompt deve exigir critérios no formato Dado/Quando/Então"
        )

    def test_prompt_has_few_shot_examples(self, prompt):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt["system_prompt"]
        assert re.search(r"exemplo", system_prompt, re.IGNORECASE), "Prompt não contém a seção de exemplos"

        entradas = re.findall(r"^\s*entrada:", system_prompt, re.IGNORECASE | re.MULTILINE)
        saidas = re.findall(r"^\s*sa[íi]da:", system_prompt, re.IGNORECASE | re.MULTILINE)

        assert len(entradas) >= 2, f"Few-shot exige ao menos 2 exemplos de entrada, encontrados: {len(entradas)}"
        assert len(saidas) >= 2, f"Few-shot exige ao menos 2 exemplos de saída, encontrados: {len(saidas)}"
        assert len(entradas) == len(saidas), "Cada exemplo precisa ter uma entrada e uma saída correspondente"

    def test_prompt_no_todos(self, prompt, full_text):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        assert "[TODO]" not in full_text, "Prompt ainda contém marcador [TODO]"
        assert not re.search(r"\bTODO\b", full_text), "Prompt ainda contém a palavra TODO"
        assert "[TODO]" not in str(prompt.get("description", "")), "Descrição contém [TODO]"

    def test_minimum_techniques(self, prompt):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt.get("techniques_applied")
        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, f"Mínimo de 2 técnicas, encontradas: {len(techniques)}"
        assert all(isinstance(t, str) and t.strip() for t in techniques), "Técnicas devem ser strings não vazias"

        # Few-shot é obrigatório pelo desafio
        assert any("few-shot" in t.lower() or "few shot" in t.lower() for t in techniques), (
            "Few-shot Learning é obrigatório e deve estar listado em techniques_applied"
        )

    # ------------------------------------------------------------------
    # Testes extras (além dos 6 obrigatórios)
    # ------------------------------------------------------------------

    def test_prompt_structure_is_valid(self, prompt):
        """Valida a estrutura usando a função utilitária do projeto."""
        is_valid, errors = validate_prompt_structure(prompt)
        assert is_valid, f"Estrutura inválida: {errors}"

    def test_bug_report_variable_only_in_user_prompt(self, prompt):
        """A variável {bug_report} deve estar no user_prompt e NÃO no system_prompt (bug do v1)."""
        assert "{bug_report}" in prompt.get("user_prompt", ""), "user_prompt deve conter {bug_report}"
        assert "{bug_report}" not in prompt["system_prompt"], "system_prompt não deve conter {bug_report}"

    def test_prompt_template_compiles(self, prompt):
        """O template deve compilar no ChatPromptTemplate com apenas a variável bug_report."""
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [("system", prompt["system_prompt"]), ("user", prompt["user_prompt"])]
        )
        assert template.input_variables == ["bug_report"], (
            f"Variáveis inesperadas no template: {template.input_variables}"
        )
        messages = template.format_messages(bug_report="Botão não funciona.")
        assert len(messages) == 2

    def test_prompt_handles_edge_cases(self, prompt):
        """O prompt deve tratar explicitamente casos especiais (edge cases)."""
        system_prompt = prompt["system_prompt"].lower()
        assert "edge case" in system_prompt or "casos especiais" in system_prompt
        assert "vago" in system_prompt or "incompleto" in system_prompt, "Deve tratar relatos vagos/incompletos"
        assert "não invente" in system_prompt or "nunca invente" in system_prompt, "Deve proibir alucinações"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
