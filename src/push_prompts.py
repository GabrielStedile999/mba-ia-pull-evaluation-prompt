"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub ({USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2)
4. Adiciona metadados (tags, descrição, técnicas utilizadas)
"""

import os
import re
import sys

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from langchain_core.prompts import ChatPromptTemplate

from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

PROMPTS_FILE = "prompts/bug_to_user_story_v2.yml"
REQUIRED_INPUT_VARIABLE = "bug_report"


def _slugify_tag(value: str) -> str:
    """Normaliza uma tag para o formato aceito pelo Hub (minúsculas, hífens)."""
    return (
        value.strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
    )


def build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    """Monta o ChatPromptTemplate (system + user) a partir do YAML."""
    system_prompt = prompt_data["system_prompt"].strip()
    user_prompt = prompt_data.get("user_prompt", "{bug_report}").strip()

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )


def build_description(prompt_data: dict) -> str:
    """Descrição publicada no Hub, incluindo as técnicas utilizadas."""
    techniques = prompt_data.get("techniques_applied", [])
    description = prompt_data.get("description", "").strip()
    if techniques:
        description += f" | Técnicas: {', '.join(techniques)}"
    return description[:1000]


def build_tags(prompt_data: dict) -> list[str]:
    """Tags do prompt: tags do YAML + técnicas aplicadas + versão."""
    tags = [_slugify_tag(t) for t in prompt_data.get("tags", [])]
    tags += [_slugify_tag(t) for t in prompt_data.get("techniques_applied", [])]
    tags.append(_slugify_tag(prompt_data.get("version", "v2")))
    # Remove duplicados preservando a ordem
    return list(dict.fromkeys(t for t in tags if t))


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt (ex.: bug_to_user_story_v2)
        prompt_data: Dados do prompt (conteúdo do YAML)

    Returns:
        True se sucesso, False caso contrário
    """
    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    identifier = f"{username}/{prompt_name}"

    client = Client()
    chat_prompt = build_chat_prompt(prompt_data)
    description = build_description(prompt_data)
    tags = build_tags(prompt_data)
    techniques = prompt_data.get("techniques_applied", [])

    readme = prompt_data.get("readme") or (
        f"# {prompt_name}\n\n{prompt_data.get('description', '')}\n\n"
        f"## Técnicas aplicadas\n" + "\n".join(f"- {t}" for t in techniques)
    )

    print(f"Fazendo push: {identifier}")
    print(f"   Variáveis de entrada: {chat_prompt.input_variables}")
    print(f"   Tags: {', '.join(tags)}")

    try:
        url = client.push_prompt(
            identifier,
            object=chat_prompt,
            is_public=True,
            description=description,
            readme=readme,
            tags=tags,
            commit_description=prompt_data.get("changelog", "Prompt otimizado (v2)")[:1000],
        )
        print(f"   ✓ Push concluído: {url}")
        return True

    except LangSmithConflictError:
        # O conteúdo é idêntico ao último commit: metadados foram atualizados,
        # mas não há nada novo para commitar.
        print("   ✓ Prompt já está atualizado no Hub (nenhuma alteração de conteúdo).")
        print(f"   ✓ https://smith.langchain.com/prompts/{prompt_name}")
        return True

    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        print(f"❌ Falha ao fazer push do prompt: {message}")

        lowered = message.lower()
        if "handle" in lowered or "public" in lowered or "403" in lowered or "404" in lowered:
            print("\nDica: para publicar prompts públicos é preciso ter um handle do Hub.")
            print("Veja as instruções em .env.example (USERNAME_LANGSMITH_HUB).")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    if not isinstance(prompt_data, dict):
        return False, ["Prompt inválido: esperado um dicionário"]

    for field in ("description", "system_prompt", "user_prompt", "version"):
        if not str(prompt_data.get(field, "")).strip():
            errors.append(f"Campo obrigatório faltando ou vazio: {field}")

    system_prompt = str(prompt_data.get("system_prompt", ""))
    user_prompt = str(prompt_data.get("user_prompt", ""))

    if re.search(r"\[TODO\]|\bTODO\b", system_prompt + user_prompt):
        errors.append("Prompt ainda contém marcadores TODO")

    if f"{{{REQUIRED_INPUT_VARIABLE}}}" not in user_prompt:
        errors.append(f"user_prompt precisa conter a variável {{{REQUIRED_INPUT_VARIABLE}}}")

    if f"{{{REQUIRED_INPUT_VARIABLE}}}" in system_prompt:
        errors.append(
            f"system_prompt não deve conter {{{REQUIRED_INPUT_VARIABLE}}} "
            "(a variável só deve aparecer no user_prompt)"
        )

    techniques = prompt_data.get("techniques_applied", [])
    if not isinstance(techniques, list) or len(techniques) < 2:
        errors.append(
            f"Mínimo de 2 técnicas requeridas em techniques_applied, encontradas: "
            f"{len(techniques) if isinstance(techniques, list) else 0}"
        )

    # Garante que o template compila (chaves não balanceadas quebram o push)
    if not errors:
        try:
            chat_prompt = build_chat_prompt(prompt_data)
            extra_vars = set(chat_prompt.input_variables) - {REQUIRED_INPUT_VARIABLE}
            if extra_vars:
                errors.append(
                    f"Variáveis inesperadas no template: {sorted(extra_vars)}. "
                    "Use chaves duplas {{ }} para texto literal."
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Template inválido: {exc}")

    return (len(errors) == 0, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS PARA O LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    prompts = load_yaml(PROMPTS_FILE)
    if not prompts:
        print(f"❌ Nenhum prompt encontrado em {PROMPTS_FILE}")
        return 1

    all_ok = True
    pushed = []

    for prompt_name, prompt_data in prompts.items():
        print(f"\n▶ Prompt: {prompt_name}")

        is_valid, errors = validate_prompt(prompt_data)
        if not is_valid:
            all_ok = False
            print("❌ Prompt inválido:")
            for error in errors:
                print(f"   - {error}")
            continue

        print("   ✓ Validação OK")

        if push_prompt_to_langsmith(prompt_name, prompt_data):
            pushed.append(f"{os.getenv('USERNAME_LANGSMITH_HUB')}/{prompt_name}")
        else:
            all_ok = False

    print("\n" + "=" * 50)
    print(f"Prompts publicados: {len(pushed)}")
    for name in pushed:
        print(f"  - {name}")

    if all_ok:
        print("\nPróximos passos:")
        print("1. Confira o prompt em https://smith.langchain.com/prompts")
        print("2. Execute a avaliação: python src/evaluate.py")
        return 0

    print("\n⚠️  Alguns prompts não foram publicados. Corrija os erros acima e tente novamente.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
