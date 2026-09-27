"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull do prompt semente do desafio (leonanluppi/bug_to_user_story_v1)
3. Salva localmente em prompts/bug_to_user_story_v1.yml
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# Prompt semente do desafio (dono explícito -> exige dangerously_pull_public_prompt=True)
SEED_PROMPT_IDENTIFIER = "leonanluppi/bug_to_user_story_v1"
LOCAL_PROMPT_KEY = "bug_to_user_story_v1"
OUTPUT_PATH = Path("prompts") / "bug_to_user_story_v1.yml"


def _extract_message_content(message) -> str:
    """
    Extrai o texto de uma mensagem do ChatPromptTemplate.

    Mensagens podem ser templates (SystemMessagePromptTemplate,
    HumanMessagePromptTemplate -> `.prompt.template`) ou mensagens fixas
    (SystemMessage, HumanMessage -> `.content`).
    """
    prompt = getattr(message, "prompt", None)
    if prompt is not None:
        # Template simples: .prompt.template
        template = getattr(prompt, "template", None)
        if isinstance(template, str):
            return template
        # Template composto (lista de prompts, ex.: multimodal)
        if isinstance(prompt, list):
            return "\n".join(
                getattr(p, "template", "") for p in prompt if getattr(p, "template", None)
            )

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    return str(message)


def _message_role(message) -> str:
    """Identifica o papel (system/user/ai) de uma mensagem do template."""
    name = type(message).__name__.lower()
    if "system" in name:
        return "system"
    if "human" in name or "user" in name:
        return "user"
    if "ai" in name or "assistant" in name:
        return "ai"
    return "other"


def prompt_template_to_dict(prompt) -> dict:
    """
    Converte um ChatPromptTemplate em um dicionário serializável em YAML,
    seguindo a mesma estrutura usada em prompts/bug_to_user_story_v1.yml.
    """
    system_parts, user_parts, other_parts = [], [], []

    for message in getattr(prompt, "messages", []):
        role = _message_role(message)
        content = _extract_message_content(message)

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        else:
            other_parts.append(f"[{role}] {content}")

    metadata = getattr(prompt, "metadata", None) or {}

    data = {
        "description": metadata.get(
            "description", "Prompt para converter relatos de bugs em User Stories"
        ),
        "system_prompt": "\n\n".join(system_parts),
        "user_prompt": "\n\n".join(user_parts) or "{bug_report}",
        "version": "v1",
        "input_variables": list(getattr(prompt, "input_variables", []) or []),
        "tags": ["bug-analysis", "user-story", "product-management"],
        "source": SEED_PROMPT_IDENTIFIER,
        "pulled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if other_parts:
        data["other_messages"] = other_parts

    return data


def pull_prompts_from_langsmith() -> bool:
    """
    Faz pull do prompt semente do LangSmith Hub e salva em YAML.

    Returns:
        True se sucesso, False caso contrário
    """
    client = Client()

    print(f"Fazendo pull do prompt: {SEED_PROMPT_IDENTIFIER}")

    try:
        prompt = client.pull_prompt(
            SEED_PROMPT_IDENTIFIER,
            dangerously_pull_public_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Falha ao fazer pull do prompt: {exc}")
        print("\nVerifique:")
        print("- LANGSMITH_API_KEY está correta no .env")
        print("- Sua conexão com a internet está funcionando")
        return False

    print(f"   ✓ Prompt carregado ({type(prompt).__name__})")

    data = prompt_template_to_dict(prompt)

    print("\nConteúdo do prompt:")
    print("-" * 50)
    print(f"[system]\n{data['system_prompt']}\n")
    print(f"[user]\n{data['user_prompt']}")
    print("-" * 50)

    if not save_yaml({LOCAL_PROMPT_KEY: data}, str(OUTPUT_PATH)):
        return False

    print(f"\n✓ Prompt salvo em: {OUTPUT_PATH}")
    return True


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    if not pull_prompts_from_langsmith():
        return 1

    print("\nPróximos passos:")
    print("1. Analise o prompt em prompts/bug_to_user_story_v1.yml")
    print("2. Crie a versão otimizada em prompts/bug_to_user_story_v2.yml")
    print("3. Faça push: python src/push_prompts.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
