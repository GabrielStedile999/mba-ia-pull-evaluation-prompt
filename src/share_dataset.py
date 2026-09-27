"""
Gera (ou recupera) o link PÚBLICO do dataset de avaliação no LangSmith.

O link impresso por src/evaluate.py só abre para quem tem acesso ao workspace.
Compartilhar o dataset expõe junto os experimentos rodados contra ele, o que
serve como evidência pública para o desafio.

Uso:
    python src/share_dataset.py

Atenção: rode uma vez e guarde o endereço. Ao compartilhar de novo, o link muda.
"""

import os
import sys

from dotenv import load_dotenv
from langsmith import Client

from utils import check_env_vars, print_section_header

load_dotenv()


def main():
    print_section_header("COMPARTILHAR DATASET DE AVALIAÇÃO")

    if not check_env_vars(["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]):
        return 1

    dataset_name = f"{os.getenv('LANGSMITH_PROJECT')}-eval"
    client = Client()

    try:
        # Se já foi compartilhado, reaproveita o link existente (não gera um novo)
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        dataset = next((d for d in datasets if d.name == dataset_name), None)
        if dataset is None:
            print(f"❌ Dataset '{dataset_name}' não encontrado. Rode antes: python src/evaluate.py")
            return 1

        try:
            shared = client.read_dataset_shared_schema(dataset_id=dataset.id)
            print(f"✓ Dataset já estava compartilhado.")
        except Exception:  # noqa: BLE001 - ainda não compartilhado
            shared = client.share_dataset(dataset_id=dataset.id)
            print("✓ Dataset compartilhado publicamente.")

        url = shared["url"] if isinstance(shared, dict) else getattr(shared, "url", shared)
        print(f"\nLink público do dataset ({dataset_name}):\n  {url}\n")
        print("Cole este link na seção 'Resultados Finais' do README.md.")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"❌ Falha ao compartilhar o dataset: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
