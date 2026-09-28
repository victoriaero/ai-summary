# Google AI Overview Study

Este repositório organiza a seleção de outcomes, a geração determinística de queries e a coleta manual de respostas do Google AI Overview. Não há uso de LLM no pipeline.

A query usada em todas as condições é:

```text
What factors influence [OUTCOME] for [GROUP] in [DOMAIN]?
```

## Estrutura

```text
.
├── artifacts/
│   ├── annotation_inputs/       # planilhas originais dos anotadores
│   └── annotation_results/      # concordância e top 3 por domínio
├── annotations/
│   ├── annotator_1/
│   │   └── google_aio_collection/
│   ├── annotator_2/
│   │   └── google_aio_collection/
│   └── annotator_3/
│       └── google_aio_collection/
├── google_aio_collection/       # fonte original preservada
├── scripts/
│   ├── generate_queries.py      # gera os formulários das três coletas
│   └── gwet.py                  # calcula concordância e seleciona outcomes
└── requirements.txt
```

Cada pasta `google_aio_collection` contém uma subpasta por grupo, 21 arquivos de coleta por grupo e um `query_manifest.csv`. São 273 queries para cada anotador: 13 grupos/condições × 7 domínios × 3 outcomes.

A pasta `google_aio_collection/` da raiz é mantida como fonte original. A coleta manual deve ser feita somente nas três cópias dentro de `annotations/`.

## Instalação

Requer Python 3.10 ou mais recente.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Os scripts resolvem os caminhos a partir da raiz do repositório e podem ser executados de qualquer diretório.

## 1. Concordância e seleção dos outcomes

Os CSVs originais ficam em `artifacts/annotation_inputs/`. Para recalcular a concordância e selecionar os três outcomes de cada domínio:

```bash
python scripts/gwet.py
```

O script grava:

- `artifacts/annotation_results/annotator_agreement_full.csv`;
- `artifacts/annotation_results/selected_top3_outcomes.csv`.

## 2. Preparar a coleta manual

```bash
python scripts/generate_queries.py
```

O script prepara três cópias independentes em `annotations/annotator_1`, `annotator_2` e `annotator_3`. Arquivos `.txt` já existentes são preservados para não apagar coleta manual; apenas arquivos ausentes são criados. O manifesto de cada anotador é atualizado com a lista completa de queries.

Cada arquivo contém a query, 15 espaços para links, uma área para o texto do AI Overview, dados da sessão de coleta e metadados que não devem ser editados.

## Fluxo recomendado

1. Mantenha as planilhas originais em `artifacts/annotation_inputs/`.
2. Execute `gwet.py` quando precisar recalcular a seleção de outcomes.
3. Execute `generate_queries.py` antes do início da coleta.
4. Distribua uma pasta de `annotations/annotator_N/` para cada anotador.
5. Durante a coleta, edite somente a cópia correspondente ao anotador.

Evite copiar pastas manualmente depois que a coleta começar. O gerador não sobrescreve arquivos existentes, mas cada pasta deve continuar atribuída a uma única pessoa.
