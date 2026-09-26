# AIO como mecanismo de governança da informação

Pipeline auditável para gerar templates de busca neutros, congelar uma escolha humana e criar condições demográficas pareadas. A geração usa o placeholder literal `{GROUP}` e nunca recebe o catálogo de grupos.

## Estrutura

- `artifacts/config.yaml`: configuração operacional e caminhos do pipeline.
- `artifacts/study_input.example.json`: outcomes e catálogo demográfico de exemplo.
- `artifacts/prompts/`: system prompt e template do user prompt.
- `artifacts/runs/<run_id>/`: artefatos imutáveis de cada execução.
- `scripts/`: geração, congelamento e expansão.

O arquivo `study_input.example.json` demonstra o contrato, mas não representa o catálogo final do estudo.

## Instalação

Requer Python 3.10 ou mais recente:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Execute os comandos a partir da raiz do repositório.

## Configuração

Todos os comandos usam `artifacts/config.yaml`. Caminhos relativos são resolvidos a partir da pasta em que o YAML está salvo, independentemente do diretório corrente.

Antes da primeira geração:

1. copie ou renomeie `study_input.example.json` quando os dados finais estiverem disponíveis;
2. ajuste `paths.study_input`;
3. preencha `model.id` com o ID do modelo instruction/chat no Hugging Face;
4. preferencialmente fixe `model.revision` em um commit para reprodução.

Os principais campos são:

```yaml
paths:
  study_input: study_input.example.json
  system_prompt: prompts/system_prompt.txt
  user_prompt: prompts/user_prompt.txt
  runs_dir: runs

pipeline:
  prompt_version: v1
  candidates_per_outcome: 5

model:
  id: null  # preencher antes da geração
  revision: null
  tokenizer_id: null  # null reutiliza model.id
  tokenizer_revision: null
  device: auto
  dtype: auto
  trust_remote_code: false
  tokenizer_use_fast: true

generation:
  seed: 42
  parameters:
    do_sample: true
    temperature: 0.7
    top_p: 0.9
    repetition_penalty: 1.0
    max_new_tokens: 768
```

`trust_remote_code` deve permanecer `false`, exceto quando o modelo escolhido realmente exigir código customizado e esse código tiver sido revisado.

O template de user prompt usa variáveis no formato `$nome`. As variáveis disponíveis são `$candidate_count`, `$domain`, `$outcome_id`, `$outcome`, `$definition`, `$geographic_scope`, `$query_intent`, `$candidate_id_start` e `$candidate_id_end`. O placeholder demográfico continua sendo o texto literal `{GROUP}`.

## 1. Gerar candidatos

```bash
python -m scripts.generate_candidates \
  --config artifacts/config.yaml \
  --run-id pilot-001
```

A execução grava `manifest.json`, `raw_generations.jsonl`, `candidates.jsonl`, `candidates.csv` e um esqueleto de `selection.json`. O manifesto preserva snapshots do YAML e do JSON, caminhos dos prompts, modelo, tokenizer, revisions, hiperparâmetros e versões das dependências. Os prompts renderizados também ficam registrados em cada geração.

Se a resposta do modelo for inválida, a resposta bruta permanece em `raw_generations.jsonl`, o manifesto marca a geração como falha e o comando termina com erro.

## 2. Selecionar e congelar

Edite `artifacts/runs/pilot-001/selection.json` e informe exatamente um candidato por outcome:

```json
{
  "outcome_id": "HOU-02",
  "selected_candidate_id": "HOU-02-Q01",
  "final_template": null,
  "notes": "Formulação mais natural e neutra."
}
```

Com `final_template: null`, o texto original é mantido. Para uma correção manual, informe o texto completo preservando exatamente um `{GROUP}`.

```bash
python -m scripts.freeze_templates \
  --config artifacts/config.yaml \
  --run-id pilot-001
```

## 3. Expandir condições

```bash
python -m scripts.expand_queries \
  --config artifacts/config.yaml \
  --run-id pilot-001
```

O comando produz `queries.jsonl` e `queries.csv`. Cada variação linguística gera uma condição distinta; o controle `people` aparece uma única vez por outcome.

## Limites desta versão

O pipeline verifica integridade estrutural: YAML/JSON válidos, cinco candidatos com IDs esperados, um `{GROUP}` por template, uma seleção por outcome e substituição completa. Neutralidade e equivalência semântica continuam sendo decisões de pesquisa. A consolidação e concordância dos CSVs de anotadores também ficam fora do escopo.

## Testes

Os testes usam um gerador simulado e não baixam modelos:

```bash
pytest -q
```
