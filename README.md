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

## Conjunto de robustez: um novo outcome por domínio

Para selecionar o outcome mais bem ranqueado fora do Top 3 original em cada um
dos sete domínios e preparar as 13 condições nas três localidades:

```bash
python scripts/generate_robustness_queries.py --dry-run
python scripts/generate_robustness_queries.py
```

O primeiro comando apenas mostra a seleção. O segundo cria 91 arquivos por
localidade (273 no total) em `annotations/robustness/v1_dallas/`,
`annotations/robustness/v2_ny/` e `annotations/robustness/v3_la/`, além do
manifesto de seleção. Arquivos `.txt` existentes são preservados; se o
manifesto existente corresponder a outra seleção, o script encerra sem alterar
a coleta.

Depois de preencher uma localidade, valide-a com, por exemplo:

```bash
python scripts/consistency.py \
  --collection-dir annotations/robustness/v1_dallas/google_aio_collection
```

O validador infere automaticamente pelo manifesto que são esperados sete
arquivos por grupo.

## Análise exploratória de condições e fontes

Para analisar presença de AIO, quantidade e composição das fontes, tamanho do
texto e overlap pareado entre condições:

```bash
python scripts/analyze_condition_sources.py
```

Por padrão, o script analisa `annotations/v1_dallas/google_aio_collection/` e
grava os resultados em `results/condition_source_analysis_dallas/`. Outra
coleta pode ser informada explicitamente:

```bash
python scripts/analyze_condition_sources.py \
  --collection-dir annotations/v2_ny/google_aio_collection \
  --output-dir results/condition_source_analysis_ny
```

O overlap de fontes é calculado por URL normalizada e por domínio para
`minority × majority`, `minority × generic` e `majority × generic`. As
saídas incluem resultados por query e agregações por dimensão social, domínio
do estudo e outcome.

O overlap também pode ser analisado com o overlap coefficient e em três
granularidades: URL canônica, domínio registrável e categoria da fonte. Depois
de gerar os CSVs, produza as figuras com:

```bash
python scripts/plot_condition_source_analysis.py
```

As figuras exploratórias são gravadas apenas em PNG dentro de
`results/condition_source_analysis_dallas/figures/`.

## Autoridade e substituição de fontes

O piloto de taxonomia automática classifica o papel de autoridade e o setor
institucional de cada fonte. Ele também decompõe os conjuntos pareados em
fontes compartilhadas, exclusivas da condição A e exclusivas da condição B:

```bash
python scripts/analyze_source_authority_turnover.py
```

São analisados `minority × majority`, `minority × generic` e
`majority × generic`, tanto por URL canônica quanto por domínio registrável.
As regras acionadas e sua confiança heurística são exportadas junto com cada
fonte. Os resultados e gráficos PNG ficam em
`results/source_authority_turnover_dallas/`.

O piloto também calcula expansão líquida normalizada, turnover, intervalos de
bootstrap pareado por outcome, modelos binários de categoria ajustados por
outcome, uma análise hierárquica complementar com intercepto aleatório por
outcome e a associação entre turnover documental e mudança na composição de
autoridade. Uma amostra estratificada sem rótulos humanos é exportada para
validação posterior da taxonomia.

As classificações são exploratórias e inteiramente automáticas. O controle
`people` é compartilhado pelas seis dimensões sociais; portanto, os contrastes
com controle não devem ser tratados como seis replicações independentes.

## Análise DIF-inspired por outcome

Para identificar outcomes cujo contraste minoria–maioria se afasta do padrão
geral da respectiva dimensão:

```bash
python scripts/dif_inspired_item_analysis.py --metric n_sources
```

Com apenas Dallas, o script produz contrastes, heterogeneidade entre outcomes,
escores robustos e rankings de candidatos. Intervalos específicos por item e
modelos de slope aleatório são deliberadamente adiados. Quando três ambientes
estiverem disponíveis, passe cada CSV:

```bash
python scripts/dif_inspired_item_analysis.py \
  --environment dallas=results/condition_source_analysis_dallas/condition_metrics.csv \
  --environment ny=results/condition_source_analysis_ny/condition_metrics.csv \
  --environment la=results/condition_source_analysis_la/condition_metrics.csv
```

## Heterogeneidade hierárquica dos contrastes pareados

O piloto formal de Dallas modela diretamente os 126 contrastes
minoria–maioria, tratando dimensão social como efeito fixo e outcome como
intercepto aleatório:

```bash
python scripts/analyze_outcome_heterogeneity.py
```

O script exporta a variância entre outcomes, BLUPs, intervalos aproximados,
bootstrap por clusters de outcome, análise `leave-one-dimension-out` e uma
decomposição ANOVA estritamente descritiva. Para `n_sources`, também ajusta uma
sensibilidade negative-binomial com outcome como efeito fixo e erros-padrão
agrupados por outcome. Essa sensibilidade não é apresentada como um GLMM de
random slopes.

Os gráficos mostram os seis contrastes brutos ajustados por dimensão junto aos
estimates hierárquicos e identificam explicitamente refits na fronteira. O
arquivo `model_diagnostics.json` registra a variância aleatória de outcome, a
variância residual, o ICC, o intervalo bootstrap de `tau`, a convergência e o
diagnóstico de singularidade.

## Epistemic commitment

### Análise atual de contextual hedging: Dallas + New York

O fluxo atual reextrai as claims nas duas cidades como **spans literais do AIO**;
não reutiliza as claims reescritas do piloto Dallas. A medida principal é a taxa
de contextual hedging nas sentenças que originaram ao menos uma claim válida.
O código mantém separadas as taxas por resposta inteira e por span original da
claim. A presença de cue no span é apenas um *scope proxy*.

```bash
bash scripts/run_hedging_ny_dallas_pipeline.sh
```

Esse processamento usa Phi-4 local, pode levar bastante tempo e retoma
checkpoints somente com os mesmos inputs, prompts e modelo. O relatório e os
PNGs ficam em `results/hedging_ny_dallas/`. Para validar a seleção de claims,
o escopo cue→claim e o matching, a anotação humana é uma etapa posterior; os
resultados correspondentes permanecem pendentes até então. MegaVeridicality
continua como análise suplementar independente.

### Histórico: piloto de epistemic commitment

A análise mantém hedging e veridicality como operacionalizações separadas. O
léxico de hedges é derivado exclusivamente dos spans de especulação anotados no
BioScope 1.0. Predicados recebem score apenas quando o dependency parse permite
associar lemma, frame, polaridade e configuração a uma entrada normalizada do
MegaVeridicality v2.1; casos ambíguos são exportados como `unmatched`.

```bash
python scripts/download_epistemic_resources.py
python scripts/analyze_epistemic_commitment.py
```

Os resultados ficam em `results/epistemic_commitment/`, sem agregação prévia
entre grupos ou réplicas e com os metadados experimentais preservados.

Para gerar figuras descritivas da análise:

```bash
python scripts/plot_epistemic_commitment.py
```

Os PNGs e uma nota metodológica são salvos em
`results/epistemic_commitment/figures/`.
