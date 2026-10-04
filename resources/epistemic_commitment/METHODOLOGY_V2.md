# Epistemic commitment: reproducible v2 workflow

This workflow keeps hedging and veridicality as distinct outcomes. Nothing in
these scripts combines them into an index or makes substantive claims.

## Hedging

`lexical_bioscope_overlap` is the exploratory baseline from the prior script:
every surface cue annotated as `speculation` in BioScope is phrase-matched in
the AIO, without context. It is not the principal hedge measure.

`contextual_hedging` trains a token classifier on BioScope speculation-cue
spans. It uses a deterministic logistic-regression token sequence model with
spaCy lexical, morphological, shape, and local-context features; cue inventory
is learned only from the annotated training data. Documents, not sentences, are
split into train/validation/test partitions (seed 20260930). The decision
threshold is selected on validation documents by exact cue-span F1; the held-out
test reports token precision/recall/F1, exact-span precision/recall/F1, and
sentence-level certain/uncertain accuracy/F1. The primary AIO outcome is
`contextual_hedged_sentence_rate`; hedge spans per 100 tokens is secondary.
Leave-one-cue-out sensitivity removes each of the most frequent BioScope cue
forms from training, with frequency and number of forms derived from the
training partition. Cue-family-out is intentionally omitted: BioScope does not
annotate cue families, and this implementation does not introduce manual
linguistic categories.

The AIO transfer sample is stratified proportionally across group type,
dimension, replica/location, resource-frequency band, and contextual model
prediction. `human_validation_blind.csv` excludes predictions; the matching
`human_validation_key.csv` must be kept from annotators until adjudication.
Annotate each expression with yes/no/uncertain for: “Does the highlighted
expression reduce the speaker/author’s commitment to the truth of the
proposition in this context?” The files include two annotator labels and an
adjudicated label. `evaluate_epistemic_human_validation.py` computes human-human
agreement and contextual-model precision/recall/F1 against adjudication;
uncertain cases are excluded from binary model scores but retained in the
agreement calculation.

## Veridicality

The strict analysis maps an AIO dependency-parsed clause-embedding occurrence
to an exact MegaVeridicality v2.1 normalized resource row using lemma, frame,
voice as represented in the frame, and polarity. Eventivity is not guessed: if
the complement does not distinguish an eventive from non-eventive resource
frame, it remains unmatched. The normalized TSV does not provide a separate
conditional-configuration column; conditional antecedents therefore remain
unmatched. No resource score is averaged across frames or polarities, and no
score is imputed to unmatched occurrences.

The objective `unmatched_reason` labels include `predicate_not_in_resource`,
`no_clause_complement`, `frame_unresolved`, `eventivity_unresolved`,
`polarity_mismatch`, `ambiguous_surface`, `conditional_context`, and
`parser_failure` (the latter is reserved for unavailable dependency parses).
Coverage is a separate response-level binary outcome. Score summaries are
reported only for covered responses. Sensitivity outputs include n>=2,
mean/median, per-occurrence data, leave-one-predicate-out, and exclusion of
data-derived dominant predicates (frequency greater than median plus three
median absolute deviations).

The veridicality annotation sample aims for 200 matched and 100 unmatched
candidates when available, stratified across predicates or unmatched reasons.
Annotators review predicate detection, clause status, frame, polarity, and
resource mapping only; they do not re-rate veridicality. As for hedging, the
blind sheet is separate from its key. The validation evaluator reports
agreement/accuracy when completed labels are supplied.

## Outputs and dependence

Each response/occurrence table retains query, outcome, domain, dimension,
condition, group, group type, location, replica, and response identifiers.
Never expand/reuse a `people` control as if those copies were independent
responses in inferential models; retain `response_id` and account for repeated
control IDs. The plotting script reuses people controls only for descriptive
faceting and labels that reuse.

Resource versions, parser/model versions, seeds, selected threshold, and model
checksum are recorded in the generated provenance JSONs. BioScope and
MegaVeridicality checksums are available in the resource download inventory and
should be copied into the final archive alongside generated outputs.

## Commands

Run from the repository root, with the project environment activated. These
commands are provided for the research team; they have not been run as part of
the code change.

```bash
python scripts/download_epistemic_resources.py
python scripts/run_lexical_bioscope_overlap.py
python scripts/analyze_contextual_hedging_v2.py
python scripts/run_veridicality_v2.py
```

After two annotators complete the blind sheets and the adjudicated columns,
evaluate them using:

```bash
python scripts/evaluate_epistemic_human_validation.py \
  --hedge-blind results/epistemic_commitment/contextual_hedging/human_validation_blind.csv \
  --hedge-key results/epistemic_commitment/contextual_hedging/human_validation_key.csv \
  --veridicality-blind results/epistemic_commitment/veridicality_v2/validation_blind.csv \
  --veridicality-key results/epistemic_commitment/veridicality_v2/validation_key.csv
```

The validation script accepts the completed blind sheets as-is. Keep key files
restricted until labels and adjudication are final.
