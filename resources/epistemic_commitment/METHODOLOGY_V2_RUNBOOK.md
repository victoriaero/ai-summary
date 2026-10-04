# Epistemic commitment v2 — runbook

Use the following executable entry points (from repository root). The lexical
script also writes the previous legacy veridicality artifacts in its own
baseline directory; use `run_veridicality_v2.py` as the authoritative strict
MegaVeridicality output.

```bash
python scripts/download_epistemic_resources.py
python scripts/run_lexical_bioscope_overlap.py
python scripts/analyze_contextual_hedging_v2.py
python scripts/run_veridicality_v2.py
python scripts/compare_hedging_sensitivity.py
python scripts/plot_epistemic_commitment_v2.py
```

`analyze_contextual_hedging_v2.py` uses a document-level BioScope split,
validation-selected exact-span threshold, held-out cue/sentence metrics, AIO
predictions, stratified blind/key files, and top-frequency leave-one-cue-out
sensitivity. Cue-family sensitivity is omitted because the resource has no
cue-family labels and this implementation introduces no hand-coded taxonomy.

For human validation, share only the `*_blind.csv` files. Keep `*_key.csv`
restricted until both annotators have completed and the adjudicated columns
have been filled. Then run:

```bash
python scripts/evaluate_epistemic_human_validation_v2.py \
  --hedge-blind results/epistemic_commitment/contextual_hedging/human_validation_blind.csv \
  --hedge-key results/epistemic_commitment/contextual_hedging/human_validation_key.csv \
  --veridicality-blind results/epistemic_commitment/veridicality_v2/validation_blind.csv \
  --veridicality-key results/epistemic_commitment/veridicality_v2/validation_key.csv
```

The evaluator reports human-human agreement and Cohen's kappa. For hedging it
also computes model precision/recall/F1 against adjudication, excluding
`uncertain` labels from binary metrics. For veridicality it reports precision
of strict matches, frame mapping accuracy, polarity accuracy, predicate
detection accuracy, and clause-status accuracy from the adjudicated checklist.
It does not ask annotators to rate the human veridicality score.

All figures produced by `plot_epistemic_commitment_v2.py` are exploratory PNGs;
it does not emit publication PDFs. The people control is repeated only for
descriptive dimension facets, never as a set of independent observations.
