# Epistemic commitment v2: Dallas pilot

This analysis treats contextual hedging and MegaVeridicality as separate
operationalizations. It makes no composite index or causal claim. Outcomes are
described as differences in hedging, strict-predicate coverage, predicate
veridicality, and proximity of the People control to other conditions.

## Analysis population and dependence

The current complete paired replica is Dallas (`v1_dallas`): 21 outcomes, one
People control and Minority/Majority responses in each of six demographic
dimensions, for 273 unique AIO responses. Los Angeles is listed in
`tables/replica_inventory.csv` but excluded from the paired models because only
one condition is available. New replicas enter the models only after the same
13 cells are present for all 21 outcomes.

The same People response can serve as a descriptive reference in six figure
panels. It is included **once** in every fitted model. Outcome fixed effects
control the common query subject, and inference uses outcome-cluster sandwich
standard errors or outcome-cluster bootstrap. With 21 outcome clusters, all
intervals are exploratory.

## Principal metrics and models

- Contextual hedging: `n_hedged_sentences / n_sentences`, fitted as a grouped
  binomial GLM. `contextual_hedges_per_100_tokens` is secondary. The BioScope
  lexical-overlap baseline is a measurement comparison, not the main hedge
  measure.
- MegaVeridicality coverage: binary indicator that the response contains at
  least one strict resource-compatible predicate, fitted as a logistic GLM.
- MegaVeridicality score: mean normalized human resource score, fitted among
  covered responses only by response-level OLS. Missing scores are not imputed.
  An occurrence-level sensitivity model attempts a response random intercept
  with outcome fixed effects; when singular, it falls back to occurrence-level
  OLS with outcome-cluster robust standard errors. Dallas used the documented
  fallback; see `models/occurrence_model_diagnostics.json`.

All main models use the 13 observed condition cells, equivalent to estimable
condition × demographic-dimension effects with a shared People reference,
plus outcome fixed effects. If multiple complete replicas become available, a
replica fixed effect is added. Planned contrasts are Minority − People,
Majority − People, and Minority − Majority only. Binomial results include odds
ratios and outcome-adjusted probability differences; score results use resource
score units. The forest plots display model-adjusted differences and 95%
intervals; bootstrap paired differences appear in
`tables/paired_bootstrap_contrasts.csv`. P-values are supplementary.

`people_proximity_results.csv` defines proximity difference as
`|People − Majority| − |People − Minority|`. Positive values mean People is
closer to Minority on that metric. Veridicality proximity requires all three
responses to have strict-score coverage; the table records how many complete
triplets remain.

## Sensitivity and validation

Hedging sensitivity covers the lexical baseline, sentence rate versus per-100
token rate, fixed higher cue-probability cutoffs (0.95 and 0.99), and minimum
sentence counts derived before comparing outcomes from the 5th and 10th
percentiles of response length. `robustness/leave_one_cue_out.csv` removes each
detected cue surface from AIO predictions and recomputes the paired effect,
retaining a sentence if another detected cue remains. This is a detected-cue
removal analysis; it does not retrain the BioScope classifier. The earlier
BioScope retraining sensitivity was not completed and is not silently treated
as equivalent to this table.

Veridicality sensitivity includes responses with at least two matches, response
means versus medians, occurrence-level scores, each strict predicate removed
in turn, exclusion of data-derived dominant predicates, and positive versus
negative polarity. The predicate-removal figure compares estimates on the
**same covered pairs** before and after removal. Ambiguous frames are already
unmatched under the strict rule. spaCy provides no calibrated per-occurrence
parse confidence here, so a low-parser-confidence subset is marked not
estimable rather than fabricated.

The detector has no externally annotated cue-family labels, so no cue-family
stacked bar is produced. Assigning labels such as “epistemic predicate” from a
hand-written mapping would violate the taxonomy constraint.

The validation folder contains a 400-item hedge sheet and 300-item predicate
sheet, plus restricted keys. The predicate sheet asks annotators about
identification, clause, frame, polarity, negation, strict decision, and resource
mapping; it does not ask them to re-rate veridicality. Human agreement,
condition-specific detector precision/recall/F1, cue-frequency-bin metrics,
and differential measurement error are computed after adjudicated labels are
filled. At present the tables correctly say `pending_annotation`. The sheets
are preserved on reruns; sample-ID changes cause a hard error instead of
overwriting labels.

## Files and figures

`tables/` holds the requested model, contrast, proximity, comparison, and
measurement-validation CSVs. `models/` holds model summaries, convergence
diagnostics, package versions, input SHA-256 hashes, seeds and the analysis
manifest. `robustness/` holds cue, predicate and threshold sensitivities.
`validation/` holds annotation sheets and measurement checks.

The 14 `figures/main/` PNGs include response distributions, model means with
95% intervals, coverage, paired-outcome lines, paired-effect forests, People
proximity and the two-stage veridicality figure. The 6
`figures/supplementary/` PNGs cover lexical comparison, predicate occurrences,
frequent matched predicates per 100 responses and item-removal robustness.
Only PNGs are generated for this exploratory stage.

## Reproduction

From the repository root in the project Python environment:

```bash
python scripts/run_lexical_bioscope_overlap.py
python scripts/analyze_contextual_hedging_v2.py --loo-top-cues 0
python scripts/run_veridicality_v2.py
python scripts/finalize_epistemic_tables.py
python scripts/run_epistemic_commitment_v2.py
```

The last command generates the complete v2 directory and figures. Re-run
`python scripts/epistemic_commitment_v2_validation.py` after adjudicating the
human sheets to update measurement-validation results without refitting the
models. The v2 runner sets `PYTHONHASHSEED=0` and uses numerical seed
`20260930`, 4,000 outcome bootstraps, and 4,000 coefficient draws. Exact local
package versions and source checksums are in `models/analysis_manifest.json`.
