# Dallas claim-level epistemic-commitment pilot

Run from the repository root:

```bash
python scripts/analyze_claim_epistemic_pilot.py
python scripts/plot_claim_epistemic_pilot.py
python -m unittest tests.test_claim_epistemic_pilot
```

This script makes no LLM calls and does not change the existing claim,
hedging, or veridicality outputs. It links the **current, provisional** Phi-4
`valid_claims.csv` to previously computed contextual hedges and strict
MegaVeridicality v2 occurrences using `response_id` and literal offsets in the
original Dallas AIO. It never measures either construct in `claim_text` or
`claim_text_with_context_markers`.

## Definitions

- `H_span_proxy`: 1 if a contextual hedge cue falls entirely inside an original
  source span of a claim, 0 otherwise. This is a **scope candidate**, not proof
  that the cue semantically modifies the claim. `hedge_scope_verified` is always
  false. `hedge_sentence_context` is a separate, broader flag; it is never used
  as the primary claim-level H measure. Multi-span claims are treated as a
  union of separate intervals, not one continuous interval.
- `V_score`: the human MegaVeridicality score of exactly one strict resource
  predicate whose complement can be uniquely located in the original AIO and
  conservatively linked to the claim. Accepted links require either that every
  selected source span is inside the complement, or that the complement is
  inside a source span and the candidate claim begins literally with the
  complement proposition (after optional `that`). The latter uses `claim_text`
  only to confirm linkage; the score still comes exclusively from the
  predicate in the original AIO. Partial overlaps, a predicate/complement that
  cannot be located uniquely, and claims with multiple eligible predicates
  remain unscored. The exact decisions are in `claim_veridicality_links.csv`
  and `strict_occurrence_alignment_audit.csv`. These are conservative lexical
  links, not human-confirmed semantic scope.
- Absence of a cue in an analyzed response is a valid zero for H. Absence of a
  conservatively linked strict predicate is **missing V**, not a score of zero.
  Coverage and conditional mean V are therefore reported separately.

## Files

- `claim_epistemic_metrics.csv`: one row per distinct accepted claim; original
  sentence/span pointers, metadata, H context/candidate flags, V score and
  coverage.
- `claim_hedge_links.csv`: every cue in the claim's original sentence, with
  `inside_source_span_scope_unverified` or `same_sentence_outside_span` and a
  count of claims sharing the same cue inside their spans.
- `claim_veridicality_links.csv`: predicate/complement link decisions and
  unscored alternatives within the original sentence.
- `all_claims_summary.csv`: descriptive summaries by condition, social
  dimension, domain and outcome. People is counted once in the overall rows.
- `all_claims_outcome_contrasts.csv`: outcome/dimension differences between
  complete claim sets. The shared People control is joined to each dimension
  here **only** for paired comparison, not duplicated in the source table.
- `matched_claim_pairs.csv`: all eight provisionally equivalent cross-condition
  pairs; no independent matching validation has been done.
- `coverage_diagnostic.csv`: requested all-claims versus unique matched-claims
  comparison, with n claims, H analysis coverage/rate, V coverage and mean V.
- `unit_granularity_diagnostic.csv`: response-level, claim-bearing-sentence,
  and claim-span H rates plus response and claim V coverage. These have
  different denominators and are deliberately labeled separately.
- `run_summary.json`: top-level counts.
- `figures/`: four descriptive PNGs: hedging across response/sentence/claim
  units, response versus claim veridicality coverage, 21 outcome-level hedge
  contrasts in each dimension, and feasibility of provisional matched pairs.
  Recreate them with `python scripts/plot_claim_epistemic_pilot.py`. They are
  diagnostic visualizations, not inferential estimates.

## Initial diagnostic, not a substantive group result

All 2,408 claims have contextual-hedging analysis coverage; 364 (15.1%) have
a cue inside a source span. Only 72 (3.0%) have one conservatively linked
strict veridicality score. The eight provisionally matched pairs cover twelve
unique claims; none has V scores on both sides and all have H=0 on both sides.
They cannot support a matched-claim estimate of differential epistemic
commitment. Claims from the same response/sentence and pairs sharing a People
claim are not independent observations. No significance tests are run.

These outputs test measurement feasibility with the existing claims. They do
not establish whether the AIO claims are true, linguistically faithful, or
semantically equivalent across conditions. Those judgments need human review.
