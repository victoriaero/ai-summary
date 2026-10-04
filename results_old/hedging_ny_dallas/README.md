# Contextual hedging: Dallas and New York

This is an exploratory observational analysis, not a formal replication or a measure of general epistemic commitment.
The primary outcome is contextual hedging in sentences that produced at least one automatically accepted, literal-source claim.

## Data inventory

| Location | Responses with AIO | Outcomes | Conditions | People controls | Literal claims |
|---|---:|---:|---:|---:|---:|
| Dallas | 273 | 21 | 13 | 21 | 1872 |
| New York | 273 | 21 | 13 | 21 | 1903 |

The research team confirmed that all NY responses were collected in New York using the same IP. The collection files do not independently verify that assertion: VPN location and IP metadata are incomplete. People is counted once per outcome/location in the models.

## Operational definitions

- All-response rate: hedged AIO sentences / all AIO sentences.
- Primary claim-bearing-sentence rate: hedged sentences originating ≥1 accepted literal claim / distinct claim-bearing sentences.
- Claim-source-span rate: accepted claims with a contextual cue inside an original source span / accepted claims. This is an unverified semantic-scope proxy, not a claim-level hedge judgment.
- Claims are selected literal fragments with checked offsets. No normalized or rewritten claim is measured.

## Model specification

Grouped-binomial GLM on response counts, with 13 observed condition cells × location and outcome fixed effects. Location is fixed; the unique People control is shared across demographic dimensions. Contrasts are standardized equally over the 21 outcomes. Intervals are percentile CIs from outcome-cluster bootstrap resampling both locations and all conditions together. Installed statsmodels does not provide a stable frequentist grouped-binomial random-intercept fit here, so this prespecified fallback is used. See `analysis_manifest.json` for convergence, warnings, seeds and successful replicates.

## Descriptive observations

The following rates pool counts within each location/condition; their denominators differ by unit and they are not model-adjusted.

| Location | Condition | All sentences | Claim-bearing sentences | Claim source spans |
|---|---|---:|---:|---:|
| Dallas | majority | 17.2% | 15.5% | 15.2% |
| Dallas | minority | 18.7% | 15.3% | 14.7% |
| Dallas | people | 14.4% | 13.7% | 13.7% |
| New York | majority | 17.4% | 17.2% | 16.4% |
| New York | minority | 15.6% | 12.5% | 11.8% |
| New York | people | 9.1% | 9.0% | 8.9% |

## Model-based inference and cross-location consistency

Estimates below are adjusted probability differences in percentage points; CIs come from the outcome-cluster bootstrap. Examine signs, magnitudes and interval overlap rather than isolated p-values.

| Dimension | Contrast | Dallas [95% CI] | New York [95% CI] | Pooled [95% CI] |
|---|---|---:|---:|---:|
| Disability | majority-people | -0.2 [-7.3, +7.2] | +6.5 [+0.8, +12.3] | +3.1 [-2.6, +8.9] |
| Disability | minority-majority | -0.8 [-9.5, +6.8] | -2.4 [-10.8, +4.9] | -1.6 [-8.5, +4.4] |
| Disability | minority-people | -1.0 [-9.9, +6.4] | +4.1 [-2.3, +9.6] | +1.5 [-4.0, +5.6] |
| Ethnicity | majority-people | -6.4 [-13.0, -0.6] | +3.1 [-4.8, +9.8] | -1.7 [-6.6, +2.3] |
| Ethnicity | minority-majority | +8.4 [+0.6, +14.7] | +1.2 [-5.6, +8.7] | +4.8 [-0.2, +9.8] |
| Ethnicity | minority-people | +2.0 [-6.8, +10.0] | +4.2 [-3.8, +12.4] | +3.1 [-3.4, +9.4] |
| Gender | majority-people | +7.9 [-0.1, +17.3] | +8.6 [+0.6, +16.7] | +8.2 [+2.1, +13.8] |
| Gender | minority-majority | -7.6 [-16.4, -0.8] | -10.1 [-17.8, -2.4] | -8.9 [-15.6, -3.0] |
| Gender | minority-people | +0.2 [-9.3, +8.0] | -1.5 [-7.9, +4.1] | -0.6 [-4.8, +3.2] |
| Gender Identity | majority-people | -0.3 [-5.7, +4.7] | +9.4 [+1.9, +16.9] | +4.5 [-0.6, +9.3] |
| Gender Identity | minority-majority | +2.9 [-3.6, +8.7] | -3.2 [-13.4, +5.6] | -0.2 [-7.0, +4.9] |
| Gender Identity | minority-people | +2.6 [-5.4, +8.2] | +6.1 [-1.0, +15.0] | +4.4 [-1.3, +9.4] |
| Race | majority-people | +4.7 [-0.3, +11.8] | +6.6 [-0.2, +15.0] | +5.7 [+1.0, +10.9] |
| Race | minority-majority | -2.2 [-11.5, +4.6] | -0.6 [-9.0, +7.9] | -1.4 [-9.1, +5.0] |
| Race | minority-people | +2.5 [-5.6, +9.8] | +6.0 [-0.4, +11.7] | +4.3 [-0.6, +9.3] |
| Sexual Orientation | majority-people | +5.3 [-3.3, +12.0] | +10.0 [+2.5, +17.3] | +7.6 [+1.3, +12.7] |
| Sexual Orientation | minority-majority | -0.2 [-12.5, +12.0] | -9.6 [-15.9, -0.6] | -4.9 [-12.3, +3.5] |
| Sexual Orientation | minority-people | +5.1 [-7.2, +16.3] | +0.4 [-5.0, +6.2] | +2.8 [-4.1, +8.8] |

## Claim-bearing and claim-span sensitivity

The all-response and claim-source-span model contrasts are in `hedging_model_results.csv`; Figure B displays descriptive rates with different denominators. `hedging_robustness.csv` reports paired-outcome sensitivity to detected-cue removal, one-outcome removal, short responses, and per-100-token scaling. These are descriptive checks, not refitted model CIs; they are not used to select a favorable specification.

## Measurement validation

A blind sample of 250 positive cue-in-span cases is in `hedge_scope_validation_sample.csv`. Cue-to-proposition scope and claim-extraction quality remain unvalidated by humans. Run the separate evaluator only after both annotators and adjudication have filled the labels; until then, `hedge_scope_validation_results.csv` says `pending_annotation`.

## Matched-claim feasibility

32704 embedding-generated, Phi-4-classified candidate pairs are provisional. `matched_claim_validated.csv` remains empty until human adjudication. No matched-claim inference is produced below 30 validated pairs per contrast/location spanning at least 10 outcomes.

## Limitations

The primary analysis restricts to claim-bearing sentences but does not guarantee identical factual content between conditions. The claim detector and acceptance checks use the same model; literal offsets ensure textual fidelity, not factual truth or semantic-scope validity. Shared People controls, claims from the same response, 21 outcome clusters and observational Google outputs limit inference. Higher hedging probability is not evidence of a guardrail, intention or generalized epistemic caution. MegaVeridicality is a separate supplementary analysis.

## Reproduce

```bash
python scripts/run_literal_claim_extraction.py --input-dir annotations/v1_dallas/google_aio_collection
python scripts/run_literal_claim_extraction.py --input-dir annotations/v2_ny/google_aio_collection
python scripts/analyze_hedging_ny_dallas.py
```
