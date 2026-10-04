# Historical hedging pipelines

The current Dallas–New York pipeline is `scripts/run_hedging_ny_dallas_pipeline.sh`.
The following older entrypoints and their historical outputs are retained for
reproducibility, not for the current contextual-hedging analysis:

- `scripts/analyze_contextual_hedging_v2.py` — earlier detector training and
  Dallas-centred output; its saved BioScope model is reused to score Dallas and
  New York with the same threshold.
- `scripts/analyze_claim_epistemic_pilot.py` and
  `scripts/plot_claim_epistemic_pilot.py` — Dallas pilot using the older,
  sometimes rewritten Phi-4 claims.
- `scripts/run_epistemic_commitment_v2.py` — combined historical hedging and
  separate MegaVeridicality analyses.

These files remain at their original import paths because historical scripts
and tests depend on them. Moving or deleting them would break reproducibility.
They are superseded as *entrypoints* by the new workflow, and no old claim
outputs are accepted by the Dallas–New York analysis.
