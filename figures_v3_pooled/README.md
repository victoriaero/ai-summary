# Three-location pooled paper figures

This folder adapts the `figures_v2` visualizations to **Dallas + New York + Los Angeles as one combined analysis**.

## Why this is not a naive pool

The three geographic repetitions are repeated measurements, not 63 independent outcomes.
For paired focal-vs-comparison effects, the scripts therefore:

1. compute the focal-comparison effect inside each `outcome x location`;
2. average Dallas, New York, and Los Angeles repetitions **within the same outcome**;
3. perform inference across outcomes.

The plotted point estimate is consequently the mean three-location effect, and the 95% CI is an **outcome-level bootstrap CI** around that cross-location summary.

For dimension rows:
- citation count: two-sided Wilcoxon test, Holm correction across the six dimensions;
- source-set, semantic, sentence-level, and evidence-gap contrasts: paired sign-flip test, Holm correction across the six dimensions;
- Overall rows are the single prespecified outcome-level aggregate and are not put into the six-dimension Holm family.

For Figures 4.4 and 4.5, condition-level metrics are first averaged across locations and the blocked rank association is recomputed on those means. The scripts use:
- ranks centered within outcome;
- within-outcome permutation test;
- 5,000 outcome-cluster bootstrap samples for the 95% CI;
- Holm correction across the same six pipeline associations used by the evidence-synthesis analysis.

For evidence alignment, source recovery can differ across cities. A condition is averaged over the locations in which it passes the existing 50% recovery + >=3 usable-source rule. `n_locations` is preserved in the exported CSV so this missingness remains inspectable.

## Expected result folders

Under `/scratch/victoria.estanislau/ai-summary/results/`:

- `link_count_analysis_dallas`, `link_count_analysis_ny`, `link_count_analysis_la`
- `source_overlap_analysis_dallas`, `source_overlap_analysis_ny`, `source_overlap_analysis_la`
- `semantic_embedding_analysis_dallas`, `semantic_embedding_analysis_ny`, `semantic_embedding_analysis_la`
- `evidence_synthesis_analysis_dallas_v2`, `evidence_synthesis_analysis_ny_v2`, `evidence_synthesis_analysis_la_v2`

## Run everything

```bash
cd /scratch/victoria.estanislau/ai-summary/figures_v3_pooled
"$VIRTUAL_ENV/bin/python" RUN_FIGURES.py
```

Default output:

```text
/scratch/victoria.estanislau/ai-summary/figures_three_locations_pooled/
```

This directory contains the four PDFs/PNGs plus CSV files with the exact pooled effects, confidence intervals, p-values, Holm-adjusted p-values, and sample composition used by the figures.

## Figures

- `figure_4_2_evidentiary_pathway_pooled3.pdf`
- `figure_4_3_semantic_shift_anatomy_pooled3.pdf`
- `figure_4_4_pathway_surface_coupling_pooled3.pdf`
- `figure_4_5_grounding_pooled3.pdf`

## Paper wording

If these pooled figures are used in the main paper, the Statistical Analysis section should state that figures summarize the three locations by averaging location-specific matched effects within outcome before inference, while location-specific replications are retained as robustness results / supplementary material.
