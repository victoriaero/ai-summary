# Three-location paper figure scripts (Sections 4.2-4.5)

These scripts regenerate the paper figures using **Dallas, New York, and Los Angeles** simultaneously and encode the per-location inferential results directly in the figure.

## Expected results folders

The scripts expect the outputs produced by the final audit pipeline:

```text
/scratch/victoria.estanislau/ai-summary/results/
├── link_count_analysis_dallas/all_results.json
├── link_count_analysis_ny/all_results.json
├── link_count_analysis_la/all_results.json
├── source_overlap_analysis_dallas/all_results.json
├── source_overlap_analysis_ny/all_results.json
├── source_overlap_analysis_la/all_results.json
├── semantic_embedding_analysis_dallas/all_results.json
├── semantic_embedding_analysis_ny/all_results.json
├── semantic_embedding_analysis_la/all_results.json
├── evidence_synthesis_analysis_dallas_v2/all_results_v2.json
├── evidence_synthesis_analysis_ny_v2/all_results_v2.json
└── evidence_synthesis_analysis_la_v2/all_results_v2.json
```

## Run everything

```bash
cd /scratch/victoria.estanislau/ai-summary/three_location_figure_scripts
python RUN_FIGURES.py
```

or with the virtualenv explicitly:

```bash
"$VIRTUAL_ENV/bin/python" RUN_FIGURES.py
```

Outputs go to:

```text
/scratch/victoria.estanislau/ai-summary/figures_three_locations/
```

## Statistical encoding

Across figures:

- **horizontal lines** = 95% confidence intervals from the original analysis;
- **filled city marker** = statistically significant result;
- **open city marker** = not statistically significant;
- for the six dimension-level contrasts, significance uses the **Holm-corrected p-value within metric and location**;
- for the single prespecified `Overall` aggregate row, significance uses the aggregate raw p-value because it is one aggregate test per metric/location;
- figure 4.4 and the right panel of 4.5 use the original **blocked within-outcome correlations**, outcome-cluster bootstrap CIs, and Holm-corrected permutation p-values.

The scripts also save CSV files containing the exact effect estimate, CI, p-value used in the figure, and significance flag. This makes it possible to audit every visual marker without reading values from the PDF.

## Figures

### Figure 4.2 — Evidentiary pathway
Two forest panels:
1. focal-comparison citation-volume effect;
2. focal-comparison URL-Jaccard displacement from the matched generic source set.

Each dimension contains separate Dallas, New York, and Los Angeles estimates and CIs, plus the outcome-level aggregate row.

### Figure 4.3 — Semantic shift anatomy
Three forest panels:
1. subject-normalized full-answer semantic displacement;
2. generic-content displacement;
3. group-specific novelty.

All three locations and their corrected significance are shown independently.

### Figure 4.4 — Pathway-surface coupling
Shows the prespecified within-outcome blocked correlation between URL source-set displacement and subject-normalized answer displacement for each city, with cluster-bootstrap CI and Holm-corrected permutation significance.

### Figure 4.5 — Grounding
Left: focal-comparison evidence semantic gap at the 50% recovery threshold for each dimension/location plus the aggregate effect.

Right: the blocked correlation between answer displacement and **evidence alignment** for each location. The stored pipeline statistic is based on semantic gap; the plotting code flips the sign (and CI endpoints) so positive values mean stronger alignment.

## Files

```text
figure_common.py
plot_4_2_three_locations.py
plot_4_3_three_locations.py
plot_4_4_three_locations.py
plot_4_5_three_locations.py
RUN_FIGURES.py
```
