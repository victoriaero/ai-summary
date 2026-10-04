# Final Google AI Overview Audit Pipeline — Dallas + NYC + Los Angeles

This is the minimal script set for the **current paper**: query-design agreement plus Sections 4.2–4.5 (citation volume, source-set displacement, answer semantic displacement, pathway/surface coupling, and alignment with cited evidence).

It intentionally **does not include** the institutional source-type or claim-survival pipelines. Those analyses depend on model-derived categorical labels that were not manually validated and are not required for the current main-paper claims.

## Expected project layout

```text
/scratch/victoria.estanislau/ai-summary/
├── annotations/
│   ├── v1_dallas/google_aio_collection/
│   ├── v2_ny/google_aio_collection/
│   └── v3_la/google_aio_collection/
├── artifacts/
│   └── annotation_inputs/
│       ├── outcomes_annotator1.csv      # optional but needed to rerun Gwet
│       └── outcomes_annotator2.csv
├── results/
└── scripts/                              # put these files here
```

If your version-folder names differ, pass them to `RUN.py`.

## One-command run

```bash
cd /scratch/victoria.estanislau/ai-summary
python scripts/RUN.py
```

By default this runs all three locations:

- `v1_dallas` → Dallas, Texas
- `v2_ny` → New York City, New York
- `v3_la` → Los Angeles, California

and performs, in order:

1. collection consistency validation;
2. citation/link-count analysis;
3. source-set overlap/displacement analysis;
4. answer semantic displacement analysis;
5. cited-source download/extraction (resumable);
6. evidence-to-synthesis semantic alignment analysis;
7. a compact three-location results summary;
8. a **secondary** cross-location pooled analysis that averages location repetitions within outcome before inference.

The source downloader is resumable, so rerunning `RUN.py` does not redownload URLs already recorded in each location's `source_corpus` metadata.

## Statistics retained from the final methodology

### Citation volume
- paired focal-vs-comparison contrasts within outcome;
- two-sided Wilcoxon signed-rank tests;
- paired rank-biserial effect sizes;
- **20,000 paired-outcome bootstrap samples for 95% CIs**;
- Holm correction across the six social dimensions;
- aggregate outcome-level contrast.

### Source-set displacement
- URL-level Jaccard distance from the matched generic condition;
- hostname/domain robustness diagnostics;
- paired sign-flip permutation tests (exact when small enough; otherwise 200,000 assignments);
- 20,000 paired-outcome bootstrap CIs;
- paired rank-biserial effect sizes;
- Holm correction across dimensions;
- aggregate and global tests.

### Answer semantic displacement
- primary subject-normalized dense semantic distance from the generic answer;
- raw dense and TF–IDF robustness checks;
- sentence-level generic coverage, group novelty, and symmetric sentence distance;
- paired sign-flip tests, Wilcoxon checks, 20,000 bootstrap CIs, Holm correction;
- global/post-hoc diagnostics.

### Evidence alignment
- source-balanced evidence semantic gap;
- reference threshold: at least 50% of cited sources recovered and at least 3 usable sources;
- sensitivity thresholds: 40%, 50%, 60%, 70%, 80%;
- paired sign-flip tests, Wilcoxon checks, 20,000 bootstrap CIs, Holm correction;
- coverage diagnostics;
- blocked rank correlations for pathway → answer and evidence alignment → answer displacement;
- 20,000 blocked permutations and 5,000 outcome-cluster bootstrap samples for correlation CIs.

### Query-selection agreement
If the two annotation CSVs exist, `RUN.py` also reruns:
- quadratically weighted Gwet's AC2 + 95% CI for relevance;
- Gwet's AC1 + 95% CI for comparability;
- selected-set agreement descriptives.

The selected-set statistics are saved explicitly as **descriptive only**, not as an independent post-selection reliability estimate.

## Important outputs

Each location gets separate result folders, e.g. Dallas:

```text
results/
├── link_count_analysis_dallas/
├── source_overlap_analysis_dallas/
├── semantic_embedding_analysis_dallas/
├── source_collection_dallas/
└── evidence_synthesis_analysis_dallas_v2/
```

The same structure is generated for `ny` and `la`.

Final convenience outputs:

```text
results/three_location_summary/
├── primary_dimension_effects_all_locations.csv
├── primary_aggregate_effects_all_locations.csv
├── pipeline_correlations_all_locations.csv
├── evidence_collection_quality_all_locations.csv
├── replication_direction_summary.csv
└── all_locations_raw_bundle.json
```

The optional secondary pooled analysis is written to:

```text
results/three_location_pooled_secondary/
```

**Primary paper inference should remain the separate per-location analyses.** The pooled folder is a secondary robustness summary; it first averages geographic repetitions within each outcome so the three cities are not treated as independent outcomes.

## Useful options

Already downloaded all source corpora:

```bash
python scripts/RUN.py --skip-source-download
```

Run only semantic displacement for all cities:

```bash
python scripts/RUN.py --only semantic_displacement
```

Run only New York:

```bash
python scripts/RUN.py --locations ny
```

Custom collection version names:

```bash
python scripts/RUN.py \
  --dallas-version v1_dallas \
  --ny-version v2_ny \
  --la-version v3_la
```

Skip the secondary pooled analysis:

```bash
python scripts/RUN.py --skip-pooled-secondary
```

## Dependencies

Python packages are listed in `requirements.txt`. The source collector also requires `curl` on the system PATH.

The scripts use CUDA automatically when available for embedding models.
