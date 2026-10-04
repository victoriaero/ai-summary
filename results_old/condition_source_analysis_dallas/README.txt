Condition and source exploratory analysis
=========================================

Collection: /scratch/victoria.estanislau/ai-summary/annotations/v1_dallas/google_aio_collection
Queries: 273
Source occurrences: 2127
Condition pairs: 378
Overlap rows (three granularities): 1134
Complete overlap rows: 1134 / 1134

Main outputs:
- condition_metrics.csv
- source_observations.csv
- source_category_composition.csv
- source_overlap_per_query.csv
- condition_metrics_aggregated.csv
- source_category_composition_aggregated.csv
- source_overlap_aggregated.csv
- analysis_summary.json

Source overlap is reported at canonical-URL, registrable-domain, and category levels:
J(A, B) = |A intersection B| / |A union B|.
O(A, B) = |A intersection B| / min(|A|, |B|).

Canonical URLs remove fragments, common tracking parameters, default ports,
and superficial www/trailing-slash differences. Registrable domains use the
bundled Public Suffix List snapshot from tldextract. Category overlap compares
the set of source categories present, not their frequencies.

The overlap coefficient is undefined when either set is empty and is exported
as a missing value in that case.

Source categories are deterministic heuristics based on hostname suffixes.
They are intended for exploration and should be reviewed before confirmatory analysis.
