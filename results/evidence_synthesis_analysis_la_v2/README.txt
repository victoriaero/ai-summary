Google AI Overview — Evidence -> Synthesis V2
==============================================

MAIN PURPOSE
------------

Estimate semantic alignment between query-relevant information
in cited sources and the final Google AI Overview.

This is NOT a formal claim-survival analysis.

The central metric should be described as an
"evidence-AIO semantic gap" or
"semantic evidence alignment".

Do NOT call 1-cosine a literal omission probability.


WHY V2
------

V1 had two important problems:

1. English source passages could be compared against
   Portuguese AIO text using an English-centric model.

2. Broken headings / formatting could cause almost an entire
   AIO to be treated as one "sentence".

V2 uses:

- sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- token-based source chunks
- token-based AIO chunks
- source-balanced aggregation
- exact experimental-label normalization
- coverage sensitivity analyses


SOURCE CHUNKS
-------------

Maximum:
    112 tokens

Overlap:
    40 tokens

Selected per cited source:
    3


AIO CHUNKS
----------

Maximum:
    96 tokens

Overlap:
    24 tokens


PRIMARY / REFERENCE SAMPLE
--------------------------

Reference threshold:
    >= 50%
    of cited sources successfully extracted

AND:
    >= 3
    usable cited sources.


SENSITIVITY
-----------

Coverage thresholds:

    40%, 50%, 60%, 70%, 80%


PRIMARY METRIC
--------------

evidence_semantic_gap_source_balanced

For each source:

1. Identify top query-relevant passages.
2. Match each passage to its most similar AIO chunk.
3. Average passage similarities within source.
4. Average sources equally.
5. Gap = 1 - alignment.


INFERENCE
---------

Within each social dimension:

minority/focal condition
vs
majority/comparison condition

matched on the same 21 outcomes.

Tests:

- paired sign-flip permutation
- bootstrap CI
- Wilcoxon confirmation
- Holm correction across six dimensions


PIPELINE CORRELATIONS
---------------------

Repeated observations are blocked by outcome.

The inferential correlation:

1. ranks variables within outcome;
2. centers ranks within outcome;
3. measures within-outcome rank association;
4. permutes Y only within outcome;
5. cluster-bootstraps entire outcomes.


IMPORTANT FILES
---------------

query_evidence_alignment_metrics_v2.csv

source_evidence_alignment_metrics_v2.csv

selected_query_relevant_evidence_v2.csv

aio_chunks_v2.csv

evidence_passage_matches_v2.csv

aio_chunk_support_v2.csv

evidence_alignment_by_group_v2.csv

reference_50pct_dimension_tests_v2.csv

sensitivity_dimension_tests_v2.csv

coverage_threshold_stability_v2.csv

aggregate_sensitivity_v2.csv

answer_support_sensitivity_v2.csv

coverage_bias_tests_v2.csv

metric_vs_coverage_diagnostics_v2.csv

mechanism_map_v2.csv

pipeline_blocked_correlations_v2.csv

weakly_aligned_relevant_evidence.csv

strongly_aligned_relevant_evidence.csv

all_results_v2.json
