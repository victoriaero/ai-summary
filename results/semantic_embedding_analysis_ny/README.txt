Google AI Overview Semantic Analysis
====================================

Collection:
v2_ny

Embedding model:
sentence-transformers/all-mpnet-base-v2

Primary metric:
dense_distance_subject_normalized

Primary inferential test:
paired sign-flip permutation test

Multiple testing:
Holm correction across the six planned social-dimension contrasts

Statistical unit:
matched domain/outcome

Output files
------------

all_results.json
    Master JSON containing essentially all derived metrics
    and statistical results.

SUMMARY.txt
    Human-readable high-level output.

document_metadata.csv
    Metadata and verbosity measures for all 273 documents.

group_vs_generic_semantic_metrics.csv
    Main 252 explicit-group vs generic matched comparisons.

semantic_metrics_by_group.csv
    Descriptive semantic metrics by group.

semantic_metrics_by_domain.csv
    Descriptive semantic metrics by domain.

semantic_metrics_by_outcome.csv
    Descriptive semantic metrics by outcome.

primary_dimension_tests_dense_subject_normalized.csv
    Main minority-vs-majority tests.

robustness_dimension_tests_dense_raw.csv
    Same tests using unnormalized presented text.

robustness_dimension_tests_tfidf.csv
    Lexical TF-IDF robustness check.

secondary_dimension_tests_generic_coverage.csv
    Sentence-level loss of generic content.

secondary_dimension_tests_group_novelty.csv
    Sentence-level content added by group marking.

secondary_dimension_tests_sentence_set_distance.csv
    Symmetric sentence-set distance.

aggregate_minority_vs_majority.csv
    Outcome-level aggregate comparisons.

global_friedman_tests.csv
    Global group effects.

exploratory_posthoc_group_tests.csv
    All 66 explicit-group comparisons, Holm corrected.

semantic_triangle_metrics.csv
    Minority/generic/majority geometry for every outcome.

semantic_triangle_summary.csv
    Triangle results summarized by dimension.

length_confound_diagnostics.csv
    Association of semantic distance with response-length change.

metric_agreement.csv
    Agreement between dense, lexical, and sentence metrics.

document_embeddings.npz
    Raw and subject-normalized document embeddings.
