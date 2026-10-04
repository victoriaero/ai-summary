Google AI Overview — Source Overlap Analysis
=============================================

Collection:
v3_la

Primary questions
-----------------

1. How much does the exact URL set change when a social
   identity is explicitly mentioned relative to the matched
   generic "people" query?

2. How much does the publisher/domain set change?

3. Is the displacement systematically different for
   minority-marked and majority/comparison-marked conditions?

Primary measures
----------------

URL Jaccard similarity:

    |URL_group ∩ URL_people|
    -------------------------
    |URL_group ∪ URL_people|

URL displacement:

    1 - URL Jaccard similarity

Domain displacement:

    1 - Domain Jaccard similarity

Statistical design
------------------

The statistical unit is the matched domain/outcome.

Within each social dimension, the 21 minority observations
are paired with the same 21 majority/comparison observations.

Primary test:
paired sign-flip permutation test.

Secondary confirmation:
Wilcoxon signed-rank test.

Multiple comparisons:
Holm correction across the six social dimensions.

Files
-----

all_results.json
    Complete analysis output.

SUMMARY.txt
    Human-readable result summary.

source_overlap_vs_generic.csv
    All 252 explicit-group vs generic comparisons.

source_overlap_by_group.csv
    Descriptive results by social group.

source_overlap_by_domain.csv
    Descriptive results by domain.

source_overlap_by_outcome.csv
    Descriptive results by outcome.

minority_vs_majority_url_distance.csv
    Main URL-overlap statistical tests.

minority_vs_majority_domain_distance.csv
    Main domain-overlap statistical tests.

minority_vs_majority_url_generic_coverage.csv
    Fraction of generic URLs preserved.

minority_vs_majority_domain_generic_coverage.csv
    Fraction of generic domains preserved.

aggregate_minority_vs_majority.csv
    Aggregate 21-outcome minority-vs-majority tests.

global_friedman_tests.csv
    Global group effects.

zero_source_cases.csv
    Responses containing no sources.

most_different_url_sets.csv
    Largest source-set displacements.

most_similar_url_sets.csv
    Smallest source-set displacements.

most_different_domain_sets.csv
    Largest publisher/domain-set displacements.
