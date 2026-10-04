Automatic source authority and turnover pilot
=============================================

Input: /scratch/victoria.estanislau/ai-summary/results/condition_source_analysis_dallas
Source occurrences: 2127
Unique canonical URLs: 1595
Unique registrable domains: 720
Turnover rows: 756

Authority taxonomy
------------------
The authority taxonomy is fully automatic and deterministic. It uses known
domain families, public/academic suffixes, domain-name tokens, and fallback
suffix rules. No manual source annotations are used.

classification_rule records the rule used for every source.
classification_confidence describes rule specificity, not validated accuracy.
Low-confidence and fallback assignments must be treated as exploratory.

Composition uses unique canonical URLs within each query. For zero-source
queries, category shares are missing rather than zero.

Expansion and substitution
--------------------------
All three paired comparisons are included:
- minority_majority
- minority_generic
- majority_generic

For each pair and granularity:
- shared = A intersection B
- a_only = A minus B
- b_only = B minus A
- replacement_mass = min(a_only, b_only)
- net_a_minus_b = a_only minus b_only
- E = (a_only minus b_only) / union
- T = (a_only plus b_only) / union = 1 minus Jaccard

Uncertainty and inference
-------------------------
Mean authority-share contrasts and normalized turnover metrics receive
percentile confidence intervals from 10,000 paired outcome bootstraps.

Category-specific binary GLMs preserve each social dimension separately and
adjust for outcome fixed effects. Standard errors are clustered by query.
These models estimate composition conditional on a source being present; they
do not model whether an AIO or source list exists.
Rare-category models with quasi-complete separation are flagged and receive no
p-values. They must not be interpreted as evidence of very large effects.

A complementary variational-Bayes binomial mixed model uses a random intercept
by outcome. Its intervals are approximate posterior credible intervals under
the mean-field approximation and are reported as a sensitivity analysis.

authority_total_variation and normalized Jensen-Shannon divergence measure
how much the complete authority-composition vector changes. Their association
with source turnover is summarized with Spearman correlations and outcome-
clustered bootstrap intervals.

Taxonomy validation
-------------------
artifacts/annotation_inputs/authority_taxonomy_validation_sample.csv is a
deterministic stratified sample with empty human-label columns. Strata are
automatic category by heuristic confidence. Because rare strata are
oversampled, validation metrics should use sampling_probability or inverse-
probability weights.

The control query ("people") is reused across social dimensions. Comparisons
against control are substantively useful but are not independent replications
across dimensions.

Main files
----------
- source_authority_observations.csv
- automatic_authority_domain_crosswalk.csv
- authority_composition_by_query.csv
- authority_paired_contrasts.csv
- authority_contrasts_aggregated.csv
- authority_contrast_bootstrap.csv
- authority_binary_models.csv
- authority_hierarchical_models.csv
- artifacts/annotation_inputs/authority_taxonomy_validation_sample.csv
- source_turnover_by_pair.csv
- source_turnover_aggregated.csv
- source_turnover_bootstrap.csv
- source_set_membership.csv
- authority_by_set_membership.csv
- authority_distance_by_pair.csv
- source_turnover_authority_link.csv
- source_turnover_authority_association.csv
- analysis_summary.json
- authority_composition_by_condition.png
- authority_share_difference_by_dimension.png
- normalized_source_expansion_turnover.png
- source_set_components.png
- source_turnover_vs_authority_change.png
