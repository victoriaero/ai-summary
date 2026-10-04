DIF-inspired item-level group interaction analysis
==================================================

Metric: n_sources
Environments: dallas

The item is a domain/outcome query template. Delta is minority minus majority.
Items are ranked by their deviation from the mean delta of their social
dimension. robust_z uses the dimension median and MAD.

In the item-level figure, the black x marks the control on the same difference
scale: people minus the majority-group value for that outcome. Zero therefore
remains the majority-group reference.

A second figure uses the same layout and axis limits but omits the control x,
so the paired minority-majority contrasts can be read without that overlay.

With one environment, item-specific confidence intervals and random-slope
variance are not identifiable. They are intentionally left missing. When at
least three environments are supplied, the script also fits:

    value ~ group_minority + C(environment)
    random effects: (1 + group_minority | outcome)

Files:
- environment_item_contrasts.csv
- dif_item_summary.csv
- dimension_heterogeneity.csv
- top_3_candidates_by_dimension.csv
- model_status.json
- paired_group_differences_by_outcome.png
- paired_group_differences_by_outcome_without_control.png
