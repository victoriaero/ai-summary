# Three-location outcome heterogeneity and variance pilot

This is an exploratory analysis and is intentionally not part of the final-paper
pipeline. It uses source count (`n_links`) from Dallas, New York, and Los Angeles.

## Random-slope model

`n_sources ~ minority * social_dimension + location + (1 + minority | outcome)`

The random slope measures how much the minority-majority source-count contrast
varies systematically by outcome after controlling for dimension and location.

- outcome random-slope variance: 0.960395
- outcome random-slope SD (tau): 0.979997
- outcome-bootstrap 95% CI for tau: [0.176991, 1.570093]
- residual variance: 10.098796
- converged: True
- bootstrap fits: 500/500 successful

The bootstrap interval is a cluster-bootstrap uncertainty diagnostic, not a
formal boundary test for a zero variance component. Outcome-specific intervals
are conditional approximations.

## Variance decomposition: all 13 query conditions

Location is treated as a fixed additive replication factor. The remaining
balanced variance is decomposed by method of moments.

- Outcome: 11.42%
- Query Condition: 14.06%
- Outcome × Query Condition: 26.22%
- Cross-location residual + unresolved error: 48.30%

## Variance decomposition: paired minority-majority differences

This decomposition directly assesses whether group sensitivity is shared by an
outcome across dimensions or specific to an Outcome × Social Dimension pairing.

- Outcome: 1.06%
- Social Dimension: 4.31%
- Outcome × Social Dimension: 29.31%
- Cross-location residual + unresolved error: 65.31%

## Interpretation limits

- Three locations identify cross-location residual variation but are not a random
  sample of all possible locations.
- Source counts are discrete; the Gaussian mixed model is a working heterogeneity
  model, not the paper's primary count inference.
- The interaction component may include stable unmeasured features tied to each
  outcome-dimension pair.
- The residual still combines location interactions, collection instability, and
  measurement error because there is one observation per city and query cell.
