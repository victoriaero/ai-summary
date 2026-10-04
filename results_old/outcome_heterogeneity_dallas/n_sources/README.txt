Outcome-level paired-difference heterogeneity pilot
===================================================

Metric: n_sources

Primary Dallas model
--------------------
paired_difference = minority value minus majority value

    paired_difference ~ C(social_dimension) + (1 | outcome)

Social dimension is a fixed effect. Outcome receives a random intercept.
The model contains 126 paired contrasts: 21 outcomes by 6 dimensions.
Optimizer: powell

Estimated outcome variance: 0.227937
Estimated residual variance: 18.870476
Outcome ICC: 0.011935
Estimated outcome SD (tau): 0.477427
Bootstrap 95% CI for tau: [0.000040, 1.543371]
Convergence status: converged
Singular fit: False
Approximate boundary-mixture p-value: 0.417389

The approximate p-value compares the ML mixed model with a fixed-dimension-only
OLS model. Variance-component null hypotheses lie on the boundary, so this is
reported as a diagnostic rather than definitive evidence.

Outcome-specific effects are empirical-Bayes/BLUP estimates. Their intervals
are approximate conditional intervals and do not include every source of
variance-component uncertainty.

Bootstrap
---------
Outcome clusters were resampled 500 times.
Successful fits: 500
Failed fits: 0

Negative-binomial robustness
----------------------------
Status: fitted

The feasible sensitivity model uses outcome fixed effects and standard errors
clustered by outcome. It is not a negative-binomial random-slope GLMM and must
not be described as one.

Descriptive variance decomposition
----------------------------------
The balanced-ANOVA decomposition is exploratory. Its residual combines
outcome-by-dimension heterogeneity, measurement noise, Google instability, and
other unobserved variation. It is not a final Generalizability Theory result.
The third component is therefore named "Outcome × Social Dimension +
Unresolved Error" rather than treated as a separately identified interaction.

Leave-one-dimension-out audit
-----------------------------
Near-white heatmap columns are not missing observations. They occur when the
refitted outcome variance reaches the numerical boundary. Boundary fits are
marked in the plot and in leave_one_dimension_out_models.csv.

Files
-----
- paired_contrasts.csv
- model_a_dimension_effects.csv
- model_a_outcome_effects.csv
- model_a_variance_components.csv
- model_a_variance_bootstrap.csv
- leave_one_dimension_out_estimates.csv
- leave_one_dimension_out_stability.csv
- leave_one_dimension_out_models.csv
- descriptive_variance_decomposition.csv
- negative_binomial_robustness.csv
- model_status.json
- model_diagnostics.json
- outcome_shrunken_effects.png
- outcome_random_deviations.png
- leave_one_dimension_out.png
- variance_decomposition_pilot.png
