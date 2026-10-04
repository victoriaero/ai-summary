from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
import seaborn as sns
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "condition_source_analysis_dallas"
    / "condition_metrics.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "outcome_heterogeneity_dallas"

DIMENSION_ORDER = (
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
)

BOOTSTRAP_RESAMPLES = 500
RANDOM_SEED = 20260929
SINGULAR_VARIANCE_RATIO_TOLERANCE = 1e-6


def is_singular_fit(outcome_variance: float, residual_variance: float) -> bool:
    if residual_variance <= 0:
        return outcome_variance <= 0
    return (
        outcome_variance / residual_variance
        <= SINGULAR_VARIANCE_RATIO_TOLERANCE
    )


def slugify(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in value.lower()
    ).strip("_")


def load_data(path: Path, metric: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "query_id",
        "dimension",
        "condition",
        "domain",
        "outcome",
        metric,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data = data.copy()
    data["value"] = pd.to_numeric(data[metric], errors="coerce")
    data["outcome_id"] = data["domain"] + " :: " + data["outcome"]
    return data


def build_paired_contrasts(observations: pd.DataFrame) -> pd.DataFrame:
    social = observations.loc[
        observations["condition"].isin(["minority", "majority"])
    ].copy()
    keys = ["dimension", "domain", "outcome", "outcome_id"]
    minority = social.loc[
        social["condition"] == "minority",
        keys + ["group", "query_id", "value"],
    ].rename(
        columns={
            "group": "minority_group",
            "query_id": "minority_query_id",
            "value": "minority_value",
        }
    )
    majority = social.loc[
        social["condition"] == "majority",
        keys + ["group", "query_id", "value"],
    ].rename(
        columns={
            "group": "majority_group",
            "query_id": "majority_query_id",
            "value": "majority_value",
        }
    )
    contrasts = minority.merge(majority, on=keys, validate="one_to_one")
    contrasts["paired_difference"] = (
        contrasts["minority_value"] - contrasts["majority_value"]
    )
    dimension_rank = {
        dimension: index for index, dimension in enumerate(DIMENSION_ORDER)
    }
    contrasts["_dimension_rank"] = contrasts["dimension"].map(dimension_rank)
    return (
        contrasts.sort_values(["_dimension_rank", "domain", "outcome"])
        .drop(columns="_dimension_rank")
        .reset_index(drop=True)
    )


def fit_delta_mixed_model(
    contrasts: pd.DataFrame,
    reml: bool = True,
):
    model = smf.mixedlm(
        "paired_difference ~ C(dimension)",
        contrasts,
        groups=contrasts["outcome_id"],
        re_formula="1",
    )
    errors = []
    for method in ("powell", "bfgs", "cg", "nm", "lbfgs"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(
                    reml=reml,
                    method=method,
                    maxiter=2_000,
                    disp=False,
                )
            if result.converged:
                return model, result, method
            errors.append(f"{method}: did not converge")
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    raise RuntimeError("Mixed model failed. " + " | ".join(errors))


def fixed_design_vector(result, dimension: str) -> np.ndarray:
    names = list(result.fe_params.index)
    vector = np.zeros(len(names))
    vector[names.index("Intercept")] = 1
    term = f"C(dimension)[T.{dimension}]"
    if term in names:
        vector[names.index(term)] = 1
    return vector


def summarize_dimension_effects(result) -> pd.DataFrame:
    covariance = result.cov_params().loc[
        result.fe_params.index,
        result.fe_params.index,
    ].to_numpy(dtype=float)
    rows = []
    for dimension in DIMENSION_ORDER:
        vector = fixed_design_vector(result, dimension)
        estimate = float(vector @ result.fe_params.to_numpy())
        standard_error = float(np.sqrt(max(vector @ covariance @ vector, 0)))
        rows.append(
            {
                "dimension": dimension,
                "estimated_mean_paired_difference": estimate,
                "standard_error": standard_error,
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
            }
        )
    return pd.DataFrame(rows)


def average_fixed_effect(result) -> tuple[float, float]:
    vectors = np.vstack(
        [fixed_design_vector(result, dimension) for dimension in DIMENSION_ORDER]
    )
    average_vector = vectors.mean(axis=0)
    parameters = result.fe_params.to_numpy(dtype=float)
    covariance = result.cov_params().loc[
        result.fe_params.index,
        result.fe_params.index,
    ].to_numpy(dtype=float)
    estimate = float(average_vector @ parameters)
    variance = float(average_vector @ covariance @ average_vector)
    return estimate, max(variance, 0)


def summarize_outcome_effects(
    contrasts: pd.DataFrame,
    result,
) -> pd.DataFrame:
    average_effect, average_variance = average_fixed_effect(result)
    outcome_metadata = contrasts[
        ["outcome_id", "domain", "outcome"]
    ].drop_duplicates()
    rows = []
    for row in outcome_metadata.itertuples(index=False):
        random_effect = float(result.random_effects[row.outcome_id].iloc[0])
        conditional_variance = float(
            result.random_effects_cov[row.outcome_id].iloc[0, 0]
        )
        random_se = np.sqrt(max(conditional_variance, 0))
        total_se = np.sqrt(max(average_variance + conditional_variance, 0))
        rows.append(
            {
                "domain": row.domain,
                "outcome": row.outcome,
                "outcome_id": row.outcome_id,
                "raw_mean_paired_difference": contrasts.loc[
                    contrasts["outcome_id"] == row.outcome_id,
                    "paired_difference",
                ].mean(),
                "outcome_random_deviation": random_effect,
                "random_deviation_se": random_se,
                "random_deviation_ci_low": random_effect - 1.96 * random_se,
                "random_deviation_ci_high": random_effect + 1.96 * random_se,
                "shrunken_average_paired_difference": (
                    average_effect + random_effect
                ),
                "shrunken_effect_se_approx": total_se,
                "shrunken_effect_ci_low_approx": (
                    average_effect + random_effect - 1.96 * total_se
                ),
                "shrunken_effect_ci_high_approx": (
                    average_effect + random_effect + 1.96 * total_se
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["absolute_random_deviation_rank"] = (
        summary["outcome_random_deviation"]
        .abs()
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary.sort_values("outcome_random_deviation").reset_index(drop=True)


def bootstrap_variance_components(
    contrasts: pd.DataFrame,
    n_resamples: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    outcome_ids = contrasts["outcome_id"].unique()
    rows = []
    failures = []
    for iteration in range(n_resamples):
        sampled = rng.choice(outcome_ids, size=len(outcome_ids), replace=True)
        parts = []
        for draw, outcome_id in enumerate(sampled):
            part = contrasts.loc[
                contrasts["outcome_id"] == outcome_id
            ].copy()
            part["outcome_id"] = f"draw_{draw:02d}::{outcome_id}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        try:
            _, result, _ = fit_delta_mixed_model(sample, reml=True)
            outcome_variance = float(result.cov_re.iloc[0, 0])
            residual_variance = float(result.scale)
            rows.append(
                {
                    "bootstrap_iteration": iteration,
                    "outcome_variance": outcome_variance,
                    "outcome_sd": np.sqrt(max(outcome_variance, 0)),
                    "residual_variance": residual_variance,
                    "outcome_icc": (
                        outcome_variance
                        / (outcome_variance + residual_variance)
                    ),
                }
            )
        except Exception as exc:
            failures.append(
                {"bootstrap_iteration": iteration, "error": str(exc)}
            )
    return pd.DataFrame(rows), {
        "requested_resamples": n_resamples,
        "successful_resamples": len(rows),
        "failed_resamples": len(failures),
        "first_failures": failures[:5],
    }


def summarize_variance_components(
    reml_result,
    ml_result,
    contrasts: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    outcome_variance = float(reml_result.cov_re.iloc[0, 0])
    residual_variance = float(reml_result.scale)
    ols = smf.ols(
        "paired_difference ~ C(dimension)",
        data=contrasts,
    ).fit()
    likelihood_ratio = max(0.0, 2 * (ml_result.llf - ols.llf))
    boundary_mixture_p = (
        0.5 * chi2.sf(likelihood_ratio, df=1)
        if likelihood_ratio > 0
        else 1.0
    )

    row = {
        "n_contrasts": len(contrasts),
        "n_outcomes": contrasts["outcome_id"].nunique(),
        "n_dimensions": contrasts["dimension"].nunique(),
        "outcome_variance": outcome_variance,
        "outcome_random_variance": outcome_variance,
        "outcome_sd": np.sqrt(max(outcome_variance, 0)),
        "residual_variance": residual_variance,
        "residual_sd": np.sqrt(max(residual_variance, 0)),
        "outcome_icc": outcome_variance
        / (outcome_variance + residual_variance),
        "ICC_outcome": outcome_variance
        / (outcome_variance + residual_variance),
        "convergence_status": (
            "converged" if reml_result.converged else "not_converged"
        ),
        "singular_fit": is_singular_fit(
            outcome_variance,
            residual_variance,
        ),
        "singular_variance_ratio_tolerance": (
            SINGULAR_VARIANCE_RATIO_TOLERANCE
        ),
        "likelihood_ratio_vs_fixed_only": likelihood_ratio,
        "boundary_mixture_p_approx": boundary_mixture_p,
        "bootstrap_successful_n": len(bootstrap),
    }
    for column in (
        "outcome_variance",
        "outcome_sd",
        "residual_variance",
        "outcome_icc",
    ):
        values = bootstrap[column].dropna()
        row[f"{column}_bootstrap_ci_low"] = values.quantile(0.025)
        row[f"{column}_bootstrap_ci_high"] = values.quantile(0.975)
    row["bootstrap_CI_tau_low"] = row[
        "outcome_sd_bootstrap_ci_low"
    ]
    row["bootstrap_CI_tau_high"] = row[
        "outcome_sd_bootstrap_ci_high"
    ]
    return pd.DataFrame([row])


def leave_one_dimension_out(
    contrasts: pd.DataFrame,
    full_outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    model_rows = []
    full_lookup = full_outcomes.set_index("outcome_id")[
        "outcome_random_deviation"
    ]
    for omitted in DIMENSION_ORDER:
        subset = contrasts.loc[contrasts["dimension"] != omitted].copy()
        _, result, method = fit_delta_mixed_model(subset, reml=True)
        deviations = pd.Series(
            {
                outcome_id: float(effect.iloc[0])
                for outcome_id, effect in result.random_effects.items()
            },
            name="outcome_random_deviation",
        )
        correlation = spearmanr(
            full_lookup.loc[deviations.index],
            deviations,
        ).statistic
        model_rows.append(
            {
                "omitted_dimension": omitted,
                "converged": bool(result.converged),
                "optimizer": method,
                "outcome_variance": float(result.cov_re.iloc[0, 0]),
                "tau_squared_outcome": float(result.cov_re.iloc[0, 0]),
                "residual_variance": float(result.scale),
                "outcome_icc": (
                    float(result.cov_re.iloc[0, 0])
                    / (float(result.cov_re.iloc[0, 0]) + float(result.scale))
                ),
                "singular_fit": is_singular_fit(
                    float(result.cov_re.iloc[0, 0]),
                    float(result.scale),
                ),
                "spearman_with_full_model": correlation,
            }
        )
        for outcome_id, deviation in deviations.items():
            rows.append(
                {
                    "omitted_dimension": omitted,
                    "outcome_id": outcome_id,
                    "outcome_random_deviation": deviation,
                    "full_model_random_deviation": full_lookup[outcome_id],
                    "absolute_deviation_rank": int(
                        deviations.abs()
                        .rank(method="min", ascending=False)
                        .loc[outcome_id]
                    ),
                }
            )
    estimates = pd.DataFrame(rows)
    metadata = full_outcomes[
        ["outcome_id", "domain", "outcome"]
    ].drop_duplicates()
    estimates = estimates.merge(metadata, on="outcome_id", validate="many_to_one")
    stability = (
        estimates.groupby(["outcome_id", "domain", "outcome"], as_index=False)
        .agg(
            mean_loo_random_deviation=("outcome_random_deviation", "mean"),
            sd_loo_random_deviation=("outcome_random_deviation", "std"),
            min_loo_random_deviation=("outcome_random_deviation", "min"),
            max_loo_random_deviation=("outcome_random_deviation", "max"),
            mean_absolute_rank=("absolute_deviation_rank", "mean"),
            top_3_count=("absolute_deviation_rank", lambda values: int((values <= 3).sum())),
        )
    )
    stability = stability.merge(
        full_outcomes[
            [
                "outcome_id",
                "outcome_random_deviation",
                "absolute_random_deviation_rank",
            ]
        ],
        on="outcome_id",
        validate="one_to_one",
    )
    return estimates, stability, pd.DataFrame(model_rows)


def balanced_anova_decomposition(
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    outcome_count = contrasts["outcome_id"].nunique()
    dimension_count = contrasts["dimension"].nunique()
    grand_mean = contrasts["paired_difference"].mean()
    outcome_means = contrasts.groupby("outcome_id")[
        "paired_difference"
    ].mean()
    dimension_means = contrasts.groupby("dimension")[
        "paired_difference"
    ].mean()
    fitted = contrasts["outcome_id"].map(outcome_means)
    fitted += contrasts["dimension"].map(dimension_means)
    fitted -= grand_mean
    residuals = contrasts["paired_difference"] - fitted
    components = {
        "outcome": dimension_count
        * float(((outcome_means - grand_mean) ** 2).sum()),
        "social_dimension": outcome_count
        * float(((dimension_means - grand_mean) ** 2).sum()),
        "Outcome × Social Dimension + Unresolved Error": float(
            (residuals**2).sum()
        ),
    }
    total = sum(components.values())
    return pd.DataFrame(
        [
            {
                "component": component,
                "sum_of_squares": value,
                "share_of_total_sum_of_squares": value / total,
                "interpretation": (
                    "Descriptive only; the residual combines outcome-by-"
                    "dimension heterogeneity, measurement noise, Google "
                    "instability, and other unobserved variation."
                    if component
                    == "Outcome × Social Dimension + Unresolved Error"
                    else "Descriptive balanced-ANOVA component."
                ),
            }
            for component, value in components.items()
        ]
    )


def negative_binomial_robustness(
    observations: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, dict]:
    if metric != "n_sources":
        return pd.DataFrame(), {
            "status": "not_fitted",
            "reason": "Negative-binomial robustness is limited to n_sources.",
        }
    social = observations.loc[
        observations["condition"].isin(["minority", "majority"])
    ].copy()
    social["group_minority"] = (
        social["condition"] == "minority"
    ).astype(int)
    formula = (
        "value ~ group_minority * C(dimension) + C(outcome_id)"
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smf.negativebinomial(formula, social).fit(
                disp=False,
                maxiter=1_000,
                cov_type="cluster",
                cov_kwds={"groups": social["outcome_id"]},
            )
        covariance = result.cov_params()
        rows = []
        parameter_names = list(result.params.index)
        for dimension in DIMENSION_ORDER:
            vector = np.zeros(len(parameter_names))
            vector[parameter_names.index("group_minority")] = 1
            interaction = (
                f"group_minority:C(dimension)[T.{dimension}]"
            )
            if interaction in parameter_names:
                vector[parameter_names.index(interaction)] = 1
            estimate = float(vector @ result.params.to_numpy())
            variance = float(vector @ covariance.to_numpy() @ vector)
            standard_error = np.sqrt(max(variance, 0))

            prediction_rows = []
            for outcome_id in social["outcome_id"].unique():
                domain, outcome = outcome_id.split(" :: ", 1)
                for condition, group_minority in (
                    ("majority", 0),
                    ("minority", 1),
                ):
                    prediction_rows.append(
                        {
                            "dimension": dimension,
                            "domain": domain,
                            "outcome": outcome,
                            "outcome_id": outcome_id,
                            "condition": condition,
                            "group_minority": group_minority,
                        }
                    )
            predictions = pd.DataFrame(prediction_rows)
            predictions["predicted"] = result.predict(predictions)
            means = predictions.groupby("condition")["predicted"].mean()
            rows.append(
                {
                    "dimension": dimension,
                    "log_rate_ratio_minority_majority": estimate,
                    "standard_error": standard_error,
                    "ci_low": estimate - 1.96 * standard_error,
                    "ci_high": estimate + 1.96 * standard_error,
                    "rate_ratio": np.exp(estimate),
                    "rate_ratio_ci_low": np.exp(
                        estimate - 1.96 * standard_error
                    ),
                    "rate_ratio_ci_high": np.exp(
                        estimate + 1.96 * standard_error
                    ),
                    "average_predicted_sources_minority": means["minority"],
                    "average_predicted_sources_majority": means["majority"],
                    "average_predicted_difference": (
                        means["minority"] - means["majority"]
                    ),
                }
            )
        return pd.DataFrame(rows), {
            "status": "fitted",
            "converged": bool(result.mle_retvals.get("converged", False)),
            "formula": formula,
            "distribution": "negative binomial NB2",
            "outcome_handling": "fixed effects",
            "standard_errors": "clustered by outcome",
            "alpha": float(result.params["alpha"]),
            "aic": float(result.aic),
            "important_limitation": (
                "This is not the requested negative-binomial random-slope "
                "GLMM. It is a feasible Dallas sensitivity analysis with "
                "outcome fixed effects and outcome-clustered standard errors."
            ),
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "failed",
            "error": str(exc),
        }


def plot_shrunken_effects(
    outcomes: pd.DataFrame,
    contrasts: pd.DataFrame,
    dimension_effects: pd.DataFrame,
    output_dir: Path,
) -> None:
    data = outcomes.sort_values("shrunken_average_paired_difference").copy()
    order = data["outcome_id"].tolist()
    y_lookup = {outcome_id: index for index, outcome_id in enumerate(order)}
    dimension_means = dimension_effects.set_index("dimension")[
        "estimated_mean_paired_difference"
    ]
    grand_mean = float(dimension_means.mean())
    raw = contrasts.copy()
    raw["dimension_adjusted_paired_difference"] = (
        raw["paired_difference"]
        - raw["dimension"].map(dimension_means)
        + grand_mean
    )
    offsets = dict(
        zip(DIMENSION_ORDER, np.linspace(-0.18, 0.18, len(DIMENSION_ORDER)))
    )
    colors = dict(
        zip(DIMENSION_ORDER, sns.color_palette("colorblind", 6))
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(11, 11))
    for dimension in DIMENSION_ORDER:
        subset = raw.loc[raw["dimension"] == dimension]
        y = subset["outcome_id"].map(y_lookup).to_numpy(dtype=float)
        y += offsets[dimension]
        ax.scatter(
            subset["dimension_adjusted_paired_difference"],
            y,
            s=22,
            alpha=0.55,
            color=colors[dimension],
            edgecolors="none",
            label=dimension,
            zorder=2,
        )

    y = np.arange(len(data))
    errors = np.vstack(
        [
            data["shrunken_average_paired_difference"]
            - data["shrunken_effect_ci_low_approx"],
            data["shrunken_effect_ci_high_approx"]
            - data["shrunken_average_paired_difference"],
        ]
    )
    ax.errorbar(
        data["shrunken_average_paired_difference"],
        y,
        xerr=errors,
        fmt="D",
        markersize=6,
        color="#111111",
        ecolor="#555555",
        capsize=2,
        linewidth=1.2,
        label="Hierarchical estimate",
        zorder=4,
    )
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y, data["outcome_id"], fontsize=8)
    ax.set_xlabel("Minority − majority source-count difference")
    ax.set_ylabel("")
    ax.set_title(
        "Dimension-adjusted minority–majority source-count differences by outcome",
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.005,
        "Small circles: six adjusted raw contrasts; diamonds: hierarchical estimates",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        frameon=False,
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(
        output_dir / "outcome_shrunken_effects.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_outcome_effects(
    outcomes: pd.DataFrame,
    variance: pd.DataFrame,
    output_dir: Path,
) -> None:
    data = outcomes.sort_values("outcome_random_deviation").copy()
    y = np.arange(len(data))
    errors = np.vstack(
        [
            data["outcome_random_deviation"]
            - data["random_deviation_ci_low"],
            data["random_deviation_ci_high"]
            - data["outcome_random_deviation"],
        ]
    )
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.errorbar(
        data["outcome_random_deviation"],
        y,
        xerr=errors,
        fmt="o",
        color="#7B2CBF",
        ecolor="#999999",
        capsize=2,
    )
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y, data["outcome_id"], fontsize=8)
    ax.set_xlabel("Outcome random deviation from the dimension-adjusted mean")
    ax.set_ylabel("")
    variance_row = variance.iloc[0]
    ax.set_title(
        "Outcome-level deviations after controlling for social dimension\n"
        "Deviation from the dimension-adjusted grand mean\n"
        rf"$\hat{{\tau}}_{{outcome}}$ = {variance_row['outcome_sd']:.3f} "
        "(95% cluster-bootstrap CI "
        f"{variance_row['bootstrap_CI_tau_low']:.3f}–"
        f"{variance_row['bootstrap_CI_tau_high']:.3f})",
        fontweight="bold",
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(
        output_dir / "outcome_random_deviations.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_leave_one_out(
    estimates: pd.DataFrame,
    models: pd.DataFrame,
    output_dir: Path,
) -> None:
    heatmap = estimates.pivot(
        index="outcome_id",
        columns="omitted_dimension",
        values="outcome_random_deviation",
    ).reindex(columns=DIMENSION_ORDER)
    max_abs = np.nanmax(np.abs(heatmap.to_numpy()))
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        heatmap,
        cmap="vlag",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.3,
        cbar_kws={"label": "Outcome random deviation"},
        ax=ax,
    )
    model_lookup = models.set_index("omitted_dimension")
    labels = [
        (
            f"{dimension}\n"
            f"τ² = {model_lookup.loc[dimension, 'tau_squared_outcome']:.3g}"
            + (
                "\n(boundary)"
                if bool(model_lookup.loc[dimension, "singular_fit"])
                else ""
            )
        )
        for dimension in DIMENSION_ORDER
    ]
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Omitted social dimension")
    ax.set_ylabel("")
    ax.set_title(
        "Leave-one-dimension-out stability of outcome deviations\n"
        "Near-white boundary columns reflect τ² ≈ 0, not missing data",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "leave_one_dimension_out.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_variance(
    variance: pd.DataFrame,
    anova: pd.DataFrame,
    output_dir: Path,
) -> None:
    model_shares = pd.Series(
        {
            "Outcome": variance["outcome_variance"].iloc[0],
            "Residual": variance["residual_variance"].iloc[0],
        }
    )
    model_shares /= model_shares.sum()
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        model_shares.index,
        model_shares.values,
        color=["#7B2CBF", "#BBBBBB"],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of modeled random variance")
    axes[0].set_title("Paired-difference mixed model")
    anova_labels = {
        "outcome": "Outcome",
        "social_dimension": "Social Dimension",
        "Outcome × Social Dimension + Unresolved Error": (
            "Outcome × Social Dimension\n+ Unresolved Error"
        ),
    }
    axes[1].bar(
        anova["component"].map(anova_labels),
        anova["share_of_total_sum_of_squares"],
        color=["#2A6FBB", "#D95F02", "#BBBBBB"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share of total sum of squares")
    axes[1].set_title("Descriptive balanced-ANOVA decomposition")
    axes[1].tick_params(axis="x", rotation=12)
    fig.suptitle(
        "Pilot heterogeneity decomposition",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "variance_decomposition_pilot.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def write_readme(
    output_dir: Path,
    metric: str,
    optimizer: str,
    variance: pd.DataFrame,
    bootstrap_status: dict,
    nb_status: dict,
) -> None:
    row = variance.iloc[0]
    text = f"""Outcome-level paired-difference heterogeneity pilot
===================================================

Metric: {metric}

Primary Dallas model
--------------------
paired_difference = minority value minus majority value

    paired_difference ~ C(social_dimension) + (1 | outcome)

Social dimension is a fixed effect. Outcome receives a random intercept.
The model contains 126 paired contrasts: 21 outcomes by 6 dimensions.
Optimizer: {optimizer}

Estimated outcome variance: {row['outcome_variance']:.6f}
Estimated residual variance: {row['residual_variance']:.6f}
Outcome ICC: {row['outcome_icc']:.6f}
Estimated outcome SD (tau): {row['outcome_sd']:.6f}
Bootstrap 95% CI for tau: [{row['bootstrap_CI_tau_low']:.6f}, {row['bootstrap_CI_tau_high']:.6f}]
Convergence status: {row['convergence_status']}
Singular fit: {row['singular_fit']}
Approximate boundary-mixture p-value: {row['boundary_mixture_p_approx']:.6g}

The approximate p-value compares the ML mixed model with a fixed-dimension-only
OLS model. Variance-component null hypotheses lie on the boundary, so this is
reported as a diagnostic rather than definitive evidence.

Outcome-specific effects are empirical-Bayes/BLUP estimates. Their intervals
are approximate conditional intervals and do not include every source of
variance-component uncertainty.

Bootstrap
---------
Outcome clusters were resampled {bootstrap_status['requested_resamples']} times.
Successful fits: {bootstrap_status['successful_resamples']}
Failed fits: {bootstrap_status['failed_resamples']}

Negative-binomial robustness
----------------------------
Status: {nb_status.get('status')}

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
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def run_analysis(
    input_path: Path,
    output_root: Path,
    metric: str,
    bootstrap_resamples: int,
) -> Path:
    observations = load_data(input_path, metric)
    contrasts = build_paired_contrasts(observations)
    _, reml_result, optimizer = fit_delta_mixed_model(contrasts, reml=True)
    _, ml_result, ml_optimizer = fit_delta_mixed_model(contrasts, reml=False)
    dimension_effects = summarize_dimension_effects(reml_result)
    dimension_mean_lookup = dimension_effects.set_index("dimension")[
        "estimated_mean_paired_difference"
    ]
    contrasts["dimension_adjusted_paired_difference"] = (
        contrasts["paired_difference"]
        - contrasts["dimension"].map(dimension_mean_lookup)
        + float(dimension_mean_lookup.mean())
    )
    outcome_effects = summarize_outcome_effects(contrasts, reml_result)
    bootstrap, bootstrap_status = bootstrap_variance_components(
        contrasts,
        bootstrap_resamples,
    )
    variance = summarize_variance_components(
        reml_result,
        ml_result,
        contrasts,
        bootstrap,
    )
    loo_estimates, loo_stability, loo_models = leave_one_dimension_out(
        contrasts,
        outcome_effects,
    )
    anova = balanced_anova_decomposition(contrasts)
    nb_results, nb_status = negative_binomial_robustness(
        observations,
        metric,
    )

    output_dir = output_root / slugify(metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(output_dir / "paired_contrasts.csv", index=False)
    dimension_effects.to_csv(
        output_dir / "model_a_dimension_effects.csv",
        index=False,
    )
    outcome_effects.to_csv(
        output_dir / "model_a_outcome_effects.csv",
        index=False,
    )
    variance.to_csv(
        output_dir / "model_a_variance_components.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "model_a_variance_bootstrap.csv",
        index=False,
    )
    loo_estimates.to_csv(
        output_dir / "leave_one_dimension_out_estimates.csv",
        index=False,
    )
    loo_stability.to_csv(
        output_dir / "leave_one_dimension_out_stability.csv",
        index=False,
    )
    loo_models.to_csv(
        output_dir / "leave_one_dimension_out_models.csv",
        index=False,
    )
    anova.to_csv(
        output_dir / "descriptive_variance_decomposition.csv",
        index=False,
    )
    nb_results.to_csv(
        output_dir / "negative_binomial_robustness.csv",
        index=False,
    )

    model_status = {
        "primary_model": {
            "status": "fitted",
            "formula": (
                "paired_difference ~ C(dimension) + (1 | outcome)"
            ),
            "reml_optimizer": optimizer,
            "ml_optimizer": ml_optimizer,
            "converged_reml": bool(reml_result.converged),
            "converged_ml": bool(ml_result.converged),
            "singular_fit": bool(variance["singular_fit"].iloc[0]),
        },
        "variance_bootstrap": bootstrap_status,
        "negative_binomial_robustness": nb_status,
    }
    (output_dir / "model_status.json").write_text(
        json.dumps(model_status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    variance_row = variance.iloc[0]
    model_diagnostics = {
        "outcome_random_variance": float(
            variance_row["outcome_random_variance"]
        ),
        "residual_variance": float(variance_row["residual_variance"]),
        "ICC_outcome": float(variance_row["ICC_outcome"]),
        "tau_outcome": float(variance_row["outcome_sd"]),
        "bootstrap_CI_tau": [
            float(variance_row["bootstrap_CI_tau_low"]),
            float(variance_row["bootstrap_CI_tau_high"]),
        ],
        "convergence_status": str(variance_row["convergence_status"]),
        "singular_fit": bool(variance_row["singular_fit"]),
        "singular_fit_definition": (
            "outcome_random_variance / residual_variance <= "
            f"{SINGULAR_VARIANCE_RATIO_TOLERANCE:g}"
        ),
    }
    (output_dir / "model_diagnostics.json").write_text(
        json.dumps(model_diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_shrunken_effects(
        outcome_effects,
        contrasts,
        dimension_effects,
        output_dir,
    )
    plot_outcome_effects(outcome_effects, variance, output_dir)
    plot_leave_one_out(loo_estimates, loo_models, output_dir)
    plot_variance(variance, anova, output_dir)
    write_readme(
        output_dir,
        metric,
        optimizer,
        variance,
        bootstrap_status,
        nb_status,
    )
    print(f"Paired contrasts: {len(contrasts)}")
    print(f"Outcome variance: {variance['outcome_variance'].iloc[0]:.6f}")
    print(f"Outcome ICC: {variance['outcome_icc'].iloc[0]:.6f}")
    print(f"Results written to: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Model Dallas outcome heterogeneity using paired group contrasts."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--metric",
        default="n_sources",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
    )
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    run_analysis(
        args.input.resolve(),
        args.output_root.resolve(),
        args.metric,
        args.bootstrap_resamples,
    )


if __name__ == "__main__":
    main()
