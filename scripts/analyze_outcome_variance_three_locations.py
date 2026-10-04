#!/usr/bin/env python3
"""Exploratory three-location outcome heterogeneity analysis for source counts.

This analysis is deliberately separate from ``ai_summary_final_scripts``.  It
uses the three geographic collections as repeated measurements to estimate
quantities that could not be separated in the earlier Dallas-only pilot:

1. variation in the minority-majority slope across outcomes;
2. stable outcome, condition, and outcome-by-condition variance components;
3. outcome-by-social-dimension heterogeneity in paired differences; and
4. residual cross-location variation.

The models are Gaussian working models for counts and are therefore
exploratory dependence/heterogeneity diagnostics, not the primary count-model
inference used by the paper.
"""
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
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCATIONS = ("dallas", "ny", "la")
DIMENSIONS = (
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
)
RANDOM_SEED = 20261004
DEFAULT_BOOTSTRAP = 500


def load_source_counts(base: Path) -> pd.DataFrame:
    frames = []
    for location in LOCATIONS:
        path = base / "results" / f"link_count_analysis_{location}" / "clean_aio_link_data.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "query_id", "dimension", "condition", "group", "domain",
            "outcome", "outcome_id", "n_links",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        frame = frame.copy()
        frame["location"] = location
        frame["n_sources"] = pd.to_numeric(frame["n_links"], errors="raise")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["condition_id"] = data["query_id"].str.split("__", n=1).str[0]
    # Across repository versions the shared People response has been named
    # ``control``, ``generic``, or ``people``.
    control_labels = ["control", "generic", "people"]
    people = data["condition"].isin(control_labels)
    data.loc[people, "condition_id"] = "people"
    expected = len(LOCATIONS) * 21 * 13
    if len(data) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(data)}")
    duplicate = data.duplicated(["location", "outcome_id", "condition_id"])
    if duplicate.any():
        raise ValueError("Duplicate location x outcome x condition cells found")
    return data


def inventory(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for location, frame in data.groupby("location", sort=False):
        rows.append({
            "location": location,
            "n_responses": len(frame),
            "n_outcomes": frame["outcome_id"].nunique(),
            "n_conditions": frame["condition_id"].nunique(),
            "n_dimensions_excluding_people": frame.loc[
                ~frame.condition.isin(["control", "generic", "people"]), "dimension"
            ].nunique(),
            "n_people_controls": int(
                frame.condition.isin(["control", "generic", "people"]).sum()
            ),
            "missing_source_counts": int(frame.n_sources.isna().sum()),
        })
    return pd.DataFrame(rows)


def paired_differences(data: pd.DataFrame) -> pd.DataFrame:
    social = data.loc[data.condition.isin(["minority", "majority"])].copy()
    index = ["location", "dimension", "domain", "outcome", "outcome_id"]
    wide = social.pivot(index=index, columns="condition", values="n_sources").reset_index()
    if wide[["minority", "majority"]].isna().any().any():
        raise ValueError("Incomplete minority-majority pairs")
    wide["paired_difference"] = wide["minority"] - wide["majority"]
    wide["outcome_dimension"] = wide["outcome_id"] + " || " + wide["dimension"]
    return wide.sort_values(["location", "dimension", "outcome_id"]).reset_index(drop=True)


def fit_random_slope(data: pd.DataFrame):
    social = data.loc[data.condition.isin(["minority", "majority"])].copy()
    social["minority_indicator"] = social.condition.eq("minority").astype(float)
    group_column = "bootstrap_outcome" if "bootstrap_outcome" in social else "outcome_id"
    model = smf.mixedlm(
        "n_sources ~ minority_indicator * C(dimension) + C(location)",
        social,
        groups=social[group_column],
        re_formula="1 + minority_indicator",
    )
    errors = []
    for method in ("lbfgs", "bfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(reml=True, method=method, maxiter=3_000, disp=False)
            if result.converged:
                return result, method, social
            errors.append(f"{method}: not converged")
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    raise RuntimeError("Random-slope model failed: " + " | ".join(errors))


def average_slope_vector(result) -> np.ndarray:
    names = list(result.fe_params.index)
    vector = np.zeros(len(names))
    vector[names.index("minority_indicator")] = 1.0
    for dimension in DIMENSIONS:
        term = f"minority_indicator:C(dimension)[T.{dimension}]"
        if term in names:
            vector[names.index(term)] += 1.0 / len(DIMENSIONS)
    # The reference dimension contributes no interaction term.  Every present
    # non-reference interaction must also be averaged, including whichever
    # dimension patsy selected as the reference.
    reference_weight = 1.0 / len(DIMENSIONS)
    vector[names.index("minority_indicator")] = reference_weight * len(DIMENSIONS)
    return vector


def dimension_slopes(result) -> pd.DataFrame:
    names = list(result.fe_params.index)
    covariance = result.cov_params().loc[names, names].to_numpy(float)
    rows = []
    for dimension in DIMENSIONS:
        vector = np.zeros(len(names))
        vector[names.index("minority_indicator")] = 1
        term = f"minority_indicator:C(dimension)[T.{dimension}]"
        if term in names:
            vector[names.index(term)] = 1
        estimate = float(vector @ result.fe_params.to_numpy(float))
        se = float(np.sqrt(max(vector @ covariance @ vector, 0)))
        rows.append({
            "dimension": dimension,
            "estimated_minority_majority_difference": estimate,
            "standard_error": se,
            "ci_low_wald": estimate - 1.96 * se,
            "ci_high_wald": estimate + 1.96 * se,
        })
    return pd.DataFrame(rows)


def outcome_slopes(result, paired: pd.DataFrame) -> pd.DataFrame:
    dimension_summary = dimension_slopes(result)
    mean_fixed = float(dimension_summary.estimated_minority_majority_difference.mean())
    names = list(result.fe_params.index)
    fixed_vector = np.zeros(len(names))
    for dimension in DIMENSIONS:
        vector = np.zeros(len(names))
        vector[names.index("minority_indicator")] = 1
        term = f"minority_indicator:C(dimension)[T.{dimension}]"
        if term in names:
            vector[names.index(term)] = 1
        fixed_vector += vector / len(DIMENSIONS)
    fixed_cov = result.cov_params().loc[names, names].to_numpy(float)
    fixed_var = float(fixed_vector @ fixed_cov @ fixed_vector)
    metadata = paired[["outcome_id", "domain", "outcome"]].drop_duplicates()
    rows = []
    for row in metadata.itertuples(index=False):
        effect = result.random_effects[row.outcome_id]
        slope = float(effect.get("minority_indicator", np.nan))
        conditional = result.random_effects_cov[row.outcome_id]
        slope_var = float(conditional.loc["minority_indicator", "minority_indicator"])
        estimate = mean_fixed + slope
        se = np.sqrt(max(fixed_var + slope_var, 0))
        raw = paired.loc[paired.outcome_id.eq(row.outcome_id)]
        rows.append({
            "domain": row.domain,
            "outcome": row.outcome,
            "outcome_id": row.outcome_id,
            "raw_mean_difference_all_dimensions_locations": float(raw.paired_difference.mean()),
            "random_slope_deviation": slope,
            "shrunken_minority_majority_difference": estimate,
            "conditional_se_approx": se,
            "conditional_ci_low_approx": estimate - 1.96 * se,
            "conditional_ci_high_approx": estimate + 1.96 * se,
        })
    return pd.DataFrame(rows).sort_values("shrunken_minority_majority_difference").reset_index(drop=True)


def covariance_components(result) -> dict:
    cov = result.cov_re
    intercept_var = float(cov.iloc[0, 0])
    slope_var = float(cov.loc["minority_indicator", "minority_indicator"])
    covariance = float(cov.iloc[0, cov.columns.get_loc("minority_indicator")])
    denominator = np.sqrt(max(intercept_var * slope_var, 0))
    return {
        "outcome_intercept_variance": intercept_var,
        "outcome_intercept_sd": np.sqrt(max(intercept_var, 0)),
        "outcome_group_slope_variance": slope_var,
        "outcome_group_slope_sd_tau": np.sqrt(max(slope_var, 0)),
        "intercept_slope_covariance": covariance,
        "intercept_slope_correlation": covariance / denominator if denominator else np.nan,
        "residual_variance": float(result.scale),
        "residual_sd": np.sqrt(max(float(result.scale), 0)),
    }


def bootstrap_random_slope(data: pd.DataFrame, n_resamples: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    outcomes = data.outcome_id.unique()
    rows, failures = [], []
    for iteration in range(n_resamples):
        sampled = rng.choice(outcomes, size=len(outcomes), replace=True)
        pieces = []
        for draw, outcome in enumerate(sampled):
            piece = data.loc[data.outcome_id.eq(outcome)].copy()
            piece["bootstrap_outcome"] = f"draw_{draw:02d}::{outcome}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            fitted, method, _ = fit_random_slope(sample)
            row = covariance_components(fitted)
            row.update({"bootstrap_iteration": iteration, "optimizer": method})
            rows.append(row)
        except Exception as exc:
            failures.append({"bootstrap_iteration": iteration, "error": str(exc)})
    return pd.DataFrame(rows), {
        "requested": n_resamples,
        "successful": len(rows),
        "failed": len(failures),
        "first_failures": failures[:5],
    }


def summarize_random_slope(result, optimizer: str, bootstrap: pd.DataFrame) -> pd.DataFrame:
    row = covariance_components(result)
    row.update({
        "converged": bool(result.converged),
        "optimizer": optimizer,
        "n_outcomes": 21,
        "n_dimensions": 6,
        "n_locations": 3,
        "n_observations": int(result.nobs),
    })
    for column in (
        "outcome_intercept_variance", "outcome_group_slope_variance",
        "outcome_group_slope_sd_tau", "residual_variance",
    ):
        values = bootstrap[column].dropna()
        row[f"{column}_bootstrap_ci_low"] = float(values.quantile(.025))
        row[f"{column}_bootstrap_ci_high"] = float(values.quantile(.975))
    return pd.DataFrame([row])


def balanced_variance_decomposition(
    data: pd.DataFrame,
    *,
    row: str,
    column: str,
    replicate: str,
    value: str,
    row_label: str,
    column_label: str,
) -> pd.DataFrame:
    """Method-of-moments crossed random-effects decomposition.

    Location is removed as a fixed additive effect first.  With three complete
    observations per row x column cell, interaction and residual variability
    are separately estimable.
    """
    frame = data[[row, column, replicate, value]].dropna().copy()
    counts = frame.groupby([row, column]).size()
    if counts.nunique() != 1:
        raise ValueError("Variance decomposition requires balanced replication")
    n_rep = int(counts.iloc[0])
    n_row = frame[row].nunique()
    n_col = frame[column].nunique()
    if n_rep < 2:
        raise ValueError("At least two location replicates are required")
    grand_raw = float(frame[value].mean())
    location_means = frame.groupby(replicate)[value].mean()
    frame["adjusted"] = frame[value] - frame[replicate].map(location_means) + grand_raw
    grand = float(frame.adjusted.mean())
    row_means = frame.groupby(row).adjusted.mean()
    col_means = frame.groupby(column).adjusted.mean()
    cell_means = frame.groupby([row, column]).adjusted.mean()
    ss_row = n_col * n_rep * float(((row_means - grand) ** 2).sum())
    ss_col = n_row * n_rep * float(((col_means - grand) ** 2).sum())
    interaction_terms = []
    for (r, c), mean in cell_means.items():
        interaction_terms.append(mean - row_means[r] - col_means[c] + grand)
    ss_interaction = n_rep * float(np.square(interaction_terms).sum())
    frame["cell_mean"] = [cell_means[(r, c)] for r, c in zip(frame[row], frame[column])]
    ss_residual = float(np.square(frame.adjusted - frame.cell_mean).sum())
    df_row, df_col = n_row - 1, n_col - 1
    df_interaction = df_row * df_col
    df_residual = n_row * n_col * (n_rep - 1)
    ms_row = ss_row / df_row
    ms_col = ss_col / df_col
    ms_interaction = ss_interaction / df_interaction
    ms_residual = ss_residual / df_residual
    raw_components = {
        row_label: (ms_row - ms_interaction) / (n_col * n_rep),
        column_label: (ms_col - ms_interaction) / (n_row * n_rep),
        f"{row_label} × {column_label}": (ms_interaction - ms_residual) / n_rep,
        "Cross-location residual + unresolved error": ms_residual,
    }
    nonnegative = {key: max(value_, 0.0) for key, value_ in raw_components.items()}
    total = sum(nonnegative.values())
    ss_lookup = {
        row_label: ss_row,
        column_label: ss_col,
        f"{row_label} × {column_label}": ss_interaction,
        "Cross-location residual + unresolved error": ss_residual,
    }
    df_lookup = {
        row_label: df_row,
        column_label: df_col,
        f"{row_label} × {column_label}": df_interaction,
        "Cross-location residual + unresolved error": df_residual,
    }
    return pd.DataFrame([
        {
            "component": key,
            "variance_estimate_untruncated": raw_components[key],
            "variance_estimate_nonnegative": nonnegative[key],
            "share_of_nonnegative_total_variance": nonnegative[key] / total if total else np.nan,
            "sum_of_squares": ss_lookup[key],
            "degrees_of_freedom": df_lookup[key],
            "mean_square": ss_lookup[key] / df_lookup[key],
            "location_handling": "fixed additive location mean removed before decomposition",
            "n_row_levels": n_row,
            "n_column_levels": n_col,
            "n_location_replicates_per_cell": n_rep,
        }
        for key in raw_components
    ])


def sensitivity_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = [("full", data)]
    specifications += [(f"omit_location_{loc}", data.loc[data.location.ne(loc)]) for loc in LOCATIONS]
    specifications += [(f"omit_dimension_{dim}", data.loc[data.dimension.ne(dim)]) for dim in DIMENSIONS]
    for name, frame in specifications:
        result, method, _ = fit_random_slope(frame)
        row = covariance_components(result)
        row.update({
            "specification": name,
            "n_observations": int(result.nobs),
            "converged": bool(result.converged),
            "optimizer": method,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outcome_slopes(outcomes: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    ordered = outcomes.sort_values("shrunken_minority_majority_difference").copy()
    lookup = {value: i for i, value in enumerate(ordered.outcome_id)}
    fig, ax = plt.subplots(figsize=(11, 10))
    colors = plt.cm.tab10(np.linspace(0, .8, len(DIMENSIONS)))
    dimension_means = paired.groupby(["outcome_id", "dimension"], as_index=False).paired_difference.mean()
    offsets = dict(zip(DIMENSIONS, np.linspace(-.2, .2, len(DIMENSIONS))))
    for color, dimension in zip(colors, DIMENSIONS):
        part = dimension_means.loc[dimension_means.dimension.eq(dimension)]
        y = part.outcome_id.map(lookup).to_numpy(float) + offsets[dimension]
        ax.scatter(part.paired_difference, y, s=20, alpha=.55, color=color, label=dimension)
    y = np.arange(len(ordered))
    ax.errorbar(
        ordered.shrunken_minority_majority_difference,
        y,
        xerr=np.vstack([
            ordered.shrunken_minority_majority_difference - ordered.conditional_ci_low_approx,
            ordered.conditional_ci_high_approx - ordered.shrunken_minority_majority_difference,
        ]),
        fmt="D", color="black", ecolor="#555", capsize=2,
        markersize=5, linewidth=1.1, label="Random-slope estimate",
    )
    ax.axvline(0, color="#777", lw=1)
    ax.set_yticks(y, ordered.outcome_id, fontsize=8)
    ax.set_xlabel("Minority − majority source-count difference")
    ax.set_title("Outcome-specific minority–majority source-count differences\nThree-location random-slope model", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.07), ncol=4, frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "outcome_random_slope_effects.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_decompositions(full: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, frame, title in (
        (axes[0], full, "All 13 query conditions"),
        (axes[1], paired, "Minority–majority paired differences"),
    ):
        labels = frame.component.str.replace("Cross-location residual + unresolved error", "Cross-location residual\n+ unresolved error", regex=False)
        values = 100 * frame.share_of_nonnegative_total_variance
        ax.bar(labels, values, color=["#35689a", "#d27b35", "#7a4fa3", "#aaaaaa"])
        ax.set_ylim(0, max(100, float(values.max()) * 1.12))
        ax.set_ylabel("Share of estimated variance (%)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        for i, value in enumerate(values):
            ax.text(i, value + 1.5, f"{value:.1f}%", ha="center", fontsize=8)
    fig.suptitle("Exploratory variance decomposition of source counts", fontweight="bold")
    fig.text(.5, .01, "Additive location means removed; residual is variation across cities plus unresolved error.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(output / "variance_decomposition_three_locations.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    data = sensitivity.copy()
    data["tau"] = data.outcome_group_slope_sd_tau
    baseline = float(summary.outcome_group_slope_sd_tau.iloc[0])
    lo = float(summary.outcome_group_slope_sd_tau_bootstrap_ci_low.iloc[0])
    hi = float(summary.outcome_group_slope_sd_tau_bootstrap_ci_high.iloc[0])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(data))
    ax.axvspan(lo, hi, color="#b7d5e8", alpha=.45, label="Full-model outcome-bootstrap 95% interval")
    ax.axvline(baseline, color="#1f4e79", lw=1.5, label="Full-model estimate")
    ax.scatter(data.tau, y, color=np.where(data.specification.str.startswith("omit_location"), "#d9782d", "#5b5b5b"), s=55)
    ax.set_yticks(y, data.specification.str.replace("_", " "))
    ax.invert_yaxis()
    ax.set_xlabel("Outcome random-slope SD (τ)")
    ax.set_title("Stability of outcome-level group sensitivity", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "random_slope_sensitivity.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readme(output: Path, summary: pd.DataFrame, full_dec: pd.DataFrame, pair_dec: pd.DataFrame, boot_status: dict) -> None:
    row = summary.iloc[0]
    def shares(frame: pd.DataFrame) -> str:
        return "\n".join(
            f"- {r.component}: {100*r.share_of_nonnegative_total_variance:.2f}%"
            for r in frame.itertuples(index=False)
        )
    text = f"""# Three-location outcome heterogeneity and variance pilot

This is an exploratory analysis and is intentionally not part of the final-paper
pipeline. It uses source count (`n_links`) from Dallas, New York, and Los Angeles.

## Random-slope model

`n_sources ~ minority * social_dimension + location + (1 + minority | outcome)`

The random slope measures how much the minority-majority source-count contrast
varies systematically by outcome after controlling for dimension and location.

- outcome random-slope variance: {row.outcome_group_slope_variance:.6f}
- outcome random-slope SD (tau): {row.outcome_group_slope_sd_tau:.6f}
- outcome-bootstrap 95% CI for tau: [{row.outcome_group_slope_sd_tau_bootstrap_ci_low:.6f}, {row.outcome_group_slope_sd_tau_bootstrap_ci_high:.6f}]
- residual variance: {row.residual_variance:.6f}
- converged: {row.converged}
- bootstrap fits: {boot_status['successful']}/{boot_status['requested']} successful

The bootstrap interval is a cluster-bootstrap uncertainty diagnostic, not a
formal boundary test for a zero variance component. Outcome-specific intervals
are conditional approximations.

## Variance decomposition: all 13 query conditions

Location is treated as a fixed additive replication factor. The remaining
balanced variance is decomposed by method of moments.

{shares(full_dec)}

## Variance decomposition: paired minority-majority differences

This decomposition directly assesses whether group sensitivity is shared by an
outcome across dimensions or specific to an Outcome × Social Dimension pairing.

{shares(pair_dec)}

## Interpretation limits

- Three locations identify cross-location residual variation but are not a random
  sample of all possible locations.
- Source counts are discrete; the Gaussian mixed model is a working heterogeneity
  model, not the paper's primary count inference.
- The interaction component may include stable unmeasured features tied to each
  outcome-dimension pair.
- The residual still combines location interactions, collection instability, and
  measurement error because there is one observation per city and query cell.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def run(base: Path, output: Path, n_bootstrap: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = load_source_counts(base)
    inv = inventory(data)
    paired = paired_differences(data)
    result, optimizer, _ = fit_random_slope(data)
    dim = dimension_slopes(result)
    outcomes = outcome_slopes(result, paired)
    bootstrap, boot_status = bootstrap_random_slope(data, n_bootstrap)
    summary = summarize_random_slope(result, optimizer, bootstrap)
    sensitivity = sensitivity_models(data)
    full_dec = balanced_variance_decomposition(
        data, row="outcome_id", column="condition_id", replicate="location",
        value="n_sources", row_label="Outcome", column_label="Query Condition",
    )
    pair_dec = balanced_variance_decomposition(
        paired, row="outcome_id", column="dimension", replicate="location",
        value="paired_difference", row_label="Outcome", column_label="Social Dimension",
    )

    inv.to_csv(output / "analysis_inventory.csv", index=False)
    paired.to_csv(output / "paired_differences_by_location.csv", index=False)
    dim.to_csv(output / "random_slope_dimension_effects.csv", index=False)
    outcomes.to_csv(output / "random_slope_outcome_effects.csv", index=False)
    bootstrap.to_csv(output / "random_slope_bootstrap.csv", index=False)
    summary.to_csv(output / "random_slope_variance_components.csv", index=False)
    sensitivity.to_csv(output / "random_slope_sensitivity.csv", index=False)
    full_dec.to_csv(output / "variance_decomposition_all_conditions.csv", index=False)
    pair_dec.to_csv(output / "variance_decomposition_paired_differences.csv", index=False)
    (output / "analysis_manifest.json").write_text(json.dumps({
        "status": "exploratory_three_location_variance_analysis",
        "metric": "n_sources",
        "locations": list(LOCATIONS),
        "random_seed": RANDOM_SEED,
        "bootstrap": boot_status,
        "primary_paper_pipeline": False,
    }, indent=2) + "\n", encoding="utf-8")

    plot_outcome_slopes(outcomes, paired, output)
    plot_decompositions(full_dec, pair_dec, output)
    plot_sensitivity(sensitivity, summary, output)
    write_readme(output, summary, full_dec, pair_dec, boot_status)
    print(f"Results written to {output}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        # ``results`` may be owned by the collaborator who generated the final
        # paper outputs.  Keep this explicitly exploratory analysis separate so
        # it never mutates that frozen tree.
        default=PROJECT_ROOT / "exploratory_results" / "outcome_variance_three_locations" / "n_sources",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    run(args.base_dir.resolve(), args.output_dir.resolve(), args.bootstrap_resamples)


if __name__ == "__main__":
    main()
