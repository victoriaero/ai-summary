from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "results"
    / "condition_source_analysis_dallas"
    / "condition_metrics.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "dif_inspired_item_analysis"

DIMENSION_ORDER = (
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
)

SUPPORTED_METRICS = (
    "n_sources",
    "n_unique_sources",
    "n_unique_domains",
    "aio_chars",
    "aio_words",
    "government_source_share",
)

METRIC_LABELS = {
    "n_sources": "number of sources",
    "n_unique_sources": "number of unique sources",
    "n_unique_domains": "number of unique domains",
    "aio_chars": "AIO length (characters)",
    "aio_words": "AIO length (words)",
    "government_source_share": "government-source share",
}


def parse_environment_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use ENVIRONMENT=/path/to/condition_metrics.csv")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Environment name and CSV path are required")
    return name.strip(), Path(raw_path).expanduser().resolve()


def load_environment(
    environment: str,
    metrics_path: Path,
    metric: str,
) -> pd.DataFrame:
    data = pd.read_csv(metrics_path)
    required = {"query_id", "dimension", "condition", "group", "domain", "outcome"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")

    if metric == "government_source_share":
        composition_path = metrics_path.with_name("source_category_composition.csv")
        composition = pd.read_csv(composition_path)
        government = composition.loc[
            composition["source_category"] == "government",
            ["query_id", "category_source_share"],
        ].rename(columns={"category_source_share": metric})
        data = data.merge(government, on="query_id", validate="one_to_one")
        data[metric] = data[metric].fillna(0.0)
    elif metric not in data.columns:
        raise ValueError(f"Metric {metric!r} is not present in {metrics_path}")

    data = data.copy()
    data["environment"] = environment
    data["value"] = pd.to_numeric(data[metric], errors="coerce")
    return data


def build_contrasts(observations: pd.DataFrame) -> pd.DataFrame:
    social = observations.loc[
        observations["condition"].isin(["minority", "majority"])
    ].copy()
    keys = ["environment", "dimension", "domain", "outcome"]
    minority = social.loc[
        social["condition"] == "minority",
        keys + ["group", "value"],
    ].rename(columns={"group": "minority_group", "value": "minority_value"})
    majority = social.loc[
        social["condition"] == "majority",
        keys + ["group", "value"],
    ].rename(columns={"group": "majority_group", "value": "majority_value"})
    contrasts = minority.merge(majority, on=keys, validate="one_to_one")
    control_keys = ["environment", "domain", "outcome"]
    control = observations.loc[
        observations["condition"] == "control",
        control_keys + ["group", "value"],
    ].rename(columns={"group": "control_group", "value": "control_value"})
    contrasts = contrasts.merge(
        control,
        on=control_keys,
        validate="many_to_one",
    )
    contrasts["delta"] = contrasts["minority_value"] - contrasts["majority_value"]
    contrasts["control_minus_majority"] = (
        contrasts["control_value"] - contrasts["majority_value"]
    )
    contrasts["outcome_id"] = contrasts["domain"] + " :: " + contrasts["outcome"]
    return contrasts


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    critical = t.ppf((1 + confidence) / 2, df=len(values) - 1)
    mean = values.mean()
    return mean - critical * standard_error, mean + critical * standard_error


def summarize_items(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "dimension",
        "domain",
        "outcome",
        "outcome_id",
        "minority_group",
        "majority_group",
    ]
    for keys, group in contrasts.groupby(group_columns, sort=False):
        values = group["delta"].dropna()
        control_differences = group["control_minus_majority"].dropna()
        ci_low, ci_high = mean_ci(values)
        mean_delta = values.mean()
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_environments": len(values),
                "mean_delta": mean_delta,
                "median_delta": values.median(),
                "control_group": group["control_group"].iloc[0],
                "mean_control_value": group["control_value"].mean(),
                "mean_control_minus_majority": control_differences.mean(),
                "sd_across_environments": (
                    values.std(ddof=1) if len(values) > 1 else np.nan
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "positive_environments": int((values > 0).sum()),
                "equal_environments": int((values == 0).sum()),
                "negative_environments": int((values < 0).sum()),
                "direction_consistency": (
                    max((values > 0).mean(), (values < 0).mean())
                    if len(values)
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)
    scored = []
    for dimension, group in summary.groupby("dimension", sort=False):
        group = group.copy()
        deltas = group["mean_delta"]
        dimension_mean = deltas.mean()
        dimension_median = deltas.median()
        sd = deltas.std(ddof=1)
        mad = np.median(np.abs(deltas - dimension_median))
        robust_scale = 1.4826 * mad
        group["dimension_mean_delta"] = dimension_mean
        group["dimension_median_delta"] = dimension_median
        group["centered_delta"] = deltas - dimension_mean
        group["standardized_deviation"] = (
            group["centered_delta"] / sd if sd > 0 else np.nan
        )
        group["robust_z"] = (
            (deltas - dimension_median) / robust_scale
            if robust_scale > 0
            else np.nan
        )
        n_items = len(group)
        group["leave_one_out_dimension_mean"] = (
            (deltas.sum() - deltas) / (n_items - 1)
            if n_items > 1
            else np.nan
        )
        group["deviation_from_leave_one_out"] = (
            deltas - group["leave_one_out_dimension_mean"]
        )
        group["absolute_deviation_rank"] = (
            group["centered_delta"].abs().rank(method="min", ascending=False).astype(int)
        )
        group["robust_outlier"] = group["robust_z"].abs() >= 2
        group["top_3_extreme"] = group["absolute_deviation_rank"] <= 3
        scored.append(group)
    return pd.concat(scored, ignore_index=True)


def summarize_heterogeneity(item_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, group in item_summary.groupby("dimension", sort=False):
        values = group["mean_delta"].dropna()
        median = values.median()
        rows.append(
            {
                "dimension": dimension,
                "n_outcomes": len(values),
                "mean_delta": values.mean(),
                "median_delta": median,
                "between_outcome_variance": values.var(ddof=1),
                "between_outcome_sd": values.std(ddof=1),
                "mad": np.median(np.abs(values - median)),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
                "min_delta": values.min(),
                "max_delta": values.max(),
                "range": values.max() - values.min(),
                "minority_higher_n": int((values > 0).sum()),
                "equal_n": int((values == 0).sum()),
                "majority_higher_n": int((values < 0).sum()),
                "robust_outlier_n": int(group["robust_outlier"].sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_random_slope_models(
    observations: pd.DataFrame,
    output_dir: Path,
) -> dict:
    environments = observations["environment"].nunique()
    if environments < 3:
        return {
            "status": "not_fitted",
            "reason": (
                "At least three replication environments are required before "
                "estimating outcome-specific random slopes."
            ),
            "n_environments": int(environments),
        }

    import statsmodels.formula.api as smf

    model_rows = []
    random_rows = []
    failures = []
    social = observations.loc[
        observations["condition"].isin(["minority", "majority"])
    ].copy()
    social["group_minority"] = (social["condition"] == "minority").astype(int)
    social["outcome_id"] = social["domain"] + " :: " + social["outcome"]

    for dimension in DIMENSION_ORDER:
        data = social.loc[social["dimension"] == dimension].dropna(subset=["value"])
        try:
            model = smf.mixedlm(
                "value ~ group_minority + C(environment)",
                data,
                groups=data["outcome_id"],
                re_formula="~group_minority",
            )
            result = model.fit(reml=True, method="lbfgs")
            fixed_effect = float(result.params["group_minority"])
            slope_variance = float(
                result.cov_re.loc["group_minority", "group_minority"]
            )
            model_rows.append(
                {
                    "dimension": dimension,
                    "converged": bool(result.converged),
                    "n_observations": len(data),
                    "n_outcomes": data["outcome_id"].nunique(),
                    "fixed_group_effect": fixed_effect,
                    "group_slope_variance": slope_variance,
                    "group_slope_sd": np.sqrt(max(slope_variance, 0)),
                    "aic": result.aic,
                    "bic": result.bic,
                }
            )
            for outcome_id, effects in result.random_effects.items():
                random_slope = float(effects.get("group_minority", np.nan))
                random_rows.append(
                    {
                        "dimension": dimension,
                        "outcome_id": outcome_id,
                        "fixed_group_effect": fixed_effect,
                        "outcome_random_slope": random_slope,
                        "outcome_specific_effect": fixed_effect + random_slope,
                    }
                )
        except Exception as exc:
            failures.append({"dimension": dimension, "error": str(exc)})

    pd.DataFrame(model_rows).to_csv(
        output_dir / "random_slope_model_summary.csv", index=False
    )
    pd.DataFrame(random_rows).to_csv(
        output_dir / "random_slope_outcome_effects.csv", index=False
    )
    return {
        "status": "fitted" if model_rows else "failed",
        "n_environments": int(environments),
        "fitted_dimensions": len(model_rows),
        "failures": failures,
    }


def plot_item_effects(
    item_summary: pd.DataFrame,
    output_dir: Path,
    metric: str,
    show_control: bool,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(18, 15), sharex=False)
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))
    for ax, dimension in zip(axes.flat, DIMENSION_ORDER):
        data = item_summary.loc[item_summary["dimension"] == dimension].copy()
        data = data.sort_values("mean_delta")
        minority_group = data["minority_group"].iloc[0]
        majority_group = data["majority_group"].iloc[0]
        y = np.arange(len(data))
        colors = np.where(data["mean_delta"] >= 0, "#7B2CBF", "#2A6FBB")
        ax.hlines(y, 0, data["mean_delta"], color="#BBBBBB", linewidth=1)
        ax.scatter(data["mean_delta"], y, color=colors, s=34, zorder=3)
        if show_control:
            ax.scatter(
                data["mean_control_minus_majority"],
                y,
                marker="x",
                color="#111111",
                linewidths=1.3,
                s=42,
                zorder=4,
                label="People − majority",
            )
        replicated = data["n_environments"].min() >= 2
        if replicated:
            errors = np.vstack(
                [
                    data["mean_delta"] - data["ci_low"],
                    data["ci_high"] - data["mean_delta"],
                ]
            )
            ax.errorbar(
                data["mean_delta"],
                y,
                xerr=errors,
                fmt="none",
                ecolor="#666666",
                alpha=0.7,
                capsize=2,
            )
        ax.axvline(0, color="#222222", linewidth=1)
        ax.axvline(
            data["dimension_mean_delta"].iloc[0],
            color="#D95F02",
            linestyle="--",
            linewidth=1.2,
            label="Mean across outcomes",
        )
        x_extent = max(
            data["mean_delta"].abs().max(),
            data["mean_control_minus_majority"].abs().max(),
            abs(data["dimension_mean_delta"].iloc[0]),
            1,
        )
        ax.set_xlim(-1.12 * x_extent, 1.12 * x_extent)
        ax.set_yticks(y, data["outcome"], fontsize=7)
        ax.set_title(dimension, pad=34, fontweight="bold")
        ax.text(
            0.01,
            1.01,
            f"← More sources for\n{majority_group}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#2A6FBB",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            0.99,
            1.01,
            f"More sources for →\n{minority_group}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#7B2CBF",
            fontsize=8,
            fontweight="bold",
        )
        ax.set_xlabel(f"Paired difference in {metric_label}")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        f"Paired differences in {metric_label} across outcomes",
        fontsize=15,
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    filename = (
        "paired_group_differences_by_outcome.png"
        if show_control
        else "paired_group_differences_by_outcome_without_control.png"
    )
    fig.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run_analysis(
    environment_paths: list[tuple[str, Path]],
    metric: str,
    output_root: Path,
) -> Path:
    observations = pd.concat(
        [
            load_environment(environment, path, metric)
            for environment, path in environment_paths
        ],
        ignore_index=True,
    )
    contrasts = build_contrasts(observations)
    item_summary = summarize_items(contrasts)
    heterogeneity = summarize_heterogeneity(item_summary)
    output_dir = output_root / slugify(metric)
    output_dir.mkdir(parents=True, exist_ok=True)

    contrasts.to_csv(output_dir / "environment_item_contrasts.csv", index=False)
    item_summary.to_csv(output_dir / "dif_item_summary.csv", index=False)
    heterogeneity.to_csv(output_dir / "dimension_heterogeneity.csv", index=False)
    (
        item_summary.sort_values(
            ["dimension", "absolute_deviation_rank"]
        )
        .groupby("dimension", sort=False)
        .head(3)
        .to_csv(output_dir / "top_3_candidates_by_dimension.csv", index=False)
    )
    model_status = fit_random_slope_models(observations, output_dir)
    (output_dir / "model_status.json").write_text(
        json.dumps(model_status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_item_effects(item_summary, output_dir, metric, show_control=True)
    plot_item_effects(item_summary, output_dir, metric, show_control=False)

    readme = f"""DIF-inspired item-level group interaction analysis
==================================================

Metric: {metric}
Environments: {', '.join(name for name, _ in environment_paths)}

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
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DIF-inspired item-level minority-majority analysis."
    )
    parser.add_argument(
        "--environment",
        action="append",
        type=parse_environment_argument,
        help="Repeatable ENVIRONMENT=/path/to/condition_metrics.csv.",
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="n_sources",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    environments = args.environment or [("dallas", DEFAULT_METRICS)]
    output_dir = run_analysis(environments, args.metric, args.output_root.resolve())
    print(f"DIF-inspired results written to: {output_dir}")


if __name__ == "__main__":
    main()
