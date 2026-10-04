from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "condition_source_analysis_dallas"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "figures"

DIMENSION_ORDER = (
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
)

COMPARISON_ORDER = (
    "minority_majority",
    "minority_generic",
    "majority_generic",
)

COMPARISON_LABELS = {
    "minority_majority": "Minority × majority",
    "minority_generic": "Minority × generic",
    "majority_generic": "Majority × generic",
}

GRANULARITY_ORDER = (
    "canonical_url",
    "registrable_domain",
    "source_category",
)

GRANULARITY_LABELS = {
    "canonical_url": "Canonical URL",
    "registrable_domain": "Registrable domain",
    "source_category": "Source category",
}

COMPARISON_COLORS = {
    "minority_majority": "#7B2CBF",
    "minority_generic": "#D95F02",
    "majority_generic": "#1B9E77",
}

CONDITION_COLORS = {
    "minority": "#7B2CBF",
    "majority": "#2A6FBB",
    "control": "#555555",
}

SHORT_GROUP_LABELS = {
    "Black people": "Black",
    "White people": "White",
    "Latino people": "Latino",
    "non-Latino people": "non-Latino",
    "Women": "Women",
    "Men": "Men",
    "People with disabilities": "With disabilities",
    "People without disabilities": "Without disabilities",
    "homosexual people": "Homosexual",
    "heterosexual people": "Heterosexual",
    "transgender people": "Transgender",
    "cisgender people": "Cisgender",
    "people": "Generic people",
}


def slugify(value: str) -> str:
    value = value.strip().lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_dimension_data(metrics: pd.DataFrame, dimension: str) -> pd.DataFrame:
    subset = metrics.loc[metrics["dimension"] == dimension]
    minority = subset.loc[
        subset["condition"] == "minority",
        ["domain", "outcome", "group", "n_sources"],
    ].rename(
        columns={
            "group": "minority_group",
            "n_sources": "minority_sources",
        }
    )
    majority = subset.loc[
        subset["condition"] == "majority",
        ["domain", "outcome", "group", "n_sources"],
    ].rename(
        columns={
            "group": "majority_group",
            "n_sources": "majority_sources",
        }
    )
    paired = minority.merge(
        majority,
        on=["domain", "outcome"],
        validate="one_to_one",
    )
    paired["difference"] = paired["minority_sources"] - paired["majority_sources"]
    paired["dimension"] = dimension
    return paired


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def plot_forest(
    metrics: pd.DataFrame,
    output_dir: Path,
    n_boot: int,
) -> None:
    rows = []
    for index, dimension in enumerate(DIMENSION_ORDER):
        paired = paired_dimension_data(metrics, dimension)
        differences = paired["difference"].to_numpy()
        ci_low, ci_high = bootstrap_mean_ci(differences, n_boot, 42 + index)
        rows.append(
            {
                "dimension": dimension,
                "minority_group": paired["minority_group"].iloc[0],
                "majority_group": paired["majority_group"].iloc[0],
                "n_pairs": len(paired),
                "mean_difference": differences.mean(),
                "median_difference": np.median(differences),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "minority_higher_n": int((differences > 0).sum()),
                "equal_n": int((differences == 0).sum()),
                "majority_higher_n": int((differences < 0).sum()),
            }
        )

    forest = pd.DataFrame(rows)
    forest.to_csv(output_dir / "forest_paired_effects_data.csv", index=False)

    plot_data = forest.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(plot_data))
    means = plot_data["mean_difference"].to_numpy()
    errors = np.vstack(
        [
            means - plot_data["ci_low"].to_numpy(),
            plot_data["ci_high"].to_numpy() - means,
        ]
    )
    ax.errorbar(
        means,
        y,
        xerr=errors,
        fmt="o",
        color="#5B2A86",
        ecolor="#5B2A86",
        capsize=4,
        markersize=7,
        linewidth=1.8,
    )
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    labels = [
        f"{row.dimension}\n{SHORT_GROUP_LABELS.get(row.minority_group, row.minority_group)} − "
        f"{SHORT_GROUP_LABELS.get(row.majority_group, row.majority_group)}"
        for row in plot_data.itertuples()
    ]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired mean difference in number of sources")
    ax.set_title("Minority–majority paired effects by social dimension")
    ax.text(
        0.99,
        0.02,
        f"Points: mean difference; bars: 95% bootstrap CI ({n_boot:,} resamples)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_1_forest_paired_source_effects")


def ordered_groups(metrics: pd.DataFrame) -> list[str]:
    groups = []
    for dimension in DIMENSION_ORDER:
        for condition in ("minority", "majority"):
            values = metrics.loc[
                (metrics["dimension"] == dimension)
                & (metrics["condition"] == condition),
                "group",
            ].unique()
            if len(values) == 1:
                groups.append(values[0])
    return groups


def plot_heatmap(metrics: pd.DataFrame, output_dir: Path) -> None:
    metrics = metrics.copy()
    metrics["outcome_id"] = metrics["domain"] + " :: " + metrics["outcome"]
    matrix = metrics.pivot(
        index="outcome_id",
        columns="group",
        values="n_sources",
    )
    control = matrix["people"]
    groups = ordered_groups(metrics)
    differences = matrix[groups].sub(control, axis=0)

    domain_order = list(dict.fromkeys(metrics["domain"]))
    outcome_order = []
    for domain in domain_order:
        outcome_order.extend(
            metrics.loc[metrics["domain"] == domain, "outcome_id"].drop_duplicates()
        )
    differences = differences.reindex(index=outcome_order, columns=groups)
    differences.columns = [
        SHORT_GROUP_LABELS.get(group, group) for group in differences.columns
    ]
    differences.to_csv(output_dir / "heatmap_difference_from_generic_data.csv")

    limit = float(np.nanmax(np.abs(differences.to_numpy())))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        differences,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 6},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Sources relative to generic ‘people’"},
        ax=ax,
    )
    ax.set_xlabel("Explicit group condition")
    ax.set_ylabel("Domain and outcome")
    ax.set_title("Source-count difference from the generic control")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_2_heatmap_difference_from_generic")


def plot_dumbbells(metrics: pd.DataFrame, output_dir: Path) -> None:
    all_pairs = []
    for dimension in DIMENSION_ORDER:
        paired = paired_dimension_data(metrics, dimension).sort_values("difference")
        all_pairs.append(paired)

        minority_group = paired["minority_group"].iloc[0]
        majority_group = paired["majority_group"].iloc[0]
        y = np.arange(len(paired))
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.hlines(
            y,
            paired["majority_sources"],
            paired["minority_sources"],
            color="#B8B8B8",
            linewidth=1.3,
            zorder=1,
        )
        ax.scatter(
            paired["minority_sources"],
            y,
            color=CONDITION_COLORS["minority"],
            s=42,
            label=minority_group,
            zorder=2,
        )
        ax.scatter(
            paired["majority_sources"],
            y,
            color=CONDITION_COLORS["majority"],
            s=42,
            label=majority_group,
            zorder=2,
        )
        labels = [
            f"{row.domain} — {row.outcome}" for row in paired.itertuples()
        ]
        ax.set_yticks(y, labels)
        ax.set_xlabel("Number of sources")
        ax.set_ylabel("")
        ax.set_title(f"Paired source counts: {dimension}")
        ax.legend(loc="lower right", frameon=True)
        ax.xaxis.grid(True, alpha=0.3)
        ax.yaxis.grid(False)
        sns.despine(ax=ax, left=True)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir,
            f"figure_3_dumbbell_{slugify(dimension)}",
        )

    pd.concat(all_pairs, ignore_index=True).to_csv(
        output_dir / "dumbbell_paired_outcomes_data.csv",
        index=False,
    )


def plot_multiscale_overlap(overlaps: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(19, 10),
        sharex=True,
        sharey="row",
    )
    metric_rows = (
        ("jaccard", "Jaccard"),
        ("overlap_coefficient", "Overlap coefficient"),
    )
    offsets = dict(zip(COMPARISON_ORDER, (-0.24, 0.0, 0.24)))

    for row_index, (metric, metric_label) in enumerate(metric_rows):
        for column_index, granularity in enumerate(GRANULARITY_ORDER):
            ax = axes[row_index, column_index]
            panel = overlaps.loc[overlaps["granularity"] == granularity]
            for comparison in COMPARISON_ORDER:
                color = COMPARISON_COLORS[comparison]
                for dimension_index, dimension in enumerate(DIMENSION_ORDER):
                    values = panel.loc[
                        (panel["comparison"] == comparison)
                        & (panel["dimension"] == dimension),
                        metric,
                    ].dropna()
                    x_center = dimension_index + offsets[comparison]
                    jitter = rng.uniform(-0.045, 0.045, len(values))
                    ax.scatter(
                        np.full(len(values), x_center) + jitter,
                        values,
                        s=15,
                        alpha=0.38,
                        color=color,
                        edgecolors="none",
                    )
                    if len(values):
                        ax.scatter(
                            x_center,
                            values.mean(),
                            marker="D",
                            s=45,
                            color=color,
                            edgecolor="white",
                            linewidth=0.7,
                            zorder=4,
                        )
                        ax.scatter(
                            x_center,
                            values.median(),
                            marker="_",
                            s=150,
                            color="#111111",
                            linewidth=1.4,
                            zorder=5,
                        )

            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(
                np.arange(len(DIMENSION_ORDER)),
                DIMENSION_ORDER,
                rotation=35,
                ha="right",
            )
            ax.set_title(GRANULARITY_LABELS[granularity])
            ax.set_ylabel(metric_label if column_index == 0 else "")
            ax.set_xlabel("")
            ax.grid(axis="x", visible=False)

    comparison_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=COMPARISON_COLORS[item],
            label=COMPARISON_LABELS[item],
        )
        for item in COMPARISON_ORDER
    ]
    summary_handles = [
        Line2D([0], [0], marker="D", linestyle="", color="#555555", label="Mean"),
        Line2D([0], [0], marker="_", linestyle="", color="#111111", label="Median"),
    ]
    fig.legend(
        handles=comparison_handles + summary_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Paired source overlap across three granularities",
        y=1.05,
        fontsize=15,
    )
    fig.text(
        0.5,
        -0.01,
        "Small dots are the 21 matched outcomes in each dimension.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_4_multiscale_overlap")


def scatter_with_group_means(
    metrics: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    stem: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for condition in ("minority", "majority", "control"):
        subset = metrics.loc[metrics["condition"] == condition]
        ax.scatter(
            subset[x],
            subset[y],
            color=CONDITION_COLORS[condition],
            alpha=0.28,
            s=25,
            label=condition.title(),
        )

    group_means = (
        metrics.groupby(["group", "condition"], as_index=False)[[x, y]].mean()
    )
    for row in group_means.itertuples(index=False):
        color = CONDITION_COLORS[row.condition]
        x_value = getattr(row, x)
        y_value = getattr(row, y)
        ax.scatter(
            x_value,
            y_value,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=75,
            zorder=4,
        )
        ax.annotate(
            SHORT_GROUP_LABELS.get(row.group, row.group),
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    correlation = metrics[[x, y]].corr().iloc[0, 1]
    ax.text(
        0.02,
        0.98,
        f"Pearson r = {correlation:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def plot_scatters(metrics: pd.DataFrame, output_dir: Path) -> None:
    scatter_with_group_means(
        metrics,
        x="n_sources",
        y="aio_chars",
        xlabel="Number of source occurrences",
        ylabel="AI Overview length (characters)",
        title="AI Overview length and source count",
        stem="figure_6_aio_length_vs_source_count",
        output_dir=output_dir,
    )


def write_readme(output_dir: Path, n_boot: int) -> None:
    text = f"""Condition/source analysis figures
=================================

Exploratory figures are saved as PNG (300 dpi). Vector PDFs are reserved for
figures selected for the final paper.

1. Forest plot of paired minority-majority source-count differences.
   Intervals are percentile bootstrap 95% CIs with {n_boot:,} resamples.
2. Outcome-by-condition heatmap relative to the generic people control.
3. Six dimension-specific paired dumbbell plots.
4. Jaccard and overlap coefficient across URL, registrable-domain, and category levels.
6. AI Overview length versus source occurrences.

Small points represent individual queries/outcomes. Larger annotated points in
scatter plots are group means. Plot-ready CSVs are included for auditing.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-oriented condition/source figures."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(input_dir / "condition_metrics.csv")
    overlaps = pd.read_csv(input_dir / "source_overlap_per_query.csv")
    configure_style()
    plot_forest(metrics, output_dir, args.bootstrap_resamples)
    plot_heatmap(metrics, output_dir)
    plot_dumbbells(metrics, output_dir)
    plot_multiscale_overlap(overlaps, output_dir)
    plot_scatters(metrics, output_dir)
    write_readme(output_dir, args.bootstrap_resamples)
    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
