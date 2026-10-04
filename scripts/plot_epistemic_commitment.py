from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "results" / "epistemic_commitment"
OUTPUT_DIR = INPUT_DIR / "figures"

DIMENSION_ORDER = [
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
]
GROUP_ORDER = ["people", "minority", "majority"]
GROUP_LABELS = {
    "people": "Generic control",
    "minority": "Minority / focal",
    "majority": "Majority / comparison",
}
PALETTE = {
    "Generic control": "#555555",
    "Minority / focal": "#2878B5",
    "Majority / comparison": "#E07A2D",
}


def load_complete_dallas(metric_file: str) -> pd.DataFrame:
    data = pd.read_csv(INPUT_DIR / metric_file)
    data = data.loc[data["replica_id"].eq("v1_dallas")].copy()
    if data.empty:
        raise ValueError("No complete Dallas records found")
    return data


def expand_people_control(data: pd.DataFrame) -> pd.DataFrame:
    people = data.loc[
        data["group_type"].eq("people")
        & data["dimension"].eq("Control")
    ].copy()
    people_parts = []
    for dimension in DIMENSION_ORDER:
        part = people.copy()
        part["dimension"] = dimension
        people_parts.append(part)
    explicit = data.loc[data["group_type"].isin(["minority", "majority"])]
    expanded = pd.concat([explicit, *people_parts], ignore_index=True)
    expanded["group_label"] = expanded["group_type"].map(GROUP_LABELS)
    expanded["dimension"] = pd.Categorical(
        expanded["dimension"],
        categories=DIMENSION_ORDER,
        ordered=True,
    )
    expanded["group_label"] = pd.Categorical(
        expanded["group_label"],
        categories=[GROUP_LABELS[item] for item in GROUP_ORDER],
        ordered=True,
    )
    expected = len(DIMENSION_ORDER) * 21 * 3
    if len(expanded) != expected:
        raise ValueError(
            f"Expected {expected} expanded Dallas records; got {len(expanded)}"
        )
    cell_counts = expanded.groupby(
        ["dimension", "group_type"],
        observed=False,
    ).size()
    if not cell_counts.eq(21).all():
        raise ValueError("Every plotted Dallas cell must have 21 outcomes")
    return expanded


def set_common_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_response_metric(
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    reference_zero: bool = False,
    note: str = "",
) -> None:
    data = data.loc[data[metric].notna()].copy()
    set_common_style()
    fig, axes = plt.subplots(
        nrows=len(DIMENSION_ORDER),
        ncols=1,
        figsize=(12, 18),
        sharex=True,
        sharey=True,
    )
    for ax, dimension in zip(axes, DIMENSION_ORDER):
        subset = data.loc[data["dimension"].eq(dimension)]
        sns.boxplot(
            data=subset,
            x="group_label",
            y=metric,
            order=[GROUP_LABELS[item] for item in GROUP_ORDER],
            hue="group_label",
            hue_order=[GROUP_LABELS[item] for item in GROUP_ORDER],
            palette=PALETTE,
            showfliers=False,
            width=0.56,
            linewidth=1.2,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="group_label",
            y=metric,
            order=[GROUP_LABELS[item] for item in GROUP_ORDER],
            hue="group_label",
            hue_order=[GROUP_LABELS[item] for item in GROUP_ORDER],
            palette=PALETTE,
            jitter=0.16,
            size=4.1,
            alpha=0.72,
            dodge=False,
            legend=False,
            ax=ax,
        )
        if reference_zero:
            ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
        counts = (
            subset.groupby("group_type", observed=False)[metric]
            .count()
            .reindex(GROUP_ORDER, fill_value=0)
        )
        ax.set_title(
            f"{dimension}  ·  n = "
            + " / ".join(str(counts[group]) for group in GROUP_ORDER),
            loc="left",
            fontsize=11,
            pad=4,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if dimension == DIMENSION_ORDER[2] else "")
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
        ax.grid(axis="x", visible=False)

    fig.suptitle(title, y=0.995, fontsize=17, fontweight="bold")
    if note:
        fig.text(0.5, 0.972, note, ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.025, 0.99, 0.96))
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_veridicality_coverage(data: pd.DataFrame) -> None:
    coverage = (
        data.assign(has_score=data["n_veridical_predicates"].gt(0))
        .groupby(["dimension", "group_type"], observed=False)["has_score"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    coverage["group_label"] = coverage["group_type"].map(GROUP_LABELS)
    coverage["coverage_percent"] = 100 * coverage["mean"]
    coverage["cell_label"] = coverage.apply(
        lambda row: f"{int(row['sum'])}/{int(row['count'])}",
        axis=1,
    )
    matrix = coverage.pivot(
        index="dimension",
        columns="group_label",
        values="coverage_percent",
    ).reindex(
        index=DIMENSION_ORDER,
        columns=[GROUP_LABELS[group] for group in GROUP_ORDER],
    )
    annot = coverage.pivot(
        index="dimension",
        columns="group_label",
        values="cell_label",
    ).reindex(
        index=DIMENSION_ORDER,
        columns=[GROUP_LABELS[group] for group in GROUP_ORDER],
    )
    set_common_style()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Responses with ≥1 matched predicate (%)"},
        ax=ax,
    )
    ax.set_title(
        "Coverage of MegaVeridicality scores in Dallas AIOs",
        loc="left",
        pad=14,
        fontsize=16,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=10)
    fig.text(
        0.5,
        0.015,
        "Cell labels show responses with ≥1 matched predicate / 21 outcomes. "
        "The generic control is reused for each dimension.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.98))
    fig.savefig(
        OUTPUT_DIR / "veridicality_score_coverage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_observed_hedge_cues() -> None:
    occurrences = pd.read_csv(INPUT_DIR / "hedge_occurrences.csv")
    occurrences = occurrences.loc[
        occurrences["replica_id"].eq("v1_dallas")
    ]
    counts = occurrences["normalized_cue"].value_counts().head(18)
    plot_data = counts.rename_axis("cue").reset_index(name="occurrences")
    plot_data["cue_label"] = plot_data["cue"].map(
        lambda cue: "asterisk (*)" if cue == "*" else cue
    )
    set_common_style()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.barplot(
        data=plot_data.sort_values("occurrences"),
        x="occurrences",
        y="cue_label",
        color="#4C78A8",
        ax=ax,
    )
    ax.set_title(
        "Most frequent BioScope cue matches in Dallas AIOs",
        loc="left",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Matched occurrences across responses")
    ax.set_ylabel("BioScope annotated cue")
    ax.text(
        0,
        -0.15,
        "Counts include every BioScope speculation span; no manual cue pruning. "
        "These are lexical matches, not contextual classifications.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "most_frequent_hedge_cues.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_readme(dallas_data: pd.DataFrame) -> None:
    n_scored = int(dallas_data["n_veridical_predicates"].gt(0).sum())
    text = f"""Epistemic commitment figures
=============================

All condition-comparison figures use the balanced Dallas collection only.
There are 21 outcomes for each explicit condition in each social dimension.
The generic people control has 21 outcomes and is displayed in every dimension
panel by reusing the same control response for that outcome.

Figures
-------
hedging_by_condition.png
    Response-level hedge occurrences per 100 tokens, shown by group type and
    social dimension. Boxes show the interquartile range and median; points are
    individual outcome responses.

veridicality_by_condition.png
    Response-level mean MegaVeridicality score among responses with at least one
    matched predicate. It contains {n_scored} scored responses before the
    generic-control reuse. Panel headings give the scored response count for
    each group type. Scores are not imputed for responses without matched
    predicates.

veridicality_score_coverage.png
    Number and percent of responses with at least one matched predicate, out of
    21 outcomes per group-type cell.

most_frequent_hedge_cues.png
    Eighteen most frequent BioScope lexical matches in Dallas. All annotated
    cues are retained, including broad and formatting cues. This figure reports
    literal lexicon matches, not contextual uncertainty classification.

No combined hedging/veridicality index is computed. Los Angeles has only one
21-response group and New York has no completed responses, so neither is mixed
into the balanced Dallas condition figures.

Regenerate from the repository root:

    python scripts/plot_epistemic_commitment.py
"""
    (OUTPUT_DIR / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hedging = load_complete_dallas("hedging_response_metrics.csv")
    veridicality = load_complete_dallas("veridicality_response_metrics.csv")
    hedging_expanded = expand_people_control(hedging)
    veridicality_expanded = expand_people_control(veridicality)

    plot_response_metric(
        hedging_expanded,
        metric="hedge_per_100_tokens",
        title="BioScope hedge cue matches in Google AI Overviews",
        ylabel="Hedge matches per 100 tokens",
        filename="hedging_by_condition.png",
        note=(
            "Each point is one of 21 outcomes per cell; boxes summarize the "
            "outcome distribution. Generic control responses are reused."
        ),
    )
    plot_response_metric(
        veridicality_expanded,
        metric="mean_veridicality",
        title="Matched predicate veridicality in Google AI Overviews",
        ylabel="Mean normalized human score per response",
        filename="veridicality_by_condition.png",
        reference_zero=True,
        note=(
            "Only responses with ≥1 confidently matched predicate have a score; "
            "panel headings report n."
        ),
    )
    plot_veridicality_coverage(veridicality_expanded)
    plot_observed_hedge_cues()
    write_readme(veridicality)
    print(f"Figures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
