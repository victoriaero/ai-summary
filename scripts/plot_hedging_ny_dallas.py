"""PNG figures for the Dallas–NY contextual-hedging study."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forest(effects: pd.DataFrame, folder: Path) -> None:
    data = effects.loc[effects.unit.eq("claim_sentence") &
                       effects.comparison.isin(["minority-people", "majority-people"])].copy()
    dims = sorted(data.demographic_dimension.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharex=True)
    colors = {"v1_dallas": "#3277a8", "v2_ny": "#d67a2c", "pooled": "#263747"}
    shifts = {"v1_dallas": -0.22, "v2_ny": 0, "pooled": 0.22}
    for ax, contrast in zip(axes, ["minority-people", "majority-people"]):
        sub = data.loc[data.comparison.eq(contrast)]
        for key, label in [("v1_dallas", "Dallas"), ("v2_ny", "New York"), ("pooled", "Pooled")]:
            part = sub.loc[sub.replica_id.eq(key)].set_index("demographic_dimension").reindex(dims)
            x = part.estimate.to_numpy() * 100
            lo = part.ci_low.to_numpy() * 100
            hi = part.ci_high.to_numpy() * 100
            y = np.arange(len(dims)) + shifts[key]
            ax.errorbar(x, y, xerr=np.vstack((np.maximum(0, x - lo), np.maximum(0, hi - x))), fmt="o", markersize=5,
                        capsize=2, color=colors[key], label=label)
        ax.axvline(0, color="#777", lw=1)
        ax.set_yticks(np.arange(len(dims)), dims)
        ax.set_title(contrast.replace("-", " − ").title())
        ax.set_xlabel("Adjusted probability difference (percentage points)")
        ax.grid(axis="x", alpha=.2)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Contextual hedging in claim-bearing sentences: cross-location contrasts")
    fig.text(.5, .015, "95% outcome-cluster bootstrap intervals. People is one shared control per outcome/location, not six independent observations.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    save(fig, folder / "figure_A_cross_location_forest.png")


def across_units(root: Path, folder: Path) -> None:
    response = pd.read_csv(root / "response_level_hedging.csv")
    sentence = pd.read_csv(root / "claim_sentence_hedging.csv")
    span = pd.read_csv(root / "claim_span_response_rates.csv")
    merged = response[["response_id", "replica_id", "group_type", "n_sentences", "n_hedged_sentences"]].merge(
        sentence[["response_id", "n_claim_sentences", "n_hedged_claim_sentences"]], on="response_id")
    merged = merged.merge(span[["response_id", "n_valid_claim_spans", "n_claim_spans_with_hedge"]], on="response_id")
    names = [("All response sentences", "n_hedged_sentences", "n_sentences"),
             ("Claim-bearing sentences", "n_hedged_claim_sentences", "n_claim_sentences"),
             ("Claim source spans", "n_claim_spans_with_hedge", "n_valid_claim_spans")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (replica, title) in zip(axes, [("v1_dallas", "Dallas"), ("v2_ny", "New York")]):
        part = merged.loc[merged.replica_id.eq(replica)]
        for shift, (name, num, den) in zip((-.23, 0, .23), names):
            values = part.groupby("group_type")[[num, den]].sum().reindex(["people", "minority", "majority"])
            rate = 100 * values[num] / values[den].replace(0, np.nan)
            ax.scatter(np.arange(3) + shift, rate, s=75, label=name)
        ax.set_xticks(range(3), ["People", "Minority", "Majority"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Observed rate (%)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Contextual hedging across analysis units")
    fig.text(.5, .02, "Each unit has a different denominator; rates are descriptive and not directly interchangeable.\nSource-span cue presence is a scope proxy, not a verified claim-level hedge.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94))
    save(fig, folder / "figure_B_hedging_across_units.png")


def probabilities(effects: pd.DataFrame, folder: Path) -> None:
    data = effects.loc[effects.unit.eq("claim_sentence") & effects.comparison.eq("minority-people") &
                       effects.replica_id.ne("pooled")].copy()
    dims = sorted(data.demographic_dimension.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, replica, title in zip(axes, ["v1_dallas", "v2_ny"], ["Dallas", "New York"]):
        part = data.loc[data.replica_id.eq(replica)].set_index("demographic_dimension").reindex(dims)
        majority = effects.loc[effects.unit.eq("claim_sentence") & effects.comparison.eq("majority-people") &
                               effects.replica_id.eq(replica)].set_index("demographic_dimension").reindex(dims)
        for shift, frame, value, low, high, label, color in [
            (-.18, part, "predicted_probability_b", "probability_b_ci_low", "probability_b_ci_high", "People", "#666"),
            (0, part, "predicted_probability_a", "probability_a_ci_low", "probability_a_ci_high", "Minority", "#317ca9"),
            (.18, majority, "predicted_probability_a", "probability_a_ci_low", "probability_a_ci_high", "Majority", "#d17c31")]:
            vals = frame[value].to_numpy() * 100
            lo = frame[low].to_numpy() * 100
            hi = frame[high].to_numpy() * 100
            ax.errorbar(np.arange(len(dims)) + shift, vals,
                        yerr=np.vstack((np.maximum(0, vals - lo), np.maximum(0, hi - vals))),
                        fmt="o", markersize=5, capsize=2, label=label, color=color)
        ax.set_xticks(range(len(dims)), dims, rotation=40, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Outcome-adjusted predicted probability (%)")
    axes[1].legend(frameon=False)
    fig.suptitle("Predicted contextual hedging in claim-bearing sentences (95% CI)")
    fig.text(.5, .015, "People predictions repeat visually across dimensions but derive from one shared control per outcome/location.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    save(fig, folder / "figure_C_predicted_probabilities.png")


def consistency(effects: pd.DataFrame, folder: Path) -> None:
    data = effects.loc[effects.unit.eq("claim_sentence") & effects.replica_id.ne("pooled") &
                       effects.comparison.isin(["minority-people", "majority-people"])].copy()
    wide = data.pivot(index=["demographic_dimension", "comparison"], columns="replica_id", values="estimate")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    label_offsets = {
        "minority-people": {
            "Race": (5, -11),
            "Gender Identity": (5, 7),
            "Gender": (5, 6),
        },
        "majority-people": {
            "Sexual Orientation": (5, -15),
            "Gender": (-62, 5),
        },
    }
    for ax, contrast in zip(axes, ["minority-people", "majority-people"]):
        sub = wide.loc[pd.IndexSlice[:, contrast], :]
        x, y = sub.v1_dallas.to_numpy() * 100, sub.v2_ny.to_numpy() * 100
        ax.scatter(x, y, color="#347da6", s=70)
        for dim, xx, yy in zip(sub.index.get_level_values(0), x, y):
            offset = label_offsets.get(contrast, {}).get(dim, (5, 5))
            ax.annotate(dim, (xx, yy), xytext=offset, textcoords="offset points", fontsize=8)
        ax.axhline(0, color="#777", lw=1)
        ax.axvline(0, color="#777", lw=1)
        ax.set_title(contrast.replace("-", " − ").title())
        ax.set_xlabel("Dallas difference (percentage points)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("New York difference (percentage points)")
    fig.suptitle("Cross-location consistency of claim-sentence contrasts")
    fig.text(.5, .02, "Six dimensions per contrast: direction and magnitude are descriptive; this is not a correlation estimate or formal replication.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .94))
    save(fig, folder / "figure_D_location_consistency.png")


def generate_figures(root: Path) -> None:
    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(root / "hedging_model_results.csv")
    forest(effects, folder)
    across_units(root, folder)
    probabilities(effects, folder)
    consistency(effects, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/hedging_ny_dallas")
    args = parser.parse_args()
    generate_figures(args.results_dir)
