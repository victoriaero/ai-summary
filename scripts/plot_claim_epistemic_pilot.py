"""Descriptive PNG figures for the Dallas claim-level epistemic pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "claim_epistemic_pilot_dallas"
COLORS = {"people": "#666666", "minority": "#2678a8", "majority": "#d27b30"}
LABELS = {"people": "People", "minority": "Minority", "majority": "Majority"}
ORDER = ["people", "minority", "majority"]


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_granularity(data: pd.DataFrame, output: Path) -> None:
    data = data.set_index("group_type").loc[ORDER]
    series = [
        ("All response sentences", "response_mean_hedged_sentence_rate_all_sentences", -0.22, "o"),
        ("Claim-bearing sentences", "claim_bearing_sentence_hedge_context_rate", 0, "s"),
        ("Claim source spans", "claim_span_hedge_candidate_rate", 0.22, "D"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, column, shift, marker in series:
        ax.scatter(np.arange(3) + shift, data[column] * 100, s=90, marker=marker, label=label, zorder=3)
    ax.set_xticks(range(3), [LABELS[x] for x in ORDER])
    ax.set_ylabel("Hedge rate (%)")
    ax.set_title("Hedging across units of analysis — Dallas")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, -0.035, "Different denominators: response mean over all sentences; unique claim-bearing sentences; claims.\nSpan cue = unverified semantic scope; these rates are not directly interchangeable.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, output / "01_hedging_by_unit.png")


def fig_coverage(data: pd.DataFrame, output: Path) -> None:
    data = data.set_index("group_type").loc[ORDER]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(3)
    for shift, column, label in [(-0.15, "response_veridicality_coverage", "Response"), (0.15, "claim_veridicality_coverage_conservative", "Claim")]:
        values = data[column].to_numpy() * 100
        ax.scatter(x + shift, values, s=95, label=label, zorder=3)
        for i, value in enumerate(values):
            denominator = int(data.iloc[i]["n_responses" if label == "Response" else "n_claims"])
            numerator = round(value * denominator / 100)
            ax.annotate(f"{numerator}/{denominator}", (x[i] + shift, value), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xticks(x, [LABELS[z] for z in ORDER])
    ax.set_ylim(-5, 90)
    ax.set_ylabel("Veridicality coverage (%)")
    ax.set_title("Strict veridicality coverage drops at claim level")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.025, "Response: any strict scored predicate. Claim: one conservatively linked predicate/complement.\nUnscored claims are missing, not zero-veridicality claims.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, output / "02_veridicality_coverage.png")


def fig_outcomes(data: pd.DataFrame, output: Path) -> None:
    data = data[data["comparison"] == "minority-majority"].copy()
    dims = sorted(data["demographic_dimension"].dropna().unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(26)
    for i, dim in enumerate(dims):
        values = data.loc[data["demographic_dimension"] == dim, "delta_H_span_proxy_rate"].dropna().to_numpy() * 100
        ax.scatter(values, i + rng.uniform(-0.22, 0.22, len(values)), s=27, alpha=0.55, color=COLORS["minority"])
        ax.scatter(np.median(values), i, s=115, marker="D", color="#172a3a", zorder=4)
        ax.annotate(f"n={len(values)}", (max(55, values.max() + 3), i), va="center", fontsize=9)
    ax.axvline(0, color="#666666", lw=1)
    ax.set_yticks(range(len(dims)), dims)
    ax.set_xlim(-105, 80)
    ax.set_xlabel("Minority − majority hedge-cue rate (percentage points)")
    ax.set_title("Outcome-level differences in claim-span hedge cues")
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.5, 0.025, "Each small point = one outcome; diamond = dimension median. Complete claim sets, not matched content.\nCue in source span is a scope proxy, not verified modification of the claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, output / "03_outcome_hedge_contrasts.png")


def fig_matched(data: pd.DataFrame, output: Path) -> None:
    counts = data.groupby("comparison").agg(n_pairs=("pair_id", "size"), n_h_different=("delta_H", lambda x: int((x != 0).sum())), n_both_v=("both_V_covered", "sum"))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = ["Minority–majority", "Minority–People", "Majority–People"]
    keys = ["minority-majority", "minority-people", "majority-people"]
    x = np.arange(3)
    n = [int(counts.loc[k, "n_pairs"]) if k in counts.index else 0 for k in keys]
    h = [int(counts.loc[k, "n_h_different"]) if k in counts.index else 0 for k in keys]
    v = [int(counts.loc[k, "n_both_v"]) if k in counts.index else 0 for k in keys]
    width = 0.25
    for offset, vals, label, color in [(-width, n, "Provisional pairs", "#73818b"), (0, h, "Different H", COLORS["minority"]), (width, v, "Both V scored", COLORS["majority"])]:
        ax.bar(x + offset, vals, width, label=label, color=color)
        for xx, yy in zip(x + offset, vals):
            ax.annotate(str(yy), (xx, yy), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(n + [1]) + 1.4)
    ax.set_ylabel("Number of pairs")
    ax.set_title("Matched-claim analysis: feasibility, not an effect estimate")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.025, "Equivalence is provisional; pairs are not independent and may share a People claim.\nNo pair supports a paired veridicality contrast in this pilot.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, output / "04_matched_claim_feasibility.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    granularity = pd.read_csv(input_dir / "unit_granularity_diagnostic.csv")
    fig_granularity(granularity[granularity["summary_level"] == "condition"], output_dir)
    fig_coverage(granularity[granularity["summary_level"] == "condition"], output_dir)
    fig_outcomes(pd.read_csv(input_dir / "all_claims_outcome_contrasts.csv"), output_dir)
    fig_matched(pd.read_csv(input_dir / "matched_claim_pairs.csv"), output_dir)
    print(f"Four PNG figures written to {output_dir}")


if __name__ == "__main__":
    main()
