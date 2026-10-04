"""Evaluate completed two-annotator cue-to-claim scope labels; never edit the annotation sheet."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score


LABELS = {"YES", "NO", "UNCERTAIN"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/hedging_ny_dallas")
    args = parser.parse_args()
    root = args.results_dir
    sample = pd.read_csv(root / "hedge_scope_validation_sample.csv").fillna("")
    key = pd.read_csv(root / "hedge_scope_validation_key.csv").fillna("")
    sample = sample.merge(key[["annotation_id", "group_type", "replica_id"]],
                          on="annotation_id", validate="one_to_one")
    if sample.empty:
        raise ValueError("No positive cue-to-claim cases to validate")
    for column in ("human_label_1", "human_label_2", "adjudicated_label"):
        values = set(sample[column].str.upper().str.strip())
        if not values.issubset(LABELS):
            raise ValueError(f"Complete {column} with YES, NO or UNCERTAIN before evaluation; found {values - LABELS}")
        sample[column] = sample[column].str.upper().str.strip()
    agreement = float((sample.human_label_1 == sample.human_label_2).mean())
    kappa = float(cohen_kappa_score(sample.human_label_1, sample.human_label_2, labels=sorted(LABELS)))
    determinate = sample.loc[sample.adjudicated_label.isin(["YES", "NO"])].copy()
    rows = []
    for category, column in (("all", None), ("cue_surface", "cue_surface"),
                             ("condition", "group_type"), ("location", "replica_id")):
        groups = [("all", determinate)] if column is None else determinate.groupby(column)
        for value, subset in groups:
            rows.append({"status": "complete", "category": category, "value": value,
                         "n_determinate": len(subset), "n_uncertain_excluded": int(
                             (sample.adjudicated_label.eq("UNCERTAIN")).sum()) if category == "all" else None,
                         "precision_yes": float(subset.adjudicated_label.eq("YES").mean()) if len(subset) else None,
                         "human_human_agreement": agreement if category == "all" else None,
                         "human_human_cohen_kappa": kappa if category == "all" else None})
    result = pd.DataFrame(rows)
    result.to_csv(root / "hedge_scope_validation_results.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, category, title in zip(axes, ["cue_surface", "condition", "location"],
                                   ["Cue surface (10 most sampled)", "Condition", "Location"]):
        part = result.loc[result.category.eq(category)]
        if category == "cue_surface":
            part = part.sort_values("n_determinate", ascending=False).head(10)
        ax.bar(part.value, part.precision_yes * 100, color="#4c83a4")
        for i, row in enumerate(part.itertuples()):
            ax.annotate(f"n={row.n_determinate}", (i, row.precision_yes * 100), ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=45 if category == "cue_surface" else 20)
    axes[0].set_ylabel("Adjudicated YES among determinate cases (%)")
    fig.suptitle("Does the detected cue qualify the claim proposition? (no external cue-family taxonomy)")
    fig.tight_layout()
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "figure_E_scope_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Scope validation results written to {root}")


if __name__ == "__main__":
    main()
