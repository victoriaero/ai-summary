"""Finalize human-adjudicated proposition matches; no inference on sparse pairs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ALLOWED = {"SAME_PROPOSITION", "PARTIAL_OVERLAP", "DIFFERENT", "UNCERTAIN"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/hedging_ny_dallas")
    args = parser.parse_args()
    root = args.results_dir
    candidates = pd.read_csv(root / "matched_claim_candidates.csv").fillna("")
    if candidates.empty:
        raise ValueError("No cross-condition candidate pairs")
    candidates["adjudicated_label"] = candidates.adjudicated_label.astype(str).str.upper().str.strip()
    bad = set(candidates.adjudicated_label) - (ALLOWED | {""})
    if bad:
        raise ValueError(f"Invalid adjudicated labels: {bad}")
    candidates["comparison"] = candidates.apply(
        lambda r: "-".join(sorted((r.group_type_a, r.group_type_b))), axis=1)
    validated = candidates.loc[candidates.adjudicated_label.eq("SAME_PROPOSITION")].copy()
    claims = pd.read_csv(root / "claim_span_hedging.csv")
    hedge = claims.set_index("claim_id").claim_span_contains_contextual_hedge.to_dict()
    validated["span_cue_a"] = validated.claim_a_id.map(hedge)
    validated["span_cue_b"] = validated.claim_b_id.map(hedge)
    validated["different_span_cue_presence"] = validated.span_cue_a.ne(validated.span_cue_b)
    validated.to_csv(root / "matched_claim_validated.csv", index=False)
    rows = []
    for (replica, comparison), part in candidates.groupby(["replica_id", "comparison"]):
        yes = part.loc[part.adjudicated_label.eq("SAME_PROPOSITION")]
        rows.append({"replica_id": replica, "comparison": comparison,
                     "n_candidates": len(part), "n_adjudicated": int(part.adjudicated_label.ne("").sum()),
                     "n_same_proposition": len(yes), "n_outcomes_with_same_proposition": yes.outcome.nunique(),
                     "n_different_span_cue_presence": int(validated.loc[
                         validated.replica_id.eq(replica) & validated.comparison.eq(comparison),
                         "different_span_cue_presence"].sum()),
                     "eligible_for_matched_inference": len(yes) >= 30 and yes.outcome.nunique() >= 10})
    feasibility = pd.DataFrame(rows)
    feasibility.to_csv(root / "matched_claim_feasibility.csv", index=False)
    eligible = feasibility.loc[feasibility.eligible_for_matched_inference]
    if eligible.empty:
        print("No contrast/location reaches 30 human-validated pairs over 10 outcomes; Figure F withheld")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = eligible.replica_id + " / " + eligible.comparison
    x = range(len(eligible))
    ax.bar(x, eligible.n_candidates, color="#aab6be", label="Candidates")
    ax.bar(x, eligible.n_same_proposition, color="#267ba5", label="Validated same proposition")
    ax.bar(x, eligible.n_different_span_cue_presence, color="#d07837",
           label="Different cue presence in source spans")
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.set_ylabel("Pairs")
    ax.set_title("Matched-claim feasibility after human adjudication")
    ax.legend(frameon=False)
    fig.tight_layout()
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "figure_F_matched_claims.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
