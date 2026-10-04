from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support


def binary_scores(pred, gold):
    keep = pd.Series(gold).isin(["yes", "no"]).to_numpy()
    y_true = (pd.Series(gold).to_numpy()[keep] == "yes").astype(int)
    y_pred = (pd.Series(pred).to_numpy()[keep] == "yes").astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"n_binary": int(keep.sum()), "precision": float(p), "recall": float(r), "f1": float(f), "uncertain_excluded": int(len(keep) - keep.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score completed blind epistemic-commitment validation sheets.")
    parser.add_argument("--hedge-blind", type=Path, required=True)
    parser.add_argument("--hedge-key", type=Path, required=True)
    parser.add_argument("--veridicality-blind", type=Path, required=True)
    parser.add_argument("--veridicality-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/epistemic_commitment/validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hedge = pd.read_csv(args.hedge_blind).fillna("")
    hkey = pd.read_csv(args.hedge_key).fillna("")
    if "annotation_id" not in hedge or "annotation_id" not in hkey:
        raise ValueError("Hedge blind/key sheets must share annotation_id")
    h = hedge.merge(hkey[["annotation_id", "model_prediction"]], on="annotation_id", validate="one_to_one")
    complete = h.loc[h.human_label_1.ne("") & h.human_label_2.ne("") & h.adjudicated_label.ne("")]
    if len(complete):
        rows.append({"analysis": "contextual_hedging", "sample_n": len(complete),
            "human_human_agreement": float((complete.human_label_1 == complete.human_label_2).mean()),
            "human_human_cohen_kappa": float(cohen_kappa_score(complete.human_label_1, complete.human_label_2)),
            **binary_scores(complete.model_prediction, complete.adjudicated_label)})
    v = pd.read_csv(args.veridicality_blind).fillna("")
    vkey = pd.read_csv(args.veridicality_key).fillna("")
    if "annotation_id" not in v or "annotation_id" not in vkey:
        raise ValueError("Veridicality blind/key sheets must share annotation_id")
    v = v.merge(vkey, on="annotation_id", validate="one_to_one", suffixes=("", "_key"))
    for field in ("predicate_correct", "introduces_clause", "frame_correct", "polarity_correct", "resource_mapping_appropriate"):
        a, b, gold = f"{field}_annotator_1", f"{field}_annotator_2", f"{field}_adjudicated"
        complete = v.loc[v[a].ne("") & v[b].ne("") & v[gold].ne("")]
        if len(complete):
            rows.append({"analysis": "veridicality", "validation_item": field, "sample_n": len(complete),
                "human_human_agreement": float((complete[a] == complete[b]).mean()),
                "human_human_cohen_kappa": float(cohen_kappa_score(complete[a], complete[b]))})
    pd.DataFrame(rows).to_csv(args.output_dir / "human_validation_metrics.csv", index=False)
    print(f"Human-validation metrics written to {args.output_dir / 'human_validation_metrics.csv'}")


if __name__ == "__main__":
    main()
