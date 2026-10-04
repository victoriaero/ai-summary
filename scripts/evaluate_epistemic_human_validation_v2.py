from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support


def agreement_metrics(data, left, right):
    valid = data.loc[data[left].ne("") & data[right].ne("")]
    if valid.empty:
        return {}
    return {"n_double_coded": len(valid), "human_human_agreement": float((valid[left] == valid[right]).mean()),
            "human_human_cohen_kappa": float(cohen_kappa_score(valid[left], valid[right]))}


def main():
    parser = argparse.ArgumentParser(description="Calculate agreement and external-validation metrics from adjudicated blind sheets.")
    parser.add_argument("--hedge-blind", type=Path, required=True)
    parser.add_argument("--hedge-key", type=Path, required=True)
    parser.add_argument("--veridicality-blind", type=Path, required=True)
    parser.add_argument("--veridicality-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/epistemic_commitment/validation"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    hedge = pd.read_csv(args.hedge_blind).fillna("").merge(pd.read_csv(args.hedge_key).fillna(""), on="annotation_id", validate="one_to_one")
    done = hedge.loc[hedge.human_label_1.ne("") & hedge.human_label_2.ne("") & hedge.adjudicated_label.ne("")]
    if len(done):
        binary = done.loc[done.adjudicated_label.isin(["yes", "no"])]
        y_true = binary.adjudicated_label.eq("yes").astype(int)
        y_pred = binary.model_prediction.eq("yes").astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        summaries.append({"analysis": "contextual_hedging", "validation_item": "commitment_reducing_cue", "n": len(done),
            **agreement_metrics(done, "human_label_1", "human_label_2"), "model_precision": float(p), "model_recall": float(r),
            "model_f1": float(f), "n_uncertain_excluded_from_binary_metrics": int(len(done) - len(binary))})

    ver = pd.read_csv(args.veridicality_blind).fillna("").merge(pd.read_csv(args.veridicality_key).fillna(""), on="annotation_id", validate="one_to_one", suffixes=("", "_key"))
    fields = [("predicate_correct", "predicate_detection_accuracy", None),
              ("introduces_clause", "clause_status_accuracy", None),
              ("frame_correct", "frame_mapping_accuracy", None),
              ("polarity_correct", "polarity_accuracy", None),
              ("resource_mapping_appropriate", "match_precision", "matched")]
    for field, metric, required_class in fields:
        a, b, gold = f"{field}_annotator_1", f"{field}_annotator_2", f"{field}_adjudicated"
        subset = ver
        if required_class:
            subset = subset.loc[subset.validation_class.eq(required_class)]
        complete = subset.loc[subset[a].ne("") & subset[b].ne("") & subset[gold].ne("")]
        if not complete.empty:
            summaries.append({"analysis": "veridicality", "validation_item": metric, "n": len(complete),
                "estimate": float(complete[gold].astype(str).str.casefold().eq("yes").mean()),
                **agreement_metrics(complete, a, b)})
    pd.DataFrame(summaries).to_csv(args.output_dir / "human_validation_metrics.csv", index=False)
    print(f"Human validation results written to {args.output_dir / 'human_validation_metrics.csv'}")


if __name__ == "__main__":
    main()
