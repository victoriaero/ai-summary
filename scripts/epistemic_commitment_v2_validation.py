"""Prepare and score the AIO transfer-validation sheets without overwriting labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "results" / "epistemic_commitment"
OUTPUT = PROJECT_ROOT / "results" / "epistemic_commitment_v2"


def preserve_sheet(path: Path, template: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        current = pd.read_csv(path).fillna("")
        if set(zip(current.annotation_id, current.response_id)) != set(zip(template.annotation_id, template.response_id)):
            raise ValueError(f"Annotation IDs have changed; refusing to overwrite existing sheet: {path}")
        for col in template.columns:
            if col not in current:
                current[col] = template.set_index("annotation_id").loc[current.annotation_id, col].to_numpy()
        current.to_csv(path, index=False)
        return current
    template.to_csv(path, index=False)
    return template


def agreement(frame: pd.DataFrame, label: str) -> tuple[float, float]:
    a, b = f"{label}_annotator_1", f"{label}_annotator_2"
    if a not in frame or b not in frame:
        return np.nan, np.nan
    double = frame.loc[frame[a].isin(["yes", "no", "uncertain"]) & frame[b].isin(["yes", "no", "uncertain"])]
    if len(double) < 2:
        return np.nan, np.nan
    return float((double[a] == double[b]).mean()), float(cohen_kappa_score(double[a], double[b]))


def safe_binomial_metrics(gold: pd.Series, predicted: pd.Series) -> tuple[float, float, float, float]:
    if not len(gold):
        return np.nan, np.nan, np.nan, np.nan
    truth = gold.eq("yes").astype(int)
    guess = predicted.eq("yes").astype(int)
    p, r, f, _ = precision_recall_fscore_support(truth, guess, average="binary", zero_division=0)
    return float(p), float(r), float(f), float((truth != guess).mean())


def enrich_veridicality_key(key: pd.DataFrame, resource: pd.DataFrame) -> pd.DataFrame:
    required = {"predicate_lemma", "syntactic_frame", "polarity", "resource_entry_id"}
    if required - set(resource.columns):
        return key
    lookup = resource[list(required)].drop_duplicates(["predicate_lemma", "syntactic_frame", "polarity"])
    augmented = key.merge(lookup, on=["predicate_lemma", "syntactic_frame", "polarity"], how="left", validate="many_to_one")
    if "matched_resource_entry" not in augmented:
        augmented["matched_resource_entry"] = ""
    empty = augmented.matched_resource_entry.fillna("").eq("") & augmented.match_status.eq("matched_strict")
    augmented.loc[empty, "matched_resource_entry"] = augmented.loc[empty, "resource_entry_id"]
    return augmented.drop(columns=["resource_entry_id"])


def validation_tables(source: Path, output: Path, contextual_data: pd.DataFrame) -> dict:
    directory = output / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    hedge_template = pd.read_csv(source / "contextual_hedging" / "human_validation_blind.csv").fillna("")
    hedge_key = pd.read_csv(source / "contextual_hedging" / "human_validation_key.csv").fillna("")
    ver_template = pd.read_csv(source / "veridicality_v2" / "validation_blind.csv").fillna("")
    ver_key = pd.read_csv(source / "veridicality_v2" / "validation_key.csv").fillna("")
    resource = pd.read_csv(source / "veridicality_v2" / "megaveridicality_resource_entries.csv").fillna("")
    ver_key = enrich_veridicality_key(ver_key, resource)

    for field in ("negation_correct", "strict_decision_correct"):
        for role in ("annotator_1", "annotator_2", "adjudicated"):
            col = f"{field}_{role}"
            if col not in ver_template:
                ver_template[col] = ""
    proposed = ver_key[["annotation_id", "syntactic_frame", "polarity", "match_status", "matched_resource_entry"]].rename(columns={
        "syntactic_frame": "proposed_frame", "polarity": "proposed_polarity",
        "match_status": "proposed_match_status", "matched_resource_entry": "proposed_resource_entry"})
    for col in proposed.columns.drop("annotation_id"):
        if col not in ver_template:
            ver_template[col] = ver_template.annotation_id.map(proposed.set_index("annotation_id")[col]).fillna("")

    hedge = preserve_sheet(directory / "hedging_annotation_sheet.csv", hedge_template)
    ver = preserve_sheet(directory / "veridicality_annotation_sheet.csv", ver_template)
    for path, key in ((directory / "hedging_key_restricted.csv", hedge_key),
                      (directory / "veridicality_key_restricted.csv", ver_key)):
        if path.exists():
            old = pd.read_csv(path).fillna("")
            if set(zip(old.annotation_id, old.response_id)) != set(zip(key.annotation_id, key.response_id)):
                raise ValueError(f"Validation key changed: {path}")
        key.to_csv(path, index=False)

    lookup = contextual_data[["response_id", "group_type", "dimension", "replica_id"]].drop_duplicates("response_id")
    hedge = hedge.merge(hedge_key, on=["annotation_id", "response_id"], validate="one_to_one").merge(
        lookup, on="response_id", how="left", validate="many_to_one")
    ver = ver.merge(ver_key, on=["annotation_id", "response_id"], validate="one_to_one", suffixes=("", "_key")).merge(
        lookup, on="response_id", how="left", validate="many_to_one")
    rows = []
    for condition in ("all", "people", "minority", "majority"):
        group = hedge if condition == "all" else hedge.loc[hedge.group_type.eq(condition)]
        for frequency in ("all", "frequent", "rare"):
            subset = group if frequency == "all" else group.loc[group.cue_frequency_band.eq(frequency)]
            labeled = subset.loc[subset.adjudicated_label.isin(["yes", "no"])]
            p, r, f, error = safe_binomial_metrics(labeled.adjudicated_label, labeled.model_prediction)
            double = subset.loc[subset.human_label_1.isin(["yes", "no", "uncertain"]) &
                                subset.human_label_2.isin(["yes", "no", "uncertain"])]
            rows.append({"analysis": "contextual_hedging", "stage": "cue_detection", "group_type": condition,
                "frequency_bin": frequency, "n_sampled": len(subset), "n_labeled_binary": len(labeled),
                "precision": p, "recall": r, "f1": f, "error_rate": error,
                "human_human_agreement": float((double.human_label_1 == double.human_label_2).mean()) if len(double) else np.nan,
                "human_human_kappa": float(cohen_kappa_score(double.human_label_1, double.human_label_2)) if len(double) >= 2 else np.nan,
                "status": "evaluated" if len(labeled) else "pending_annotation"})

    fields = ("predicate_correct", "introduces_clause", "frame_correct", "polarity_correct", "negation_correct",
              "resource_mapping_appropriate", "strict_decision_correct")
    for condition in ("all", "people", "minority", "majority"):
        group = ver if condition == "all" else ver.loc[ver.group_type.eq(condition)]
        for field in fields:
            chosen = group.loc[group.validation_class.eq("matched")] if field == "resource_mapping_appropriate" else group
            labeled = chosen.loc[chosen[f"{field}_adjudicated"].isin(["yes", "no"])]
            a, kappa = agreement(chosen, field)
            rows.append({"analysis": "veridicality", "stage": field, "group_type": condition,
                "frequency_bin": "", "n_sampled": len(chosen), "n_labeled_binary": len(labeled),
                "accuracy_or_match_precision": float(labeled[f"{field}_adjudicated"].eq("yes").mean()) if len(labeled) else np.nan,
                "human_human_agreement": a, "human_human_kappa": kappa,
                "status": "evaluated" if len(labeled) else "pending_annotation"})
    pd.DataFrame(rows).to_csv(directory / "measurement_validation.csv", index=False)

    differential = []
    hedge_labeled = hedge.loc[hedge.adjudicated_label.isin(["yes", "no"])].copy()
    hedge_labeled["error"] = hedge_labeled.adjudicated_label.ne(hedge_labeled.model_prediction)
    for analysis, frame, error_col in (("contextual_hedging", hedge_labeled, "error"),):
        people = frame.loc[frame.group_type.eq("people")]
        for other in ("minority", "majority"):
            target = frame.loc[frame.group_type.eq(other)]
            row = {"analysis": analysis, "stage": "cue_detection", "comparison": f"{other} - people",
                   "n_people": len(people), "n_other": len(target), "status": "pending_annotation"}
            if len(people) and len(target):
                table = [[int(target[error_col].sum()), int((~target[error_col]).sum())],
                         [int(people[error_col].sum()), int((~people[error_col]).sum())]]
                row.update({"error_rate_difference": float(target[error_col].mean() - people[error_col].mean()),
                            "fisher_exact_p_value": float(fisher_exact(table)[1]), "status": "evaluated"})
            differential.append(row)
    for field in ("predicate_correct", "frame_correct", "polarity_correct", "resource_mapping_appropriate"):
        selected = ver.loc[ver.validation_class.eq("matched")] if field == "resource_mapping_appropriate" else ver
        selected = selected.loc[selected[f"{field}_adjudicated"].isin(["yes", "no"])].copy()
        selected["error"] = selected[f"{field}_adjudicated"].eq("no")
        people = selected.loc[selected.group_type.eq("people")]
        for other in ("minority", "majority"):
            target = selected.loc[selected.group_type.eq(other)]
            row = {"analysis": "veridicality", "stage": field, "comparison": f"{other} - people",
                   "n_people": len(people), "n_other": len(target), "status": "pending_annotation"}
            if len(people) and len(target):
                table = [[int(target.error.sum()), int((~target.error).sum())],
                         [int(people.error.sum()), int((~people.error).sum())]]
                row.update({"error_rate_difference": float(target.error.mean() - people.error.mean()),
                            "fisher_exact_p_value": float(fisher_exact(table)[1]), "status": "evaluated"})
            differential.append(row)
    pd.DataFrame(differential).to_csv(directory / "differential_measurement_error.csv", index=False)
    summary = {"hedge_sample_n": len(hedge), "veridicality_sample_n": len(ver),
               "hedge_adjudicated_binary_n": int(hedge.adjudicated_label.isin(["yes", "no"]).sum()),
               "veridicality_mapping_adjudicated_n": int(ver.resource_mapping_appropriate_adjudicated.isin(["yes", "no"]).sum())}
    (directory / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/score manual validation and differential measurement error.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = pd.read_csv(args.source_dir / "contextual_hedging" / "contextual_hedging_response_metrics.csv")
    summary = validation_tables(args.source_dir.resolve(), args.output_dir.resolve(), data)
    print(f"Validation sheets and metrics written: {summary}")


if __name__ == "__main__":
    main()
