from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import analyze_contextual_hedging as h
from analyze_epistemic_commitment import discover_responses


def fit_model(rows):
    x = [h.token_features(t) for r in rows for t in r["doc"]]
    y = [label for r in rows for label in r["labels"]]
    model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=h.SEED))
    model.fit(x, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual BioScope hedging model and AIO transfer analysis.")
    parser.add_argument("--collections-root", type=Path, default=h.DEFAULT_COLLECTIONS_ROOT)
    parser.add_argument("--bioscope-dir", type=Path, default=h.DEFAULT_BIOSCOPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=h.DEFAULT_OUTPUT)
    parser.add_argument("--spacy-model", default=h.DEFAULT_SPACY_MODEL)
    parser.add_argument("--loo-top-cues", type=int, default=10)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load(args.spacy_model)

    raw = h.bioscope_sentences(args.bioscope_dir.resolve())
    rows, _, _, _ = h.sentence_data(raw, nlp)
    train_idx, dev_idx, test_idx = h.stratified_split(np.asarray([r["document_id"] for r in rows], dtype=object))
    train_rows, dev_rows, test_rows = ([rows[i] for i in indexes] for indexes in (train_idx, dev_idx, test_idx))
    model = fit_model(train_rows)
    threshold = h.choose_threshold(dev_rows, model)
    evaluation = h.evaluate_partition(test_rows, model, threshold)
    evaluation.update({"threshold": threshold, "split_seed": h.SEED, "split_unit": "BioScope document", "split_sizes": {"train": len(train_rows), "validation": len(dev_rows), "test": len(test_rows)}})
    (out / "bioscope_heldout_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")

    responses, inventory = discover_responses(args.collections_root.resolve())
    occ, metrics, candidates = h.collect_response_predictions(responses, nlp, model, threshold)
    occ.to_csv(out / "contextual_hedge_occurrences.csv", index=False)
    metrics.to_csv(out / "contextual_hedging_response_metrics.csv", index=False)
    inventory.drop(columns=["aio_text"]).to_csv(out / "input_inventory.csv", index=False)
    h.make_blind_sample(candidates, out)
    import joblib
    model_path = out / "contextual_hedging_model.joblib"
    joblib.dump({"pipeline": model, "threshold": threshold, "model_version": h.MODEL_VERSION,
                 "spacy_model": args.spacy_model, "seed": h.SEED}, model_path)

    # Leave-one-cue-out is defined from BioScope training cue counts, not a hand-written list.
    cue_counts = {}
    for row in train_rows:
        for start, end, surface in row["cues"]:
            key = h.normalize_cue(surface)
            cue_counts[key] = cue_counts.get(key, 0) + 1
    top_cues = [cue for cue, _ in sorted(cue_counts.items(), key=lambda x: (-x[1], x[0]))[:max(0, args.loo_top_cues)]]
    sensitivity = []
    for cue in top_cues:
        loo_rows = []
        for row in train_rows:
            removed = [(a, b) for a, b, surface in row["cues"] if h.normalize_cue(surface) == cue]
            labels = row["labels"].copy()
            excluded = {t.i for t in row["doc"] if any(t.idx < b and t.idx + len(t) > a for a, b in removed)}
            loo_rows.append({**row, "labels": [v for i, v in enumerate(labels) if i not in excluded], "doc": row["doc"], "excluded_token_indices": excluded})
        x, y = [], []
        for row, original in zip(loo_rows, train_rows):
            x.extend(h.token_features(t) for t in original["doc"] if t.i not in row["excluded_token_indices"])
            y.extend(v for i, v in enumerate(original["labels"]) if i not in row["excluded_token_indices"])
        if not y or len(set(y)) < 2:
            continue
        loo_model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=h.SEED))
        loo_model.fit(x, y)
        loo_threshold = h.choose_threshold(dev_rows, loo_model)
        _, loo_metrics, _ = h.collect_response_predictions(responses, nlp, loo_model, loo_threshold)
        loo_metrics.insert(0, "omitted_bioscope_cue", cue)
        loo_metrics["threshold"] = loo_threshold
        sensitivity.append(loo_metrics)
    if sensitivity:
        pd.concat(sensitivity, ignore_index=True).to_csv(out / "leave_one_cue_out_sensitivity.csv", index=False)
    else:
        pd.DataFrame(columns=["omitted_bioscope_cue", "response_id", "contextual_hedged_sentence_rate", "contextual_hedges_per_100_tokens"]).to_csv(out / "leave_one_cue_out_sensitivity.csv", index=False)

    (out / "provenance.json").write_text(json.dumps({
        "model_version": h.MODEL_VERSION, "seed": h.SEED, "spacy_version": spacy.__version__,
        "spacy_model": args.spacy_model, "threshold_selection": "maximize exact cue-span F1 on held-out BioScope validation documents",
        "document_split": {"train": len(train_rows), "validation": len(dev_rows), "test": len(test_rows)},
        "feature_description": "spaCy token/morphology/shape and ±2-token context features; logistic regression; no manual lexical lists",
        "cue_family_sensitivity": "not run: BioScope does not provide cue-family labels, and no manually assigned families were introduced",
        "leave_one_cue_out": {"selection": "top cue forms by frequency in training split", "cues": top_cues},
        "manual_validation": "blind annotation rows and key are separate files; key must not be shared with annotators",
    }, indent=2) + "\n")
    print(f"Contextual hedging outputs written to {out}")


if __name__ == "__main__":
    main()
