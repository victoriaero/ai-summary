from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from analyze_epistemic_commitment import (
    DEFAULT_COLLECTIONS_ROOT,
    DEFAULT_MEGAVERIDICALITY_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPACY_MODEL,
    METADATA_COLUMNS,
    build_resource_surface_map,
    discover_responses,
    has_to_marker,
    is_conditional_antecedent,
    is_negated,
    is_passive,
    metadata_subset,
    resolve_resource_lemma,
    subtree_span,
)


SEED = 20260930
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "veridicality_v2"
OCCURRENCE_COLS = METADATA_COLUMNS + [
    "sentence_id", "sentence_text", "predicate_surface", "predicate_lemma",
    "phrasal_predicate", "complement_text", "dependency_relation",
    "syntactic_frame", "voice", "negated", "polarity",
    "matched_resource_entry", "match_status", "unmatched_reason",
    "veridicality_score", "parser_confidence",
]


def load_resource(path: Path) -> pd.DataFrame:
    resource = pd.read_csv(path, sep="\t")
    required = {"verb", "frame", "polarity", "sentence", "veridicalitynorm"}
    if missing := required - set(resource.columns):
        raise ValueError(f"MegaVeridicality missing required columns: {sorted(missing)}")
    resource = resource.copy()
    resource["predicate_lemma"] = resource.verb.str.casefold().str.strip()
    resource["resource_entry_id"] = [f"MV21-{i:05d}" for i in range(1, len(resource) + 1)]
    return resource


def candidate_frames(predicate, complement) -> tuple[list[str], str]:
    passive = is_passive(predicate)
    if complement.dep_ == "ccomp":
        return (["NP was Ved that S"] if passive else ["NP Ved that S"], "finite_clausal_complement")
    if complement.dep_ != "xcomp" and not (complement.dep_ == "advcl" and has_to_marker(complement)):
        return [], "frame_unresolved"
    if not has_to_marker(complement):
        return [], "frame_unresolved"
    if passive:
        base = "NP was Ved to VP"
    elif any(c.dep_ in {"obj", "dobj"} and c.i < complement.i for c in predicate.children) or any(c.dep_ in {"nsubj", "nsubjpass"} for c in complement.children):
        base = "NP Ved NP to VP"
    else:
        base = "NP Ved to VP"
    if complement.lemma_.casefold() == "do":
        return [f"{base}[+eventive]"], "infinitival_eventive_do"
    if complement.lemma_.casefold() == "have":
        return [f"{base}[-eventive]"], "infinitival_noneventive_have"
    return [f"{base}[+eventive]", f"{base}[-eventive]"], "eventivity_unresolved"


def get_complements(predicate):
    return [c for c in predicate.children if c.dep_ in {"ccomp", "xcomp"} or (c.dep_ == "advcl" and has_to_marker(c))]


def decide_match(predicate, complement, lemma: str, resource: pd.DataFrame) -> tuple[str, str, str, float | None]:
    frames, frame_state = candidate_frames(predicate, complement)
    if not frames:
        return "", "unmatched", frame_state, None
    if is_conditional_antecedent(predicate):
        return " | ".join(frames), "unmatched", "conditional_context", None
    polarity = "negative" if is_negated(predicate) else "positive"
    entries = resource.loc[resource.predicate_lemma.eq(lemma)]
    same_frame = entries.loc[entries.frame.isin(frames)]
    if same_frame.empty:
        return " | ".join(frames), "unmatched", "frame_unresolved", None
    same_polarity = same_frame.loc[same_frame.polarity.str.casefold().eq(polarity)]
    if same_polarity.empty:
        return " | ".join(frames), "unmatched", "polarity_mismatch", None
    if frame_state == "eventivity_unresolved" and len(same_polarity) > 1:
        return " | ".join(frames), "unmatched", "eventivity_unresolved", None
    if len(same_polarity) != 1:
        return " | ".join(frames), "unmatched", "ambiguous_surface", None
    row = same_polarity.iloc[0]
    return str(row.frame), "matched_strict", "", float(row.veridicalitynorm)


def analyze_response(response: pd.Series, doc, resource: pd.DataFrame, surface_map: dict[str, str], ambiguous: dict[str, list[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lemmas = set(resource.predicate_lemma)
    sentences = list(doc.sents)
    sent_by_token = {token.i: (i + 1, sent) for i, sent in enumerate(sentences) for token in sent}
    candidates = []
    parser_available = bool(doc.has_annotation("DEP"))
    for predicate in doc:
        complements = get_complements(predicate) if parser_available else []
        lemma, method = resolve_resource_lemma(predicate, lemmas, surface_map)
        surface_ambiguous = predicate.lower_ in ambiguous
        # Keep resource predicates even without a clause complement for an auditable denominator.
        if lemma is not None and not complements:
            sentence_id, sentence = sent_by_token[predicate.i]
            candidates.append(make_row(response, predicate, None, sentence_id, sentence, lemma, "", "unmatched", "no_clause_complement", None, parser_available))
            continue
        # Keep clause-embedding candidates that do not exist in the rating resource.
        for complement in complements:
            sentence_id, sentence = sent_by_token[predicate.i]
            if surface_ambiguous:
                frame, status, reason, score = "", "unmatched", "ambiguous_surface", None
                resolved = lemma or predicate.lemma_.casefold()
            elif lemma is None:
                frame, status, reason, score = "", "unmatched", "predicate_not_in_resource", None
                resolved = predicate.lemma_.casefold()
            else:
                frame, status, reason, score = decide_match(predicate, complement, lemma, resource)
                resolved = lemma
            candidates.append(make_row(response, predicate, complement, sentence_id, sentence, resolved, frame, status, reason, score, parser_available))
    scores = [r["veridicality_score"] for r in candidates if r["veridicality_score"] is not None]
    metrics = {**metadata_subset(response), "predicate_coverage": int(bool(scores)), "n_veridical_predicates": len(scores),
        "n_unmatched_candidates": len(candidates) - len(scores), "mean_veridicality": float(np.mean(scores)) if scores else np.nan,
        "median_veridicality": float(np.median(scores)) if scores else np.nan, "min_veridicality": min(scores) if scores else np.nan,
        "max_veridicality": max(scores) if scores else np.nan, "n_veridical_predicates_ge2": int(len(scores) >= 2)}
    return candidates, metrics


def make_row(response, predicate, complement, sentence_id, sentence, lemma, frame, status, reason, score, parser_available):
    particles = [c.lower_ for c in predicate.children if c.dep_ == "prt"]
    phrase = "_".join([lemma, *particles]) if particles else ""
    voice = "passive" if is_passive(predicate) else "active"
    negated = is_negated(predicate)
    return {**metadata_subset(response), "sentence_id": sentence_id, "sentence_text": sentence.text,
        "predicate_surface": predicate.text, "predicate_lemma": lemma, "phrasal_predicate": phrase,
        "complement_text": subtree_span(complement).text if complement is not None else "",
        "dependency_relation": complement.dep_ if complement is not None else "",
        "syntactic_frame": frame, "voice": voice, "negated": negated,
        "polarity": "negative" if negated else "positive", "matched_resource_entry": "",
        "match_status": status, "unmatched_reason": reason,
        "veridicality_score": score, "parser_confidence": np.nan if parser_available else 0.0}


def make_validation_sample(occurrences: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    samples = []
    for matched, requested in ((True, 200), (False, 100)):
        subset = occurrences.loc[occurrences.match_status.eq("matched_strict" if matched else "unmatched")]
        if subset.empty:
            continue
        # Stratify unmatched samples by explicit unmatched reason.
        n = min(requested, len(subset))
        group_col = "unmatched_reason" if not matched else "predicate_lemma"
        quotas = (subset[group_col].value_counts(normalize=True) * n).round().astype(int)
        while quotas.sum() > n:
            eligible = quotas[quotas > 1]
            if eligible.empty: break
            quotas[eligible.idxmax()] -= 1
        picks = []
        for key, quota in quotas.items():
            part = subset.loc[subset[group_col].eq(key)]
            picks.extend(part.sample(n=min(int(quota), len(part)), random_state=SEED).index.tolist())
        if len(picks) < n:
            remaining = subset.drop(index=picks)
            picks.extend(remaining.sample(n=min(n-len(picks), len(remaining)), random_state=SEED).index.tolist())
        samples.extend(occurrences.loc[picks].assign(validation_class="matched" if matched else "unmatched").to_dict("records"))
    sample = pd.DataFrame(samples)
    if sample.empty:
        return
    sample["annotation_id"] = [f"V{i:04d}" for i in range(1, len(sample) + 1)]
    checks = ["predicate_correct", "introduces_clause", "frame_correct", "polarity_correct", "resource_mapping_appropriate"]
    blind_cols = ["annotation_id", "response_id", "sentence_id", "sentence_text", "predicate_surface", "complement_text"]
    for col in checks:
        sample[f"{col}_annotator_1"] = ""
        sample[f"{col}_annotator_2"] = ""
        sample[f"{col}_adjudicated"] = ""
        blind_cols.extend([f"{col}_annotator_1", f"{col}_annotator_2", f"{col}_adjudicated"])
    sample[blind_cols].to_csv(output_dir / "validation_blind.csv", index=False)
    sample[["annotation_id", "response_id", "validation_class", "predicate_lemma", "syntactic_frame", "polarity", "matched_resource_entry", "match_status", "unmatched_reason", "veridicality_score"]].to_csv(output_dir / "validation_key.csv", index=False)


def make_sensitivity(metrics: pd.DataFrame, occurrences: pd.DataFrame, output_dir: Path) -> None:
    covered = metrics.loc[metrics.predicate_coverage.eq(1)].copy()
    covered["metric"] = "strict_mean"
    covered["value"] = covered.mean_veridicality
    two_plus = covered.loc[covered.n_veridical_predicates.ge(2)].copy()
    two_plus["metric"] = "strict_mean_n_ge_2"
    two_plus["value"] = two_plus.mean_veridicality
    median = covered.copy()
    median["metric"] = "strict_median"
    median["value"] = median.median_veridicality
    pd.concat([covered, two_plus, median], ignore_index=True).to_csv(output_dir / "response_sensitivity_metrics.csv", index=False)
    matched = occurrences.loc[occurrences.match_status.eq("matched_strict")].copy()
    if matched.empty:
        return
    counts = matched.predicate_lemma.value_counts()
    cutoff = float(counts.median() + 3 * (counts - counts.median()).abs().median())
    dominant = set(counts[counts > cutoff].index)
    matched["dominant_predicate"] = matched.predicate_lemma.isin(dominant)
    matched.to_csv(output_dir / "occurrence_level_strict_scores.csv", index=False)
    loo = []
    for lemma in counts.index:
        omitted = matched.loc[~matched.predicate_lemma.eq(lemma)]
        for response_id, group in omitted.groupby("response_id"):
            loo.append({"omitted_predicate": lemma, "response_id": response_id, "mean_veridicality": group.veridicality_score.mean(), "median_veridicality": group.veridicality_score.median(), "n_veridical_predicates": len(group)})
    pd.DataFrame(loo).to_csv(output_dir / "leave_one_predicate_out.csv", index=False)
    matched.loc[~matched.dominant_predicate].groupby("response_id", as_index=False).agg(
        mean_veridicality_without_dominant=("veridicality_score", "mean"), n_predicates_without_dominant=("veridicality_score", "size")
    ).to_csv(output_dir / "without_dominant_predicates.csv", index=False)


def evaluate_validation(path: Path, output_dir: Path) -> None:
    if not path.exists():
        return
    data = pd.read_csv(path).fillna("")
    rows = []
    for label in ("predicate_correct", "introduces_clause", "frame_correct", "polarity_correct", "resource_mapping_appropriate"):
        a, b, gold = (f"{label}_annotator_1", f"{label}_annotator_2", f"{label}_adjudicated")
        if not {a, b, gold}.issubset(data.columns):
            continue
        valid = data.loc[data[gold].astype(str).str.strip().ne("")]
        if len(valid):
            rows.append({"validation_item": label, "n": len(valid), "annotator_agreement": accuracy_score(valid[a], valid[b]), "adjudicated_accuracy": accuracy_score(valid[gold], valid[a])})
    pd.DataFrame(rows).to_csv(output_dir / "human_validation_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict MegaVeridicality v2.1 clause-embedding analysis.")
    parser.add_argument("--collections-root", type=Path, default=DEFAULT_COLLECTIONS_ROOT)
    parser.add_argument("--resource", type=Path, default=DEFAULT_MEGAVERIDICALITY_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    parser.add_argument("--validation-file", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    resource = load_resource(args.resource.resolve())
    responses, inventory = discover_responses(args.collections_root.resolve())
    nlp = spacy.load(args.spacy_model)
    surface_map, ambiguous = build_resource_surface_map(nlp, resource)
    occurrences, metrics = [], []
    for (_, response), doc in zip(responses.iterrows(), nlp.pipe(responses.aio_text.tolist(), batch_size=32)):
        occ, metric = analyze_response(response, doc, resource, surface_map, ambiguous)
        occurrences.extend(occ); metrics.append(metric)
    occ_df = pd.DataFrame(occurrences, columns=OCCURRENCE_COLS)
    metric_df = pd.DataFrame(metrics)
    occ_df.to_csv(out / "predicate_occurrences_v2.csv", index=False)
    metric_df.to_csv(out / "veridicality_response_metrics_v2.csv", index=False)
    inventory.drop(columns=["aio_text"]).to_csv(out / "input_inventory.csv", index=False)
    resource.to_csv(out / "megaveridicality_resource_entries.csv", index=False)
    make_validation_sample(occ_df, out)
    make_sensitivity(metric_df, occ_df, out)
    if args.validation_file:
        evaluate_validation(args.validation_file.resolve(), out)
    provenance = {"resource": "MegaVeridicality v2.1", "score": "veridicalitynorm", "seed": SEED,
        "resource_note": "The normalized TSV encodes frame and polarity; voice is represented in frame strings. It has no separate conditional column, so conditional contexts are unmatched.",
        "parser": {"spacy": spacy.__version__, "model": args.spacy_model, "model_version": nlp.meta.get("version")},
        "strict_scored_occurrences": int(occ_df.veridicality_score.notna().sum()),
        "unmatched_by_reason": occ_df.loc[occ_df.match_status.eq("unmatched"), "unmatched_reason"].value_counts().to_dict(),
        "annotation_blindness": "prediction/match labels are kept in validation_key.csv, separate from validation_blind.csv"}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Strict veridicality outputs written to {out}")


if __name__ == "__main__":
    main()
