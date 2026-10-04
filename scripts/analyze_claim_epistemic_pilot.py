"""Link Dallas claim candidates to epistemic cues in their original AIO text.

This is an exploratory, non-generative analysis. A cue inside a source span is
only a scope *candidate*, never a verified claim-level hedge judgment.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "results/claim_extraction_pilot/microsoft-phi-4"
EPISTEMIC = ROOT / "results/epistemic_commitment"
OUTPUT = ROOT / "results/claim_epistemic_pilot_dallas"
CONTRASTS = (("minority", "majority"), ("minority", "people"), ("majority", "people"))


def _unique_offset(haystack: str, needle: str) -> int | None:
    if not needle:
        return None
    start = haystack.find(needle)
    return start if start >= 0 and haystack.find(needle, start + 1) < 0 else None


def _source_intervals(claim: pd.Series) -> list[tuple[int, int]]:
    text = claim.full_aio_text
    sentence_start = int(claim.sentence_start_char)
    if text[sentence_start:int(claim.sentence_end_char)] != claim.original_sentence:
        raise ValueError(f"Sentence offset mismatch: {claim.claim_id}")
    spans = json.loads(claim.verifiable_span)
    intervals = []
    for span in spans:
        start = sentence_start + int(span["start_char"])
        end = sentence_start + int(span["end_char"])
        if not start < end or text[start:end] != span["text"]:
            raise ValueError(f"Source span mismatch: {claim.claim_id}")
        intervals.append((start, end))
    if not intervals:
        raise ValueError(f"Missing source span: {claim.claim_id}")
    return intervals


def _contained(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(a <= start < end <= b for a, b in intervals)


def _overlaps(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start < b and end > a for a, b in intervals)


def _claim_begins_with_complement(claim_text: str, complement_text: str) -> bool:
    # Used only to confirm linkage. All measurements still come from the AIO.
    complement = " ".join(complement_text.split()).casefold()
    if complement.startswith("that "):
        complement = complement[5:]
    claim = " ".join(claim_text.split()).casefold()
    return bool(complement) and claim.startswith(complement)


def _strict_offsets(row: pd.Series, text: str) -> dict:
    sentence = str(row.sentence_text)
    sentence_start = _unique_offset(text, sentence)
    if sentence_start is None:
        return {"alignment_status": "sentence_not_unique"}
    predicate = _unique_offset(sentence, str(row.predicate_surface))
    complement = _unique_offset(sentence, str(row.complement_text))
    if predicate is None or complement is None:
        return {"alignment_status": "predicate_or_complement_not_unique"}
    return {"alignment_status": "unique_literal", "predicate_start": sentence_start + predicate,
            "predicate_end": sentence_start + predicate + len(str(row.predicate_surface)),
            "complement_start": sentence_start + complement,
            "complement_end": sentence_start + complement + len(str(row.complement_text))}


def _summarize(data: pd.DataFrame, keys: list[str], level: str) -> pd.DataFrame:
    rows = []
    for key, sub in data.groupby(keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row.update({"summary_level": level, "n_claims": len(sub),
                    "n_responses": sub.response_id.nunique(),
                    "n_original_sentences": sub.sentence_id.nunique(),
                    "hedge_coverage": float(sub.hedge_analysis_covered.mean()),
                    "hedge_span_candidate_rate": float(sub.H_span_proxy.mean()),
                    "hedge_sentence_context_rate": float(sub.hedge_sentence_context.mean()),
                    "veridicality_coverage": float(sub.V_covered.mean()),
                    "n_veridicality_scored": int(sub.V_covered.sum()),
                    "mean_veridicality": sub.V_score.mean() if sub.V_covered.any() else np.nan,
                    "median_veridicality": sub.V_score.median() if sub.V_covered.any() else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def _granularity_diagnostic(metrics: pd.DataFrame, epistemic_dir: Path) -> pd.DataFrame:
    hedge_responses = pd.read_csv(epistemic_dir / "contextual_hedging/contextual_hedging_response_metrics.csv")
    veridicality_responses = pd.read_csv(epistemic_dir / "veridicality_v2/veridicality_response_metrics_v2.csv")
    response_ids = set(metrics.response_id)
    hedge_responses = hedge_responses.loc[hedge_responses.response_id.isin(response_ids)].set_index("response_id")
    veridicality_responses = veridicality_responses.loc[
        veridicality_responses.response_id.isin(response_ids)].set_index("response_id")
    rows = []
    for level, keys in (("condition", ["group_type"]),
                        ("dimension_condition", ["group_type", "demographic_dimension"])):
        for key, subset in metrics.groupby(keys, dropna=False, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            ids = subset.response_id.unique()
            h = hedge_responses.reindex(ids)
            v = veridicality_responses.reindex(ids)
            sentences = subset.drop_duplicates(["response_id", "sentence_id"])
            rows.append({**dict(zip(keys, key)), "summary_level": level,
                         "n_responses": len(ids), "n_claim_bearing_sentences": len(sentences),
                         "n_claims": len(subset),
                         "response_mean_hedged_sentence_rate_all_sentences":
                             h.contextual_hedged_sentence_rate.mean(),
                         "claim_bearing_sentence_hedge_context_rate":
                             sentences.hedge_sentence_context.mean(),
                         "claim_span_hedge_candidate_rate": subset.H_span_proxy.mean(),
                         "response_veridicality_coverage": v.n_veridical_predicates.gt(0).mean(),
                         "claim_veridicality_coverage_conservative": subset.V_covered.mean(),
                         "claim_mean_veridicality_scored_only": subset.V_score.mean()})
    return pd.DataFrame(rows)


def analyze(claim_dir: Path, epistemic_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    claims = pd.read_csv(claim_dir / "valid_claims.csv")
    if claims.claim_id.duplicated().any():
        raise ValueError("claim_id must be unique")
    claims = claims.loc[claims.replica_id.eq("v1_dallas")].copy()
    claim_ids = set(claims.claim_id)
    response_text = claims.drop_duplicates("response_id").set_index("response_id").full_aio_text.to_dict()
    hedge_covered = set(pd.read_csv(epistemic_dir / "contextual_hedging/contextual_hedging_response_metrics.csv").response_id)
    veridicality_covered = set(pd.read_csv(epistemic_dir / "veridicality_v2/veridicality_response_metrics_v2.csv").response_id)
    hedges = pd.read_csv(epistemic_dir / "contextual_hedging/contextual_hedge_occurrences.csv")
    strict = pd.read_csv(epistemic_dir / "veridicality_v2/occurrence_level_strict_scores.csv")
    matches = pd.read_csv(claim_dir / "claim_matching_candidates.csv")
    hedge_by_response = {key: value for key, value in hedges.groupby("response_id")}
    strict_by_response = {key: value for key, value in strict.groupby("response_id")}
    hedge_links, veridicality_links, metric_rows = [], [], []
    cue_claims: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    strict_audit = []
    aligned_strict: dict[str, list[tuple[int, pd.Series, dict]]] = defaultdict(list)
    for response_id, group in strict_by_response.items():
        if response_id not in response_text:
            continue
        for index, occurrence in group.iterrows():
            offsets = _strict_offsets(occurrence, response_text[response_id])
            strict_audit.append({"resource_row": index, "response_id": response_id,
                                 "alignment_status": offsets["alignment_status"], **offsets})
            if offsets["alignment_status"] == "unique_literal":
                aligned_strict[response_id].append((index, occurrence, offsets))

    for _, claim in claims.iterrows():
        intervals = _source_intervals(claim)
        response_id = claim.response_id
        text = claim.full_aio_text
        sentence_start, sentence_end = int(claim.sentence_start_char), int(claim.sentence_end_char)
        sentence_cues, span_cues = [], []
        for index, cue in hedge_by_response.get(response_id, pd.DataFrame()).iterrows():
            cue_start, cue_end = int(cue.cue_start), int(cue.cue_end)
            if text[cue_start:cue_end] != str(cue.cue_surface):
                raise ValueError(f"Hedge cue offset mismatch: {response_id}, row {index}")
            if sentence_start <= cue_start < cue_end <= sentence_end:
                inside = _contained(cue_start, cue_end, intervals)
                sentence_cues.append(str(cue.cue_surface))
                if inside:
                    span_cues.append(str(cue.cue_surface))
                    cue_claims[(response_id, cue_start, cue_end)].add(claim.claim_id)
                hedge_links.append({"claim_id": claim.claim_id, "response_id": response_id,
                                    "sentence_id": claim.sentence_id, "resource_row": index,
                                    "cue": cue.cue_surface, "cue_start": cue_start, "cue_end": cue_end,
                                    "relation": "inside_source_span_scope_unverified" if inside else "same_sentence_outside_span",
                                    "cue_probability": cue.cue_probability})

        linked_scores = []
        for index, occurrence, offsets in aligned_strict.get(response_id, []):
            complement_start, complement_end = offsets["complement_start"], offsets["complement_end"]
            if not (sentence_start <= complement_start < complement_end <= sentence_end):
                continue
            source_inside_complement = all(complement_start <= a < b <= complement_end
                                           for a, b in intervals)
            complement_inside_source = _contained(complement_start, complement_end, intervals)
            if source_inside_complement or (complement_inside_source and
                                             _claim_begins_with_complement(str(claim.claim_text),
                                                                          str(occurrence.complement_text))):
                relation = ("source_span_inside_complement" if source_inside_complement
                            else "claim_begins_with_contained_complement")
                linked_scores.append(float(occurrence.veridicality_score))
            elif complement_inside_source:
                relation = "complement_in_source_span_claim_scope_unverified"
            elif _overlaps(complement_start, complement_end, intervals):
                relation = "partial_complement_overlap_unscored"
            else:
                relation = "same_sentence_only_unscored"
            veridicality_links.append({"claim_id": claim.claim_id, "response_id": response_id,
                                       "sentence_id": claim.sentence_id, "resource_row": index,
                                       "predicate_surface": occurrence.predicate_surface,
                                       "predicate_lemma": occurrence.predicate_lemma,
                                       "complement_text": occurrence.complement_text,
                                       "syntactic_frame": occurrence.syntactic_frame,
                                       "negated": occurrence.negated,
                                       "matched_resource_entry": occurrence.matched_resource_entry,
                                       "resource_score": occurrence.veridicality_score,
                                       "predicate_start": offsets["predicate_start"],
                                       "complement_start": complement_start,
                                       "complement_end": complement_end, "link_status": relation})
        metric_rows.append({"claim_id": claim.claim_id, "response_id": response_id,
                            "query_id": claim.query_id, "domain": claim.domain, "outcome": claim.outcome,
                            "demographic_dimension": claim.demographic_dimension,
                            "group": claim.group, "group_type": claim.group_type,
                            "location": claim.location, "replica_id": claim.replica_id,
                            "sentence_id": claim.sentence_id, "original_sentence": claim.original_sentence,
                            "original_span_text": " | ".join(text[a:b] for a, b in intervals),
                            "source_intervals_absolute": json.dumps(intervals),
                            "hedge_analysis_covered": response_id in hedge_covered,
                            "n_hedge_cues_in_sentence": len(sentence_cues),
                            "n_hedge_cues_in_span": len(span_cues),
                            "hedge_sentence_context": bool(sentence_cues) if response_id in hedge_covered else np.nan,
                            "H_span_proxy": bool(span_cues) if response_id in hedge_covered else np.nan,
                            "hedge_scope_verified": False,
                            "hedge_span_cues": json.dumps(span_cues, ensure_ascii=False),
                            "veridicality_analysis_covered": response_id in veridicality_covered,
                            "n_linked_strict_predicates": len(linked_scores),
                            "V_covered": len(linked_scores) == 1,
                            "V_score": linked_scores[0] if len(linked_scores) == 1 else np.nan,
                            "V_link_status": "one_conservatively_linked_strict_predicate" if len(linked_scores) == 1
                            else ("multiple_strict_complements_unscored" if len(linked_scores) > 1
                                  else "no_conservatively_linked_strict_predicate")})

    metrics = pd.DataFrame(metric_rows)
    hedge_links = pd.DataFrame(hedge_links)
    if len(hedge_links):
        hedge_links["n_claims_sharing_cue_in_span"] = hedge_links.apply(
            lambda row: len(cue_claims.get((row.response_id, row.cue_start, row.cue_end), set())), axis=1)
    veridicality_links = pd.DataFrame(veridicality_links)
    metrics.to_csv(output / "claim_epistemic_metrics.csv", index=False)
    hedge_links.to_csv(output / "claim_hedge_links.csv", index=False)
    veridicality_links.to_csv(output / "claim_veridicality_links.csv", index=False)
    pd.DataFrame(strict_audit).to_csv(output / "strict_occurrence_alignment_audit.csv", index=False)

    summary = pd.concat([
        _summarize(metrics, ["group_type"], "condition"),
        _summarize(metrics, ["group_type", "demographic_dimension"], "dimension_condition"),
        _summarize(metrics, ["group_type", "domain", "outcome"], "outcome_condition"),
        _summarize(metrics, ["group_type", "demographic_dimension", "domain", "outcome"],
                   "dimension_outcome_condition")], ignore_index=True)
    summary.to_csv(output / "all_claims_summary.csv", index=False)

    matched = matches.loc[matches.match_status.eq("EQUIVALENT") &
                          matches.claim_a_id.isin(claim_ids) & matches.claim_b_id.isin(claim_ids)].copy()
    lookup = metrics.set_index("claim_id")
    pair_rows = []
    for pair in matched.itertuples():
        left, right = lookup.loc[pair.claim_a_id], lookup.loc[pair.claim_b_id]
        if left.group_type == right.group_type or left.domain != right.domain or left.outcome != right.outcome:
            continue
        key = frozenset((left.group_type, right.group_type))
        contrast = next(((a, b) for a, b in CONTRASTS if frozenset((a, b)) == key), None)
        if contrast is None:
            continue
        a, b = (left, right) if left.group_type == contrast[0] else (right, left)
        pair_rows.append({"pair_id": pair.record_id, "comparison": "-".join(contrast),
                          "demographic_dimension": pair.demographic_dimension,
                          "domain": a.domain, "outcome": a.outcome,
                          "claim_a_id": a.name, "claim_b_id": b.name,
                          "group_type_a": a.group_type, "group_type_b": b.group_type,
                          "response_id_a": a.response_id, "response_id_b": b.response_id,
                          "H_a": a.H_span_proxy, "H_b": b.H_span_proxy,
                          "delta_H": float(a.H_span_proxy) - float(b.H_span_proxy)
                          if pd.notna(a.H_span_proxy) and pd.notna(b.H_span_proxy) else np.nan,
                          "V_a": a.V_score, "V_b": b.V_score,
                          "delta_V": a.V_score - b.V_score if pd.notna(a.V_score) and pd.notna(b.V_score) else np.nan,
                          "both_V_covered": bool(a.V_covered and b.V_covered),
                          "hedge_scope_verified": False, "match_status": pair.match_status,
                          "match_exploratory": True})
    pair_metrics = pd.DataFrame(pair_rows)
    pair_metrics.to_csv(output / "matched_claim_pairs.csv", index=False)
    matched_ids = set(pair_metrics.claim_a_id).union(pair_metrics.claim_b_id) if len(pair_metrics) else set()
    matched_claims = metrics.loc[metrics.claim_id.isin(matched_ids)]
    diagnostics = pd.concat([_summarize(metrics, ["group_type"], "all_claims"),
                             _summarize(matched_claims, ["group_type"], "matched_claims_unique")],
                            ignore_index=True)
    diagnostics.to_csv(output / "coverage_diagnostic.csv", index=False)
    _granularity_diagnostic(metrics, epistemic_dir).to_csv(
        output / "unit_granularity_diagnostic.csv", index=False)

    # Outcome-level contrasts compare entire claim sets, not equivalent claims.
    outcome_rows = []
    dimensions = sorted(metrics.loc[metrics.group_type.ne("people"), "demographic_dimension"].unique())
    for dimension in dimensions:
        dim = metrics.loc[metrics.demographic_dimension.eq(dimension) & metrics.group_type.ne("people")]
        for domain, outcome in dim[["domain", "outcome"]].drop_duplicates().itertuples(index=False, name=None):
            conditions = {kind: (metrics.loc[metrics.group_type.eq("people") & metrics.domain.eq(domain) &
                                             metrics.outcome.eq(outcome)] if kind == "people" else
                                 dim.loc[dim.group_type.eq(kind) & dim.domain.eq(domain) & dim.outcome.eq(outcome)])
                          for kind in ("people", "minority", "majority")}
            for a, b in CONTRASTS:
                left, right = conditions[a], conditions[b]
                outcome_rows.append({"demographic_dimension": dimension, "domain": domain, "outcome": outcome,
                                     "comparison": f"{a}-{b}", "n_claims_a": len(left), "n_claims_b": len(right),
                                     "delta_H_span_proxy_rate": left.H_span_proxy.mean() - right.H_span_proxy.mean()
                                     if len(left) and len(right) else np.nan,
                                     "V_coverage_a": left.V_covered.mean() if len(left) else np.nan,
                                     "V_coverage_b": right.V_covered.mean() if len(right) else np.nan,
                                     "delta_mean_V_scored": left.V_score.mean() - right.V_score.mean()
                                     if left.V_covered.any() and right.V_covered.any() else np.nan,
                                     "matched_content": False})
    pd.DataFrame(outcome_rows).to_csv(output / "all_claims_outcome_contrasts.csv", index=False)
    report = {"n_claims": len(metrics), "n_matched_pairs": len(pair_metrics),
              "n_unique_matched_claims": len(matched_claims),
              "n_hedge_span_candidates": int(metrics.H_span_proxy.sum()),
              "n_claims_with_one_strict_linked_score": int(metrics.V_covered.sum()),
              "n_strict_occurrences_dallas": len(strict_audit),
              "n_strict_occurrences_uniquely_aligned": int(sum(x["alignment_status"] == "unique_literal"
                                                           for x in strict_audit)),
              "n_matched_pairs_with_both_V": int(pair_metrics.both_V_covered.sum()) if len(pair_metrics) else 0,
              "n_matched_pairs_with_H_difference": int(pair_metrics.delta_H.notna().sum()) if len(pair_metrics) else 0,
              "source_claim_dir": str(claim_dir), "source_epistemic_dir": str(epistemic_dir)}
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-dir", type=Path, default=CLAIMS)
    parser.add_argument("--epistemic-dir", type=Path, default=EPISTEMIC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.claim_dir, args.epistemic_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
