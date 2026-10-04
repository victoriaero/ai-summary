"""Qualification-preserving, Claimify-inspired AIO claim extraction pilot.

Not a reproduction of Claimify and not a factual truth verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd

from claim_extraction_qwen14b_core import (
    DEFAULT_INPUT, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_ROOT, EMBEDDING_ID, PROMPT_VERSION, SEED,
    guard_equivalence, load_dallas, model_fingerprint, model_identity, model_slug,
    normalized_exact, preflight, segment_responses, stable_id,
    validate_decomposition, validate_disambiguation,
    validate_selection, validate_verdict,
)
from claim_extraction_qwen14b_inference import LocalModel


STAGES = ("selection", "disambiguation", "decomposition", "entailment", "qualification", "context_audit", "matching")


def save_csv(rows, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if data.empty and columns:
        data = pd.DataFrame(columns=columns)
    for column in data.select_dtypes(include="object").columns:
        if data[column].map(lambda item: isinstance(item, (list, dict))).any():
            data[column] = data[column].map(
                lambda item: json.dumps(item, ensure_ascii=False) if isinstance(item, (list, dict)) else item)
    data.to_csv(path, index=False)


def sentence_inputs(sentences: pd.DataFrame) -> list[dict]:
    rows = sentences.to_dict("records")
    for item in rows:
        item["record_id"] = item["sentence_id"]
    return rows


def _candidate_pairs(claims: pd.DataFrame, embeddings: np.ndarray, mode: str) -> list[dict]:
    """Bidirectional top-five candidate generation, without an equivalence threshold."""
    if claims.empty:
        return []
    vectors = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    rows = claims.reset_index(drop=True)
    pairs = {}
    if mode == "dedup":
        groups = rows.groupby("response_id", sort=False).indices
        group_pairs = [(np.asarray(ix), np.asarray(ix)) for ix in groups.values()]
    else:
        group_pairs = []
        for (_, _, _, replica), subset in rows.groupby(["demographic_dimension", "domain", "outcome", "replica_id"], sort=False):
            people = subset.index[subset.group_type.eq("people")].to_numpy()
            minority = subset.index[subset.group_type.eq("minority")].to_numpy()
            majority = subset.index[subset.group_type.eq("majority")].to_numpy()
            for left, right in ((people, minority), (people, majority), (minority, majority)):
                if len(left) and len(right):
                    group_pairs.append((left, right))
    for left, right in group_pairs:
        sim = vectors[left] @ vectors[right].T
        for ai, row in enumerate(sim):
            for bj in np.argsort(-row)[:5]:
                i, j = int(left[ai]), int(right[bj])
                if i != j:
                    key = tuple(sorted((i, j)))
                    pairs[key] = max(float(row[bj]), pairs.get(key, -1.0))
        if mode != "dedup":
            for bj, col in enumerate(sim.T):
                for ai in np.argsort(-col)[:5]:
                    i, j = int(left[ai]), int(right[bj])
                    key = tuple(sorted((i, j)))
                    pairs[key] = max(float(col[ai]), pairs.get(key, -1.0))
    output = []
    for (i, j), similarity in sorted(pairs.items()):
        a, b = rows.iloc[i], rows.iloc[j]
        if mode == "dedup" and a.response_id != b.response_id:
            continue
        output.append({"record_id": stable_id(mode, a.claim_id, b.claim_id), "claim_a_id": a.claim_id,
                       "claim_b_id": b.claim_id, "claim_a": a.claim_text_with_context_markers,
                       "claim_b": b.claim_text_with_context_markers, "similarity": similarity,
                       "original_query": a.original_query, "original_sentence": a.original_sentence,
                       "preceding_sentences": a.preceding_sentences, "following_sentences": a.following_sentences,
                       "pair_type": mode, "demographic_dimension": a.demographic_dimension, "domain": a.domain,
                       "outcome": a.outcome, "replica_id": a.replica_id,
                       "group_type_a": a.group_type, "group_type_b": b.group_type})
    return output


def _judge_pairs(engine: LocalModel, inputs: list[dict], output: Path, claim_index: dict) -> pd.DataFrame:
    if not inputs:
        return pd.DataFrame(columns=["claim_a_id", "claim_b_id", "similarity", "match_status", "pair_type"])
    results = engine.run_stage("matching", inputs, output, lambda v, r: v if v.get("match_status") in
                               {"EQUIVALENT", "RELATED_NOT_EQUIVALENT", "NOT_RELATED", "UNCERTAIN"}
                               else (_ for _ in ()).throw(ValueError("invalid matching verdict")))
    records = []
    for item in inputs:
        verdict = (results[item["record_id"]] or {}).get("match_status", "UNCERTAIN")
        a, b = claim_index[item["claim_a_id"]], claim_index[item["claim_b_id"]]
        verdict = guard_equivalence(a, b, verdict)
        records.append({**item, "match_status": verdict,
                        "model_reason": (results[item["record_id"]] or {}).get("reason", ""),
                        "exploratory": True})
    return pd.DataFrame(records)


def _deduplicate(valid: pd.DataFrame, embedder, engine: LocalModel, output: Path) -> pd.DataFrame:
    if valid.empty:
        return valid
    valid = valid.sort_values("claim_id").copy()
    valid["exact_key"] = valid.claim_text_with_context_markers.map(normalized_exact)
    exact = valid.drop_duplicates(["response_id", "exact_key"]).copy()
    exact = exact.drop(columns="exact_key").reset_index(drop=True)
    if len(exact) < 2:
        return exact
    vectors = np.asarray(embedder.encode(exact.claim_text_with_context_markers.tolist(), normalize_embeddings=True))
    inputs = _candidate_pairs(exact, vectors, "dedup")
    index = {row["claim_id"]: row for row in exact.to_dict("records")}
    judged = _judge_pairs(engine, inputs, output / "logs" / "dedup_judgments.jsonl", index)
    save_csv(judged, output / "deduplication_candidates.csv")
    dropped = set()
    for row in judged.itertuples():
        if row.match_status == "EQUIVALENT" and row.claim_a_id not in dropped and row.claim_b_id not in dropped:
            dropped.add(max(row.claim_a_id, row.claim_b_id))
    return exact.loc[~exact.claim_id.isin(dropped)].reset_index(drop=True)


def _presence_patterns(valid: pd.DataFrame, matching: pd.DataFrame) -> pd.DataFrame:
    """Only complete pairwise-equivalent, one-per-condition components get a pattern."""
    if valid.empty:
        return pd.DataFrame(columns=["demographic_dimension", "domain", "outcome", "claim_ids", "provisional_pattern", "status"])
    rows = []
    for (dimension, domain, outcome, replica), part in valid.groupby(["demographic_dimension", "domain", "outcome", "replica_id"], sort=False):
        ids = part.claim_id.tolist()
        types = part.set_index("claim_id").group_type.to_dict()
        edges = set()
        uncertain_ids = set()
        if not matching.empty:
            sub = matching.loc[(matching.demographic_dimension == dimension) & (matching.domain == domain) &
                               (matching.outcome == outcome) & (matching.replica_id == replica)]
            edges = {frozenset((r.claim_a_id, r.claim_b_id)) for r in sub.loc[sub.match_status.eq("EQUIVALENT")].itertuples()}
            uncertain = sub.loc[sub.match_status.eq("UNCERTAIN")]
            uncertain_ids = set(uncertain.claim_a_id).union(uncertain.claim_b_id)
        parent = {item: item for item in ids}

        def find(item):
            while parent[item] != item:
                item = parent[item]
            return item

        for edge in edges:
            a, b = tuple(edge)
            if a in parent and b in parent:
                parent[find(b)] = find(a)
        components = {}
        for item in ids:
            components.setdefault(find(item), []).append(item)
        for members in components.values():
            kinds = [types[item] for item in members]
            complete = (len(kinds) == len(set(kinds)) and all(
                frozenset((a, b)) in edges for i, a in enumerate(members) for b in members[i + 1:]))
            pattern = ("present in all 3" if len(kinds) == 3 else
                       " + ".join(k.title() for k in ("people", "minority", "majority") if k in kinds) +
                       (" only" if len(kinds) == 1 else "")) if complete and not uncertain_ids.intersection(members) else "uncertain"
            rows.append({"demographic_dimension": dimension, "domain": domain, "outcome": outcome, "replica_id": replica,
                         "claim_ids": json.dumps(sorted(members)), "provisional_pattern": pattern,
                         "status": "provisional" if pattern != "uncertain" else "ambiguous_or_uncertain"})
    return pd.DataFrame(rows)


def _matching_triplets(patterns: pd.DataFrame, claims: pd.DataFrame, matching: pd.DataFrame) -> pd.DataFrame:
    """Trace provisional components to claim text and all three pairwise similarities."""
    columns = ["demographic_dimension", "domain", "outcome", "replica_id", "provisional_pattern", "status",
               "people_claim", "minority_claim", "majority_claim", "people_claim_id", "minority_claim_id",
               "majority_claim_id", "similarity_people_minority", "similarity_people_majority",
               "similarity_minority_majority"]
    if patterns.empty:
        return pd.DataFrame(columns=columns)
    lookup = claims.drop_duplicates("claim_id").set_index("claim_id")
    similarities = {}
    for row in matching.itertuples():
        similarities[frozenset((row.claim_a_id, row.claim_b_id))] = row.similarity
    output = []
    for row in patterns.itertuples():
        ids = json.loads(row.claim_ids)
        by_type = {}
        for claim_id in ids:
            if claim_id in lookup.index:
                kind = lookup.loc[claim_id, "group_type"]
                by_type.setdefault(kind, []).append(claim_id)
        record = {"demographic_dimension": row.demographic_dimension, "domain": row.domain,
                  "outcome": row.outcome, "replica_id": row.replica_id,
                  "provisional_pattern": row.provisional_pattern, "status": row.status}
        for kind in ("people", "minority", "majority"):
            values = by_type.get(kind, [])
            record[f"{kind}_claim_id"] = json.dumps(values)
            record[f"{kind}_claim"] = json.dumps([lookup.loc[item, "claim_text_with_context_markers"]
                                                  for item in values], ensure_ascii=False)
        for left, right in (("people", "minority"), ("people", "majority"), ("minority", "majority")):
            values = [similarities.get(frozenset((a, b))) for a in by_type.get(left, []) for b in by_type.get(right, [])]
            record[f"similarity_{left}_{right}"] = max((v for v in values if v is not None), default=np.nan)
        output.append(record)
    return pd.DataFrame(output, columns=columns)


def _make_metrics(responses: pd.DataFrame, sentences: pd.DataFrame, selection: dict,
                  disambiguation: dict, claims: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    records = []
    for response in responses.to_dict("records"):
        sid = sentences.loc[sentences.response_id.eq(response["response_id"]), "sentence_id"].tolist()
        valid_selection = [item for item in sid if (selection.get(item) or {}).get("selection_status") in
                           {"HAS_VERIFIABLE_CLAIM", "NO_VERIFIABLE_CLAIM"}]
        selected = [item for item in sid if (selection.get(item) or {}).get("selection_status") == "HAS_VERIFIABLE_CLAIM"]
        unresolved = [item for item in selected if (disambiguation.get(item) or {}).get("ambiguity_status") == "UNRESOLVABLE_AMBIGUITY"]
        part = claims.loc[claims.response_id.eq(response["response_id"])] if not claims.empty else claims
        accepted = valid.loc[valid.response_id.eq(response["response_id"])] if not valid.empty else valid
        n_sent = len(sid)
        n_claim_sent = len(selected)
        words = len(re.findall(r"\b[\w'-]+\b", response["full_aio_text"]))
        records.append({**{k: response.get(k, "") for k in ("response_id", "query_id", "outcome", "domain",
                         "demographic_dimension", "group", "condition", "group_type", "location", "replica_id")},
                        "n_sentences": n_sent, "n_sentences_with_verifiable_claim": n_claim_sent,
                        "n_sentences_with_valid_selection": len(valid_selection),
                        "n_selection_invalid": n_sent - len(valid_selection),
                        "proportion_sentences_with_claim": n_claim_sent / len(valid_selection) if valid_selection else np.nan,
                        "n_extracted_claims": len(part), "n_valid_claims": len(accepted),
                        "claims_per_sentence": len(part) / n_sent if n_sent else np.nan,
                        "claims_per_100_words": len(part) / words * 100 if words else np.nan,
                        "mean_claims_per_claim_sentence": len(part) / n_claim_sent if n_claim_sent else np.nan,
                        "n_unresolvable_sentences": len(unresolved),
                        "n_failed_entailment": int(part.entailment.eq("NOT_ENTAILED").sum()) if len(part) else 0,
                        "n_failed_qualification_preservation": int(part.qualification_preservation.eq("FAIL").sum()) if len(part) else 0,
                        "n_hallucinated_context": int(part.context_status.eq("HALLUCINATED_CONTEXT").sum()) if len(part) else 0})
    return pd.DataFrame(records)


def _validation_sample(sentences: pd.DataFrame, selection: dict, ambiguity: dict, claims: pd.DataFrame,
                       n: int = 250) -> pd.DataFrame:
    sample = sentences.drop(columns=["full_aio_text"], errors="ignore").copy()
    sample["selection_model"] = sample.sentence_id.map(lambda i: (selection.get(i) or {}).get("selection_status", "INVALID"))
    sample["ambiguity_status"] = sample.sentence_id.map(lambda i: (ambiguity.get(i) or {}).get("ambiguity_status", "NOT_RUN"))
    by_sentence = claims.groupby("sentence_id").size().to_dict() if not claims.empty else {}
    sample["n_extracted_claims"] = sample.sentence_id.map(lambda i: by_sentence.get(i, 0))
    sample["claim_count_bucket"] = sample.n_extracted_claims.clip(upper=2)
    keys = ["group_type", "demographic_dimension", "selection_model", "ambiguity_status", "claim_count_bucket"]
    baseline = sample.groupby(["outcome", "group_type"], dropna=False, group_keys=False).sample(n=1, random_state=SEED) if len(sample) else sample
    strata = sample.groupby(keys, dropna=False, group_keys=False).sample(n=1, random_state=SEED) if len(sample) else sample
    chosen = pd.concat([baseline, strata]).drop_duplicates("sentence_id")
    if len(chosen) > n:
        optional = chosen.loc[~chosen.sentence_id.isin(baseline.sentence_id)]
        chosen = pd.concat([baseline, optional.sample(n=max(0, n - len(baseline)), random_state=SEED)])
    extra = sample.loc[~sample.sentence_id.isin(chosen.sentence_id)]
    if len(chosen) < n and len(extra):
        chosen = pd.concat([chosen, extra.sample(n=min(n - len(chosen), len(extra)), random_state=SEED)])
    if not chosen.empty:
        grouped = claims.groupby("sentence_id").apply(lambda x: json.dumps(x[["claim_text", "entailment", "qualification_preservation"]].to_dict("records"), ensure_ascii=False), include_groups=False).to_dict() if not claims.empty else {}
        chosen["context"] = chosen.apply(lambda r: json.dumps({"preceding": json.loads(r.preceding_sentences),
            "following": json.loads(r.following_sentences)}, ensure_ascii=False), axis=1)
        chosen["extracted_claims"] = chosen.sentence_id.map(lambda i: grouped.get(i, "[]"))
        chosen["entailment_model"] = chosen.extracted_claims
        chosen["qualifier_preservation_model"] = chosen.extracted_claims
    for field in ("human_selection", "human_claim_quality", "human_entailment", "human_qualifier_preservation", "notes"):
        chosen[field] = ""
    return chosen


def run(args) -> None:
    # Fail before writes and before GPU allocation if either checkpoint/dependency is missing.
    model_path, embedding_path = preflight(
        args.model_path, args.embedding_model_path, args.tensor_parallel_size, args.embedding_model_id)
    model_id = model_identity(model_path, args.model_id)
    fingerprint = model_fingerprint(model_path)
    output = args.output_dir or DEFAULT_OUTPUT_ROOT / model_slug(model_id)
    configuration = {"model_id": model_id, "model_path": str(model_path), "model_fingerprint": fingerprint,
                     "embedding_model_id": args.embedding_model_id, "embedding_path": str(embedding_path),
                     "embedding_fingerprint": model_fingerprint(embedding_path),
                     "prompt_version": PROMPT_VERSION}
    configuration_path = output / "logs" / "run_configuration.json"
    if configuration_path.exists():
        previous = json.loads(configuration_path.read_text(encoding="utf-8"))
        if previous != configuration:
            raise RuntimeError(f"Output directory belongs to a different model/embedding/prompt configuration: {output}. "
                               "Choose a new --output-dir; existing checkpoints will not be mixed.")
    responses = load_dallas(args.input_dir)
    sentences = segment_responses(responses)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    configuration_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    save_csv(responses, output / "response_inventory.csv")
    save_csv(sentences.drop(columns=["full_aio_text"], errors="ignore"), output / "sentences.csv")
    engine = LocalModel(model_path, model_id, batch_size=args.batch_size,
                        max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        trust_remote_code=args.trust_remote_code)
    sent = sentence_inputs(sentences)
    selection = engine.run_stage("selection", sent, output / "selection_outputs.jsonl",
                                 lambda v, r: validate_selection(v, r["original_sentence"]))
    selected = [{**r, "verifiable_spans": selection[r["record_id"]]["verifiable_spans"]} for r in sent
                if selection[r["record_id"]] and selection[r["record_id"]]["selection_status"] == "HAS_VERIFIABLE_CLAIM"]
    ambiguity = engine.run_stage("disambiguation", selected, output / "disambiguation_outputs.jsonl",
                                  lambda v, r: validate_disambiguation(v, r["original_sentence"]))
    decomposable = [{**r, "resolutions": ambiguity[r["record_id"]].get("resolutions", [])} for r in selected
                    if ambiguity[r["record_id"]] and ambiguity[r["record_id"]]["ambiguity_status"] != "UNRESOLVABLE_AMBIGUITY"]
    decomposition = engine.run_stage("decomposition", decomposable, output / "logs" / "decomposition_outputs.jsonl",
                                      lambda v, r: validate_decomposition(v, len(r["verifiable_spans"]), bool(r["resolutions"])))
    claims = []
    for row in decomposable:
        result = decomposition[row["record_id"]]
        if not result:
            continue
        for number, claim in enumerate(result["claims"]):
            spans = [row["verifiable_spans"][i] for i in claim["source_span_indices"]]
            claim_id = "c_" + stable_id(row["response_id"], row["sentence_id"], number, claim["claim_text"])
            claims.append({**row, **claim, "claim_id": claim_id, "record_id": claim_id,
                           "verifiable_span": json.dumps(spans, ensure_ascii=False),
                           "span_start_char": min(row["sentence_start_char"] + s["start_char"] for s in spans),
                           "span_end_char": max(row["sentence_start_char"] + s["end_char"] for s in spans),
                           "selection_status": "HAS_VERIFIABLE_CLAIM",
                           "selection_confidence": selection[row["sentence_id"]].get("confidence"),
                           "ambiguity_status": ambiguity[row["sentence_id"]]["ambiguity_status"],
                           "ambiguity_type": json.dumps([s.get("ambiguity_type") for s in row["resolutions"]]),
                           "model_name": model_id, "model_version": fingerprint, "generation_seed": SEED})
    for stage, field in (("entailment", "entailment"), ("qualification", "qualification_preservation"),
                         ("context_audit", "context_status")):
        results = engine.run_stage(stage, claims, output / f"{stage}_validation.jsonl",
                                   lambda v, r, s=stage: validate_verdict(v, s))
        for claim in claims:
            result = results[claim["claim_id"]] or {}
            claim[field] = result.get(field, "INVALID")
            if stage == "qualification":
                claim["missing_qualifiers"] = result.get("missing_qualifiers", [])
    claim_df = pd.DataFrame(claims)
    save_csv(claim_df, output / "extracted_claims.csv")
    save_csv(claim_df[["claim_id", "response_id", "sentence_id", "entailment"]] if len(claim_df) else [],
             output / "entailment_validation.csv", ["claim_id", "response_id", "sentence_id", "entailment"])
    save_csv(claim_df[["claim_id", "response_id", "sentence_id", "qualification_preservation", "missing_qualifiers", "context_status"]]
             if len(claim_df) else [], output / "qualification_validation.csv",
             ["claim_id", "response_id", "sentence_id", "qualification_preservation", "missing_qualifiers", "context_status"])
    accepted = claim_df.loc[claim_df.entailment.eq("ENTAILED") & claim_df.qualification_preservation.eq("PASS") &
                            claim_df.context_status.eq("GROUNDED")].copy() if len(claim_df) else claim_df
    from sentence_transformers import SentenceTransformer
    # A resolved filesystem path prevents an implicit hub lookup, across
    # sentence-transformers versions that differ in local_files_only support.
    embedder = SentenceTransformer(str(embedding_path))
    accepted = _deduplicate(accepted, embedder, engine, output)
    save_csv(accepted, output / "valid_claims.csv")
    metrics = _make_metrics(responses, sentences, selection, ambiguity, claim_df, accepted)
    save_csv(metrics, output / "response_metrics.csv")
    save_csv(_validation_sample(sentences, selection, ambiguity, claim_df), output / "human_validation_sample.csv")
    if len(accepted):
        # Shared People responses are deliberately repeated only in this comparison view.
        expanded = []
        for dimension in sorted(metrics.demographic_dimension.dropna().unique()):
            non_people = accepted.loc[(accepted.demographic_dimension == dimension) & accepted.group_type.ne("people")]
            for domain, outcome in non_people[["domain", "outcome"]].drop_duplicates().itertuples(index=False, name=None):
                people = accepted.loc[accepted.group_type.eq("people") & accepted.domain.eq(domain) & accepted.outcome.eq(outcome)]
                expanded.extend(people.assign(demographic_dimension=dimension).to_dict("records"))
        matching_claims = pd.concat([accepted.loc[accepted.group_type.ne("people")], pd.DataFrame(expanded)], ignore_index=True)
        vectors = np.asarray(embedder.encode(matching_claims.claim_text_with_context_markers.tolist(), normalize_embeddings=True))
        match_inputs = _candidate_pairs(matching_claims, vectors, "cross_condition")
        index = {r["claim_id"]: r for r in matching_claims.to_dict("records")}
        matched = _judge_pairs(engine, match_inputs, output / "logs" / "matching_judgments.jsonl", index)
    else:
        matched = pd.DataFrame(columns=["claim_a_id", "claim_b_id", "similarity", "match_status", "pair_type"])
        matching_claims = accepted
    save_csv(matched, output / "claim_matching_candidates.csv")
    patterns = _presence_patterns(matching_claims, matched)
    save_csv(patterns, output / "claim_presence_patterns.csv")
    save_csv(_matching_triplets(patterns, matching_claims, matched), output / "claim_matching_triplets.csv")
    from plot_claim_extraction_qwen14b import plot_all
    plot_all(output)
    manifest = {"pipeline": "qualification-preserving Claimify-inspired pilot", "claimify_reproduction": False,
                "truth_verification": False, **configuration,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len, "trust_remote_code": args.trust_remote_code,
                "seed": SEED, "prompt_version": PROMPT_VERSION, "python": platform.python_version(),
                "n_responses": len(responses), "n_sentences": len(sentences), "n_valid_claims": len(accepted),
                "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
                    if (model_path / "config.json").exists() else None,
                "input_file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(args.input_dir.glob("*/*.txt"))}}
    (output / "logs").mkdir(exist_ok=True)
    (output / "logs" / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Claim pilot completed: {output} ({len(responses)} responses, {len(accepted)} valid claims)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: results/claim_extraction_pilot/<model-slug>")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-id", type=str, help="Scientific model identifier; inferred from local config/path if omitted")
    parser.add_argument("--embedding-model-path", type=str)
    parser.add_argument("--embedding-model-id", type=str, default=EMBEDDING_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare sentences without model inference")
    parser.add_argument("--preflight", action="store_true", help="Check local dependencies/checkpoints without writes")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization < 1 or args.max_model_len < 1 or args.batch_size < 1:
        parser.error("GPU memory fraction must be in (0,1), and lengths/batch size must be positive")
    if args.preflight:
        model_path, embedding_path = preflight(
            args.model_path, args.embedding_model_path, args.tensor_parallel_size, args.embedding_model_id)
        print(json.dumps({"model_path": str(model_path), "model_id": model_identity(model_path, args.model_id),
                          "embedding_path": str(embedding_path), "tensor_parallel_size": args.tensor_parallel_size}, indent=2))
    elif args.prepare_only:
        responses = load_dallas(args.input_dir)
        sentences = segment_responses(responses)
        path = Path(args.model_path).expanduser().resolve()
        output = args.output_dir or DEFAULT_OUTPUT_ROOT / model_slug(model_identity(path, args.model_id))
        output.mkdir(parents=True, exist_ok=True)
        save_csv(responses, output / "response_inventory.csv")
        save_csv(sentences.drop(columns=["full_aio_text"], errors="ignore"), output / "sentences.csv")
        print(f"Prepared {len(responses)} responses and {len(sentences)} sentence-like units: {output}")
    else:
        run(args)


if __name__ == "__main__":
    main()
