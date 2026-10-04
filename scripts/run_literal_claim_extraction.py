"""Re-extract faithful, literal AIO claims with local Phi-4 (or another vLLM model).

Run once per location. Outputs are isolated from the earlier rewriting pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from claim_extraction_qwen14b_core import (
    DEFAULT_MODEL_PATH, EMBEDDING_ID, SEED, load_dallas, model_fingerprint,
    model_identity, model_slug, preflight, segment_responses, stable_id,
)
from claim_extraction_qwen14b_inference import LocalModel
from literal_claims_core import PROMPT_VERSION, SCHEMAS, SYSTEM, literal_text, prompt, validate_stage
from run_claim_extraction_qwen14b import _candidate_pairs, save_csv as _save_csv, sentence_inputs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING = Path("/scratch/victoria.estanislau/models/all-mpnet-base-v2")
LOCATIONS = {"v1_dallas": "Dallas", "v2_ny": "New York"}


def save_csv(rows, path: Path, columns=None) -> None:
    # The legacy writer serializes object columns in place. Protect the in-memory
    # literal lists, which are still needed for matching and audit sampling.
    _save_csv(rows.copy(deep=True) if isinstance(rows, pd.DataFrame) else rows, path, columns)


def input_hashes(input_dir: Path) -> dict[str, str]:
    return {str(path.relative_to(input_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(input_dir.glob("*/*.txt"))}


def prepare(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    responses = load_dallas(input_dir)
    replica = input_dir.resolve().parents[0].name
    if replica not in LOCATIONS or set(responses.replica_id) != {replica}:
        raise ValueError(f"Expected one Dallas/NY replica in {input_dir}")
    if len(responses) != 273 or responses.query_id.duplicated().any():
        raise ValueError(f"Incomplete or duplicate AIO responses in {replica}: {len(responses)}")
    if responses.loc[responses.group_type.eq("people")].shape[0] != 21:
        raise ValueError("Expected exactly 21 unique People controls")
    return responses, segment_responses(responses)


def deduplicate_literal_candidates(candidates: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    """Collapse repeated model proposals with identical literal source spans.

    The claim ID is derived only from response, sentence, and absolute span
    offsets. Repeated proposals therefore represent the same measured unit,
    not independent claims. The first proposal in deterministic model-output
    order is retained and every proposal remains visible in the audit table.
    """
    groups: dict[str, list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(candidate["claim_id"], []).append(candidate)
    unique, audit = [], []
    for claim_id, proposals in groups.items():
        first = proposals[0].copy()
        identity = (first["response_id"], first["sentence_id"], first["absolute_source_spans"])
        for proposal in proposals:
            other = (proposal["response_id"], proposal["sentence_id"], proposal["absolute_source_spans"])
            if other != identity:
                raise ValueError(f"Literal claim ID collision with different spans: {claim_id}")
        notes = list(dict.fromkeys(str(item.get("resolution_note", "")) for item in proposals))
        first["n_model_proposals_same_spans"] = len(proposals)
        first["all_resolution_notes"] = notes
        unique.append(first)
        for index, proposal in enumerate(proposals, start=1):
            audit.append({"claim_id": claim_id, "response_id": proposal["response_id"],
                          "sentence_id": proposal["sentence_id"], "proposal_order": index,
                          "duplicate_group_size": len(proposals), "retained": index == 1,
                          "source_span_texts": proposal["source_span_texts"],
                          "absolute_source_spans": proposal["absolute_source_spans"],
                          "resolution_note": proposal.get("resolution_note", ""),
                          "deduplication_rule": "same response + sentence + exact absolute source spans"})
    return unique, pd.DataFrame(audit)


def run(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    replica = input_dir.parents[0].name
    if replica not in LOCATIONS:
        raise ValueError(f"Unsupported collection: {input_dir}")
    output = args.output_dir or ROOT / "results/claim_extraction_literal_v2" / model_slug(
        args.model_id or model_identity(args.model_path.resolve())) / replica
    output = output.resolve()
    if output.is_relative_to(ROOT / "results/claim_extraction_pilot"):
        raise ValueError("Literal extraction must not write into the old claim-extraction pilot")
    if args.preflight:
        model_path, embedding_path = preflight(str(args.model_path), str(args.embedding_model_path),
                                               args.tensor_parallel_size, args.embedding_model_id)
        print(json.dumps({"model_path": str(model_path), "embedding_path": str(embedding_path),
                          "input_dir": str(input_dir), "output_dir": str(output),
                          "prompt_version": PROMPT_VERSION}, indent=2))
        return
    responses, sentences = prepare(input_dir)
    if args.prepare_only:
        print(f"{replica}: {len(responses)} responses, {len(sentences)} sentence-like units; no files written")
        return
    model_path, embedding_path = preflight(str(args.model_path), str(args.embedding_model_path),
                                           args.tensor_parallel_size, args.embedding_model_id)
    model_id = model_identity(model_path, args.model_id)
    config = {"prompt_version": PROMPT_VERSION, "model_id": model_id,
              "model_path": str(model_path), "model_fingerprint": model_fingerprint(model_path),
              "embedding_model_id": args.embedding_model_id, "embedding_path": str(embedding_path),
              "embedding_fingerprint": model_fingerprint(embedding_path), "input_hashes": input_hashes(input_dir),
              "replica_id": replica, "seed": SEED}
    config_path = output / "logs/run_configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Input, prompt or model changed; choose a new --output-dir to avoid mixed checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    save_csv(responses, output / "response_inventory.csv")
    save_csv(sentences.drop(columns=["full_aio_text"], errors="ignore"), output / "sentences.csv")
    engine = LocalModel(model_path, model_id, batch_size=args.batch_size,
                        tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        trust_remote_code=args.trust_remote_code, prompt_fn=prompt,
                        prompt_version=PROMPT_VERSION, system=SYSTEM, schemas=SCHEMAS)
    sent = sentence_inputs(sentences)
    selection = engine.run_stage("selection", sent, output / "logs/selection.jsonl",
                                 lambda v, r: validate_stage("selection", v, r))
    selected = [{**r, "verifiable_spans": selection[r["record_id"]]["verifiable_spans"]}
                for r in sent if selection[r["record_id"]] and
                selection[r["record_id"]]["selection_status"] == "HAS_VERIFIABLE_CLAIM"]
    ambiguity = engine.run_stage("disambiguation", selected, output / "logs/disambiguation.jsonl",
                                 lambda v, r: validate_stage("disambiguation", v, r))
    eligible = [{**r, "resolutions": ambiguity[r["record_id"]].get("resolutions", [])}
                for r in selected if ambiguity[r["record_id"]] and
                ambiguity[r["record_id"]]["ambiguity_status"] != "UNRESOLVABLE_AMBIGUITY"]
    decomp = engine.run_stage("literal_decomposition", eligible, output / "logs/literal_decomposition.jsonl",
                              lambda v, r: validate_stage("literal_decomposition", v, r))
    candidates = []
    for row in eligible:
        result = decomp[row["record_id"]]
        if not result:
            continue
        for claim in result["claims"]:
            spans = claim["source_spans"]
            absolute = [{**span, "start_char": row["sentence_start_char"] + span["start_char"],
                         "end_char": row["sentence_start_char"] + span["end_char"]} for span in spans]
            fragments = literal_text(row["original_sentence"], spans)
            claim_id = "lc_" + stable_id(row["response_id"], row["sentence_id"],
                                           [(s["start_char"], s["end_char"]) for s in absolute])
            candidates.append({**row, "claim_id": claim_id, "record_id": claim_id,
                               "source_spans": spans, "absolute_source_spans": absolute,
                               "source_span_texts": fragments, "resolution_note": claim.get("resolution_note", ""),
                               "prompt_version": PROMPT_VERSION, "model_id": model_id})
    candidates, candidate_audit = deduplicate_literal_candidates(candidates)
    save_csv(candidate_audit, output / "literal_claim_proposal_audit.csv")
    for stage in ("entailment", "qualification", "context_audit"):
        result = engine.run_stage(stage, candidates, output / f"logs/{stage}.jsonl",
                                  lambda v, r, s=stage: validate_stage(s, v, r))
        field = {"entailment": "entailment", "qualification": "qualification_preservation",
                 "context_audit": "context_status"}[stage]
        for claim in candidates:
            claim[field] = (result[claim["claim_id"]] or {}).get(field, "INVALID")
            if stage == "qualification":
                claim["missing_qualifiers"] = (result[claim["claim_id"]] or {}).get("missing_qualifiers", [])
    all_claims = pd.DataFrame(candidates)
    save_csv(all_claims, output / "extracted_claims.csv")
    valid = all_claims.loc[all_claims.entailment.eq("ENTAILED") &
                           all_claims.qualification_preservation.eq("PASS") &
                           all_claims.context_status.eq("GROUNDED")].copy() if len(all_claims) else all_claims
    if len(valid):
        valid["span_key"] = valid.absolute_source_spans.map(lambda x: json.dumps(x, sort_keys=True))
        valid = valid.drop_duplicates(["response_id", "span_key"]).drop(columns="span_key")
    save_csv(valid, output / "valid_claims.csv")
    # Matching is provisional. It never changes the literal-claim denominator.
    matching = []
    if len(valid):
        from sentence_transformers import SentenceTransformer
        people = valid.loc[valid.group_type.eq("people")]
        expanded = [valid.loc[valid.group_type.ne("people")]]
        for dimension in sorted(valid.loc[valid.group_type.ne("people"), "demographic_dimension"].unique()):
            expanded.append(people.assign(demographic_dimension=dimension))
        match_frame = pd.concat(expanded, ignore_index=True)
        match_frame["claim_text_with_context_markers"] = match_frame.source_span_texts.map(
            lambda x: json.dumps(x, ensure_ascii=False))
        embedder = SentenceTransformer(str(embedding_path))
        vectors = np.asarray(embedder.encode(match_frame.claim_text_with_context_markers.tolist(),
                                             normalize_embeddings=True))
        inputs = _candidate_pairs(match_frame, vectors, "cross_condition")
        claim_lookup = match_frame.drop_duplicates("claim_id").set_index("claim_id")
        for item in inputs:
            other = claim_lookup.loc[item["claim_b_id"]]
            item["original_query_b"] = other.original_query
            item["original_sentence_b"] = other.original_sentence
            item["preceding_sentences_b"] = other.preceding_sentences
            item["following_sentences_b"] = other.following_sentences
        verdicts = engine.run_stage("matching", inputs, output / "logs/matching.jsonl",
                                    lambda v, r: validate_stage("matching", v, r))
        for item in inputs:
            verdict = verdicts[item["record_id"]] or {}
            matching.append({**item, "match_status": verdict.get("match_status", "UNCERTAIN"),
                             "modality_difference": verdict.get("modality_difference"),
                             "model_reason": verdict.get("reason", ""), "human_label_1": "",
                             "human_label_2": "", "adjudicated_label": ""})
    save_csv(matching, output / "matched_claim_candidates.csv",
             ["claim_a_id", "claim_b_id", "match_status", "human_label_1", "human_label_2", "adjudicated_label"])
    sample = (valid.groupby(["group_type", "domain"], group_keys=False).sample(n=1, random_state=SEED)
              if len(valid) else valid.copy())
    remainder = valid.loc[~valid.claim_id.isin(sample.claim_id)] if len(valid) else valid
    if len(sample) < 100 and len(remainder):
        sample = pd.concat([sample, remainder.sample(n=min(100 - len(sample), len(remainder)), random_state=SEED)])
    for field in ("human_literal_fidelity_1", "human_literal_fidelity_2", "human_atomicity_1",
                  "human_atomicity_2", "adjudicated_fidelity", "notes"):
        sample[field] = ""
    save_csv(sample, output / "human_claim_quality_sample.csv")
    (output / "logs/run_manifest.json").write_text(json.dumps({**config, "python": platform.python_version(),
        "n_responses": len(responses), "n_sentences": len(sentences), "n_valid_claims": len(valid),
        "n_matching_candidates": len(matching), "human_validation": "pending_annotation"}, indent=2) + "\n")
    print(f"Literal claims written to {output}: {len(valid)} valid claims in {len(responses)} responses")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-id")
    parser.add_argument("--embedding-model-path", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--embedding-model-id", default=EMBEDDING_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check local dependencies and checkpoints without writes")
    args = parser.parse_args()
    if args.batch_size < 1 or args.tensor_parallel_size < 1 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Invalid batch, tensor parallelism or GPU memory fraction")
    run(args)


if __name__ == "__main__":
    main()
