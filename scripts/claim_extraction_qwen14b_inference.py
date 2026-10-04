"""Deterministic batched vLLM inference with audited JSONL checkpoints."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from claim_extraction_qwen14b_core import PROMPT_VERSION, SEED, model_fingerprint, stable_id, text_hash
from claim_extraction_qwen14b_prompts import SYSTEM, prompt


SCHEMAS = {
    "selection": {"type": "object", "required": ["selection_status", "verifiable_spans", "confidence"], "properties": {
        "selection_status": {"enum": ["HAS_VERIFIABLE_CLAIM", "NO_VERIFIABLE_CLAIM"]},
        "verifiable_spans": {"type": "array", "items": {"type": "object", "required": ["text", "start_char", "end_char"], "properties": {
            "text": {"type": "string"}, "start_char": {"type": "integer"}, "end_char": {"type": "integer"}}}},
        "confidence": {"enum": ["high", "medium", "low"]}}},
    "disambiguation": {"type": "object", "required": ["ambiguity_status", "resolutions"], "properties": {
        "ambiguity_status": {"enum": ["NO_AMBIGUITY", "RESOLVABLE_AMBIGUITY", "UNRESOLVABLE_AMBIGUITY"]},
        "resolutions": {"type": "array", "items": {"type": "object"}}}},
    "decomposition": {"type": "object", "required": ["claims"], "properties": {"claims": {"type": "array", "items": {"type": "object"}}}},
    "entailment": {"type": "object", "required": ["entailment"], "properties": {
        "entailment": {"enum": ["ENTAILED", "NOT_ENTAILED", "UNCERTAIN"]}}},
    "qualification": {"type": "object", "required": ["qualification_preservation", "missing_qualifiers"], "properties": {
        "qualification_preservation": {"enum": ["PASS", "FAIL", "UNCERTAIN"]},
        "missing_qualifiers": {"type": "array", "items": {"type": "string"}}}},
    "context_audit": {"type": "object", "required": ["context_status"], "properties": {
        "context_status": {"enum": ["GROUNDED", "HALLUCINATED_CONTEXT", "UNCERTAIN"]}}},
    "matching": {"type": "object", "required": ["match_status"], "properties": {
        "match_status": {"enum": ["EQUIVALENT", "RELATED_NOT_EQUIVALENT", "NOT_RELATED", "UNCERTAIN"]}}},
}


def parse_json(raw: str) -> dict:
    value = raw.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(value[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def checkpoint_records(path: Path) -> dict[str, dict]:
    found = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    if item.get("status") == "valid":
                        found[item["record_id"]] = item
                except (ValueError, KeyError):
                    continue
    return found


def recover_literal_selection(path: Path, inputs: list[dict], identity: str,
                              validator, prompt_fn=prompt, prompt_version=PROMPT_VERSION) -> dict[str, dict]:
    """Append auditable, strictly literal offset repairs from prior invalid tries."""
    previous = checkpoint_records(path)
    if not path.exists():
        return previous
    expected = {stable_id("selection", row["record_id"], prompt_version,
                          identity, prompt_fn("selection", row)): row for row in inputs}
    recovered = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
                key = item.get("record_id")
                if (item.get("stage") != "selection" or item.get("status") != "invalid" or
                        key not in expected or key in previous):
                    continue
                parsed = validator(parse_json(item["raw_output"]), expected[key])
                if not any(span.get("offset_alignment") == "deterministic_unique_literal_match"
                           for span in parsed.get("verifiable_spans", [])):
                    continue
                repaired = {**item, "status": "valid", "parsed": parsed,
                            "recovery_method": "deterministic_unique_literal_match",
                            "recovered_from_attempt": item.get("attempt")}
                repaired.pop("error", None)
                previous[key] = repaired
                recovered.append(repaired)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
    if recovered:
        with path.open("a", encoding="utf-8") as stream:
            for item in recovered:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return previous


class LocalModel:
    def __init__(self, model_path: Path, model_id: str, batch_size: int = 32,
                 max_model_len: int = 8192, tensor_parallel_size: int = 2,
                 gpu_memory_utilization: float = 0.65, trust_remote_code: bool = False,
                 prompt_fn=prompt, prompt_version=PROMPT_VERSION, system=SYSTEM,
                 schemas=None):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.path = model_path
        self.model_id = model_id
        self.checkpoint_identity = f"{model_id}:{model_path}:{model_fingerprint(model_path)}"
        self.batch_size = batch_size
        self.SamplingParams = SamplingParams
        self.prompt_fn = prompt_fn
        self.prompt_version = prompt_version
        self.system = system
        self.schemas = schemas or SCHEMAS
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=trust_remote_code)
        self.llm = LLM(model=str(model_path), trust_remote_code=trust_remote_code,
                       max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization, seed=SEED)

    def _sampling(self, stage: str):
        kwargs = {"temperature": 0.0, "max_tokens": 1200 if stage in {"decomposition", "literal_decomposition"} else 420, "seed": SEED}
        signature = inspect.signature(self.SamplingParams)
        if "structured_outputs" in signature.parameters:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                kwargs["structured_outputs"] = StructuredOutputsParams(json=self.schemas[stage])
            except (ImportError, TypeError):
                pass
        elif "guided_decoding" in signature.parameters:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(json=self.schemas[stage])
            except (ImportError, TypeError):
                pass
        return self.SamplingParams(**kwargs)

    def _generate(self, stage: str, texts: list[str]) -> list[str]:
        rendered = [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system}, {"role": "user", "content": item}],
            tokenize=False, add_generation_prompt=True) for item in texts]
        results = self.llm.generate(rendered, self._sampling(stage), use_tqdm=False)
        return [item.outputs[0].text for item in results]

    def run_stage(self, stage: str, inputs: list[dict], output_path: Path, validator) -> dict[str, dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        identity = getattr(self, "checkpoint_identity", str(self.path))
        prompt_fn = getattr(self, "prompt_fn", prompt)
        prompt_version = getattr(self, "prompt_version", PROMPT_VERSION)
        system = getattr(self, "system", SYSTEM)
        previous = (recover_literal_selection(output_path, inputs, identity, validator,
                                              prompt_fn, prompt_version)
                    if stage == "selection" else checkpoint_records(output_path))
        pending = []
        for row in inputs:
            prompt_text = prompt_fn(stage, row)
            key = stable_id(stage, row["record_id"], prompt_version, identity, prompt_text)
            if key not in previous:
                pending.append((key, row, prompt_text))
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            remaining = batch
            for attempt in range(3):
                if not remaining:
                    break
                outputs = self._generate(stage, [item[2] for item in remaining])
                retry = []
                with output_path.open("a", encoding="utf-8") as stream:
                    for (key, row, prompt_text), raw in zip(remaining, outputs):
                        result = {"record_id": key, "source_record_id": row["record_id"], "stage": stage,
                                  "prompt_hash": text_hash(system + prompt_text), "prompt_version": prompt_version,
                                  "model_path": str(self.path), "model_id": getattr(self, "model_id", str(self.path)),
                                  "model_fingerprint": identity, "seed": SEED, "attempt": attempt + 1,
                                  "raw_output": raw}
                        try:
                            parsed = validator(parse_json(raw), row)
                            result.update({"status": "valid", "parsed": parsed})
                            previous[key] = result
                        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                            result.update({"status": "invalid", "error": str(exc)})
                            if attempt < 2:
                                correction = ("\n\nFORMAT CORRECTION: The preceding output was invalid (" + str(exc)[:180] +
                                              "). Return only a JSON object satisfying the requested fields and "
                                              "exact offsets; do not change the original text.")
                                retry.append((key, row, prompt_text + correction))
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                remaining = retry
        return {row["record_id"]: previous.get(stable_id(
            stage, row["record_id"], prompt_version, identity,
            prompt_fn(stage, row)), {}).get("parsed")
            for row in inputs}


# Compatibility for earlier imports; the implementation itself is model-agnostic.
LocalQwen = LocalModel
