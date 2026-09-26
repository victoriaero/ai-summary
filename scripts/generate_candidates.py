from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Protocol

from scripts.common import (
    PipelineError,
    append_jsonl,
    load_and_validate_study,
    load_pipeline_config,
    read_jsonl,
    require_model_id,
    update_manifest,
    utc_now,
    validate_template,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.prompts import build_messages, load_prompt_templates


class TextGenerator(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def generate(
        self, messages: list[dict[str, str]], parameters: dict[str, Any], seed: int
    ) -> str: ...


class HuggingFaceGenerator:
    """Small adapter around a Transformers causal instruction model."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_use_fast: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise PipelineError(
                "Generation requires torch and transformers; install requirements.txt"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.requested_revision = revision
        self.tokenizer_id = tokenizer_id or model_id
        self.requested_tokenizer_revision = tokenizer_revision or revision
        self.trust_remote_code = trust_remote_code
        self.tokenizer_use_fast = tokenizer_use_fast
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.requested_tokenizer_revision,
            trust_remote_code=trust_remote_code,
            use_fast=tokenizer_use_fast,
        )
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if dtype != "auto":
            if not hasattr(torch, dtype):
                raise PipelineError(f"Unknown torch dtype: {dtype}")
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.model.to(resolved_device)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_id": self.tokenizer_id,
            "requested_tokenizer_revision": self.requested_tokenizer_revision,
            "trust_remote_code": self.trust_remote_code,
            "tokenizer_use_fast": self.tokenizer_use_fast,
            "resolved_tokenizer_revision": getattr(
                self.tokenizer, "_commit_hash", None
            ),
        }

    def generate(
        self, messages: list[dict[str, str]], parameters: dict[str, Any], seed: int
    ) -> str:
        from transformers import set_seed

        set_seed(seed)
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, ValueError) as exc:
            raise PipelineError(
                "The selected tokenizer must provide a chat template"
            ) from exc
        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, **parameters)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_candidates(raw_response: str, outcome_id: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Model returned invalid JSON for {outcome_id}: {exc}") from exc
    if not isinstance(parsed, list) or len(parsed) != 5:
        count = len(parsed) if isinstance(parsed, list) else "non-array"
        raise PipelineError(
            f"Expected 5 candidates for {outcome_id}; received {count}"
        )

    expected_ids = [f"{outcome_id}-Q{number:02d}" for number in range(1, 6)]
    candidate_ids: list[str] = []
    validated: list[dict[str, str]] = []
    for index, candidate in enumerate(parsed):
        if not isinstance(candidate, dict) or set(candidate) != {"candidate_id", "query"}:
            raise PipelineError(
                f"Candidate {index + 1} for {outcome_id} must contain only "
                "candidate_id and query"
            )
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise PipelineError(f"Candidate {index + 1} has an invalid candidate_id")
        query = validate_template(candidate.get("query"), f"query for {candidate_id}")
        candidate_ids.append(candidate_id)
        validated.append({"candidate_id": candidate_id, "query": query})

    if len(set(candidate_ids)) != 5:
        raise PipelineError(f"Duplicate candidate IDs for {outcome_id}")
    if candidate_ids != expected_ids:
        raise PipelineError(
            f"Candidate IDs for {outcome_id} must be {expected_ids}; got {candidate_ids}"
        )
    return validated


def generate_run(
    config_path: Path,
    run_id: str,
    generator: TextGenerator,
    runs_dir: Path | None = None,
) -> Path:
    pipeline_config, paths = load_pipeline_config(config_path)
    study_input = load_and_validate_study(paths["study_input"])
    system_prompt, user_template = load_prompt_templates(
        paths["system_prompt"], paths["user_prompt"]
    )
    pipeline = pipeline_config["pipeline"]
    prompt_version = pipeline["prompt_version"]
    runs_dir = runs_dir or paths["runs_dir"]

    if not run_id or Path(run_id).name != run_id:
        raise PipelineError("run_id must be a non-empty directory name")
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise PipelineError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    generation = pipeline_config["generation"]
    metadata = generator.metadata
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "prompt_version": prompt_version,
        "model": metadata,
        "generation": generation,
        "config_file": str(config_path.resolve()),
        "config_snapshot": pipeline_config,
        "study_input_file": str(paths["study_input"]),
        "study_input_snapshot": study_input,
        "prompt_files": {
            "system": str(paths["system_prompt"]),
            "user": str(paths["user_prompt"]),
        },
        "packages": {
            name: _package_version(name) for name in ("python", "torch", "transformers")
        },
        "stages": {"generation": {"status": "running", "timestamp": utc_now()}},
    }
    write_json(run_dir / "manifest.json", manifest)

    all_candidates: list[dict[str, Any]] = []
    try:
        for outcome in study_input["outcomes"]:
            messages = build_messages(
                outcome,
                study_input["study"],
                system_prompt,
                user_template,
                pipeline["candidates_per_outcome"],
            )
            raw = generator.generate(
                messages, generation["parameters"], generation["seed"]
            )
            raw_record: dict[str, Any] = {
                "outcome_id": outcome["outcome_id"],
                "prompt_version": prompt_version,
                "system_prompt": messages[0]["content"],
                "user_prompt": messages[1]["content"],
                "raw_response": raw,
                "generated_at": utc_now(),
                "status": "received",
            }
            append_jsonl(run_dir / "raw_generations.jsonl", raw_record)
            try:
                candidates = parse_candidates(raw, outcome["outcome_id"])
            except PipelineError as exc:
                raw_record["status"] = "invalid"
                raw_record["error"] = str(exc)
                raw_rows = read_jsonl(run_dir / "raw_generations.jsonl")
                raw_rows[-1] = raw_record
                write_jsonl(run_dir / "raw_generations.jsonl", raw_rows)
                raise

            for candidate in candidates:
                all_candidates.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "domain": outcome["domain"],
                        "outcome_id": outcome["outcome_id"],
                        "outcome": outcome["outcome"],
                        "definition": outcome["definition"],
                        "geographic_scope": study_input["study"]["geographic_scope"],
                        "query_intent": study_input["study"]["query_intent"],
                        "generation_model": metadata.get("model_id"),
                        "generation_revision": metadata.get("resolved_revision")
                        or metadata.get("requested_revision"),
                        "generation_tokenizer": metadata.get("tokenizer_id"),
                        "generation_prompt_version": prompt_version,
                        "system_prompt": messages[0]["content"],
                        "user_prompt": messages[1]["content"],
                        "generation_parameters": generation,
                        "raw_response": raw,
                        "raw_query": candidate["query"],
                        "validation_status": "structural_passed",
                    }
                )

        write_jsonl(run_dir / "candidates.jsonl", all_candidates)
        write_csv(run_dir / "candidates.csv", all_candidates)
        selection = [
            {
                "outcome_id": outcome["outcome_id"],
                "selected_candidate_id": None,
                "final_template": None,
                "notes": "",
            }
            for outcome in study_input["outcomes"]
        ]
        write_json(run_dir / "selection.json", selection)
        update_manifest(run_dir, "generation", "complete")
    except Exception:
        update_manifest(run_dir, "generation", "failed")
        raise
    return run_dir

def _package_version(name: str) -> str | None:
    if name == "python":
        import platform

        return platform.python_version()
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate query-template candidates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config, _ = load_pipeline_config(args.config)
    model = config["model"]
    generator = HuggingFaceGenerator(
        model_id=require_model_id(config),
        revision=model["revision"],
        tokenizer_id=model["tokenizer_id"],
        tokenizer_revision=model["tokenizer_revision"],
        device=model["device"],
        dtype=model["dtype"],
        trust_remote_code=model["trust_remote_code"],
        tokenizer_use_fast=model["tokenizer_use_fast"],
    )
    run_dir = generate_run(args.config, args.run_id, generator)
    print(f"Candidates written to {run_dir}")


if __name__ == "__main__":
    main()

