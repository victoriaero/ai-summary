from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PLACEHOLDER = "{GROUP}"


class PipelineError(ValueError):
    """Raised when an input or artifact violates the pipeline contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise PipelineError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON in {path}: {exc}") from exc


def load_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise PipelineError(f"File not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise PipelineError(f"Invalid YAML in {path}: {exc}") from exc




def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PipelineError(
                        f"Invalid JSON in {path}, line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise PipelineError(
                        f"Expected an object in {path}, line {line_number}"
                    )
                rows.append(value)
    except FileNotFoundError as exc:
        raise PipelineError(f"File not found: {path}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{field} must be a non-empty string")
    return value.strip()


def validate_template(text: Any, field: str = "template") -> str:
    value = require_nonempty_string(text, field)
    count = value.count(PLACEHOLDER)
    if count != 1:
        raise PipelineError(
            f"{field} must contain {PLACEHOLDER} exactly once; found {count}"
        )
    return value


def validate_study_input(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise PipelineError("Study input must be a JSON object")

    study = config.get("study")
    if not isinstance(study, dict):
        raise PipelineError("study must be an object")
    require_nonempty_string(study.get("geographic_scope"), "study.geographic_scope")
    require_nonempty_string(study.get("query_intent"), "study.query_intent")
    if study.get("group_placeholder") != PLACEHOLDER:
        raise PipelineError(f"study.group_placeholder must be {PLACEHOLDER}")

    outcomes = config.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise PipelineError("outcomes must be a non-empty array")
    seen_outcomes: set[str] = set()
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            raise PipelineError(f"outcomes[{index}] must be an object")
        for field in ("outcome_id", "domain", "outcome", "definition"):
            require_nonempty_string(outcome.get(field), f"outcomes[{index}].{field}")
        outcome_id = outcome["outcome_id"]
        if outcome_id in seen_outcomes:
            raise PipelineError(f"Duplicate outcome_id: {outcome_id}")
        seen_outcomes.add(outcome_id)

    control = config.get("control")
    if not isinstance(control, dict):
        raise PipelineError("control must be an object")
    require_nonempty_string(control.get("condition_id"), "control.condition_id")
    if control.get("label") != "people":
        raise PipelineError('control.label must be "people"')

    demographics = config.get("demographics")
    if not isinstance(demographics, list) or not demographics:
        raise PipelineError("demographics must be a non-empty array")
    seen_dimensions: set[str] = set()
    seen_groups: set[str] = set()
    seen_conditions: set[tuple[str, str]] = set()
    for d_index, dimension in enumerate(demographics):
        if not isinstance(dimension, dict):
            raise PipelineError(f"demographics[{d_index}] must be an object")
        dimension_id = require_nonempty_string(
            dimension.get("dimension_id"), f"demographics[{d_index}].dimension_id"
        )
        require_nonempty_string(
            dimension.get("dimension"), f"demographics[{d_index}].dimension"
        )
        if dimension_id in seen_dimensions:
            raise PipelineError(f"Duplicate dimension_id: {dimension_id}")
        seen_dimensions.add(dimension_id)
        groups = dimension.get("groups")
        if not isinstance(groups, list) or not groups:
            raise PipelineError(f"demographics[{d_index}].groups must be non-empty")
        for g_index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise PipelineError(
                    f"demographics[{d_index}].groups[{g_index}] must be an object"
                )
            group_id = require_nonempty_string(
                group.get("group_id"),
                f"demographics[{d_index}].groups[{g_index}].group_id",
            )
            require_nonempty_string(
                group.get("group"),
                f"demographics[{d_index}].groups[{g_index}].group",
            )
            if group_id in seen_groups:
                raise PipelineError(f"group_id must be globally unique: {group_id}")
            seen_groups.add(group_id)
            variants = group.get("variants")
            if not isinstance(variants, list) or not variants:
                raise PipelineError(f"Group {group_id} must contain variants")
            for v_index, variant in enumerate(variants):
                if not isinstance(variant, dict):
                    raise PipelineError(
                        f"Variant {v_index} in group {group_id} must be an object"
                    )
                variant_id = require_nonempty_string(
                    variant.get("variant_id"), f"variant_id in group {group_id}"
                )
                label = require_nonempty_string(
                    variant.get("label"), f"label for {group_id}/{variant_id}"
                )
                if PLACEHOLDER in label:
                    raise PipelineError(
                        f"Demographic label cannot contain {PLACEHOLDER}: {label}"
                    )
                condition_key = (group_id, variant_id)
                if condition_key in seen_conditions:
                    raise PipelineError(
                        f"Duplicate group/variant pair: {group_id}/{variant_id}"
                    )
                seen_conditions.add(condition_key)

    return config


def load_and_validate_study(path: Path) -> dict[str, Any]:
    return validate_study_input(load_json(path))


def load_pipeline_config(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_yaml(path)
    if not isinstance(config, dict):
        raise PipelineError("Pipeline config must be a YAML object")

    paths_config = config.get("paths")
    if not isinstance(paths_config, dict):
        raise PipelineError("paths must be an object")
    resolved_paths: dict[str, Path] = {}
    for name in ("study_input", "system_prompt", "user_prompt", "runs_dir"):
        raw_path = require_nonempty_string(paths_config.get(name), f"paths.{name}")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        resolved_paths[name] = candidate.resolve()

    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict):
        raise PipelineError("pipeline must be an object")
    if pipeline.get("candidates_per_outcome") != 5:
        raise PipelineError("pipeline.candidates_per_outcome must be 5")
    require_nonempty_string(pipeline.get("prompt_version"), "pipeline.prompt_version")

    model = config.get("model")
    if not isinstance(model, dict):
        raise PipelineError("model must be an object")
    for field in ("revision", "tokenizer_id", "tokenizer_revision"):
        if model.get(field) is not None and not isinstance(model[field], str):
            raise PipelineError(f"model.{field} must be a string or null")
    for field in ("device", "dtype"):
        require_nonempty_string(model.get(field), f"model.{field}")
    for field in ("trust_remote_code", "tokenizer_use_fast"):
        if not isinstance(model.get(field), bool):
            raise PipelineError(f"model.{field} must be a boolean")

    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise PipelineError("generation must be an object")
    seed = generation.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PipelineError("generation.seed must be an integer")
    parameters = generation.get("parameters")
    if not isinstance(parameters, dict):
        raise PipelineError("generation.parameters must be an object")
    if not isinstance(parameters.get("do_sample"), bool):
        raise PipelineError("generation.parameters.do_sample must be a boolean")
    _require_positive_number(parameters.get("temperature"), "temperature")
    top_p = _require_positive_number(parameters.get("top_p"), "top_p")
    if top_p > 1:
        raise PipelineError("generation.parameters.top_p must be at most 1")
    _require_positive_number(
        parameters.get("repetition_penalty"), "repetition_penalty"
    )
    max_new_tokens = parameters.get("max_new_tokens")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise PipelineError("generation.parameters.max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise PipelineError("generation.parameters.max_new_tokens must be positive")

    return config, resolved_paths


def require_model_id(config: dict[str, Any]) -> str:
    model_id = config["model"].get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PipelineError(
            "model.id is not configured; set a Hugging Face model ID in config.yaml"
        )
    return model_id.strip()


def _require_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PipelineError(f"generation.parameters.{field} must be positive")
    return float(value)


def update_manifest(run_dir: Path, stage: str, status: str) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    stages = manifest.setdefault("stages", {})
    stages[stage] = {"status": status, "timestamp": utc_now()}
    write_json(manifest_path, manifest)

