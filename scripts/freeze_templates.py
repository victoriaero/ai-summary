from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.common import (
    PipelineError,
    load_json,
    load_pipeline_config,
    read_jsonl,
    update_manifest,
    utc_now,
    validate_template,
    write_csv,
    write_jsonl,
)


def freeze_templates(run_dir: Path, selection_path: Path | None = None) -> Path:
    candidates = read_jsonl(run_dir / "candidates.jsonl")
    if not candidates:
        raise PipelineError("No candidates found")
    selection_path = selection_path or run_dir / "selection.json"
    selections = load_json(selection_path)
    if not isinstance(selections, list):
        raise PipelineError("selection.json must contain an array")

    by_candidate: dict[str, dict[str, Any]] = {}
    expected_outcomes: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in by_candidate:
            raise PipelineError(f"Invalid or duplicate candidate_id: {candidate_id}")
        by_candidate[candidate_id] = candidate
        expected_outcomes.add(candidate["outcome_id"])

    by_outcome: dict[str, dict[str, Any]] = {}
    for index, selection in enumerate(selections):
        if not isinstance(selection, dict):
            raise PipelineError(f"Selection {index + 1} must be an object")
        outcome_id = selection.get("outcome_id")
        if not isinstance(outcome_id, str) or not outcome_id:
            raise PipelineError(f"Selection {index + 1} has an invalid outcome_id")
        if outcome_id in by_outcome:
            raise PipelineError(f"More than one selection for {outcome_id}")
        by_outcome[outcome_id] = selection

    selected_outcomes = set(by_outcome)
    if selected_outcomes != expected_outcomes:
        missing = sorted(expected_outcomes - selected_outcomes)
        extra = sorted(selected_outcomes - expected_outcomes)
        raise PipelineError(f"Selection outcomes mismatch; missing={missing}, extra={extra}")

    frozen: list[dict[str, Any]] = []
    for outcome_id in sorted(expected_outcomes):
        selection = by_outcome[outcome_id]
        selected_id = selection.get("selected_candidate_id")
        if not isinstance(selected_id, str) or not selected_id:
            raise PipelineError(f"Select exactly one candidate for {outcome_id}")
        candidate = by_candidate.get(selected_id)
        if candidate is None:
            raise PipelineError(f"Unknown selected_candidate_id: {selected_id}")
        if candidate["outcome_id"] != outcome_id:
            raise PipelineError(f"Candidate {selected_id} does not belong to {outcome_id}")

        edited = selection.get("final_template")
        final_template = candidate["raw_query"] if edited is None else edited
        final_template = validate_template(
            final_template, f"final_template for {outcome_id}"
        )
        notes = selection.get("notes", "")
        if not isinstance(notes, str):
            raise PipelineError(f"notes for {outcome_id} must be a string")

        frozen.append(
            {
                **candidate,
                "template_id": selected_id,
                "selected_candidate_id": selected_id,
                "selection_notes": notes,
                "final_template": final_template,
                "frozen_at": utc_now(),
            }
        )

    output = run_dir / "frozen_templates.jsonl"
    if output.exists():
        raise PipelineError(f"Frozen templates already exist: {output}")
    write_jsonl(output, frozen)
    write_csv(run_dir / "frozen_templates.csv", frozen)
    update_manifest(run_dir, "freeze", "complete")
    return output


def freeze_from_config(
    config_path: Path, run_id: str, selection_path: Path | None = None
) -> Path:
    _, paths = load_pipeline_config(config_path)
    return freeze_templates(paths["runs_dir"] / run_id, selection_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one selected template per outcome")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    output = freeze_from_config(args.config, args.run_id, args.selection)
    print(f"Frozen templates written to {output}")


if __name__ == "__main__":
    main()

