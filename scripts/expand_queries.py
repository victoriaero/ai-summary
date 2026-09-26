from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.common import (
    PLACEHOLDER,
    PipelineError,
    load_and_validate_study,
    load_pipeline_config,
    read_jsonl,
    update_manifest,
    write_csv,
    write_jsonl,
)


def expand_queries(config_path: Path, run_dir: Path) -> Path:
    config = load_and_validate_study(config_path)
    templates = read_jsonl(run_dir / "frozen_templates.jsonl")
    if not templates:
        raise PipelineError("No frozen templates found")
    expected_outcomes = {item["outcome_id"] for item in config["outcomes"]}
    actual_outcomes = {item.get("outcome_id") for item in templates}
    if actual_outcomes != expected_outcomes or len(templates) != len(expected_outcomes):
        raise PipelineError("Frozen templates must contain exactly one row per outcome")

    output_path = run_dir / "queries.jsonl"
    if output_path.exists():
        raise PipelineError(f"Expanded queries already exist: {output_path}")

    rows: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for template in templates:
        final_template = template.get("final_template")
        if not isinstance(final_template, str) or final_template.count(PLACEHOLDER) != 1:
            raise PipelineError(f"Invalid frozen template: {template.get('template_id')}")
        template_id = template["template_id"]
        provenance = dict(template)

        for dimension in config["demographics"]:
            for group in dimension["groups"]:
                for variant in group["variants"]:
                    query_id = (
                        f"{template_id}__{group['group_id']}__{variant['variant_id']}"
                    )
                    row = {
                        **provenance,
                        "query_id": query_id,
                        "condition_type": "experimental",
                        "social_dimension": dimension["dimension"],
                        "social_dimension_id": dimension["dimension_id"],
                        "group_id": group["group_id"],
                        "group": group["group"],
                        "variant_id": variant["variant_id"],
                        "group_condition": variant["label"],
                        "query": final_template.replace(PLACEHOLDER, variant["label"]),
                    }
                    _add_query(rows, query_ids, row)

        control = config["control"]
        control_id = f"{template_id}__{control['condition_id']}"
        control_row = {
            **provenance,
            "query_id": control_id,
            "condition_type": "control",
            "social_dimension": None,
            "social_dimension_id": None,
            "group_id": None,
            "group": None,
            "variant_id": None,
            "group_condition": control["label"],
            "query": final_template.replace(PLACEHOLDER, control["label"]),
        }
        _add_query(rows, query_ids, control_row)

    write_jsonl(output_path, rows)
    write_csv(run_dir / "queries.csv", rows)
    update_manifest(run_dir, "expansion", "complete")
    return output_path


def expand_from_config(config_path: Path, run_id: str) -> Path:
    _, paths = load_pipeline_config(config_path)
    return expand_queries(paths["study_input"], paths["runs_dir"] / run_id)


def _add_query(
    rows: list[dict[str, Any]], query_ids: set[str], row: dict[str, Any]
) -> None:
    query_id = row["query_id"]
    if query_id in query_ids:
        raise PipelineError(f"Duplicate query_id: {query_id}")
    if PLACEHOLDER in row["query"]:
        raise PipelineError(f"Unresolved placeholder in {query_id}")
    query_ids.add(query_id)
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand frozen templates into queries")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output = expand_from_config(args.config, args.run_id)
    print(f"Expanded queries written to {output}")


if __name__ == "__main__":
    main()

