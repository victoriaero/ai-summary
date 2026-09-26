from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.common import PLACEHOLDER, PipelineError, load_json, read_jsonl
from scripts.expand_queries import expand_queries
from scripts.freeze_templates import freeze_templates
from scripts.generate_candidates import generate_run, parse_candidates


class FakeGenerator:
    def __init__(self, demographic_labels: list[str]) -> None:
        self.demographic_labels = demographic_labels
        self.seen_messages: list[list[dict[str, str]]] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": "fake/model",
            "requested_revision": "test",
            "resolved_revision": "fake-commit",
            "tokenizer_id": "fake/tokenizer",
        }

    def generate(
        self, messages: list[dict[str, str]], parameters: dict[str, Any], seed: int
    ) -> str:
        rendered = "\n".join(message["content"] for message in messages)
        for label in self.demographic_labels:
            assert label not in rendered
        assert messages[0]["content"].startswith("EXTERNAL SYSTEM PROMPT")
        assert parameters["temperature"] == 0.7
        self.seen_messages.append(messages)
        user_prompt = messages[1]["content"]
        outcome_id = user_prompt.split("Outcome ID: ", 1)[1].splitlines()[0]
        return json.dumps(
            [
                {
                    "candidate_id": f"{outcome_id}-Q{number:02d}",
                    "query": f"What factors ({number}) influence the outcome for {PLACEHOLDER} in the United States?",
                }
                for number in range(1, 6)
            ]
        )


def _study_input() -> dict[str, Any]:
    outcomes = [
        ("Housing", "HOU-02", "Housing affordability", "Housing costs relative to household resources."),
        ("Employment", "EMP-04", "Wages / pay", "Monetary compensation for work."),
        ("Healthcare", "HC-01", "Access to needed medical care", "Ability to obtain needed medical care."),
    ]
    return {
        "study": {
            "geographic_scope": "United States",
            "query_intent": "explanatory",
            "group_placeholder": PLACEHOLDER,
        },
        "control": {"condition_id": "control-people", "label": "people"},
        "outcomes": [
            {
                "domain": domain,
                "outcome_id": outcome_id,
                "outcome": outcome,
                "definition": definition,
            }
            for domain, outcome_id, outcome, definition in outcomes
        ],
        "demographics": [
            {
                "dimension_id": "race",
                "dimension": "Race",
                "groups": [
                    {
                        "group_id": "race-black",
                        "group": "Black",
                        "variants": [
                            {"variant_id": "black-people", "label": "Black people"},
                            {"variant_id": "black-americans", "label": "Black Americans"},
                        ],
                    }
                ],
            },
            {
                "dimension_id": "gender",
                "dimension": "Gender",
                "groups": [
                    {
                        "group_id": "gender-women",
                        "group": "Women",
                        "variants": [{"variant_id": "women", "label": "women"}],
                    }
                ],
            },
        ],
    }


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    study_path = tmp_path / "study_input.json"
    study_path.write_text(json.dumps(_study_input()), encoding="utf-8")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.txt").write_text(
        "EXTERNAL SYSTEM PROMPT with {GROUP}", encoding="utf-8"
    )
    (prompts / "user.txt").write_text(
        "Generate $candidate_count queries.\nOutcome ID: $outcome_id\n"
        "Domain: $domain\nOutcome: $outcome\nDefinition: $definition\n"
        "Scope: $geographic_scope\nIntent: $query_intent\n"
        "Use {GROUP}. IDs: $candidate_id_start through $candidate_id_end.",
        encoding="utf-8",
    )
    config = {
        "paths": {
            "study_input": "study_input.json",
            "system_prompt": "prompts/system.txt",
            "user_prompt": "prompts/user.txt",
            "runs_dir": "runs",
        },
        "pipeline": {"prompt_version": "test-v1", "candidates_per_outcome": 5},
        "model": {
            "id": None,
            "revision": None,
            "tokenizer_id": None,
            "tokenizer_revision": None,
            "device": "auto",
            "dtype": "auto",
            "trust_remote_code": False,
            "tokenizer_use_fast": True,
        },
        "generation": {
            "seed": 42,
            "parameters": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "max_new_tokens": 512,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, study_path


def _valid_raw(outcome_id: str = "HOU-02") -> str:
    return json.dumps(
        [
            {
                "candidate_id": f"{outcome_id}-Q{number:02d}",
                "query": f"Question {number} about {PLACEHOLDER}?",
            }
            for number in range(1, 6)
        ]
    )


def test_parse_candidates_requires_valid_json_and_five_unique_ids() -> None:
    with pytest.raises(PipelineError, match="invalid JSON"):
        parse_candidates("not-json", "HOU-02")
    with pytest.raises(PipelineError, match="Expected 5"):
        parse_candidates("[]", "HOU-02")

    duplicated = json.loads(_valid_raw())
    duplicated[1]["candidate_id"] = duplicated[0]["candidate_id"]
    with pytest.raises(PipelineError, match="Duplicate"):
        parse_candidates(json.dumps(duplicated), "HOU-02")


@pytest.mark.parametrize(
    "query",
    ["No placeholder here", "Two {GROUP} placeholders for {GROUP}"],
)
def test_parse_candidates_requires_exactly_one_placeholder(query: str) -> None:
    rows = json.loads(_valid_raw())
    rows[0]["query"] = query
    with pytest.raises(PipelineError, match="exactly once"):
        parse_candidates(json.dumps(rows), "HOU-02")


def test_end_to_end_pipeline_and_provenance(tmp_path: Path) -> None:
    config_path, study_path = _write_config(tmp_path)
    labels = ["Black people", "Black Americans", "women"]
    generator = FakeGenerator(labels)
    run_dir = generate_run(config_path, "test-run", generator)

    assert run_dir == tmp_path / "runs" / "test-run"
    assert len(generator.seen_messages) == 3
    candidates = read_jsonl(run_dir / "candidates.jsonl")
    assert len(candidates) == 15
    assert candidates[0]["candidate_id"] == "HOU-02-Q01"
    manifest = load_json(run_dir / "manifest.json")
    assert manifest["prompt_version"] == "test-v1"
    assert manifest["config_snapshot"]["generation"]["parameters"]["top_p"] == 0.9
    assert manifest["prompt_files"]["system"].endswith("prompts/system.txt")

    selection_path = run_dir / "selection.json"
    selection = load_json(selection_path)
    for item in selection:
        item["selected_candidate_id"] = f"{item['outcome_id']}-Q01"
        item["final_template"] = None
        item["notes"] = "Selected in test"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    freeze_templates(run_dir)
    output = expand_queries(study_path, run_dir)
    queries = read_jsonl(output)
    assert len(queries) == 12
    assert all(PLACEHOLDER not in row["query"] for row in queries)
    assert all("raw_response" in row and "system_prompt" in row for row in queries)

    for outcome_id in ("HOU-02", "EMP-04", "HC-01"):
        controls = [
            row
            for row in queries
            if row["outcome_id"] == outcome_id
            and row["condition_type"] == "control"
        ]
        assert len(controls) == 1
        assert controls[0]["group_condition"] == "people"
        assert controls[0]["social_dimension"] is None

    with (run_dir / "queries.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(queries)
    assert [row["query_id"] for row in csv_rows] == [
        row["query_id"] for row in queries
    ]


def test_freeze_rejects_unknown_selection(tmp_path: Path) -> None:
    config_path, _ = _write_config(tmp_path)
    generator = FakeGenerator(["Black people", "Black Americans", "women"])
    run_dir = generate_run(config_path, "test-run", generator)
    selection_path = run_dir / "selection.json"
    selection = load_json(selection_path)
    selection[0]["selected_candidate_id"] = "DOES-NOT-EXIST"
    for item in selection[1:]:
        item["selected_candidate_id"] = f"{item['outcome_id']}-Q01"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(PipelineError, match="Unknown"):
        freeze_templates(run_dir)
