from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

from scripts.common import PLACEHOLDER, PipelineError


def load_prompt_templates(
    system_path: Path, user_path: Path
) -> tuple[str, Template]:
    system_prompt = _read_prompt(system_path)
    user_prompt = _read_prompt(user_path)
    if PLACEHOLDER not in system_prompt or PLACEHOLDER not in user_prompt:
        raise PipelineError(
            f"Both prompt files must contain the literal placeholder {PLACEHOLDER}"
        )
    return system_prompt, Template(user_prompt)


def build_messages(
    outcome: dict[str, Any],
    study: dict[str, Any],
    system_prompt: str,
    user_template: Template,
    candidates_per_outcome: int,
) -> list[dict[str, str]]:
    values = {
        "candidate_count": str(candidates_per_outcome),
        "domain": outcome["domain"],
        "outcome_id": outcome["outcome_id"],
        "outcome": outcome["outcome"],
        "definition": outcome["definition"],
        "geographic_scope": study["geographic_scope"],
        "query_intent": study["query_intent"],
        "candidate_id_start": f"{outcome['outcome_id']}-Q01",
        "candidate_id_end": f"{outcome['outcome_id']}-Q{candidates_per_outcome:02d}",
    }
    try:
        user_prompt = user_template.substitute(values)
    except (KeyError, ValueError) as exc:
        raise PipelineError(f"Invalid user prompt template: {exc}") from exc
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _read_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise PipelineError(f"Prompt file not found: {path}") from exc
    if not prompt:
        raise PipelineError(f"Prompt file is empty: {path}")
    return prompt
