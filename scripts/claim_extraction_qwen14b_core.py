"""Pure preparation, validation and local-model helpers for the AIO claim pilot."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from analyze_epistemic_commitment import parse_collection_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "annotations" / "v1_dallas" / "google_aio_collection"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "claim_extraction_pilot"
DEFAULT_MODEL_PATH = Path("/scratch/LLMs/models/microsoft/phi-4")
DEFAULT_MODEL_ID = "microsoft/phi-4"
EMBEDDING_ID = "sentence-transformers/all-mpnet-base-v2"
SEED = 20261001
PROMPT_VERSION = "aio-claims-preserving-v1"


def stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:20]


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_dallas(input_dir: Path = DEFAULT_INPUT) -> pd.DataFrame:
    root = input_dir.resolve().parents[1]
    rows = [parse_collection_file(path, root) for path in sorted(input_dir.glob("*/*.txt"))]
    if not rows:
        raise FileNotFoundError(f"No collection files under {input_dir}")
    data = pd.DataFrame(rows).rename(columns={"query": "original_query", "aio_text": "full_aio_text", "dimension": "demographic_dimension"})
    data = data.loc[data.full_aio_text.fillna("").str.strip().ne("")].copy()
    if data.response_id.duplicated().any():
        raise ValueError("Response IDs are not unique; People must be processed once")
    return data.sort_values("response_id").reset_index(drop=True)


def _heading(text: str) -> bool:
    value = text.strip().rstrip(":")
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", value)
    title_like = bool(words and len(words) >= 2 and all(
        word.casefold() in {"and", "or", "the", "of", "to", "in", "for", "a", "an"} or word[0].isupper()
        for word in words))
    return bool(value and len(value.split()) <= 8 and not re.search(r"[.!?;]", value) and
                (text.rstrip().endswith(":") or value.isupper() or title_like))


def segment_responses(data: pd.DataFrame) -> pd.DataFrame:
    """Preserve offsets in the extracted AIO text; never repair collapsed formatting."""
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    rows = []
    for response in data.to_dict("records"):
        raw = response["full_aio_text"]
        line_start = 0
        paragraph = 0
        pending_heading = ""
        for line in raw.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            if not body.strip():
                paragraph += 1
                line_start += len(line)
                continue
            stripped = body.strip()
            if _heading(stripped):
                pending_heading = stripped
                line_start += len(line)
                continue
            bullet = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", body)
            prefix = len(bullet.group(0)) if bullet else 0
            content = body[prefix:]
            for sent in nlp(content).sents:
                local_start = prefix + sent.start_char
                local_end = prefix + sent.end_char
                while local_start < local_end and body[local_start].isspace():
                    local_start += 1
                while local_end > local_start and body[local_end - 1].isspace():
                    local_end -= 1
                if local_start == local_end:
                    continue
                start, end = line_start + local_start, line_start + local_end
                value = raw[start:end]
                rows.append({
                    **{k: response.get(k, "") for k in ("response_id", "query_id", "original_query", "outcome", "domain",
                        "demographic_dimension", "group", "condition", "group_type", "location", "replica_id", "source_file")},
                    "sentence_id": f"s_{stable_id(response['response_id'], start, end)}",
                    "original_sentence": value, "sentence_start_char": start, "sentence_end_char": end,
                    "paragraph_id": paragraph, "heading": pending_heading, "is_bullet": bool(bullet),
                    "structure_warning": bool(re.search(r"[a-z][.!?][A-Z]|[a-z][A-Z][a-z]", value)),
                    "full_aio_text": raw,
                })
            pending_heading = ""
            line_start += len(line)
    sentences = pd.DataFrame(rows)
    if sentences.empty:
        return sentences
    sentences = sentences.sort_values(["response_id", "sentence_start_char"]).reset_index(drop=True)
    for response_id, indices in sentences.groupby("response_id", sort=False).groups.items():
        order = list(indices)
        for pos, index in enumerate(order):
            sentences.at[index, "preceding_sentences"] = json.dumps(
                sentences.loc[order[max(0, pos - 5):pos], "original_sentence"].tolist(), ensure_ascii=False)
            sentences.at[index, "following_sentences"] = json.dumps(
                sentences.loc[order[pos + 1:pos + 6], "original_sentence"].tolist(), ensure_ascii=False)
            sentences.at[index, "sentence_order"] = pos
    for row in sentences.itertuples():
        assert row.full_aio_text[row.sentence_start_char:row.sentence_end_char] == row.original_sentence
    return sentences


def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.casefold()).strip("-") or "local-model"


def model_identity(path: Path, explicit_id: str | None = None) -> str:
    if explicit_id:
        return explicit_id
    if path.resolve() == DEFAULT_MODEL_PATH.resolve():
        return DEFAULT_MODEL_ID
    config = path / "config.json"
    if config.exists():
        name = json.loads(config.read_text(encoding="utf-8")).get("_name_or_path")
        if isinstance(name, str) and name and not Path(name).is_absolute():
            return name
    return path.name


def model_fingerprint(path: Path) -> str:
    """Bind checkpoints to local model files without hashing tens of GB of weights."""
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        stat = file.stat()
        digest.update(f"{file.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        if file.suffix in {".json", ".jinja"} and stat.st_size <= 10_000_000:
            digest.update(file.read_bytes())
    return digest.hexdigest()


def resolve_local_model(value: str | None, model_id: str, purpose: str) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{purpose} checkpoint does not exist: {path}")
        if purpose == "LLM" and not (path / "config.json").exists():
            raise FileNotFoundError(f"LLM checkpoint lacks config.json: {path}")
        return path
    try:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model_id, local_files_only=True)).resolve()
        return path
    except Exception as exc:
        raise RuntimeError(
            f"{purpose} checkpoint is not available locally: {model_id}. "
            f"Pass --{'model' if purpose == 'LLM' else 'embedding-model'}-path PATH to the existing checkpoint. "
            "No automatic download will be attempted."
        ) from exc


def preflight(model_path: str | None, embedding_path: str | None,
              tensor_parallel_size: int = 2, embedding_model_id: str = EMBEDDING_ID) -> tuple[Path, Path]:
    if tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be positive")
    missing = [name for name in ("torch", "vllm", "sentence_transformers") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing Python packages: " + ", ".join(missing) +
                           ". Install them in the GPU environment before running; no models were downloaded.")
    import torch
    if torch.cuda.device_count() < tensor_parallel_size:
        raise RuntimeError(f"Requested {tensor_parallel_size} GPUs, but only {torch.cuda.device_count()} are visible")
    return (resolve_local_model(model_path or str(DEFAULT_MODEL_PATH), DEFAULT_MODEL_ID, "LLM"),
            resolve_local_model(embedding_path, embedding_model_id, "Embedding"))


def validate_selection(value: dict, sentence: str) -> dict:
    status = value.get("selection_status")
    if status not in {"HAS_VERIFIABLE_CLAIM", "NO_VERIFIABLE_CLAIM"}:
        raise ValueError("invalid selection status")
    spans = value.get("verifiable_spans")
    if not isinstance(spans, list) or (status == "NO_VERIFIABLE_CLAIM" and spans) or (status == "HAS_VERIFIABLE_CLAIM" and not spans):
        raise ValueError("invalid span list")
    for span in spans:
        start, end, text = span.get("start_char"), span.get("end_char"), span.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("invalid span text")
        if not (isinstance(start, int) and not isinstance(start, bool) and
                isinstance(end, int) and not isinstance(end, bool) and
                0 <= start < end <= len(sentence) and sentence[start:end] == text):
            # Only repair a unique, literal occurrence. Never accept a paraphrase
            # or guess between repeated occurrences of the same text.
            aligned_start = sentence.find(text)
            if aligned_start < 0 or sentence.find(text, aligned_start + 1) >= 0:
                raise ValueError("span text does not uniquely match original sentence")
            span["start_char"] = aligned_start
            span["end_char"] = aligned_start + len(text)
            span["offset_alignment"] = "deterministic_unique_literal_match"
    if value.get("confidence") not in {"high", "medium", "low"}:
        raise ValueError("invalid confidence")
    return value


def validate_disambiguation(value: dict, sentence: str) -> dict:
    if value.get("ambiguity_status") not in {"NO_AMBIGUITY", "RESOLVABLE_AMBIGUITY", "UNRESOLVABLE_AMBIGUITY"}:
        raise ValueError("invalid ambiguity status")
    resolutions = value.get("resolutions", [])
    if not isinstance(resolutions, list):
        raise ValueError("invalid resolutions")
    if value["ambiguity_status"] == "RESOLVABLE_AMBIGUITY" and not resolutions:
        raise ValueError("resolvable ambiguity requires explicit resolutions")
    if value["ambiguity_status"] == "NO_AMBIGUITY" and resolutions:
        raise ValueError("no-ambiguity decision cannot include resolutions")
    for item in resolutions:
        if item.get("ambiguity_type") not in {"referential", "structural"}:
            raise ValueError("invalid ambiguity type")
        if item.get("original_expression") and item["original_expression"] not in sentence:
            raise ValueError("resolution expression is not in original sentence")
        if value["ambiguity_status"] == "RESOLVABLE_AMBIGUITY" and not item.get("resolution_source"):
            raise ValueError("missing resolution source")
    return value


def validate_decomposition(value: dict, n_spans: int, has_resolution: bool) -> dict:
    claims = value.get("claims")
    if not isinstance(claims, list):
        raise ValueError("missing claims list")
    for claim in claims:
        if not isinstance(claim.get("claim_text"), str) or not claim["claim_text"].strip():
            raise ValueError("empty claim")
        indices = claim.get("source_span_indices")
        if not isinstance(indices, list) or not indices or not all(isinstance(i, int) and 0 <= i < n_spans for i in indices):
            raise ValueError("claim has no valid source-span linkage")
        marked = claim.get("claim_text_with_context_markers", "")
        if not isinstance(marked, str) or not marked:
            raise ValueError("missing marked claim")
        if not isinstance(claim.get("added_context"), list):
            raise ValueError("added_context must be a list")
        if claim.get("added_context") and (not has_resolution or "[" not in marked or "]" not in marked):
            raise ValueError("context addition lacks traceable resolution/markers")
        if any(not isinstance(item, str) or f"[{item}]" not in marked for item in claim["added_context"]):
            raise ValueError("every context addition must be individually bracket-marked")
        if not isinstance(claim.get("qualifiers_preserved"), list):
            raise ValueError("missing qualifier list")
        if not isinstance(claim.get("attribution_present"), bool):
            raise ValueError("missing attribution flag")
    return value


def validate_verdict(value: dict, stage: str) -> dict:
    field, allowed = {
        "entailment": ("entailment", {"ENTAILED", "NOT_ENTAILED", "UNCERTAIN"}),
        "qualification": ("qualification_preservation", {"PASS", "FAIL", "UNCERTAIN"}),
        "context_audit": ("context_status", {"GROUNDED", "HALLUCINATED_CONTEXT", "UNCERTAIN"}),
    }[stage]
    if value.get(field) not in allowed:
        raise ValueError(f"invalid {stage} verdict")
    if stage == "qualification" and not isinstance(value.get("missing_qualifiers"), list):
        raise ValueError("missing qualifier audit list")
    return value


def normalized_exact(value: str) -> str:
    return " ".join(value.casefold().split())


def guard_equivalence(a: dict, b: dict, verdict: str) -> str:
    """Conservative extra veto, never an automatic equivalence declaration."""
    if verdict != "EQUIVALENT":
        return verdict
    for key in ("modality", "negation", "quantity", "comparison", "attribution", "causal_language"):
        left, right = str(a.get(key, "")).strip().casefold(), str(b.get(key, "")).strip().casefold()
        if left != right:
            return "UNCERTAIN"
    return verdict
