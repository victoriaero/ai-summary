from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTIONS_ROOT = PROJECT_ROOT / "annotations"
DEFAULT_BIOSCOPE_DIR = (
    PROJECT_ROOT / "resources" / "epistemic_commitment" / "bioscope" / "raw"
)
DEFAULT_MEGAVERIDICALITY_FILE = (
    PROJECT_ROOT
    / "resources"
    / "epistemic_commitment"
    / "megaveridicality_v2_1"
    / "raw"
    / "mega-veridicality-v2.1"
    / "mega-veridicality-v2.1-normalized.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "epistemic_commitment"
DEFAULT_SPACY_MODEL = "en_core_web_sm"

BIOSCOPE_SOURCE_URL = (
    "https://rgai.inf.u-szeged.hu/sites/rgai.sed.hu/files/bioscope.zip"
)
MEGAVERIDICALITY_SOURCE_URL = (
    "https://megaattitude.io/projects/mega-veridicality/"
    "mega-veridicality-v2.1.zip"
)

EXPECTED_LOCATION = {
    "v1_dallas": "Dallas",
    "v2_ny": "New York",
    "v3_la": "Los Angeles",
}

METADATA_COLUMNS = [
    "response_id",
    "query_id",
    "outcome",
    "domain",
    "dimension",
    "group",
    "condition",
    "group_type",
    "replica_id",
    "expected_location",
    "vpn_location",
    "location",
    "public_ip",
    "collection_datetime",
    "language",
    "source_file",
]

HEDGE_OCCURRENCE_COLUMNS = METADATA_COLUMNS + [
    "sentence_id",
    "sentence_text",
    "cue",
    "normalized_cue",
    "start",
    "end",
    "sentence_start",
    "sentence_end",
]

PREDICATE_OCCURRENCE_COLUMNS = METADATA_COLUMNS + [
    "sentence_id",
    "sentence_text",
    "predicate_surface",
    "predicate_lemma",
    "predicate_lemma_method",
    "complement_text",
    "syntactic_frame",
    "negated",
    "conditional_antecedent",
    "matched_resource_entry",
    "matching_status",
    "veridicality_score",
]


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def get_section(text: str, header: str, next_headers: Iterable[str]) -> str:
    text = normalize_newlines(text)
    match = re.search(rf"(?m)^\s*{re.escape(header)}\s*$", text)
    if not match:
        return ""
    remainder = text[match.end() :]
    positions = []
    for next_header in next_headers:
        next_match = re.search(
            rf"(?m)^\s*{re.escape(next_header)}\s*$",
            remainder,
        )
        if next_match:
            positions.append(next_match.start())
    if positions:
        remainder = remainder[: min(positions)]
    lines = remainder.splitlines()
    while lines and (
        not lines[0].strip() or re.fullmatch(r"=+", lines[0].strip())
    ):
        lines.pop(0)
    while lines and (
        not lines[-1].strip() or re.fullmatch(r"=+", lines[-1].strip())
    ):
        lines.pop()
    return "\n".join(lines).strip()


def parse_key_values(section: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in section.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def normalize_group_type(condition: str, group: str) -> str:
    condition_normalized = condition.casefold().strip()
    group_normalized = group.casefold().strip()
    if condition_normalized in {"control", "generic", "people"}:
        return "people"
    if group_normalized == "people":
        return "people"
    if condition_normalized in {"minority", "focal"}:
        return "minority"
    if condition_normalized in {"majority", "comparison"}:
        return "majority"
    return condition_normalized or "unknown"


def parse_collection_file(
    path: Path,
    collections_root: Path,
) -> dict[str, Any]:
    path = path.resolve()
    collections_root = collections_root.resolve()
    raw = path.read_text(encoding="utf-8", errors="replace")
    aio_text = get_section(
        raw,
        "AI OVERVIEW TEXT",
        ["COLLECTION INFO", "METADATA — DO NOT EDIT"],
    )
    query = get_section(
        raw,
        "QUERY",
        ["LINKS", "AI OVERVIEW TEXT", "COLLECTION INFO"],
    )
    collection_info = parse_key_values(
        get_section(
            raw,
            "COLLECTION INFO",
            ["METADATA — DO NOT EDIT"],
        )
    )
    metadata = parse_key_values(
        get_section(raw, "METADATA — DO NOT EDIT", [])
    )
    relative = path.relative_to(collections_root)
    replica_id = relative.parts[0]
    query_id = metadata.get("Query ID") or f"{path.parent.name}__{path.stem}"
    condition = metadata.get("Condition", "")
    group = metadata.get("Group", path.parent.name.replace("_", " "))
    vpn_location = collection_info.get("VPN location", "")
    expected_location = EXPECTED_LOCATION.get(replica_id, replica_id)
    record = {
        "response_id": f"{replica_id}::{query_id}",
        "query_id": query_id,
        "outcome": metadata.get("Outcome", ""),
        "domain": metadata.get("Domain", ""),
        "dimension": metadata.get("Dimension", ""),
        "group": group,
        "condition": condition,
        "group_type": normalize_group_type(condition, group),
        "replica_id": replica_id,
        "expected_location": expected_location,
        "vpn_location": vpn_location,
        "location": vpn_location or expected_location,
        "public_ip": collection_info.get("Public IP", ""),
        "collection_datetime": collection_info.get("Date/time", ""),
        "language": collection_info.get("Language", ""),
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "query": query,
        "aio_text": aio_text,
        "recorded_aio_present": collection_info.get("AIO present", ""),
        "notes": collection_info.get("Notes", ""),
    }
    return record


def discover_responses(collections_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for collection_dir in sorted(
        collections_root.glob("v*/google_aio_collection")
    ):
        for path in sorted(collection_dir.glob("*/*.txt")):
            records.append(parse_collection_file(path, collections_root))
    if not records:
        raise FileNotFoundError(
            f"No collection TXT files found below {collections_root}"
        )
    inventory = pd.DataFrame(records)
    inventory["has_aio_text"] = inventory["aio_text"].str.strip().ne("")
    inventory["analysis_included"] = inventory["has_aio_text"]
    responses = inventory.loc[inventory["analysis_included"]].copy()
    if responses["response_id"].duplicated().any():
        duplicates = responses.loc[
            responses["response_id"].duplicated(False), "response_id"
        ].tolist()
        raise ValueError(f"Duplicate response IDs: {duplicates[:5]}")
    return responses.reset_index(drop=True), inventory.reset_index(drop=True)


def normalize_cue(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def extract_bioscope_lexicon(
    bioscope_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    expected_files = [
        bioscope_dir / "abstracts.xml",
        bioscope_dir / "full_papers.xml",
        bioscope_dir / "clinical_merger" / "clinical_records_anon.xml",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing BioScope XML files: {missing}")

    counts: Counter[tuple[str, str]] = Counter()
    files_by_cue: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    file_counts: Counter[str] = Counter()
    for path in expected_files:
        root = ET.parse(path).getroot()
        for cue_element in root.iter("cue"):
            if cue_element.attrib.get("type", "").casefold() != "speculation":
                continue
            cue = "".join(cue_element.itertext()).strip()
            if not cue:
                continue
            normalized = normalize_cue(cue)
            key = (cue, normalized)
            counts[key] += 1
            files_by_cue[key].add(path.name)
            file_counts[path.name] += 1

    rows = [
        {
            "cue": cue,
            "normalized_cue": normalized,
            "frequency_in_resource": frequency,
            "source": "BioScope 1.0",
            "resource_files": ";".join(sorted(files_by_cue[(cue, normalized)])),
        }
        for (cue, normalized), frequency in counts.items()
    ]
    lexicon = pd.DataFrame(rows).sort_values(
        ["frequency_in_resource", "normalized_cue", "cue"],
        ascending=[False, True, True],
    )
    return lexicon.reset_index(drop=True), dict(file_counts)


def build_hedge_matcher(nlp, lexicon: pd.DataFrame) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    normalized_cues = sorted(set(lexicon["normalized_cue"]))
    patterns = [nlp.make_doc(cue) for cue in normalized_cues if cue]
    matcher.add("BIOSCOPE_SPECULATION_CUE", patterns)
    return matcher


def metadata_subset(record: dict[str, Any] | pd.Series) -> dict[str, Any]:
    return {column: record[column] for column in METADATA_COLUMNS}


def analyze_hedging(
    record: pd.Series,
    doc: Doc,
    matcher: PhraseMatcher,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sentences = list(doc.sents)
    sentence_by_token = {
        token.i: sentence_index
        for sentence_index, sentence in enumerate(sentences, start=1)
        for token in sentence
    }
    matched_spans = [doc[start:end] for _, start, end in matcher(doc)]
    matched_spans = filter_spans(matched_spans)
    occurrence_rows = []
    hedged_sentence_ids = set()
    for span in matched_spans:
        sentence_id = sentence_by_token[span.start]
        sentence = sentences[sentence_id - 1]
        hedged_sentence_ids.add(sentence_id)
        occurrence_rows.append(
            {
                **metadata_subset(record),
                "sentence_id": sentence_id,
                "sentence_text": sentence.text,
                "cue": span.text,
                "normalized_cue": normalize_cue(span.text),
                "start": span.start_char,
                "end": span.end_char,
                "sentence_start": span.start_char - sentence.start_char,
                "sentence_end": span.end_char - sentence.start_char,
            }
        )

    n_tokens = sum(
        1 for token in doc if not token.is_space and not token.is_punct
    )
    n_sentences = len(sentences)
    metrics = {
        **metadata_subset(record),
        "n_sentences": n_sentences,
        "n_tokens": n_tokens,
        "n_hedge_occurrences": len(occurrence_rows),
        "n_hedged_sentences": len(hedged_sentence_ids),
        "hedge_per_100_tokens": (
            100 * len(occurrence_rows) / n_tokens if n_tokens else np.nan
        ),
        "hedged_sentence_rate": (
            len(hedged_sentence_ids) / n_sentences
            if n_sentences
            else np.nan
        ),
    }
    return metrics, occurrence_rows


def load_megaveridicality(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"MegaVeridicality file not found: {path}")
    resource = pd.read_csv(path, sep="\t")
    required = {
        "verb",
        "frame",
        "polarity",
        "sentence",
        "veridicalitynorm",
    }
    missing = required - set(resource.columns)
    if missing:
        raise ValueError(f"MegaVeridicality is missing columns: {missing}")
    resource = resource.copy()
    resource["predicate_lemma"] = resource["verb"].str.casefold()
    resource["conditional"] = False
    resource["resource_entry_id"] = (
        resource["predicate_lemma"]
        + "|"
        + resource["frame"]
        + "|"
        + resource["polarity"]
        + "|conditional=False"
    )
    if resource.duplicated(
        ["predicate_lemma", "frame", "polarity", "conditional"]
    ).any():
        raise ValueError("MegaVeridicality normalized entries are not unique")
    return resource


def build_resource_surface_map(
    nlp,
    resource: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    sentence_rows = resource[["sentence", "predicate_lemma"]].drop_duplicates()
    candidates: defaultdict[str, set[str]] = defaultdict(set)
    resource_sentences = sentence_rows["sentence"].str.replace(
        r"\'",
        "'",
        regex=False,
    )
    docs = nlp.pipe(resource_sentences.tolist(), batch_size=128)
    for (_, row), doc in zip(sentence_rows.iterrows(), docs):
        roots = [
            token
            for token in doc
            if token.dep_ == "ROOT" and token.pos_ in {"VERB", "ADJ"}
        ]
        if len(roots) != 1:
            continue
        candidates[roots[0].lower_].add(row["predicate_lemma"])
    unambiguous = {
        surface: next(iter(lemmas))
        for surface, lemmas in candidates.items()
        if len(lemmas) == 1
    }
    ambiguous = {
        surface: sorted(lemmas)
        for surface, lemmas in candidates.items()
        if len(lemmas) > 1
    }
    return unambiguous, ambiguous


def resolve_resource_lemma(
    predicate: Token,
    resource_lemmas: set[str],
    resource_surface_map: dict[str, str],
) -> tuple[str | None, str]:
    parser_lemma = predicate.lemma_.casefold()
    particles = sorted(
        child.lower_ for child in predicate.children if child.dep_ == "prt"
    )
    for particle in particles:
        phrasal_lemma = f"{parser_lemma}_{particle}"
        if phrasal_lemma in resource_lemmas:
            return phrasal_lemma, "spacy_lemma_plus_dependency_particle"
    if parser_lemma in resource_lemmas:
        return parser_lemma, "spacy_lemma"
    surface = predicate.lower_
    if surface in resource_lemmas:
        return surface, "surface_equals_resource_lemma"
    if surface in resource_surface_map:
        return resource_surface_map[surface], "resource_sentence_surface_map"
    return None, "not_in_resource"


def is_passive(predicate: Token) -> bool:
    if "Pass" in predicate.morph.get("Voice"):
        return True
    return any(
        child.dep_ in {"auxpass", "nsubjpass"}
        for child in predicate.children
    )


def is_negated(predicate: Token) -> bool:
    return any(
        child.dep_ == "neg" or child.lower_ in {"not", "n't"}
        for child in predicate.children
    )


def is_conditional_antecedent(predicate: Token) -> bool:
    nodes = [predicate, *predicate.ancestors]
    return any(
        child.dep_ == "mark" and child.lower_ in {"if", "unless"}
        for node in nodes
        for child in node.children
    )


def subtree_span(token: Token) -> Span:
    subtree = list(token.subtree)
    start = min(item.i for item in subtree)
    end = max(item.i for item in subtree) + 1
    return token.doc[start:end]


def has_to_marker(complement: Token) -> bool:
    return any(
        child.lower_ == "to" and child.dep_ in {"aux", "mark"}
        for child in complement.children
    )


def has_for_subject(complement: Token) -> bool:
    for item in complement.subtree:
        if item.lower_ == "for" and item.dep_ in {"mark", "prep", "case"}:
            return True
    return False


def has_object_before_complement(predicate: Token, complement: Token) -> bool:
    return any(
        child.dep_ in {"obj", "dobj"} and child.i < complement.i
        for child in predicate.children
    )


def has_embedded_subject(complement: Token) -> bool:
    return any(
        child.dep_ in {"nsubj", "nsubjpass"} and child.i < complement.i
        for child in complement.children
    )


def candidate_frames(predicate: Token, complement: Token) -> tuple[list[str], str]:
    passive = is_passive(predicate)
    if complement.dep_ == "ccomp" and not has_to_marker(complement):
        frame = "NP was Ved that S" if passive else "NP Ved that S"
        return [frame], "finite_clausal_complement"

    if complement.dep_ not in {"ccomp", "xcomp", "advcl"} or not has_to_marker(
        complement
    ):
        return [], "unsupported_complement_dependency"

    if has_for_subject(complement) and not passive:
        return ["NP Ved for NP to VP"], "for_to_infinitive"

    if passive:
        prefix = "NP was Ved to VP"
    elif has_object_before_complement(
        predicate,
        complement,
    ) or has_embedded_subject(complement):
        prefix = "NP Ved NP to VP"
    else:
        prefix = "NP Ved to VP"

    if complement.lemma_.casefold() == "do":
        return [f"{prefix}[+eventive]"], "to_infinitive_eventive_do"
    if complement.lemma_.casefold() == "have":
        return [f"{prefix}[-eventive]"], "to_infinitive_noneventive_have"
    return [
        f"{prefix}[+eventive]",
        f"{prefix}[-eventive]",
    ], "to_infinitive_eventivity_ambiguous"


def match_resource_entry(
    predicate: Token,
    complement: Token,
    predicate_lemma: str,
    resource: pd.DataFrame,
) -> tuple[str, str, str, float | None]:
    frames, structural_status = candidate_frames(predicate, complement)
    if not frames:
        return "unmatched", structural_status, "unmatched", None
    frame_label = " | ".join(frames)
    if is_conditional_antecedent(predicate):
        return (
            frame_label,
            "unmatched_conditional_configuration",
            "unmatched",
            None,
        )
    polarity = "negative" if is_negated(predicate) else "positive"
    candidates = resource.loc[
        resource["predicate_lemma"].eq(predicate_lemma)
        & resource["frame"].isin(frames)
        & resource["polarity"].eq(polarity)
        & resource["conditional"].eq(False)  # noqa: E712
    ]
    if len(candidates) == 1:
        row = candidates.iloc[0]
        return (
            str(row["frame"]),
            "matched",
            str(row["resource_entry_id"]),
            float(row["veridicalitynorm"]),
        )
    if len(candidates) > 1:
        return (
            frame_label,
            "unmatched_ambiguous_eventivity",
            "unmatched",
            None,
        )
    return frame_label, f"unmatched_no_resource_tuple:{structural_status}", "unmatched", None


def analyze_veridicality(
    record: pd.Series,
    doc: Doc,
    resource: pd.DataFrame,
    resource_surface_map: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resource_lemmas = set(resource["predicate_lemma"])
    sentences = list(doc.sents)
    sentence_by_token = {
        token.i: sentence_index
        for sentence_index, sentence in enumerate(sentences, start=1)
        for token in sentence
    }
    occurrences = []
    for predicate in doc:
        lemma, lemma_method = resolve_resource_lemma(
            predicate,
            resource_lemmas,
            resource_surface_map,
        )
        if lemma is None:
            continue
        complements = [
            child
            for child in predicate.children
            if child.dep_ in {"ccomp", "xcomp"}
            or (
                child.dep_ == "advcl"
                and has_to_marker(child)
                and has_for_subject(child)
            )
        ]
        for complement in complements:
            frame, status, entry, score = match_resource_entry(
                predicate,
                complement,
                lemma,
                resource,
            )
            sentence_id = sentence_by_token[predicate.i]
            occurrences.append(
                {
                    **metadata_subset(record),
                    "sentence_id": sentence_id,
                    "sentence_text": sentences[sentence_id - 1].text,
                    "predicate_surface": predicate.text,
                    "predicate_lemma": lemma,
                    "predicate_lemma_method": lemma_method,
                    "complement_text": subtree_span(complement).text,
                    "syntactic_frame": frame,
                    "negated": is_negated(predicate),
                    "conditional_antecedent": is_conditional_antecedent(
                        predicate
                    ),
                    "matched_resource_entry": entry,
                    "matching_status": status,
                    "veridicality_score": score,
                }
            )

    scores = pd.Series(
        [
            row["veridicality_score"]
            for row in occurrences
            if row["veridicality_score"] is not None
        ],
        dtype=float,
    )
    metrics = {
        **metadata_subset(record),
        "n_veridical_predicates": int(len(scores)),
        "n_unmatched_predicate_occurrences": int(
            len(occurrences) - len(scores)
        ),
        "mean_veridicality": scores.mean() if len(scores) else np.nan,
        "median_veridicality": scores.median() if len(scores) else np.nan,
        "min_veridicality": scores.min() if len(scores) else np.nan,
        "max_veridicality": scores.max() if len(scores) else np.nan,
    }
    return metrics, occurrences


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(
    output_dir: Path,
    n_inventory: int,
    n_responses: int,
    n_hedges: int,
    n_predicates: int,
    n_matched_predicates: int,
) -> None:
    text = f"""Epistemic commitment analysis
==============================

This directory contains two separate operationalizations. They are not merged
into a composite index.

Input coverage
--------------
Collection files discovered: {n_inventory}
Non-empty AIO responses analyzed: {n_responses}

Hedging
-------
The lexicon is generated only from spans explicitly annotated with
type=\"speculation\" in BioScope 1.0. No cue is added, removed, stemmed, or
lemmatized by hand. Unicode NFKC, whitespace collapse, and case-folding are used
only to create normalized_cue for matching. Phrase matching is token-based and
case-insensitive. When lexicon entries overlap, the longest non-overlapping span
is retained. Token counts exclude spaces and punctuation. Character offsets
start/end are zero-based half-open offsets in the complete AIO response;
sentence_start/sentence_end are relative to sentence_text.

BioScope is a biomedical/clinical resource and this is a direct lexical
transfer to AIO text, not a contextual hedge classifier. Broad cues and
formatting symbols remain when BioScope annotated them; no domain-adaptation or
manual pruning is applied. The English spaCy parser is used for every included
response, with the original language metadata retained for auditing.

Hedge occurrences found: {n_hedges}

Veridicality
------------
Scores come from the official normalized MegaVeridicality v2.1 file. A
predicate is considered only when its spaCy lemma occurs in the resource and it
has a dependency-parsed ccomp or xcomp complement. Matching uses predicate
lemma, syntactic frame, predicate polarity, and the non-conditional resource
configuration. Conditional-antecedent uses are unmatched because the official
normalized table contains only non-conditional tuples.

The resource distinguishes eventive and non-eventive infinitival frames. An
infinitive is matched when only one resource tuple is possible or when the
complement head is the resource's explicit do/have contrast. Otherwise it is
marked unmatched rather than assigned a score. n_veridical_predicates counts
only matched occurrences. Summary statistics are missing when that count is
zero.

Predicate candidates found: {n_predicates}
Matched and scored predicates: {n_matched_predicates}

Files
-----
- input_inventory.csv: every discovered collection file and inclusion status
- hedge_lexicon.csv: lexicon derived from BioScope annotations
- hedge_occurrences.csv: occurrence-level hedge matches
- hedging_response_metrics.csv: response-level hedging metrics
- predicate_occurrences.csv: matched and unmatched predicate candidates
- veridicality_response_metrics.csv: response-level veridicality summaries
- megaveridicality_resource_entries.csv: exact normalized tuples used
- resource_provenance.json: versions, URLs, checksums, and parser version

Reproduction
------------
Run from the repository root:

    python scripts/analyze_epistemic_commitment.py
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def run_analysis(
    collections_root: Path,
    bioscope_dir: Path,
    megaveridicality_file: Path,
    output_dir: Path,
    spacy_model: str,
) -> Path:
    lexicon, bioscope_file_counts = extract_bioscope_lexicon(bioscope_dir)
    megaveridicality = load_megaveridicality(megaveridicality_file)
    responses, inventory = discover_responses(collections_root)

    nlp = spacy.load(spacy_model)
    matcher = build_hedge_matcher(nlp, lexicon)
    resource_surface_map, ambiguous_resource_surfaces = (
        build_resource_surface_map(nlp, megaveridicality)
    )
    docs = list(nlp.pipe(responses["aio_text"].tolist(), batch_size=32))

    hedge_metrics = []
    hedge_occurrences = []
    veridicality_metrics = []
    predicate_occurrences = []
    for (_, response), doc in zip(responses.iterrows(), docs):
        hedge_summary, hedge_rows = analyze_hedging(response, doc, matcher)
        veridicality_summary, predicate_rows = analyze_veridicality(
            response,
            doc,
            megaveridicality,
            resource_surface_map,
        )
        hedge_metrics.append(hedge_summary)
        hedge_occurrences.extend(hedge_rows)
        veridicality_metrics.append(veridicality_summary)
        predicate_occurrences.extend(predicate_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.drop(columns=["aio_text"]).to_csv(
        output_dir / "input_inventory.csv",
        index=False,
    )
    lexicon.to_csv(output_dir / "hedge_lexicon.csv", index=False)
    pd.DataFrame(hedge_occurrences, columns=HEDGE_OCCURRENCE_COLUMNS).to_csv(
        output_dir / "hedge_occurrences.csv",
        index=False,
    )
    pd.DataFrame(hedge_metrics).to_csv(
        output_dir / "hedging_response_metrics.csv",
        index=False,
    )
    predicate_frame = pd.DataFrame(
        predicate_occurrences,
        columns=PREDICATE_OCCURRENCE_COLUMNS,
    )
    predicate_frame.to_csv(
        output_dir / "predicate_occurrences.csv",
        index=False,
    )
    pd.DataFrame(veridicality_metrics).to_csv(
        output_dir / "veridicality_response_metrics.csv",
        index=False,
    )
    megaveridicality.to_csv(
        output_dir / "megaveridicality_resource_entries.csv",
        index=False,
    )

    bioscope_zip = bioscope_dir.parent / "bioscope.zip"
    mega_zip = (
        megaveridicality_file.parents[2] / "mega-veridicality-v2.1.zip"
    )
    provenance = {
        "bioscope": {
            "version": "1.0",
            "source_url": BIOSCOPE_SOURCE_URL,
            "archive_sha256": sha256(bioscope_zip)
            if bioscope_zip.exists()
            else None,
            "annotated_speculation_spans_by_file": bioscope_file_counts,
            "lexicon_rows": len(lexicon),
            "unique_normalized_cues": int(lexicon["normalized_cue"].nunique()),
        },
        "megaveridicality": {
            "version": "2.1",
            "source_url": MEGAVERIDICALITY_SOURCE_URL,
            "archive_sha256": sha256(mega_zip) if mega_zip.exists() else None,
            "normalized_tuples": len(megaveridicality),
            "predicates": int(megaveridicality["predicate_lemma"].nunique()),
            "frames": sorted(megaveridicality["frame"].unique().tolist()),
            "score_column": "veridicalitynorm",
            "unambiguous_inflected_surface_mappings": len(
                resource_surface_map
            ),
            "ambiguous_inflected_surfaces_excluded": (
                ambiguous_resource_surfaces
            ),
        },
        "dependency_parser": {
            "library": "spaCy",
            "spacy_version": spacy.__version__,
            "model": spacy_model,
            "model_version": nlp.meta.get("version"),
            "language": nlp.meta.get("lang"),
        },
        "collections": {
            "files_discovered": len(inventory),
            "responses_analyzed": len(responses),
            "replicas": inventory.groupby("replica_id")["has_aio_text"]
            .agg(files="size", nonempty="sum")
            .reset_index()
            .to_dict(orient="records"),
        },
    }
    (output_dir / "resource_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(
        output_dir,
        len(inventory),
        len(responses),
        len(hedge_occurrences),
        len(predicate_occurrences),
        int(predicate_frame["veridicality_score"].notna().sum()),
    )
    print(f"Collection files discovered: {len(inventory)}")
    print(f"Non-empty AIO responses analyzed: {len(responses)}")
    print(f"BioScope lexicon rows: {len(lexicon)}")
    print(f"Hedge occurrences: {len(hedge_occurrences)}")
    print(f"Predicate candidates: {len(predicate_occurrences)}")
    print(
        "Matched veridicality predicates: "
        f"{predicate_frame['veridicality_score'].notna().sum()}"
    )
    print(f"Results written to: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure hedging and MegaVeridicality-matched predicate "
            "commitment in Google AI Overview responses."
        )
    )
    parser.add_argument(
        "--collections-root",
        type=Path,
        default=DEFAULT_COLLECTIONS_ROOT,
    )
    parser.add_argument(
        "--bioscope-dir",
        type=Path,
        default=DEFAULT_BIOSCOPE_DIR,
    )
    parser.add_argument(
        "--megaveridicality-file",
        type=Path,
        default=DEFAULT_MEGAVERIDICALITY_FILE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    args = parser.parse_args()
    run_analysis(
        args.collections_root.resolve(),
        args.bioscope_dir.resolve(),
        args.megaveridicality_file.resolve(),
        args.output_dir.resolve(),
        args.spacy_model,
    )


if __name__ == "__main__":
    main()
