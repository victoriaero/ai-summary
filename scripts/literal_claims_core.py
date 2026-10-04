"""Versioned, literal-span claim extraction rules for Dallas and New York."""
from __future__ import annotations

import json

from claim_extraction_qwen14b_core import validate_disambiguation, validate_selection
from claim_extraction_qwen14b_prompts import context as old_context, prompt as old_prompt


PROMPT_VERSION = "aio-literal-claims-v2"
SYSTEM = ("You are a careful research annotator. Return exactly one JSON object. "
          "Use only the supplied AIO and query; do not use outside knowledge. "
          "Never paraphrase, normalize, or add words to a claim's source text.")
SCHEMAS = {
    "selection": {"type": "object", "required": ["selection_status", "verifiable_spans", "confidence"],
                  "properties": {"selection_status": {"enum": ["HAS_VERIFIABLE_CLAIM", "NO_VERIFIABLE_CLAIM"]},
                                 "verifiable_spans": {"type": "array", "items": {"type": "object"}},
                                 "confidence": {"enum": ["high", "medium", "low"]}}},
    "disambiguation": {"type": "object", "required": ["ambiguity_status", "resolutions"],
                       "properties": {"ambiguity_status": {"enum": ["NO_AMBIGUITY", "RESOLVABLE_AMBIGUITY", "UNRESOLVABLE_AMBIGUITY"]},
                                      "resolutions": {"type": "array", "items": {"type": "object"}}}},
    "literal_decomposition": {"type": "object", "required": ["claims"],
                              "properties": {"claims": {"type": "array", "items": {"type": "object"}}}},
    "entailment": {"type": "object", "required": ["entailment"],
                   "properties": {"entailment": {"enum": ["ENTAILED", "NOT_ENTAILED", "UNCERTAIN"]}}},
    "qualification": {"type": "object", "required": ["qualification_preservation", "missing_qualifiers"],
                      "properties": {"qualification_preservation": {"enum": ["PASS", "FAIL", "UNCERTAIN"]},
                                     "missing_qualifiers": {"type": "array", "items": {"type": "string"}}}},
    "context_audit": {"type": "object", "required": ["context_status"],
                      "properties": {"context_status": {"enum": ["GROUNDED", "HALLUCINATED_CONTEXT", "UNCERTAIN"]}}},
    "matching": {"type": "object", "required": ["match_status"],
                 "properties": {"match_status": {"enum": ["SAME_PROPOSITION", "PARTIAL_OVERLAP", "DIFFERENT", "UNCERTAIN"]}}},
}


def literal_text(sentence: str, spans: list[dict]) -> list[str]:
    """Separate original fragments; never synthesize a replacement sentence."""
    return [sentence[item["start_char"]:item["end_char"]] for item in spans]


def prompt(stage: str, row: dict) -> str:
    if stage in {"selection", "disambiguation"}:
        return old_prompt(stage, row)
    base = old_context(row)
    if stage == "literal_decomposition":
        task = ("Selected verifiable regions: " + json.dumps(row["verifiable_spans"], ensure_ascii=False) + "\n"
                "Identify relatively atomic factual propositions. For each proposition return one or more "
                "EXACT substrings of SENTENCE OF INTEREST, with zero-based start_char and exclusive end_char. "
                "Every selected character must come from that sentence. Include governing hedges/modals, "
                "negation, attribution, quantities, comparisons, conditions and population/time scope where "
                "needed for a faithful proposition. Disjoint fragments are allowed but must remain separate; "
                "do not join them into a new sentence. Do not include a bare fragment that changes the "
                "proposition when separated from its qualifier. Return "
                "{claims:[{source_spans:[{text,start_char,end_char}], resolution_note:string}]}. "
                "Return [] if no faithful literal proposition is possible.")
    elif stage in {"entailment", "qualification", "context_audit"}:
        fragments = json.dumps(row["source_span_texts"], ensure_ascii=False)
        task = "Original literal fragments (NOT a rewritten claim): " + fragments + "\n"
        if stage == "entailment":
            task += ("Do the fragments together express one specific factual proposition grounded in the "
                     "original sentence/context, without becoming misleading through omission? "
                     "Return {entailment: ENTAILED|NOT_ENTAILED|UNCERTAIN, reason: string}.")
        elif stage == "qualification":
            task += ("Are all necessary governing qualifications preserved in these fragments: modality, "
                     "uncertainty, negation, attribution, quantities, comparisons, conditions, population/time "
                     "scope and association versus causation? Return {qualification_preservation: "
                     "PASS|FAIL|UNCERTAIN, missing_qualifiers:[string], reason:string}.")
        else:
            task += ("Does interpreting the fragments as this proposition require unsupported outside "
                     "context? Return {context_status: GROUNDED|HALLUCINATED_CONTEXT|UNCERTAIN, reason:string}.")
    elif stage == "matching":
        task = ("Claim A literal fragments: " + row["claim_a"] + "\n"
                "Claim B query: " + row.get("original_query_b", "") + "\n"
                "Claim B preceding sentences: " + row.get("preceding_sentences_b", "") + "\n"
                "Claim B original sentence: " + row.get("original_sentence_b", "") + "\n"
                "Claim B literal fragments: " + row["claim_b"] + "\n"
                "Claim B following sentences: " + row.get("following_sentences_b", "") + "\n"
                "Classify the underlying factual proposition, independently of epistemic hedging. "
                "'X may increase Y' and 'X increases Y' may share a base proposition; record the modality "
                "difference separately. Association versus causation, negation, quantities, populations "
                "or comparison changes must not be silently equated. Return "
                "{match_status:SAME_PROPOSITION|PARTIAL_OVERLAP|DIFFERENT|UNCERTAIN, "
                "modality_difference:boolean, reason:string}.")
    else:
        raise ValueError(stage)
    return base + "\n\nTASK: " + task


def validate_literal_decomposition(value: dict, sentence: str, selected: list[dict]) -> dict:
    claims = value.get("claims")
    if not isinstance(claims, list):
        raise ValueError("missing claims")
    selected_ranges = [(x["start_char"], x["end_char"]) for x in selected]
    for claim in claims:
        spans = claim.get("source_spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError("claim has no literal source spans")
        previous_end = -1
        for span in spans:
            start, end, surface = span.get("start_char"), span.get("end_char"), span.get("text")
            if not isinstance(start, int) or not isinstance(end, int) or not isinstance(surface, str):
                raise ValueError("invalid source span types")
            if not (0 <= start < end <= len(sentence) and sentence[start:end] == surface):
                raise ValueError("source span is not exact original text")
            if start < previous_end:
                raise ValueError("source spans overlap or are out of order")
            if not any(a <= start and end <= b for a, b in selected_ranges):
                raise ValueError("source span is outside selected verifiable regions")
            previous_end = end
        if not isinstance(claim.get("resolution_note", ""), str):
            raise ValueError("invalid resolution note")
    return value


def validate_stage(stage: str, value: dict, row: dict) -> dict:
    if stage == "selection":
        return validate_selection(value, row["original_sentence"])
    if stage == "disambiguation":
        return validate_disambiguation(value, row["original_sentence"])
    if stage == "literal_decomposition":
        return validate_literal_decomposition(value, row["original_sentence"], row["verifiable_spans"])
    field, allowed = {
        "entailment": ("entailment", {"ENTAILED", "NOT_ENTAILED", "UNCERTAIN"}),
        "qualification": ("qualification_preservation", {"PASS", "FAIL", "UNCERTAIN"}),
        "context_audit": ("context_status", {"GROUNDED", "HALLUCINATED_CONTEXT", "UNCERTAIN"}),
        "matching": ("match_status", {"SAME_PROPOSITION", "PARTIAL_OVERLAP", "DIFFERENT", "UNCERTAIN"}),
    }[stage]
    if value.get(field) not in allowed:
        raise ValueError(f"invalid {stage} verdict")
    if stage == "qualification" and not isinstance(value.get("missing_qualifiers"), list):
        raise ValueError("missing qualifier audit")
    return value
