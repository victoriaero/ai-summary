"""Auditable, stage-specific prompts. No external evidence or truth checking."""
from __future__ import annotations

import json

SYSTEM = ("You are a careful research annotator. Return exactly one JSON object and no prose. "
          "Use only the query, original sentence, and provided AIO context. "
          "Do not use outside knowledge. A factual claim is a specific proposition that could in principle "
          "be objectively verified as true or false; do not judge whether it is true or important.")


def context(row: dict) -> str:
    return (f"QUERY: {row['original_query']}\n"
            f"PRECEDING SENTENCES: {row['preceding_sentences']}\n"
            f"SENTENCE OF INTEREST: {row['original_sentence']}\n"
            f"FOLLOWING SENTENCES: {row['following_sentences']}")


def prompt(stage: str, row: dict) -> str:
    base = context(row)
    if stage == "selection":
        task = ("Identify exact, contiguous verifiable span(s) within SENTENCE OF INTEREST. "
                "Do not rewrite or paraphrase. Preserve modal/epistemic words, attribution, negation, "
                "quantities, comparators, conditions and population/time scope inside each selected span. "
                "Offsets are zero-based Python character offsets within the original sentence; end is exclusive. "
                "Return {selection_status: HAS_VERIFIABLE_CLAIM|NO_VERIFIABLE_CLAIM, "
                "verifiable_spans: [{text,start_char,end_char}], confidence: high|medium|low}. "
                "For no claim, spans must be []. Neither relevance nor citations decide verifiability.")
    elif stage == "disambiguation":
        task = ("Selected exact spans: " + json.dumps(row["verifiable_spans"], ensure_ascii=False) + ". "
                "Identify referential or structural ambiguity, including unresolved references and time. "
                "Do not treat vagueness alone as ambiguity. Resolve only from query/surrounding AIO text. "
                "Return {ambiguity_status: NO_AMBIGUITY|RESOLVABLE_AMBIGUITY|UNRESOLVABLE_AMBIGUITY, "
                "resolutions: [{ambiguity_type: referential|structural, original_expression, "
                "resolved_expression, resolution_source}], confidence: high|medium|low}. "
                "If any consequential ambiguity cannot be reliably resolved, choose UNRESOLVABLE_AMBIGUITY.")
    elif stage == "decomposition":
        task = ("Selected spans, indexed from zero: " + json.dumps(row["verifiable_spans"], ensure_ascii=False) + "\n"
                "Resolutions: " + json.dumps(row["resolutions"], ensure_ascii=False) + "\n"
                "Produce informative, relatively atomic, decontextualized factual claims. Preserve modality, "
                "uncertainty, negation, attribution, source named in AIO, quantity, time/population/geographic "
                "scope, conditions, comparisons and association-versus-causation. Do not decompose into "
                "trivialities. Any words supplied from context must appear in square brackets in "
                "claim_text_with_context_markers and be listed in added_context. "
                "Return {claims: [{claim_text, claim_text_with_context_markers, source_span_indices: [0], "
                "added_context: [string], qualifiers_preserved: [string], attribution_present: true|false, "
                "attribution_text: string, attributed_source: string, modality: string, negation: string, "
                "quantity: string, comparison: string, causal_language: string}]}. "
                "An empty claims list is allowed if no faithful claim can be formed.")
    elif stage == "entailment":
        task = ("Claim to audit: " + row["claim_text_with_context_markers"] + "\n"
                "Is every substantive part of this claim supported by the source sentence and allowed AIO "
                "context? Do not use world knowledge. Return {entailment: ENTAILED|NOT_ENTAILED|UNCERTAIN, "
                "reason: one short sentence}.")
    elif stage == "qualification":
        task = ("Claim to audit: " + row["claim_text_with_context_markers"] + "\n"
                "Check preservation of modal/epistemic verbs, uncertainty, negation, quantities, comparisons, "
                "temporal/population/geographic scope, conditional language and attribution from original "
                "sentence and selected spans. Return {qualification_preservation: PASS|FAIL|UNCERTAIN, "
                "missing_qualifiers: [string], reason: one short sentence}.")
    elif stage == "context_audit":
        task = ("Claim to audit: " + row["claim_text_with_context_markers"] + "\n"
                "Declared added context: " + json.dumps(row.get("added_context", []), ensure_ascii=False) + "\n"
                "Does any claim content come from neither original sentence nor supplied AIO context? "
                "Return {context_status: GROUNDED|HALLUCINATED_CONTEXT|UNCERTAIN, "
                "unsupported_content: [string], reason: one short sentence}.")
    elif stage == "matching":
        task = ("Claim A: " + row["claim_a"] + "\nClaim B: " + row["claim_b"] + "\n"
                "Are these the same verifiable proposition? Differences in may versus certainty, negation, "
                "quantities, comparison groups, attribution, or association versus causation preclude "
                "equivalence. Do not use outside knowledge. Return {match_status: EQUIVALENT|RELATED_NOT_EQUIVALENT|"
                "NOT_RELATED|UNCERTAIN, reason: one short sentence}.")
    else:
        raise ValueError(stage)
    return base + "\n\nTASK: " + task
