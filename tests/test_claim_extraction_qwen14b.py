from __future__ import annotations

import sys
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from claim_extraction_qwen14b_core import (  # noqa: E402
    DEFAULT_MODEL_PATH, guard_equivalence, model_identity, model_slug,
    segment_responses, validate_decomposition,
    validate_selection,
)
from claim_extraction_qwen14b_inference import LocalQwen, recover_literal_selection  # noqa: E402
import run_claim_extraction_qwen14b as runner  # noqa: E402
from run_claim_extraction_qwen14b import _presence_patterns  # noqa: E402


class ClaimPilotTests(unittest.TestCase):
    def test_model_identity_is_not_hardcoded_to_qwen(self):
        self.assertEqual(model_identity(DEFAULT_MODEL_PATH), "microsoft/phi-4")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(json.dumps({"_name_or_path": "Acme/Other-Model"}))
            self.assertEqual(model_identity(path), "Acme/Other-Model")
            self.assertEqual(model_slug(model_identity(path)), "acme-other-model")
            self.assertEqual(model_identity(path, "explicit/model"), "explicit/model")

    def test_segmentation_preserves_offsets_bullets_and_shared_people(self):
        text = "Heading:\n• Black applicants may face barriers.\n\nNext paragraph.\n"
        data = pd.DataFrame([{"response_id": "dallas::people", "query_id": "people__x", "original_query": "Why?",
                              "outcome": "Access", "domain": "Education", "demographic_dimension": "Control",
                              "group": "people", "condition": "control", "group_type": "people", "location": "Dallas",
                              "replica_id": "v1_dallas", "source_file": "example.txt", "full_aio_text": text}])
        output = segment_responses(data)
        self.assertEqual(len(output), 2)
        self.assertTrue(bool(output.iloc[0].is_bullet))
        self.assertEqual(output.iloc[0].heading, "Heading:")
        for row in output.itertuples():
            self.assertEqual(text[row.sentence_start_char:row.sentence_end_char], row.original_sentence)

    def test_selection_rejects_rewritten_or_wrong_offsets(self):
        sentence = "Black applicants may face barriers."
        valid = {"selection_status": "HAS_VERIFIABLE_CLAIM", "verifiable_spans": [
            {"text": sentence, "start_char": 0, "end_char": len(sentence)}], "confidence": "high"}
        self.assertEqual(validate_selection(valid, sentence), valid)
        wrong = {**valid, "verifiable_spans": [{"text": "Black applicants face barriers.",
                                                 "start_char": 0, "end_char": len(sentence)}]}
        with self.assertRaises(ValueError):
            validate_selection(wrong, sentence)

    def test_selection_aligns_only_unique_literal_span(self):
        sentence = "Heading: Black applicants may face barriers."
        value = {"selection_status": "HAS_VERIFIABLE_CLAIM", "verifiable_spans": [
            {"text": "Black applicants may face barriers.", "start_char": 0, "end_char": 37}],
            "confidence": "high"}
        parsed = validate_selection(value, sentence)
        span = parsed["verifiable_spans"][0]
        self.assertEqual(sentence[span["start_char"]:span["end_char"]], span["text"])
        self.assertEqual(span["offset_alignment"], "deterministic_unique_literal_match")
        repeated = {"selection_status": "HAS_VERIFIABLE_CLAIM", "verifiable_spans": [
            {"text": "may", "start_char": 4, "end_char": 7}], "confidence": "high"}
        with self.assertRaises(ValueError):
            validate_selection(repeated, "may or may not")

    def test_selection_checkpoint_recovery_keeps_invalid_attempt(self):
        from claim_extraction_qwen14b_core import PROMPT_VERSION, stable_id
        from claim_extraction_qwen14b_prompts import prompt
        row = {"record_id": "sentence-1", "original_query": "Why?",
               "original_sentence": "Heading: X may vary.", "preceding_sentences": "[]",
               "following_sentences": "[]"}
        identity = "synthetic:model:hash"
        key = stable_id("selection", row["record_id"], PROMPT_VERSION, identity,
                        prompt("selection", row))
        raw = json.dumps({"selection_status": "HAS_VERIFIABLE_CLAIM",
                          "verifiable_spans": [{"text": "X may vary.", "start_char": 0, "end_char": 11}],
                          "confidence": "high"})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.jsonl"
            path.write_text(json.dumps({"record_id": key, "source_record_id": row["record_id"],
                                        "stage": "selection", "status": "invalid", "attempt": 1,
                                        "raw_output": raw, "error": "span text does not match"}) + "\n")
            recovered = recover_literal_selection(
                path, [row], identity, lambda value, item: validate_selection(value, item["original_sentence"]))
            self.assertEqual(recovered[key]["status"], "valid")
            self.assertEqual(len(path.read_text().splitlines()), 2)
            recover_literal_selection(
                path, [row], identity, lambda value, item: validate_selection(value, item["original_sentence"]))
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_added_context_requires_resolution_and_markers(self):
        claim = {"claim_text": "Black women may face barriers.", "claim_text_with_context_markers": "Black women may face barriers.",
                 "source_span_indices": [0], "added_context": ["Black women"], "qualifiers_preserved": ["may"],
                 "attribution_present": False}
        with self.assertRaises(ValueError):
            validate_decomposition({"claims": [claim]}, 1, True)
        claim["claim_text_with_context_markers"] = "[Black women] may face barriers."
        self.assertEqual(len(validate_decomposition({"claims": [claim]}, 1, True)["claims"]), 1)

    def test_qualification_mismatch_vetoes_equivalence(self):
        a = {"modality": "may", "negation": "", "causal_language": "associated with"}
        b = {"modality": "", "negation": "", "causal_language": "causes"}
        self.assertEqual(guard_equivalence(a, b, "EQUIVALENT"), "UNCERTAIN")
        self.assertEqual(guard_equivalence(a, a, "EQUIVALENT"), "EQUIVALENT")

    def test_invalid_output_retry_and_checkpoint_resume(self):
        engine = object.__new__(LocalQwen)
        engine.path = Path("/synthetic/model")
        engine.batch_size = 4
        calls = []

        def generate(stage, prompts):
            calls.append(prompts)
            return ["not json"] if len(calls) == 1 else [json.dumps({
                "selection_status": "HAS_VERIFIABLE_CLAIM", "verifiable_spans": [
                    {"text": "X may vary.", "start_char": 0, "end_char": 11}], "confidence": "high"})]

        engine._generate = generate
        row = {"record_id": "sentence-1", "original_query": "Why?", "original_sentence": "X may vary.",
               "preceding_sentences": "[]", "following_sentences": "[]"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.jsonl"
            first = engine.run_stage("selection", [row], path, lambda v, r: validate_selection(v, r["original_sentence"]))
            self.assertEqual(first["sentence-1"]["selection_status"], "HAS_VERIFIABLE_CLAIM")
            self.assertEqual(len(calls), 2)
            self.assertIn("FORMAT CORRECTION", calls[1][0])
            second = engine.run_stage("selection", [row], path, lambda v, r: validate_selection(v, r["original_sentence"]))
            self.assertEqual(second, first)
            self.assertEqual(len(calls), 2)

    def test_presence_requires_consistent_triplet(self):
        claims = pd.DataFrame([{"claim_id": x, "group_type": kind, "demographic_dimension": "Race", "domain": "Employment",
                                "outcome": "Hiring", "replica_id": "v1_dallas"}
                               for x, kind in (("p", "people"), ("m", "minority"), ("a", "majority"))])
        two = pd.DataFrame([{"claim_a_id": "p", "claim_b_id": "m", "match_status": "EQUIVALENT",
                             "demographic_dimension": "Race", "domain": "Employment", "outcome": "Hiring", "replica_id": "v1_dallas"},
                            {"claim_a_id": "m", "claim_b_id": "a", "match_status": "EQUIVALENT",
                             "demographic_dimension": "Race", "domain": "Employment", "outcome": "Hiring", "replica_id": "v1_dallas"}])
        self.assertEqual(_presence_patterns(claims, two).iloc[0].status, "ambiguous_or_uncertain")
        all_three = pd.concat([two, pd.DataFrame([{"claim_a_id": "p", "claim_b_id": "a", "match_status": "EQUIVALENT",
                                                   "demographic_dimension": "Race", "domain": "Employment", "outcome": "Hiring", "replica_id": "v1_dallas"}])])
        self.assertEqual(_presence_patterns(claims, all_three).iloc[0].provisional_pattern, "present in all 3")

    def test_synthetic_end_to_end_without_gpu(self):
        class FakeQwen:
            def __init__(self, *args, **kwargs):
                pass

            def run_stage(self, stage, inputs, path, validator):
                path.parent.mkdir(parents=True, exist_ok=True)
                result = {}
                with path.open("a") as stream:
                    for row in inputs:
                        if stage == "selection":
                            text = row["original_sentence"]
                            value = {"selection_status": "HAS_VERIFIABLE_CLAIM", "verifiable_spans": [
                                {"text": text, "start_char": 0, "end_char": len(text)}], "confidence": "high"}
                        elif stage == "disambiguation":
                            value = {"ambiguity_status": "NO_AMBIGUITY", "resolutions": []}
                        elif stage == "decomposition":
                            value = {"claims": [{"claim_text": row["original_sentence"],
                                "claim_text_with_context_markers": row["original_sentence"],
                                "source_span_indices": [0], "added_context": [], "qualifiers_preserved": ["may"],
                                "attribution_present": False, "attribution_text": "", "attributed_source": "",
                                "modality": "may", "negation": "", "quantity": "", "comparison": "", "causal_language": ""}]}
                        elif stage == "entailment":
                            value = {"entailment": "ENTAILED"}
                        elif stage == "qualification":
                            value = {"qualification_preservation": "PASS", "missing_qualifiers": []}
                        elif stage == "context_audit":
                            value = {"context_status": "GROUNDED"}
                        else:
                            value = {"match_status": "EQUIVALENT"}
                        result[row["record_id"]] = validator(value, row)
                        stream.write(json.dumps({"record_id": row["record_id"], "source_record_id": row["record_id"],
                                                 "status": "valid", "parsed": value}) + "\n")
                return result

        class FakeEmbedding:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

        rows = []
        for group_type in ("people", "minority", "majority"):
            text = f"{group_type.title()} may face barriers."
            rows.append({"response_id": f"dallas::{group_type}", "query_id": group_type,
                         "original_query": "What factors influence hiring?", "outcome": "Hiring", "domain": "Employment",
                         "demographic_dimension": "Control" if group_type == "people" else "Race",
                         "group": group_type, "condition": group_type, "group_type": group_type,
                         "location": "Dallas", "replica_id": "v1_dallas", "source_file": "synthetic.txt",
                         "full_aio_text": text})
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeEmbedding
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "results"
            args = types.SimpleNamespace(input_dir=Path(temp), output_dir=out, model_path=None,
                                         model_id="synthetic-model", embedding_model_id="synthetic-embedding",
                                         embedding_model_path=None, batch_size=3, tensor_parallel_size=1,
                                         gpu_memory_utilization=.5, max_model_len=2048, trust_remote_code=False)
            with patch.object(runner, "preflight", return_value=(Path(temp), Path(temp))), \
                 patch.object(runner, "load_dallas", return_value=pd.DataFrame(rows)), \
                 patch.object(runner, "LocalModel", FakeQwen), \
                 patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                runner.run(args)
            valid = pd.read_csv(out / "valid_claims.csv")
            metrics = pd.read_csv(out / "response_metrics.csv")
            self.assertEqual(len(valid), 3)
            self.assertEqual(len(metrics), 3)
            self.assertEqual(int(metrics.group_type.eq("people").sum()), 1)
            self.assertTrue((out / "figures" / "claim_extraction_funnel.png").exists())
            self.assertEqual(pd.read_csv(out / "extracted_claims.csv").iloc[0].model_name, "synthetic-model")
            args.model_id = "different-model"
            with patch.object(runner, "preflight", return_value=(Path(temp), Path(temp))):
                with self.assertRaisesRegex(RuntimeError, "different model"):
                    runner.run(args)


if __name__ == "__main__":
    unittest.main()
