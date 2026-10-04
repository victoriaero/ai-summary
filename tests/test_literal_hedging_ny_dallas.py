"""Contract tests for literal claims and paired contextual-hedging units."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from literal_claims_core import validate_literal_decomposition, validate_stage
from claim_extraction_qwen14b_inference import LocalModel
from run_literal_claim_extraction import deduplicate_literal_candidates
from analyze_hedging_ny_dallas import add_cells, bootstrap_model, inventory, measure_claims


class LiteralClaimTests(unittest.TestCase):
    def test_duplicate_literal_proposals_are_collapsed_and_audited(self):
        base = {"claim_id": "lc_same", "response_id": "r1", "sentence_id": "s1",
                "source_span_texts": ["X may rise"],
                "absolute_source_spans": [{"text": "X may rise", "start_char": 10, "end_char": 20}]}
        candidates = [{**base, "resolution_note": "first"}, {**base, "resolution_note": "second"}]
        unique, audit = deduplicate_literal_candidates(candidates)
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["n_model_proposals_same_spans"], 2)
        self.assertEqual(unique[0]["all_resolution_notes"], ["first", "second"])
        self.assertEqual(len(audit), 2)
        self.assertEqual(int(audit.retained.sum()), 1)

    def test_claim_id_collision_with_different_spans_fails(self):
        common = {"claim_id": "lc_collision", "response_id": "r1", "sentence_id": "s1",
                  "source_span_texts": ["X"]}
        candidates = [
            {**common, "absolute_source_spans": [{"text": "X", "start_char": 0, "end_char": 1}]},
            {**common, "absolute_source_spans": [{"text": "X", "start_char": 4, "end_char": 5}]},
        ]
        with self.assertRaises(ValueError):
            deduplicate_literal_candidates(candidates)

    def test_exact_spans_and_disjoint_fragments(self):
        sentence = "Studies suggest X may increase Y, but not Z."
        selected = [{"text": sentence, "start_char": 0, "end_char": len(sentence)}]
        first = sentence.index("Studies suggest")
        second = sentence.index("X may increase Y")
        spans = [{"text": "Studies suggest", "start_char": first, "end_char": first + len("Studies suggest")},
                 {"text": "X may increase Y", "start_char": second, "end_char": second + len("X may increase Y")}]
        value = {"claims": [{"source_spans": spans, "resolution_note": ""}]}
        self.assertEqual(validate_literal_decomposition(value, sentence, selected), value)
        spans[1]["text"] = "X increases Y"
        with self.assertRaises(ValueError):
            validate_literal_decomposition(value, sentence, selected)

    def test_qualification_is_required_for_acceptance(self):
        row = {"original_sentence": "X may increase Y", "source_span_texts": ["X increases Y"]}
        with self.assertRaises(ValueError):
            validate_stage("qualification", {"qualification_preservation": "INVALID"}, row)

    def test_stage_resume_does_not_generate_twice(self):
        engine = object.__new__(LocalModel)
        engine.path = Path("/local/test-model")
        engine.model_id = "test-model"
        engine.checkpoint_identity = "test-model:fixed"
        engine.batch_size = 2
        engine.prompt_fn = lambda stage, row: row["original_sentence"]
        engine.prompt_version = "test-v2"
        engine.system = "test"
        calls = []
        engine._generate = lambda stage, texts: calls.append(len(texts)) or ['{"claims": []}' for _ in texts]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            items = [{"record_id": "one", "original_sentence": "X."}]
            check = lambda value, _: value if isinstance(value.get("claims"), list) else None
            self.assertEqual(engine.run_stage("literal_decomposition", items, path, check)["one"], {"claims": []})
            self.assertEqual(engine.run_stage("literal_decomposition", items, path, check)["one"], {"claims": []})
            self.assertEqual(calls, [1])


class AnalysisTests(unittest.TestCase):
    def test_cue_must_be_inside_original_span(self):
        text = "X may rise, while Y may fall."
        response = pd.DataFrame([{"response_id": "v1_dallas::q", "aio_text": text}])
        claim = pd.DataFrame([{"claim_id": "lc_1", "response_id": "v1_dallas::q", "query_id": "q",
                              "replica_id": "v1_dallas", "outcome": "o", "demographic_dimension": "Race",
                              "group_type": "minority", "sentence_id": "s1", "original_sentence": text,
                              "sentence_start_char": 0, "sentence_end_char": len(text),
                              "source_span_texts": json.dumps(["X may rise"]),
                              "absolute_source_spans": json.dumps([{"text": "X may rise", "start_char": 0, "end_char": 10}])}])
        cues = pd.DataFrame([{"response_id": "v1_dallas::q", "cue_surface": "may", "cue_start": 2,
                              "cue_end": 5, "cue_probability": .9},
                             {"response_id": "v1_dallas::q", "cue_surface": "may", "cue_start": 20,
                              "cue_end": 23, "cue_probability": .9}])
        claims, links, sentences, counts = measure_claims(claim, cues, response)
        self.assertEqual(len(links), 1)
        self.assertTrue(claims.iloc[0].claim_span_contains_contextual_hedge)
        self.assertEqual(int(sentences.iloc[0].n_claim_sentences), 1)
        self.assertEqual(int(counts.iloc[0].n_claim_spans_with_hedge), 1)

    def test_people_is_one_cell_and_bootstrap_preserves_paired_effect(self):
        rows = [{"response_id": f"{loc}-{outcome}-{group}", "replica_id": loc,
                 "outcome": outcome, "dimension": "Race" if group != "people" else "Control",
                 "group_type": group, "n_claim_sentences": 10,
                 "n_hedged_claim_sentences": {"people": 2, "minority": 3, "majority": 1}[group]}
                for loc in ("v1_dallas", "v2_ny") for outcome in (f"o{i}" for i in range(21))
                for group in ("people", "minority", "majority")]
        data = add_cells(pd.DataFrame(rows))
        self.assertEqual(int(data.cell.eq("people").sum()), 42)
        effects, diagnostics = bootstrap_model(data, "claim_sentence", 4, 11)
        target = effects.loc[effects.comparison.eq("minority-people") & effects.replica_id.eq("pooled")].iloc[0]
        self.assertAlmostEqual(target.estimate, .1, places=5)
        self.assertEqual(diagnostics["bootstrap_succeeded"], 4)

    def test_inventory_rejects_missing_condition(self):
        raw = pd.DataFrame([{"replica_id": "v1_dallas", "response_id": "q", "query_id": "q",
                             "has_aio_text": True, "group_type": "people", "dimension": "Control",
                             "outcome": "o", "vpn_location": "", "public_ip": ""}])
        with self.assertRaises(ValueError):
            inventory(raw)


if __name__ == "__main__":
    unittest.main()
