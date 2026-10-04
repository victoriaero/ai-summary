from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_claim_epistemic_pilot import (  # noqa: E402
    _claim_begins_with_complement, _contained, _overlaps, _source_intervals, _strict_offsets,
)


class ClaimEpistemicPilotTests(unittest.TestCase):
    def test_literal_claim_intervals_do_not_fill_gaps(self):
        sentence = "May arise here, but another fact is certain."
        row = pd.Series({"claim_id": "c1", "full_aio_text": sentence,
                         "original_sentence": sentence, "sentence_start_char": 0,
                         "sentence_end_char": len(sentence),
                         "verifiable_span": '[{"text":"May arise here","start_char":0,"end_char":14},'
                                            '{"text":"certain","start_char":37,"end_char":44}]'})
        # Use actual offsets so the test also guards provenance validation.
        row["verifiable_span"] = ('[{"text":"May arise here","start_char":0,"end_char":14},'
                                  '{"text":"certain","start_char":' + str(sentence.index("certain")) +
                                  ',"end_char":' + str(sentence.index("certain") + 7) + '}]')
        intervals = _source_intervals(row)
        self.assertTrue(_contained(0, 3, intervals))
        self.assertFalse(_contained(sentence.index("but"), sentence.index("but") + 3, intervals))
        self.assertTrue(_overlaps(0, 4, intervals))

    def test_veridicality_requires_unique_literal_alignment(self):
        text = "Research shows that access may improve."
        row = pd.Series({"sentence_text": text, "predicate_surface": "shows",
                         "complement_text": "that access may improve"})
        aligned = _strict_offsets(row, text)
        self.assertEqual(aligned["alignment_status"], "unique_literal")
        self.assertEqual(text[aligned["complement_start"]:aligned["complement_end"]],
                         "that access may improve")
        self.assertEqual(_strict_offsets(row, text + " " + text)["alignment_status"],
                         "sentence_not_unique")

    def test_complement_link_is_not_any_embedded_overlap(self):
        self.assertTrue(_claim_begins_with_complement(
            "Access may improve.", "that access may improve"))
        self.assertFalse(_claim_begins_with_complement(
            "Social norms influence what students choose to study.",
            "what students choose to study"))


if __name__ == "__main__":
    unittest.main()
