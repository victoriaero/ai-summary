from __future__ import annotations

import sys
import unittest
from pathlib import Path

import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_epistemic_commitment as analysis  # noqa: E402


class EpistemicCommitmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = spacy.load("en_core_web_sm")
        cls.resource = analysis.load_megaveridicality(
            analysis.DEFAULT_MEGAVERIDICALITY_FILE
        )
        cls.resource_lemmas = set(cls.resource["predicate_lemma"])
        cls.surface_map, _ = analysis.build_resource_surface_map(
            cls.nlp,
            cls.resource,
        )

    def predicate_match(self, text: str, surface: str):
        doc = self.nlp(text)
        predicate = next(token for token in doc if token.text == surface)
        complement = next(
            child
            for child in predicate.children
            if child.dep_ in {"ccomp", "xcomp"}
        )
        lemma, method = analysis.resolve_resource_lemma(
            predicate,
            self.resource_lemmas,
            self.surface_map,
        )
        self.assertIsNotNone(lemma)
        match = analysis.match_resource_entry(
            predicate,
            complement,
            lemma,
            self.resource,
        )
        return lemma, method, match

    def test_bioscope_lexicon_is_entirely_resource_derived(self) -> None:
        lexicon, file_counts = analysis.extract_bioscope_lexicon(
            analysis.DEFAULT_BIOSCOPE_DIR
        )
        self.assertEqual(lexicon["frequency_in_resource"].sum(), 4513)
        self.assertEqual(set(lexicon["source"]), {"BioScope 1.0"})
        self.assertEqual(sum(file_counts.values()), 4513)

    def test_finite_positive_frame_matches(self) -> None:
        lemma, _, match = self.predicate_match(
            "Researchers believe that access improves.",
            "believe",
        )
        self.assertEqual(lemma, "believe")
        self.assertEqual(match[0], "NP Ved that S")
        self.assertEqual(match[1], "matched")
        self.assertIsNotNone(match[3])

    def test_conditional_configuration_is_not_imputed(self) -> None:
        _, _, match = self.predicate_match(
            "If researchers believe that access improves, policy changes.",
            "believe",
        )
        self.assertEqual(match[1], "unmatched_conditional_configuration")
        self.assertIsNone(match[3])

    def test_ambiguous_infinitive_is_not_imputed(self) -> None:
        _, _, match = self.predicate_match(
            "Officials expected students to succeed.",
            "expected",
        )
        self.assertIn("NP Ved NP to VP", match[0])
        self.assertEqual(match[1], "unmatched_ambiguous_eventivity")
        self.assertIsNone(match[3])

    def test_phrasal_predicate_uses_dependency_particle(self) -> None:
        lemma, method, match = self.predicate_match(
            "Investigators figured out that costs matter.",
            "figured",
        )
        self.assertEqual(lemma, "figure_out")
        self.assertEqual(method, "spacy_lemma_plus_dependency_particle")
        self.assertEqual(match[1], "matched")


if __name__ == "__main__":
    unittest.main()
