import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_queries import GROUPS, generate_collection
from generate_robustness_queries import select_robustness_outcomes


class RobustnessQueryGenerationTests(unittest.TestCase):
    def setUp(self):
        self.agreement = ROOT / "artifacts/annotation_results/annotator_agreement_full.csv"
        self.top3 = ROOT / "artifacts/annotation_results/selected_top3_outcomes.csv"

    def test_selects_one_unused_outcome_per_domain(self):
        selected = select_robustness_outcomes(self.agreement, self.top3)
        original = pd.read_csv(self.top3)
        self.assertEqual(len(selected), 7)
        self.assertEqual(selected.Domain.nunique(), 7)
        self.assertTrue(selected.robustness_rank.eq(4).all())
        overlap = selected.merge(original, on=["Domain", "Outcome"])
        self.assertTrue(overlap.empty)
        expected = {
            "Healthcare": "Person-centered care",
            "Employment": "Employment benefits",
            "Education": "Completion/graduation",
            "Housing": "Neighborhood quality",
            "Credit & Financial Services": "Credit score",
            "Criminal Justice": "Police-initiated contact",
            "Government Benefits": "Administrative burden",
        }
        self.assertEqual(selected.set_index("Domain").Outcome.to_dict(), expected)

    def test_collection_has_91_files_and_preserves_existing_text(self):
        selected = select_robustness_outcomes(self.agreement, self.top3)[["Domain", "Outcome"]]
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory) / "google_aio_collection"
            created, preserved = generate_collection(selected, collection)
            self.assertEqual((created, preserved), (7 * len(GROUPS), 0))
            self.assertEqual(len(list(collection.glob("*/*.txt"))), 91)
            manifest = pd.read_csv(collection / "query_manifest.csv")
            self.assertEqual(len(manifest), 91)
            self.assertEqual(manifest.query_id.nunique(), 91)
            target = next(collection.glob("people/*.txt"))
            original = target.read_text(encoding="utf-8")
            target.write_text(original + "MANUAL ENTRY\n", encoding="utf-8")
            created, preserved = generate_collection(selected, collection)
            self.assertEqual((created, preserved), (0, 91))
            self.assertTrue(target.read_text(encoding="utf-8").endswith("MANUAL ENTRY\n"))


if __name__ == "__main__":
    unittest.main()
