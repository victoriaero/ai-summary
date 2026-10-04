from __future__ import annotations

import argparse
from pathlib import Path

from analyze_epistemic_commitment import (
    DEFAULT_BIOSCOPE_DIR,
    DEFAULT_COLLECTIONS_ROOT,
    DEFAULT_MEGAVERIDICALITY_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPACY_MODEL,
    run_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BioScope lexical-overlap baseline (exploratory only).")
    parser.add_argument("--collections-root", type=Path, default=DEFAULT_COLLECTIONS_ROOT)
    parser.add_argument("--bioscope-dir", type=Path, default=DEFAULT_BIOSCOPE_DIR)
    parser.add_argument("--megaveridicality-file", type=Path, default=DEFAULT_MEGAVERIDICALITY_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "lexical_bioscope_overlap")
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    args = parser.parse_args()
    run_analysis(args.collections_root.resolve(), args.bioscope_dir.resolve(), args.megaveridicality_file.resolve(), args.output_dir.resolve(), args.spacy_model)


if __name__ == "__main__":
    main()
