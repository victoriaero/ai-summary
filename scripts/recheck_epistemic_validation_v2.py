"""Re-evaluate adjudicated validation sheets while preserving resource IDs."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from complete_epistemic_validation_entries import complete
from epistemic_commitment_v2_validation import validation_tables


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Update human-validation metrics without refitting models.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results" / "epistemic_commitment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "epistemic_commitment_v2")
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    data = pd.read_csv(source / "contextual_hedging" / "contextual_hedging_response_metrics.csv")
    summary = validation_tables(source, output, data)
    complete(source, output)
    shutil.copyfile(output / "validation" / "measurement_validation.csv", output / "tables" / "measurement_validation.csv")
    print(f"Validation metrics updated: {summary}")


if __name__ == "__main__":
    main()
