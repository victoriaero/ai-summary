from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Add harmonized demographic_dimension metadata to generated epistemic-commitment CSV tables.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/epistemic_commitment"))
    args = parser.parse_args()
    root = args.results_dir.resolve()
    dirs = [root / "lexical_bioscope_overlap", root / "contextual_hedging", root / "veridicality_v2"]
    changed = 0
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.csv"):
            data = pd.read_csv(path)
            if "dimension" in data and "demographic_dimension" not in data:
                data.insert(data.columns.get_loc("dimension") + 1, "demographic_dimension", data["dimension"])
                data.to_csv(path, index=False)
                changed += 1
    print(f"Harmonized demographic_dimension in {changed} generated tables under {root}")


if __name__ == "__main__":
    main()
