"""Reproducible entry point for epistemic-commitment v2 tables and figures."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def stable_python_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], environment)


def same_pair_predicate_changes(output: Path) -> None:
    table = output / "robustness" / "leave_one_predicate_out.csv"
    result = pd.read_csv(table)
    if result.empty:
        return
    paired = pd.read_csv(output / "tables" / "paired_outcome_metrics.csv")
    scored = pd.read_csv(output / "tables" / "strict_predicate_occurrences.csv")
    total = scored.groupby("response_id").veridicality_score.agg(["sum", "count"])
    lookup = {}
    for predicate in result.omitted_predicate.dropna().unique():
        removed = scored.loc[scored.predicate_lemma.eq(predicate)].groupby("response_id").veridicality_score.agg(["sum", "count"])
        n = total["count"] - removed["count"].reindex(total.index, fill_value=0)
        remaining = (total["sum"] - removed["sum"].reindex(total.index, fill_value=0)) / n.where(n.gt(0))
        for dimension in paired.dimension.unique():
            part = paired.loc[paired.dimension.eq(dimension)]
            for left, right in (("Minority", "People"), ("Majority", "People"), ("Minority", "Majority")):
                l = left.lower(); r = right.lower()
                valid = part[f"response_id_{l}"].map(remaining).notna() & part[f"response_id_{r}"].map(remaining).notna()
                baseline = (part.loc[valid, f"mean_veridicality_{l}"] - part.loc[valid, f"mean_veridicality_{r}"]).mean()
                lookup[(predicate, dimension, f"{left} - {right}")] = float(baseline) if valid.any() else np.nan
    result["baseline_same_pairs"] = [lookup.get((row.omitted_predicate, row.dimension, row.comparison), np.nan) for row in result.itertuples()]
    result["effect_change_same_pairs"] = result.estimate - result.baseline_same_pairs
    result.to_csv(table, index=False)


def main() -> None:
    stable_python_hash_seed()
    parser = argparse.ArgumentParser(description="Run outcome-adjusted epistemic commitment analyses and export PNG figures.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results" / "epistemic_commitment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "epistemic_commitment_v2")
    args = parser.parse_args()
    source = args.source_dir.resolve(); output = args.output_dir.resolve()

    import epistemic_commitment_v2_stats as stats
    import epistemic_commitment_v2_validation as validation
    import epistemic_commitment_v2_figures as figures
    import finalize_epistemic_commitment_v2 as finalize

    stats.validation_tables = validation.validation_tables
    stats.main()
    same_pair_predicate_changes(output)
    figures.generate(output)
    finalize.finalize(source, output)


if __name__ == "__main__":
    main()
