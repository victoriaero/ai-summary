from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import analyze_veridicality_v2 as analysis


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=analysis.OUTPUT_DIR)
    parser.add_argument("--resource", type=Path, default=analysis.DEFAULT_MEGAVERIDICALITY_FILE)
    known, _ = parser.parse_known_args()

    analysis.main()
    out = known.output_dir.resolve()
    occurrences_path = out / "predicate_occurrences_v2.csv"
    occurrences = pd.read_csv(occurrences_path)
    resource = analysis.load_resource(known.resource.resolve())
    resource = resource[["resource_entry_id", "predicate_lemma", "frame", "polarity", "veridicalitynorm"]]
    resource = resource.rename(columns={"frame": "syntactic_frame", "veridicalitynorm": "resource_score"})
    matched = occurrences.match_status.eq("matched_strict")
    linked = occurrences.loc[matched].merge(
        resource,
        on=["predicate_lemma", "syntactic_frame", "polarity"],
        how="left",
        validate="many_to_one",
    )
    if linked.resource_entry_id.isna().any():
        raise ValueError("A strict match could not be linked to its exact MegaVeridicality row")

    # pandas inferred the initially empty column as float64; explicitly make it
    # textual before assigning resource IDs (required by newer pandas releases).
    occurrences["matched_resource_entry"] = occurrences["matched_resource_entry"].astype("string")
    occurrences.loc[matched, "matched_resource_entry"] = linked.resource_entry_id.astype("string").to_numpy()
    occurrences.to_csv(occurrences_path, index=False)


if __name__ == "__main__":
    main()
