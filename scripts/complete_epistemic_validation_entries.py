"""Fill proposed MegaVeridicality row IDs in generated audit sheets."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def complete(source: Path, output: Path) -> None:
    resource = pd.read_csv(source / "veridicality_v2" / "megaveridicality_resource_entries.csv")
    key_path = output / "validation" / "veridicality_key_restricted.csv"
    sheet_path = output / "validation" / "veridicality_annotation_sheet.csv"
    key = pd.read_csv(key_path).fillna("")
    sheet = pd.read_csv(sheet_path).fillna("")
    mapping = resource[["predicate_lemma", "frame", "polarity", "resource_entry_id"]].rename(columns={"frame": "syntactic_frame"})
    if mapping.duplicated(["predicate_lemma", "syntactic_frame", "polarity"]).any():
        raise ValueError("MegaVeridicality resource tuples are not unique; cannot assign an unambiguous row ID")
    enriched = key.merge(mapping, on=["predicate_lemma", "syntactic_frame", "polarity"], how="left", validate="many_to_one")
    strict = enriched.match_status.eq("matched_strict")
    if enriched.loc[strict, "resource_entry_id"].isna().any():
        raise ValueError("At least one strict match has no exact resource entry")
    enriched["matched_resource_entry"] = enriched.matched_resource_entry.astype("string")
    enriched.loc[strict, "matched_resource_entry"] = enriched.loc[strict, "resource_entry_id"].astype("string").to_numpy()
    enriched = enriched.drop(columns=["resource_entry_id"])
    enriched.to_csv(key_path, index=False)
    ids = enriched.set_index("annotation_id").matched_resource_entry
    if "proposed_resource_entry" not in sheet:
        sheet["proposed_resource_entry"] = ""
    sheet["proposed_resource_entry"] = sheet.annotation_id.map(ids).fillna("")
    sheet.to_csv(sheet_path, index=False)
    print(f"Completed {int(strict.sum())} strict resource-entry IDs in {key_path.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete exact resource IDs in veridicality validation sheets.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results" / "epistemic_commitment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "epistemic_commitment_v2")
    args = parser.parse_args()
    complete(args.source_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
