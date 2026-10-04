"""Generate the one-new-outcome-per-domain robustness collection.

The script selects the highest-ranked outcome in each domain that was not part
of the original Top 3. It then creates the same manual Google AIO collection
template for all 13 groups and all three locations. Existing TXT files are
never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from generate_queries import COLLECTION_DIRNAME, GROUPS, generate_collection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGREEMENT = PROJECT_ROOT / "artifacts/annotation_results/annotator_agreement_full.csv"
DEFAULT_TOP3 = PROJECT_ROOT / "artifacts/annotation_results/selected_top3_outcomes.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "annotations/robustness"
VERSIONS = ("v1_dallas", "v2_ny", "v3_la")
EXPECTED_DOMAINS = 7

RANK_COLUMNS = ["mean_score", "min_score", "abs_difference", "Outcome"]
RANK_ASCENDING = [False, False, True, True]


def select_robustness_outcomes(agreement_csv: Path, top3_csv: Path) -> pd.DataFrame:
    agreement = pd.read_csv(agreement_csv)
    top3 = pd.read_csv(top3_csv)
    required = {"Domain", "Outcome", "mean_score", "min_score", "abs_difference"}
    missing = required - set(agreement.columns)
    if missing:
        raise ValueError(f"Missing agreement columns: {sorted(missing)}")
    if not {"Domain", "Outcome"}.issubset(top3.columns):
        raise ValueError("Top-3 file must contain Domain and Outcome")
    if agreement.duplicated(["Domain", "Outcome"]).any():
        raise ValueError("Agreement table has duplicate Domain/Outcome rows")
    if top3.duplicated(["Domain", "Outcome"]).any():
        raise ValueError("Top-3 table has duplicate Domain/Outcome rows")

    domain_order = agreement["Domain"].drop_duplicates().tolist()
    if len(domain_order) != EXPECTED_DOMAINS:
        raise ValueError(f"Expected {EXPECTED_DOMAINS} domains, found {len(domain_order)}")
    top_counts = top3.groupby("Domain").size().reindex(domain_order, fill_value=0)
    if not top_counts.eq(3).all():
        raise ValueError(f"Original selection must have exactly three outcomes per domain: {top_counts.to_dict()}")

    used = set(map(tuple, top3[["Domain", "Outcome"]].itertuples(index=False, name=None)))
    candidates = agreement.loc[
        [tuple(row) not in used for row in agreement[["Domain", "Outcome"]].itertuples(index=False, name=None)]
    ].copy()
    candidates["_domain_order"] = pd.Categorical(candidates["Domain"], domain_order, ordered=True)
    candidates = candidates.sort_values(
        ["_domain_order", *RANK_COLUMNS], ascending=[True, *RANK_ASCENDING], kind="stable"
    )
    selected = candidates.groupby("Domain", sort=False, observed=True).head(1).copy()
    selected = selected.sort_values("_domain_order").drop(columns="_domain_order").reset_index(drop=True)
    if len(selected) != EXPECTED_DOMAINS or selected["Domain"].nunique() != EXPECTED_DOMAINS:
        raise ValueError("Could not select exactly one unused outcome per domain")
    if any(tuple(row) in used for row in selected[["Domain", "Outcome"]].itertuples(index=False, name=None)):
        raise AssertionError("Robustness outcome overlaps the original Top 3")

    selected.insert(2, "robustness_rank", 4)
    selected["selection_rule"] = (
        "highest-ranked outcome outside original Top 3: mean_score desc, "
        "min_score desc, abs_difference asc, Outcome asc"
    )
    return selected


def check_existing_manifest(collection_dir: Path, selected: pd.DataFrame) -> None:
    manifest_path = collection_dir / "query_manifest.csv"
    if not manifest_path.exists():
        return
    existing = pd.read_csv(manifest_path)
    expected_pairs = set(map(tuple, selected[["Domain", "Outcome"]].itertuples(index=False, name=None)))
    actual_pairs = set(map(tuple, existing[["domain", "outcome"]].drop_duplicates().itertuples(index=False, name=None)))
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            f"Existing robustness manifest has a different outcome selection: {manifest_path}. "
            "Use a new --output-root; no collected files were changed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement-csv", type=Path, default=DEFAULT_AGREEMENT)
    parser.add_argument("--top3-csv", type=Path, default=DEFAULT_TOP3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Show the selection and counts without writing files")
    args = parser.parse_args()

    selected = select_robustness_outcomes(args.agreement_csv.resolve(), args.top3_csv.resolve())
    total_per_location = len(selected) * len(GROUPS)
    print("ROBUSTNESS OUTCOME SELECTION")
    print(selected[["Domain", "Outcome", "mean_score", "min_score", "abs_difference"]].to_string(index=False))
    print(f"\nOutcomes: {len(selected)}")
    print(f"Groups: {len(GROUPS)}")
    print(f"Files per location: {total_per_location}")
    print(f"Total files across locations: {total_per_location * len(VERSIONS)}")
    if args.dry_run:
        return

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_root / "selected_robustness_outcomes.csv", index=False)
    run_summary = {
        "collection_set": "one-new-outcome-per-domain robustness set",
        "selection_rule": selected.selection_rule.iloc[0],
        "n_outcomes": len(selected),
        "n_groups": len(GROUPS),
        "n_locations": len(VERSIONS),
        "files_per_location": total_per_location,
        "locations": list(VERSIONS),
        "existing_txt_policy": "preserve; never overwrite",
    }
    (output_root / "generation_manifest.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )

    generation_input = selected[["Domain", "Outcome"]]
    for version in VERSIONS:
        collection_dir = output_root / version / COLLECTION_DIRNAME
        check_existing_manifest(collection_dir, selected)
        created, preserved = generate_collection(generation_input, collection_dir)
        print(f"{version}: {created} created, {preserved} preserved — {collection_dir}")


if __name__ == "__main__":
    main()
