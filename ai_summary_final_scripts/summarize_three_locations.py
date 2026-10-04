#!/usr/bin/env python3
"""Create compact machine-readable summaries for the three-location final audit.

This script does NOT replace the per-location inferential analyses. It gathers their
primary effect estimates, confidence intervals, corrected p-values, and pipeline
correlations into one directory so the final paper can be updated without manually
opening four different result trees for each city.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASE = Path("/scratch/victoria.estanislau/ai-summary")
LOCATIONS = {
    "dallas": {"version": "v1_dallas", "label": "Dallas"},
    "ny": {"version": "v2_ny", "label": "New York City"},
    "la": {"version": "v3_la", "label": "Los Angeles"},
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick(row: dict[str, Any], *names: str):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return np.nan


def dimension_row(analysis: str, location: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "location": location,
        "dimension": row.get("dimension"),
        "n": pick(row, "n", "n_outcomes"),
        "focal_group": pick(row, "minority_group", "focal_group"),
        "comparison_group": pick(row, "majority_group", "comparison_group"),
        "focal_mean": pick(row, "minority_mean", "mean_1"),
        "comparison_mean": pick(row, "majority_mean", "mean_2"),
        "mean_difference_focal_minus_comparison": pick(row, "mean_difference"),
        "median_difference": pick(row, "median_difference"),
        "ci_low": pick(row, "bootstrap_ci_low", "ci_low"),
        "ci_high": pick(row, "bootstrap_ci_high", "ci_high"),
        "rank_biserial": pick(row, "rank_biserial"),
        "p_raw": pick(row, "p_permutation_raw", "p_raw", "p_value"),
        "p_holm": pick(row, "p_permutation_holm", "p_holm"),
        "significant_holm": pick(row, "significant_permutation_holm", "significant_holm"),
    }


def aggregate_row(analysis: str, location: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "location": location,
        "n": pick(row, "n", "n_outcomes"),
        "coverage_threshold": pick(row, "coverage_threshold"),
        "focal_mean": pick(row, "minority_mean", "mean_1"),
        "comparison_mean": pick(row, "majority_mean", "mean_2"),
        "mean_difference_focal_minus_comparison": pick(row, "mean_difference"),
        "median_difference": pick(row, "median_difference"),
        "ci_low": pick(row, "bootstrap_ci_low", "ci_low"),
        "ci_high": pick(row, "bootstrap_ci_high", "ci_high"),
        "rank_biserial": pick(row, "rank_biserial"),
        "p_raw": pick(row, "p_permutation_raw", "p_raw", "p_value"),
        "p_holm": pick(row, "p_permutation_holm", "p_holm"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    base = args.base_dir.resolve()
    results = base / "results"
    out = results / "three_location_summary"
    out.mkdir(parents=True, exist_ok=True)

    dim_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    raw_bundle: dict[str, Any] = {"locations": {}}

    for slug, info in LOCATIONS.items():
        label = info["label"]

        link = load_json(results / f"link_count_analysis_{slug}" / "all_results.json")
        overlap = load_json(results / f"source_overlap_analysis_{slug}" / "all_results.json")
        semantic = load_json(results / f"semantic_embedding_analysis_{slug}" / "all_results.json")
        evidence = load_json(results / f"evidence_synthesis_analysis_{slug}_v2" / "all_results_v2.json")

        raw_bundle["locations"][slug] = {
            "label": label,
            "collection_version": info["version"],
            "link_count": link,
            "source_overlap": overlap,
            "semantic": semantic,
            "evidence": evidence,
        }

        for row in link["statistical_tests"]["minority_vs_majority_by_dimension"]:
            dim_rows.append(dimension_row("citation_count", label, row))
        agg_rows.append(aggregate_row("citation_count", label, link["statistical_tests"]["aggregate_minority_vs_majority"]))

        for row in overlap["minority_vs_majority"]["url_jaccard_distance"]:
            dim_rows.append(dimension_row("source_url_jaccard_distance", label, row))
        for row in overlap["minority_vs_majority"].get("aggregate", []):
            metric = row.get("metric") or row.get("metric_name") or ""
            if "url" in str(metric).lower() and "jaccard" in str(metric).lower():
                agg_rows.append(aggregate_row("source_url_jaccard_distance", label, row))

        for row in semantic["primary_analysis"]["dimension_tests"]:
            dim_rows.append(dimension_row("answer_semantic_displacement", label, row))
        agg_rows.append(aggregate_row("answer_semantic_displacement", label, semantic["primary_analysis"]["aggregate_minority_vs_majority"]))

        for row in evidence["reference_analysis_50pct"]["dimension_tests"]:
            dim_rows.append(dimension_row("evidence_semantic_gap", label, row))
        agg50 = [
            row for row in evidence["coverage_sensitivity"]["aggregate_outcome_blocked"]
            if np.isclose(float(row.get("coverage_threshold", np.nan)), 0.5)
        ]
        if agg50:
            agg_rows.append(aggregate_row("evidence_semantic_gap", label, agg50[0]))

        for row in evidence["pipeline"]["blocked_correlations"]:
            corr_rows.append({
                "location": label,
                "metric_x": row.get("metric_x"),
                "metric_y": row.get("metric_y"),
                "n_observations": row.get("n_observations"),
                "n_blocks": row.get("n_blocks"),
                "blocked_rank_correlation": row.get("blocked_rank_correlation"),
                "ci_low": row.get("cluster_bootstrap_ci_low"),
                "ci_high": row.get("cluster_bootstrap_ci_high"),
                "p_raw": row.get("p_blocked_permutation_raw"),
                "p_holm": row.get("p_blocked_permutation_holm"),
                "significant_holm": row.get("significant_holm"),
            })

        cq = evidence["collection_quality"]
        quality_rows.append({
            "location": label,
            "n_queries": cq.get("n_queries"),
            "mean_fetch_coverage": cq.get("mean_fetch_coverage"),
            "median_fetch_coverage": cq.get("median_fetch_coverage"),
            "reference_eligible_queries": cq.get("reference_eligible_queries"),
            "n_usable_source_texts": cq.get("n_usable_source_texts"),
            "n_source_chunks": cq.get("n_source_chunks"),
            "n_truncated_sources": cq.get("n_truncated_sources"),
        })

    dim = pd.DataFrame(dim_rows)
    agg = pd.DataFrame(agg_rows)
    corrs = pd.DataFrame(corr_rows)
    quality = pd.DataFrame(quality_rows)

    dim.to_csv(out / "primary_dimension_effects_all_locations.csv", index=False)
    agg.to_csv(out / "primary_aggregate_effects_all_locations.csv", index=False)
    corrs.to_csv(out / "pipeline_correlations_all_locations.csv", index=False)
    quality.to_csv(out / "evidence_collection_quality_all_locations.csv", index=False)

    # Descriptive replication summary only; no new hypothesis test here.
    rep_rows = []
    for (analysis, dimension), g in dim.groupby(["analysis", "dimension"], dropna=False):
        vals = pd.to_numeric(g["mean_difference_focal_minus_comparison"], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        n_pos = int((vals > 0).sum())
        n_neg = int((vals < 0).sum())
        rep_rows.append({
            "analysis": analysis,
            "dimension": dimension,
            "n_locations_available": int(len(vals)),
            "n_positive_effects": n_pos,
            "n_negative_effects": n_neg,
            "direction_consistent_all_locations": bool(n_pos == len(vals) or n_neg == len(vals)),
            "mean_effect_across_locations_descriptive": float(vals.mean()),
        })
    pd.DataFrame(rep_rows).to_csv(out / "replication_direction_summary.csv", index=False)

    with (out / "all_locations_raw_bundle.json").open("w", encoding="utf-8") as f:
        json.dump(raw_bundle, f, indent=2, ensure_ascii=False)

    lines = [
        "Three-location final audit summary",
        "==================================",
        "",
        "Primary inference remains the per-location matched-outcome analysis.",
        "This directory only gathers those results and reports descriptive replication directions.",
        "",
        "Files:",
        "- primary_dimension_effects_all_locations.csv",
        "- primary_aggregate_effects_all_locations.csv",
        "- pipeline_correlations_all_locations.csv",
        "- evidence_collection_quality_all_locations.csv",
        "- replication_direction_summary.csv",
        "- all_locations_raw_bundle.json",
    ]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Summary written to: {out}")
    print(f"Dimension rows: {len(dim)}")
    print(f"Aggregate rows: {len(agg)}")
    print(f"Pipeline correlation rows: {len(corrs)}")


if __name__ == "__main__":
    main()
