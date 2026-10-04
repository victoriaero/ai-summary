#!/usr/bin/env python3
"""Secondary cross-location pooled analysis.

IMPORTANT: the paper's primary analysis should remain per-location. This script
creates a secondary pooled estimate by FIRST computing each focal-comparison
paired effect within outcome x location, THEN averaging the location replicates
for the same outcome, and ONLY THEN performing inference across outcomes.
This avoids treating the three geographic repetitions as independent outcomes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

DEFAULT_BASE = Path("/scratch/victoria.estanislau/ai-summary")
LOCATIONS = {
    "dallas": "v1_dallas",
    "ny": "v2_ny",
    "la": "v3_la",
}
RANDOM_SEED = 42
N_PERMUTATIONS = 200_000
N_BOOTSTRAP = 20_000
EXACT_MAX_N = 18
ALPHA = 0.05


def stable_seed(label: str) -> int:
    return (RANDOM_SEED + zlib.crc32(str(label).encode("utf-8"))) % (2**32 - 1)


def signflip_p(diff: np.ndarray, label: str) -> tuple[float, str]:
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    diff = diff[~np.isclose(diff, 0.0)]
    n = len(diff)
    if n == 0:
        return 1.0, "all_zero"
    obs = abs(float(diff.mean()))
    if n <= EXACT_MAX_N:
        vals = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            vals.append(abs(float(np.mean(diff * np.asarray(signs)))))
        p = float(np.mean(np.asarray(vals) >= obs - 1e-15))
        return p, f"exact_signflip_2^{n}"
    rng = np.random.default_rng(stable_seed("perm::" + label))
    hits = 0
    for _ in range(N_PERMUTATIONS):
        signs = rng.choice((-1.0, 1.0), size=n)
        hits += abs(float(np.mean(diff * signs))) >= obs - 1e-15
    return float((hits + 1) / (N_PERMUTATIONS + 1)), f"monte_carlo_signflip_{N_PERMUTATIONS}"


def bootstrap_ci(diff: np.ndarray, label: str) -> tuple[float, float]:
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed("boot::" + label))
    idx = rng.integers(0, len(diff), size=(N_BOOTSTRAP, len(diff)))
    means = diff[idx].mean(axis=1)
    return tuple(map(float, np.percentile(means, [2.5, 97.5])))


def paired_rank_biserial(diff: np.ndarray) -> float:
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    diff = diff[~np.isclose(diff, 0.0)]
    if len(diff) == 0:
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    pos = float(ranks[diff > 0].sum())
    neg = float(ranks[diff < 0].sum())
    denom = pos + neg
    return (pos - neg) / denom if denom else 0.0


def test_differences(diff: pd.Series, label: str) -> dict:
    vals = pd.to_numeric(diff, errors="coerce").dropna().to_numpy(float)
    ci_low, ci_high = bootstrap_ci(vals, label)
    p_perm, method = signflip_p(vals, label)
    if len(vals) == 0 or np.allclose(vals, 0):
        W, p_w = 0.0, 1.0
    else:
        w = wilcoxon(vals, alternative="two-sided")
        W, p_w = float(w.statistic), float(w.pvalue)
    return {
        "n_outcomes": int(len(vals)),
        "mean_difference": float(np.mean(vals)) if len(vals) else np.nan,
        "median_difference": float(np.median(vals)) if len(vals) else np.nan,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "rank_biserial": paired_rank_biserial(vals),
        "p_permutation_raw": p_perm,
        "permutation_method": method,
        "wilcoxon_W": W,
        "p_wilcoxon_raw": p_w,
    }


def paired_effects(df: pd.DataFrame, value: str, location: str, eligibility=None) -> pd.DataFrame:
    d = df.copy()
    if eligibility is not None:
        d = d.loc[eligibility(d)].copy()
    d = d.loc[d["condition"].isin(["minority", "majority"])].copy()
    piv = d.pivot_table(
        index=["dimension", "outcome_id"],
        columns="condition",
        values=value,
        aggfunc="mean",
    ).dropna(subset=["minority", "majority"])
    piv["difference"] = piv["minority"] - piv["majority"]
    piv = piv.reset_index()
    piv["location"] = location
    return piv[["location", "dimension", "outcome_id", "minority", "majority", "difference"]]


def load_effects(base: Path, analysis: str) -> pd.DataFrame:
    frames = []
    for slug in LOCATIONS:
        if analysis == "citation_count":
            path = base / "results" / f"link_count_analysis_{slug}" / "clean_aio_link_data.csv"
            df = pd.read_csv(path)
            frames.append(paired_effects(df, "n_links", slug))
        elif analysis == "source_url_jaccard_distance":
            path = base / "results" / f"source_overlap_analysis_{slug}" / "source_overlap_vs_generic.csv"
            df = pd.read_csv(path)
            frames.append(paired_effects(df, "url_jaccard_distance", slug))
        elif analysis == "answer_semantic_displacement":
            path = base / "results" / f"semantic_embedding_analysis_{slug}" / "group_vs_generic_semantic_metrics.csv"
            df = pd.read_csv(path)
            frames.append(paired_effects(df, "dense_distance_subject_normalized", slug))
        elif analysis == "evidence_semantic_gap":
            path = base / "results" / f"evidence_synthesis_analysis_{slug}_v2" / "query_evidence_alignment_metrics_v2.csv"
            df = pd.read_csv(path)
            def eligible(x):
                return (
                    (pd.to_numeric(x["fetch_coverage"], errors="coerce") >= 0.50)
                    & (pd.to_numeric(x["n_available_sources"], errors="coerce") >= 3)
                    & pd.to_numeric(x["evidence_semantic_gap_source_balanced"], errors="coerce").notna()
                )
            frames.append(paired_effects(df, "evidence_semantic_gap_source_balanced", slug, eligible))
        else:
            raise ValueError(analysis)
    return pd.concat(frames, ignore_index=True)


def analyze(base: Path, analysis: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    loc_effects = load_effects(base, analysis)
    # Location repetitions are averaged BEFORE inference; outcome remains the unit.
    pooled_outcome = (
        loc_effects.groupby(["dimension", "outcome_id"], as_index=False)
        .agg(
            difference=("difference", "mean"),
            focal_mean=("minority", "mean"),
            comparison_mean=("majority", "mean"),
            n_locations=("location", "nunique"),
        )
    )
    rows = []
    for dimension, g in pooled_outcome.groupby("dimension", sort=False):
        result = test_differences(g["difference"], f"{analysis}::{dimension}")
        result.update({
            "analysis": analysis,
            "dimension": dimension,
            "focal_mean": float(g["focal_mean"].mean()),
            "comparison_mean": float(g["comparison_mean"].mean()),
            "mean_locations_per_outcome": float(g["n_locations"].mean()),
            "min_locations_per_outcome": int(g["n_locations"].min()),
            "max_locations_per_outcome": int(g["n_locations"].max()),
        })
        rows.append(result)
    tests = pd.DataFrame(rows)
    if len(tests):
        rej, padj, _, _ = multipletests(tests["p_permutation_raw"], method="holm", alpha=ALPHA)
        tests["p_permutation_holm"] = padj
        tests["significant_holm"] = rej
        rejw, padjw, _, _ = multipletests(tests["p_wilcoxon_raw"], method="holm", alpha=ALPHA)
        tests["p_wilcoxon_holm"] = padjw
        tests["significant_wilcoxon_holm"] = rejw

    # Aggregate across dimensions within outcome, then infer across outcomes.
    by_outcome = (
        pooled_outcome.groupby("outcome_id", as_index=False)
        .agg(
            difference=("difference", "mean"),
            n_dimensions=("dimension", "nunique"),
            mean_locations=("n_locations", "mean"),
        )
    )
    # Align with the evidence-analysis convention: at least 2 contributing dimensions.
    by_outcome = by_outcome.loc[by_outcome["n_dimensions"] >= 2].copy()
    aggregate = test_differences(by_outcome["difference"], f"{analysis}::aggregate")
    aggregate.update({
        "analysis": analysis,
        "minimum_dimensions_required": 2,
        "mean_dimensions_per_outcome": float(by_outcome["n_dimensions"].mean()) if len(by_outcome) else np.nan,
        "mean_location_replicates_per_outcome": float(by_outcome["mean_locations"].mean()) if len(by_outcome) else np.nan,
    })
    return tests, aggregate, loc_effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    base = args.base_dir.resolve()
    out = base / "results" / "three_location_pooled_secondary"
    out.mkdir(parents=True, exist_ok=True)

    analyses = [
        "citation_count",
        "source_url_jaccard_distance",
        "answer_semantic_displacement",
        "evidence_semantic_gap",
    ]
    all_tests, aggregates = [], []
    metadata = {
        "status": "secondary_cross_location_analysis",
        "primary_analysis_note": "Primary inference should remain the separate per-location analyses.",
        "pooling_rule": "Compute focal-comparison effect within outcome x location; average location effects within outcome; infer across outcomes.",
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "holm_family": "six social dimensions within each metric",
    }
    for analysis in analyses:
        tests, aggregate, loc_effects = analyze(base, analysis)
        all_tests.append(tests)
        aggregates.append(aggregate)
        loc_effects.to_csv(out / f"{analysis}_location_outcome_effects.csv", index=False)

    all_tests_df = pd.concat(all_tests, ignore_index=True)
    aggregate_df = pd.DataFrame(aggregates)
    all_tests_df.to_csv(out / "pooled_dimension_tests.csv", index=False)
    aggregate_df.to_csv(out / "pooled_aggregate_tests.csv", index=False)
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Secondary pooled results written to: {out}")
    print("Primary paper inference should still use the per-location analyses.")


if __name__ == "__main__":
    main()
