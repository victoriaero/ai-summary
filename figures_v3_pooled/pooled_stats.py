#!/usr/bin/env python3
from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

BASE_DEFAULT = Path('/scratch/victoria.estanislau/ai-summary')
OUTPUT_DEFAULT = BASE_DEFAULT / 'figures_three_locations_pooled'

LOCATIONS = {
    'dallas': 'Dallas',
    'ny': 'New York',
    'la': 'Los Angeles',
}

DIMENSIONS = [
    'Race',
    'Ethnicity',
    'Gender',
    'Disability',
    'Sexual Orientation',
    'Gender Identity',
]

RANDOM_SEED = 42
N_PERMUTATIONS = 200_000
N_BOOTSTRAP = 20_000
EXACT_SIGNFLIP_MAX_N = 18
N_BLOCK_PERMUTATIONS = 20_000
N_BLOCK_BOOTSTRAP = 5_000
ALPHA = 0.05


def stable_seed(label: str) -> int:
    return (RANDOM_SEED + zlib.crc32(str(label).encode('utf-8'))) % (2**32 - 1)


def holm_adjust(pvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, (m - i) * p[idx])
        running = max(running, val)
        adj_sorted[i] = running
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    reject = adj < ALPHA
    return adj, reject


def bootstrap_mean_ci(values, label: str, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed('boot::' + label))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def signflip_p(values, label: str, n_perm: int = N_PERMUTATIONS) -> tuple[float, str]:
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    d = d[~np.isclose(d, 0.0)]
    n = len(d)
    if n == 0:
        return 1.0, 'all_zero'
    obs = abs(float(d.mean()))

    if n <= EXACT_SIGNFLIP_MAX_N:
        total = 2 ** n
        extreme = 0
        powers = (1 << np.arange(n, dtype=np.uint64))
        batch = 10000
        for start in range(0, total, batch):
            stop = min(start + batch, total)
            ints = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = (ints & powers) > 0
            signs = bits.astype(float) * 2.0 - 1.0
            stats = np.abs((signs * d).mean(axis=1))
            extreme += int(np.sum(stats >= obs - 1e-15))
        return float(extreme / total), f'exact_signflip_2^{n}'

    rng = np.random.default_rng(stable_seed('perm::' + label))
    extreme = 0
    done = 0
    batch = 10000
    while done < n_perm:
        b = min(batch, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(b, n))
        stats = np.abs((signs * d).mean(axis=1))
        extreme += int(np.sum(stats >= obs - 1e-15))
        done += b
    return float((extreme + 1) / (n_perm + 1)), f'monte_carlo_signflip_{n_perm}'


def paired_rank_biserial(values) -> float:
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    d = d[~np.isclose(d, 0.0)]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method='average')
    pos = float(ranks[d > 0].sum())
    neg = float(ranks[d < 0].sum())
    return (pos - neg) / (pos + neg) if (pos + neg) else 0.0


def test_differences(values, label: str, primary: str = 'signflip') -> dict:
    d = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(float)
    lo, hi = bootstrap_mean_ci(d, label)
    p_perm, method = signflip_p(d, label)
    if len(d) == 0 or np.allclose(d, 0):
        W, p_w = 0.0, 1.0
    else:
        w = wilcoxon(d, alternative='two-sided')
        W, p_w = float(w.statistic), float(w.pvalue)
    p_primary = p_w if primary == 'wilcoxon' else p_perm
    return {
        'n_outcomes': int(len(d)),
        'mean_difference': float(np.mean(d)) if len(d) else np.nan,
        'median_difference': float(np.median(d)) if len(d) else np.nan,
        'bootstrap_ci_low': lo,
        'bootstrap_ci_high': hi,
        'rank_biserial': paired_rank_biserial(d),
        'p_permutation_raw': float(p_perm),
        'permutation_method': method,
        'wilcoxon_W': W,
        'p_wilcoxon_raw': float(p_w),
        'p_primary_raw': float(p_primary),
        'primary_test': primary,
    }


def result_path(base: Path, slug: str, analysis: str, filename: str) -> Path:
    if analysis == 'evidence':
        return base / 'results' / f'evidence_synthesis_analysis_{slug}_v2' / filename
    return base / 'results' / f'{analysis}_{slug}' / filename


def read_three_location_csv(base: Path, analysis: str, filename: str) -> pd.DataFrame:
    frames = []
    for slug in LOCATIONS:
        path = result_path(base, slug, analysis, filename)
        if not path.exists():
            raise FileNotFoundError(f'Missing required result file: {path}')
        df = pd.read_csv(path)
        df['location'] = slug
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def paired_location_effects(
    df: pd.DataFrame,
    value: str,
    *,
    eligibility: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    d = df.copy()
    if eligibility is not None:
        d = d.loc[eligibility(d)].copy()
    d = d.loc[d['condition'].isin(['minority', 'majority'])].copy()
    d[value] = pd.to_numeric(d[value], errors='coerce')
    d = d.dropna(subset=[value, 'dimension', 'outcome_id', 'location', 'condition'])
    piv = d.pivot_table(
        index=['location', 'dimension', 'outcome_id'],
        columns='condition',
        values=value,
        aggfunc='mean',
    ).dropna(subset=['minority', 'majority'])
    piv['difference'] = piv['minority'] - piv['majority']
    return piv.reset_index()


def pooled_dimension_analysis(
    df: pd.DataFrame,
    value: str,
    analysis_label: str,
    *,
    eligibility: Callable[[pd.DataFrame], pd.Series] | None = None,
    primary: str = 'signflip',
    minimum_dimensions_for_aggregate: int = 2,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    loc = paired_location_effects(df, value, eligibility=eligibility)

    # Crucial rule: repetitions across cities are averaged WITHIN outcome first.
    pooled = (
        loc.groupby(['dimension', 'outcome_id'], as_index=False)
        .agg(
            difference=('difference', 'mean'),
            focal_mean=('minority', 'mean'),
            comparison_mean=('majority', 'mean'),
            n_locations=('location', 'nunique'),
        )
    )

    rows = []
    for dimension in DIMENSIONS:
        g = pooled.loc[pooled['dimension'] == dimension]
        res = test_differences(g['difference'], f'{analysis_label}::{dimension}', primary=primary)
        res.update({
            'dimension': dimension,
            'focal_mean': float(g['focal_mean'].mean()) if len(g) else np.nan,
            'comparison_mean': float(g['comparison_mean'].mean()) if len(g) else np.nan,
            'mean_locations_per_outcome': float(g['n_locations'].mean()) if len(g) else np.nan,
            'min_locations_per_outcome': int(g['n_locations'].min()) if len(g) else 0,
            'max_locations_per_outcome': int(g['n_locations'].max()) if len(g) else 0,
        })
        rows.append(res)

    tests = pd.DataFrame(rows)
    pcol = 'p_wilcoxon_raw' if primary == 'wilcoxon' else 'p_permutation_raw'
    padj, reject = holm_adjust(tests[pcol].to_numpy(float))
    tests['p_holm'] = padj
    tests['significant_holm'] = reject
    tests['p_primary_raw'] = tests[pcol]

    # Prespecified overall summary: average dimensions within outcome after
    # city repetitions have already been averaged.
    by_outcome = (
        pooled.groupby('outcome_id', as_index=False)
        .agg(
            difference=('difference', 'mean'),
            focal_mean=('focal_mean', 'mean'),
            comparison_mean=('comparison_mean', 'mean'),
            n_dimensions=('dimension', 'nunique'),
            mean_locations=('n_locations', 'mean'),
        )
    )
    by_outcome = by_outcome.loc[
        by_outcome['n_dimensions'] >= minimum_dimensions_for_aggregate
    ].copy()
    aggregate = test_differences(
        by_outcome['difference'], f'{analysis_label}::aggregate', primary=primary
    )
    aggregate.update({
        'dimension': 'Overall',
        'focal_mean': float(by_outcome['focal_mean'].mean()) if len(by_outcome) else np.nan,
        'comparison_mean': float(by_outcome['comparison_mean'].mean()) if len(by_outcome) else np.nan,
        'n_dimensions_mean': float(by_outcome['n_dimensions'].mean()) if len(by_outcome) else np.nan,
        'mean_locations_per_outcome': float(by_outcome['mean_locations'].mean()) if len(by_outcome) else np.nan,
    })
    return tests, aggregate, loc


def evidence_eligible(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df['fetch_coverage'], errors='coerce').ge(0.50)
        & pd.to_numeric(df['n_available_sources'], errors='coerce').ge(3)
        & pd.to_numeric(df['evidence_semantic_gap_source_balanced'], errors='coerce').notna()
    )


def pooled_group_descriptives(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    links = read_three_location_csv(base, 'link_count_analysis', 'clean_aio_link_data.csv')
    overlap = read_three_location_csv(base, 'source_overlap_analysis', 'source_overlap_vs_generic.csv')

    # Equal city weighting. Since the design is balanced, this is also the
    # pooled mean when all 21 outcomes are present in each city.
    link_city = (
        links.groupby(['location', 'group'], as_index=False)['n_links'].mean()
    )
    link_group = (
        link_city.groupby('group', as_index=False)['n_links'].mean()
        .rename(columns={'n_links': 'mean_links'})
    )

    overlap['url_overlap'] = 1 - pd.to_numeric(overlap['url_jaccard_distance'], errors='coerce')
    ov_city = (
        overlap.groupby(['location', 'group'], as_index=False)['url_overlap'].mean()
    )
    ov_group = (
        ov_city.groupby('group', as_index=False)['url_overlap'].mean()
        .rename(columns={'url_overlap': 'mean_url_overlap'})
    )
    return link_group, ov_group


def pooled_mechanism(base: Path, *, evidence_only: bool = False) -> pd.DataFrame:
    frames = []
    for slug in LOCATIONS:
        path = result_path(base, slug, 'evidence', 'mechanism_map_v2.csv')
        if not path.exists():
            raise FileNotFoundError(f'Missing required result file: {path}')
        df = pd.read_csv(path)
        df['location'] = slug
        if evidence_only:
            df = df.loc[evidence_eligible(df)].copy()
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    id_cols = ['group', 'condition', 'dimension', 'domain', 'outcome', 'outcome_id']
    numeric_candidates = [
        'fetch_coverage', 'n_available_sources',
        'url_jaccard_distance', 'domain_jaccard_distance',
        'url_generic_coverage', 'domain_generic_coverage',
        'dense_distance_subject_normalized',
        'generic_coverage_distance', 'group_novelty_distance',
        'symmetric_sentence_distance',
        'evidence_alignment_source_balanced',
        'evidence_semantic_gap_source_balanced',
        'answer_support_alignment', 'answer_support_semantic_gap',
    ]
    nums = [c for c in numeric_candidates if c in all_df.columns]
    agg = {c: 'mean' for c in nums}
    agg['location'] = 'nunique'
    pooled = (
        all_df.groupby(id_cols, as_index=False)
        .agg(agg)
        .rename(columns={'location': 'n_locations'})
    )
    return pooled


def _centered_rank_arrays(df: pd.DataFrame, x: str, y: str, block: str):
    parts = []
    for block_value, g in df.groupby(block, sort=False):
        if len(g) < 3:
            continue
        xv = pd.to_numeric(g[x], errors='coerce').to_numpy(float)
        yv = pd.to_numeric(g[y], errors='coerce').to_numpy(float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]
        if len(xv) < 3:
            continue
        xr = rankdata(xv, method='average'); xr = xr - xr.mean()
        yr = rankdata(yv, method='average'); yr = yr - yr.mean()
        parts.append((block_value, xr.astype(float), yr.astype(float)))
    return parts


def blocked_rank_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    block: str = 'outcome_id',
    label: str = 'blocked',
    n_perm: int = N_BLOCK_PERMUTATIONS,
    n_boot: int = N_BLOCK_BOOTSTRAP,
) -> dict:
    temp = df[[block, x, y]].dropna().copy()
    parts = _centered_rank_arrays(temp, x, y, block)
    if len(parts) < 3:
        return {}

    nums = np.array([float(np.dot(xr, yr)) for _, xr, yr in parts])
    x2 = np.array([float(np.dot(xr, xr)) for _, xr, yr in parts])
    y2 = np.array([float(np.dot(yr, yr)) for _, xr, yr in parts])
    observed = float(nums.sum() / math.sqrt(x2.sum() * y2.sum()))

    rng = np.random.default_rng(stable_seed('blocked::' + label))
    denom = math.sqrt(x2.sum() * y2.sum())
    extreme = 0
    done = 0
    batch_size = 1000
    while done < n_perm:
        b = min(batch_size, n_perm - done)
        perm_num = np.zeros(b, dtype=float)
        for _, xr, yr in parts:
            # Random permutations of within-block rank vector.
            order = np.argsort(rng.random((b, len(yr))), axis=1)
            yp = yr[order]
            perm_num += yp @ xr
        stats = perm_num / denom
        extreme += int(np.sum(np.abs(stats) >= abs(observed) - 1e-15))
        done += b
    p = float((extreme + 1) / (n_perm + 1))

    # Cluster bootstrap across outcomes. Each outcome is resampled as a whole.
    idx = rng.integers(0, len(parts), size=(n_boot, len(parts)))
    boot_num = nums[idx].sum(axis=1)
    boot_x2 = x2[idx].sum(axis=1)
    boot_y2 = y2[idx].sum(axis=1)
    boot = boot_num / np.sqrt(boot_x2 * boot_y2)
    lo, hi = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5])

    return {
        'metric_x': x,
        'metric_y': y,
        'block': block,
        'n_observations': int(sum(len(xr) for _, xr, yr in parts)),
        'n_blocks': int(len(parts)),
        'blocked_rank_correlation': observed,
        'cluster_bootstrap_ci_low': float(lo),
        'cluster_bootstrap_ci_high': float(hi),
        'p_blocked_permutation_raw': p,
        'n_permutations': int(n_perm),
        'n_bootstrap': int(n_boot),
    }


def pooled_pipeline_correlations(base: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    all_pooled = pooled_mechanism(base, evidence_only=False)
    evidence_pooled = pooled_mechanism(base, evidence_only=True)

    pairs = [
        ('url_jaccard_distance', 'dense_distance_subject_normalized'),
        ('domain_jaccard_distance', 'dense_distance_subject_normalized'),
        ('url_jaccard_distance', 'evidence_semantic_gap_source_balanced'),
        ('domain_jaccard_distance', 'evidence_semantic_gap_source_balanced'),
        ('evidence_semantic_gap_source_balanced', 'dense_distance_subject_normalized'),
        ('answer_support_semantic_gap', 'dense_distance_subject_normalized'),
    ]

    rows = []
    for x, y in pairs:
        uses_evidence = (
            'evidence_' in x or 'evidence_' in y
            or 'answer_support' in x or 'answer_support' in y
        )
        df = evidence_pooled if uses_evidence else all_pooled
        if x not in df.columns or y not in df.columns:
            continue
        res = blocked_rank_correlation(df, x, y, label=f'{x}::{y}')
        if res:
            rows.append(res)

    corr = pd.DataFrame(rows)
    if len(corr):
        adj, rej = holm_adjust(corr['p_blocked_permutation_raw'].to_numpy(float))
        corr['p_blocked_permutation_holm'] = adj
        corr['significant_holm'] = rej
    return corr, {'all': all_pooled, 'evidence': evidence_pooled}


def ensure_pipeline_cache(base: Path, output_dir: Path, force: bool = False):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    corr_path = output_dir / 'pooled_pipeline_correlations.csv'
    all_path = output_dir / 'pooled_mechanism_all.csv'
    ev_path = output_dir / 'pooled_mechanism_evidence.csv'
    if not force and corr_path.exists() and all_path.exists() and ev_path.exists():
        return pd.read_csv(corr_path), {
            'all': pd.read_csv(all_path),
            'evidence': pd.read_csv(ev_path),
        }
    corr, maps = pooled_pipeline_correlations(base)
    corr.to_csv(corr_path, index=False)
    maps['all'].to_csv(all_path, index=False)
    maps['evidence'].to_csv(ev_path, index=False)
    return corr, maps
