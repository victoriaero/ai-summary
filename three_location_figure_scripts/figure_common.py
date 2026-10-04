#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DEFAULT = Path('/scratch/victoria.estanislau/ai-summary')
OUTPUT_DEFAULT = BASE_DEFAULT / 'figures_three_locations'

LOCATIONS = {
    'dallas': {
        'label': 'Dallas',
        'marker': 'o',
        'color': '#253142',
    },
    'ny': {
        'label': 'New York',
        'marker': 's',
        'color': '#0072B2',
    },
    'la': {
        'label': 'Los Angeles',
        'marker': '^',
        'color': '#009E73',
    },
}

DIMENSION_ORDER = [
    'Race',
    'Ethnicity',
    'Gender',
    'Disability',
    'Sexual Orientation',
    'Gender Identity',
]

# Paper palette
COL_FOCAL = '#D55E00'
COL_COMPARISON = '#0072B2'
COL_PATHWAY = '#009E73'
COL_SYNTHESIS = '#CC79A7'
COL_PURPLE = '#8E63B6'
COL_INK = '#253142'
COL_MUTED = '#667085'
COL_GRID = '#DDE3EA'
COL_LIGHT = '#F5F7FA'
COL_BG = '#FFFFFF'


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Missing required results file: {path}')
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def paths_for_location(base: Path, slug: str) -> dict[str, Path]:
    results = base / 'results'
    return {
        'link': results / f'link_count_analysis_{slug}' / 'all_results.json',
        'overlap': results / f'source_overlap_analysis_{slug}' / 'all_results.json',
        'semantic': results / f'semantic_embedding_analysis_{slug}' / 'all_results.json',
        'evidence': results / f'evidence_synthesis_analysis_{slug}_v2' / 'all_results_v2.json',
    }


def load_location_bundle(base: Path, slug: str) -> dict[str, dict[str, Any]]:
    paths = paths_for_location(base, slug)
    return {k: load_json(v) for k, v in paths.items()}


def sig_from_row(row: dict[str, Any], aggregate: bool = False) -> tuple[float, bool]:
    """Return the inferential p-value used in the figure and significance flag.

    Dimension-level contrasts use Holm-corrected p-values. Prespecified single
    aggregate tests use their raw p-value because there is only one aggregate
    test per metric/location.
    """
    if aggregate:
        for key in ('p_permutation_raw', 'p_value', 'p_raw'):
            if key in row and row[key] is not None:
                p = float(row[key])
                return p, bool(p < 0.05)
        return np.nan, False

    for key in ('p_permutation_holm', 'p_holm'):
        if key in row and row[key] is not None:
            p = float(row[key])
            return p, bool(p < 0.05)

    for key in ('significant_permutation_holm', 'significant_holm'):
        if key in row:
            return np.nan, bool(row[key])

    return np.nan, False


def effect_record(
    *,
    location_slug: str,
    location_label: str,
    metric: str,
    dimension: str,
    row: dict[str, Any],
    aggregate: bool = False,
    effect_key: str = 'mean_difference',
) -> dict[str, Any]:
    p, sig = sig_from_row(row, aggregate=aggregate)
    n = row.get('n', row.get('n_outcomes', np.nan))
    return {
        'location_slug': location_slug,
        'location': location_label,
        'metric': metric,
        'dimension': dimension,
        'n': n,
        'effect': float(row.get(effect_key, np.nan)),
        'ci_low': float(row.get('bootstrap_ci_low', np.nan)),
        'ci_high': float(row.get('bootstrap_ci_high', np.nan)),
        'p_display': p,
        'significant': sig,
        'aggregate': aggregate,
        'rank_biserial': row.get('rank_biserial', np.nan),
    }


def correlation_record(
    *,
    location_slug: str,
    location_label: str,
    metric: str,
    row: dict[str, Any],
    flip_sign: bool = False,
) -> dict[str, Any]:
    r = float(row['blocked_rank_correlation'])
    lo = float(row['cluster_bootstrap_ci_low'])
    hi = float(row['cluster_bootstrap_ci_high'])
    if flip_sign:
        r, lo, hi = -r, -hi, -lo
    p = float(row.get('p_blocked_permutation_holm', np.nan))
    return {
        'location_slug': location_slug,
        'location': location_label,
        'metric': metric,
        'n': int(row.get('n_observations', 0)),
        'n_blocks': int(row.get('n_blocks', 0)),
        'effect': r,
        'ci_low': lo,
        'ci_high': hi,
        'p_display': p,
        'significant': bool(p < 0.05),
    }


def make_location_legend(ax, *, anchor=(0.0, 1.15), ncol=3):
    handles = []
    for slug, info in LOCATIONS.items():
        handles.append(
            Line2D(
                [0], [0],
                marker=info['marker'],
                linestyle='None',
                markerfacecolor=info['color'],
                markeredgecolor=info['color'],
                markersize=7.5,
                label=info['label'],
            )
        )
    leg = ax.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=False,
        fontsize=8.7,
        handletextpad=0.45,
        columnspacing=1.2,
        borderaxespad=0,
    )
    return leg


def add_significance_legend(ax, *, anchor=(1.0, 1.15)):
    filled = Line2D(
        [0], [0], marker='o', linestyle='None',
        markerfacecolor=COL_INK, markeredgecolor=COL_INK,
        markersize=6.5, label='significant',
    )
    openm = Line2D(
        [0], [0], marker='o', linestyle='None',
        markerfacecolor='white', markeredgecolor=COL_INK,
        markeredgewidth=1.4, markersize=6.5, label='not significant',
    )
    leg = ax.legend(
        handles=[filled, openm],
        loc='upper right',
        bbox_to_anchor=anchor,
        ncol=2,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0,
    )
    return leg


def padded_xlim(values: Iterable[float], *, zero=True, min_pad=0.01) -> tuple[float, float]:
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return (-1, 1)
    lo, hi = float(vals.min()), float(vals.max())
    if zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), abs(hi), 1.0)
    pad = max(span * 0.18, min_pad)
    return lo - pad, hi + pad


def draw_multi_location_forest(
    ax,
    df: pd.DataFrame,
    *,
    row_order: list[str],
    x_label: str,
    title: str,
    title_color: str,
    xlim: tuple[float, float] | None = None,
    left_label: str | None = None,
    right_label: str | None = None,
    show_ylabels: bool = True,
    overall_band: bool = True,
    significance_note: bool = False,
) -> None:
    """Draw one forest panel with three locations per row.

    Filled markers denote statistically significant results. For dimension rows,
    significance is Holm-corrected. For Overall, significance is based on the
    single prespecified aggregate test.
    """
    offsets = {
        'dallas': +0.18,
        'ny': 0.0,
        'la': -0.18,
    }
    ybase = {label: len(row_order) - 1 - i for i, label in enumerate(row_order)}

    if xlim is None:
        vals = list(df['ci_low']) + list(df['ci_high']) + list(df['effect'])
        xlim = padded_xlim(vals, zero=True)

    # Sign background, kept intentionally subtle.
    ax.axvspan(xlim[0], 0, color=COL_FOCAL, alpha=0.025, zorder=0)
    ax.axvspan(0, xlim[1], color=COL_COMPARISON, alpha=0.018, zorder=0)
    ax.axvline(0, color=COL_GRID, linewidth=1.15, zorder=1)

    if overall_band and 'Overall' in ybase:
        oy = ybase['Overall']
        ax.axhspan(oy - 0.48, oy + 0.48, color=COL_LIGHT, zorder=0)

    for _, rec in df.iterrows():
        label = rec['dimension']
        if label not in ybase:
            continue
        slug = rec['location_slug']
        info = LOCATIONS[slug]
        yy = ybase[label] + offsets[slug]
        effect = float(rec['effect'])
        lo = float(rec['ci_low']) if np.isfinite(rec['ci_low']) else np.nan
        hi = float(rec['ci_high']) if np.isfinite(rec['ci_high']) else np.nan
        sig = bool(rec['significant'])

        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(
                yy, lo, hi,
                color=info['color'],
                linewidth=2.0 if sig else 1.35,
                alpha=1.0 if sig else 0.62,
                zorder=2,
            )

        ax.scatter(
            [effect], [yy],
            s=58 if label != 'Overall' else 72,
            marker=info['marker'],
            facecolor=info['color'] if sig else 'white',
            edgecolor=info['color'],
            linewidth=1.45,
            zorder=3,
        )

    yticks = [ybase[x] for x in row_order]
    ax.set_yticks(yticks)
    if show_ylabels:
        ax.set_yticklabels(
            row_order,
            fontsize=9.5,
            color=COL_INK,
        )
        for tick, label in zip(ax.get_yticklabels(), row_order):
            tick.set_fontweight('bold' if label == 'Overall' else 'normal')
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    ax.set_ylim(-0.65, len(row_order) - 0.35)
    ax.set_xlim(*xlim)
    ax.tick_params(axis='x', labelsize=8.4, colors=COL_MUTED)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(x_label, fontsize=9.0, color=COL_INK, labelpad=8)
    ax.set_title(title, loc='left', fontsize=10.7, fontweight='bold', color=title_color, pad=16)

    if left_label:
        ax.text(
            0.01, 1.015, left_label,
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=8.2, fontweight='bold', color=COL_FOCAL,
        )
    if right_label:
        ax.text(
            0.99, 1.015, right_label,
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=8.2, fontweight='bold', color=COL_COMPARISON,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', color=COL_GRID, linewidth=0.65, alpha=0.55, zorder=0)

    if significance_note:
        ax.text(
            0.99, 0.01,
            'filled = significant',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7.8, color=COL_MUTED,
        )


def save_figure(fig, out_dir: Path, stem: str, *, dpi: int = 350) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f'{stem}.png'
    pdf = out_dir / f'{stem}.pdf'
    fig.savefig(png, dpi=dpi, bbox_inches='tight', facecolor=COL_BG)
    fig.savefig(pdf, bbox_inches='tight', facecolor=COL_BG)
    return png, pdf
