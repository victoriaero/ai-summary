#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_common import (
    BASE_DEFAULT, OUTPUT_DEFAULT, LOCATIONS,
    COL_PATHWAY, COL_SYNTHESIS, COL_INK, COL_MUTED, COL_GRID, COL_LIGHT, COL_BG,
    load_location_bundle, correlation_record, padded_xlim, save_figure,
)


def collect(base: Path) -> pd.DataFrame:
    rows = []
    for slug, info in LOCATIONS.items():
        evidence = load_location_bundle(base, slug)['evidence']
        candidates = [
            row for row in evidence['pipeline']['blocked_correlations']
            if row.get('metric_x') == 'url_jaccard_distance'
            and row.get('metric_y') == 'dense_distance_subject_normalized'
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f'Expected one URL-displacement -> answer-displacement correlation for {slug}; found {len(candidates)}'
            )
        rows.append(correlation_record(
            location_slug=slug,
            location_label=info['label'],
            metric='source_to_answer_coupling',
            row=candidates[0],
            flip_sign=False,
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    args = ap.parse_args()

    base = args.base_dir.absolute()
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)

    df = collect(base)
    df.to_csv(out / 'figure_4_4_pathway_surface_coupling_3loc_stats.csv', index=False)

    fig, ax = plt.subplots(figsize=(9.3, 4.9), facecolor=COL_BG)
    ax.set_facecolor(COL_BG)

    vals = list(df['ci_low']) + list(df['ci_high']) + list(df['effect'])
    xlim = padded_xlim(vals, zero=True, min_pad=0.05)
    xlim = (min(xlim[0], -0.05), max(xlim[1], 0.55))

    ax.axvspan(0, xlim[1], color=COL_PATHWAY, alpha=0.045, zorder=0)
    ax.axvline(0, color=COL_GRID, linewidth=1.2, zorder=1)

    order = ['dallas', 'ny', 'la']
    y = {slug: 2 - i for i, slug in enumerate(order)}

    for _, rec in df.iterrows():
        slug = rec['location_slug']
        info = LOCATIONS[slug]
        yy = y[slug]
        sig = bool(rec['significant'])
        ax.hlines(
            yy, rec['ci_low'], rec['ci_high'],
            color=info['color'],
            linewidth=4.2 if sig else 2.2,
            alpha=1.0 if sig else 0.62,
            zorder=2,
        )
        ax.scatter(
            [rec['effect']], [yy],
            s=115,
            marker=info['marker'],
            facecolor=info['color'] if sig else 'white',
            edgecolor=info['color'],
            linewidth=1.6,
            zorder=3,
        )
        ptxt = '< .001' if rec['p_display'] < .001 else f"= {rec['p_display']:.3f}"
        ax.text(
            rec['ci_high'] + (xlim[1]-xlim[0])*0.025,
            yy,
            rf"$r_{{blocked}}={rec['effect']:.3f}$  ·  $p_{{Holm}} {ptxt}$",
            ha='left', va='center', fontsize=9.1,
            color=COL_INK if sig else COL_MUTED,
        )

    ax.set_yticks([y[s] for s in order])
    ax.set_yticklabels([LOCATIONS[s]['label'] for s in order], fontsize=10.5, color=COL_INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, 2.65)
    ax.tick_params(axis='x', labelsize=9.0, colors=COL_MUTED)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(
        'Blocked rank correlation: source-set displacement ↔ answer displacement',
        fontsize=9.7, color=COL_INK, labelpad=10,
    )
    ax.grid(axis='x', color=COL_GRID, linewidth=0.7, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)

    consistent = bool((df['effect'] > 0).all() or (df['effect'] < 0).all())
    n_sig = int(df['significant'].sum())
    direction_text = (
        f"Same direction in 3/3 locations · significant in {n_sig}/3"
        if consistent
        else f"Direction varies across locations · significant in {n_sig}/3"
    )

    fig.suptitle(
        'Source-pathway shifts track semantic shifts across locations',
        x=0.08, y=0.97, ha='left',
        fontsize=15.0, fontweight='bold', color=COL_INK,
    )
    fig.text(
        0.08, 0.91,
        direction_text,
        ha='left', va='top', fontsize=9.4,
        color=COL_SYNTHESIS if consistent else COL_MUTED,
        fontweight='bold' if consistent else 'normal',
    )
    fig.text(
        0.08, 0.052,
        'Within-outcome rank correlation; 95% outcome-cluster bootstrap CIs. Filled markers indicate Holm-corrected significance.',
        ha='left', va='bottom', fontsize=8.3, color=COL_MUTED,
    )

    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.20, top=0.78)
    png, pdf = save_figure(fig, out, 'figure_4_4_pathway_surface_coupling_3loc')
    print(f'PNG: {png}')
    print(f'PDF: {pdf}')
    plt.close(fig)


if __name__ == '__main__':
    main()
