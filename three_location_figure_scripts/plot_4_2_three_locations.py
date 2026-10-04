#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_common import (
    BASE_DEFAULT, OUTPUT_DEFAULT, LOCATIONS, DIMENSION_ORDER,
    COL_FOCAL, COL_COMPARISON, COL_PATHWAY, COL_INK, COL_MUTED, COL_BG,
    load_location_bundle, effect_record, draw_multi_location_forest,
    make_location_legend, add_significance_legend, padded_xlim, save_figure,
)


def collect(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    volume = []
    source = []

    for slug, info in LOCATIONS.items():
        bundle = load_location_bundle(base, slug)
        link = bundle['link']
        overlap = bundle['overlap']

        for row in link['statistical_tests']['minority_vs_majority_by_dimension']:
            volume.append(effect_record(
                location_slug=slug,
                location_label=info['label'],
                metric='citation_count',
                dimension=row['dimension'],
                row=row,
            ))

        volume.append(effect_record(
            location_slug=slug,
            location_label=info['label'],
            metric='citation_count',
            dimension='Overall',
            row=link['statistical_tests']['aggregate_minority_vs_majority'],
            aggregate=True,
        ))

        for row in overlap['minority_vs_majority']['url_jaccard_distance']:
            source.append(effect_record(
                location_slug=slug,
                location_label=info['label'],
                metric='source_url_jaccard_distance',
                dimension=row['dimension'],
                row=row,
            ))

        agg_candidates = [
            row for row in overlap['minority_vs_majority'].get('aggregate', [])
            if str(row.get('metric', '')).lower() == 'url_jaccard_distance'
        ]
        if len(agg_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one URL-Jaccard aggregate for {slug}; found {len(agg_candidates)}"
            )
        source.append(effect_record(
            location_slug=slug,
            location_label=info['label'],
            metric='source_url_jaccard_distance',
            dimension='Overall',
            row=agg_candidates[0],
            aggregate=True,
        ))

    return pd.DataFrame(volume), pd.DataFrame(source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    args = ap.parse_args()

    base = args.base_dir.absolute()
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)

    volume, source = collect(base)
    volume.to_csv(out / 'figure_4_2_evidentiary_pathway_3loc_stats_volume.csv', index=False)
    source.to_csv(out / 'figure_4_2_evidentiary_pathway_3loc_stats_source.csv', index=False)

    rows = DIMENSION_ORDER + ['Overall']

    volume_xlim = padded_xlim(
        list(volume['ci_low']) + list(volume['ci_high']) + list(volume['effect']),
        zero=True,
        min_pad=0.25,
    )
    source_xlim = padded_xlim(
        list(source['ci_low']) + list(source['ci_high']) + list(source['effect']),
        zero=True,
        min_pad=0.01,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.8, 6.6),
        facecolor=COL_BG,
        gridspec_kw={'wspace': 0.27},
    )

    draw_multi_location_forest(
        axes[0], volume,
        row_order=rows,
        x_label='Focal - comparison mean cited sources',
        title='EVIDENCE VOLUME',
        title_color=COL_FOCAL,
        xlim=volume_xlim,
        left_label='COMPARISON MORE',
        right_label='FOCAL MORE',
        show_ylabels=True,
    )

    draw_multi_location_forest(
        axes[1], source,
        row_order=rows,
        x_label='Focal - comparison URL-Jaccard distance from generic',
        title='SOURCE-SET DISPLACEMENT',
        title_color=COL_PATHWAY,
        xlim=source_xlim,
        left_label='COMPARISON FARTHER',
        right_label='FOCAL FARTHER',
        show_ylabels=False,
    )

    # Keep city legend and significance semantics visually separate.
    leg1 = make_location_legend(axes[0], anchor=(0.0, 1.145), ncol=3)
    axes[0].add_artist(leg1)
    add_significance_legend(axes[1], anchor=(1.0, 1.145))

    fig.suptitle(
        'Identity marking changes the amount and pathway of surfaced evidence across locations',
        x=0.08, y=0.985, ha='left',
        fontsize=15.0, fontweight='bold', color=COL_INK,
    )
    fig.text(
        0.08, 0.945,
        'Dallas, New York, and Los Angeles · 95% paired-outcome bootstrap CIs · filled markers denote statistical significance',
        ha='left', va='top', fontsize=9.3, color=COL_MUTED,
    )
    fig.text(
        0.08, 0.026,
        'Dimension rows: Holm-corrected tests within each metric and location. Overall: the single prespecified outcome-level aggregate test (uncorrected).',
        ha='left', va='bottom', fontsize=8.2, color=COL_MUTED,
    )

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.83)

    png, pdf = save_figure(fig, out, 'figure_4_2_evidentiary_pathway_3loc')
    print(f'PNG: {png}')
    print(f'PDF: {pdf}')
    plt.close(fig)


if __name__ == '__main__':
    main()
