#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from figure_common import (
    BASE_DEFAULT, OUTPUT_DEFAULT, LOCATIONS, DIMENSION_ORDER,
    COL_SYNTHESIS, COL_PURPLE, COL_INK, COL_MUTED, COL_BG,
    load_location_bundle, effect_record, draw_multi_location_forest,
    make_location_legend, add_significance_legend, padded_xlim, save_figure,
)

NOVELTY_COLOR = '#B94F7E'


def collect(base: Path) -> dict[str, pd.DataFrame]:
    out = {
        'full_answer': [],
        'generic_content': [],
        'novelty': [],
    }

    for slug, info in LOCATIONS.items():
        sem = load_location_bundle(base, slug)['semantic']

        for row in sem['primary_analysis']['dimension_tests']:
            out['full_answer'].append(effect_record(
                location_slug=slug,
                location_label=info['label'],
                metric='full_answer_displacement',
                dimension=row['dimension'],
                row=row,
            ))
        out['full_answer'].append(effect_record(
            location_slug=slug,
            location_label=info['label'],
            metric='full_answer_displacement',
            dimension='Overall',
            row=sem['primary_analysis']['aggregate_minority_vs_majority'],
            aggregate=True,
        ))

        for row in sem['sentence_level']['generic_coverage_dimension_tests']:
            out['generic_content'].append(effect_record(
                location_slug=slug,
                location_label=info['label'],
                metric='generic_content_displacement',
                dimension=row['dimension'],
                row=row,
            ))
        out['generic_content'].append(effect_record(
            location_slug=slug,
            location_label=info['label'],
            metric='generic_content_displacement',
            dimension='Overall',
            row=sem['sentence_level']['aggregate_generic_coverage'],
            aggregate=True,
        ))

        for row in sem['sentence_level']['group_novelty_dimension_tests']:
            out['novelty'].append(effect_record(
                location_slug=slug,
                location_label=info['label'],
                metric='group_specific_novelty',
                dimension=row['dimension'],
                row=row,
            ))
        out['novelty'].append(effect_record(
            location_slug=slug,
            location_label=info['label'],
            metric='group_specific_novelty',
            dimension='Overall',
            row=sem['sentence_level']['aggregate_group_novelty'],
            aggregate=True,
        ))

    return {k: pd.DataFrame(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    args = ap.parse_args()

    base = args.base_dir.absolute()
    out_dir = args.output_dir.absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(base)
    for name, df in data.items():
        df.to_csv(out_dir / f'figure_4_3_semantic_shift_3loc_stats_{name}.csv', index=False)

    rows = DIMENSION_ORDER + ['Overall']

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.0, 6.7),
        facecolor=COL_BG,
        gridspec_kw={'wspace': 0.22},
    )

    specs = [
        (
            'full_answer',
            'FULL-ANSWER DISPLACEMENT',
            COL_SYNTHESIS,
            'Focal - comparison distance from generic answer',
            'COMPARISON FARTHER',
            'FOCAL FARTHER',
        ),
        (
            'generic_content',
            'GENERIC-CONTENT DISPLACEMENT',
            COL_PURPLE,
            'Focal - comparison loss of generic content',
            'COMPARISON MORE',
            'FOCAL MORE',
        ),
        (
            'novelty',
            'GROUP-SPECIFIC NOVELTY',
            NOVELTY_COLOR,
            'Focal - comparison group-specific novelty',
            'COMPARISON MORE',
            'FOCAL MORE',
        ),
    ]

    for i, (key, title, color, xlabel, left, right) in enumerate(specs):
        df = data[key]
        xlim = padded_xlim(
            list(df['ci_low']) + list(df['ci_high']) + list(df['effect']),
            zero=True,
            min_pad=0.012,
        )
        draw_multi_location_forest(
            axes[i], df,
            row_order=rows,
            x_label=xlabel,
            title=title,
            title_color=color,
            xlim=xlim,
            left_label=left,
            right_label=right,
            show_ylabels=(i == 0),
        )

    leg1 = make_location_legend(axes[0], anchor=(0.0, 1.145), ncol=3)
    axes[0].add_artist(leg1)
    add_significance_legend(axes[2], anchor=(1.0, 1.145))

    fig.suptitle(
        'Identity-conditioned semantic shifts replicate across three U.S. locations',
        x=0.065, y=0.986, ha='left',
        fontsize=15.0, fontweight='bold', color=COL_INK,
    )
    fig.text(
        0.065, 0.946,
        'Positive effects indicate greater change for the focal condition · 95% paired-outcome bootstrap CIs',
        ha='left', va='top', fontsize=9.3, color=COL_MUTED,
    )
    fig.text(
        0.065, 0.025,
        'Dimension rows: filled markers indicate Holm-corrected significance within metric and location. Overall uses the single prespecified outcome-level aggregate test.',
        ha='left', va='bottom', fontsize=8.1, color=COL_MUTED,
    )

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.16, top=0.82)
    png, pdf = save_figure(fig, out_dir, 'figure_4_3_semantic_shift_anatomy_3loc')
    print(f'PNG: {png}')
    print(f'PDF: {pdf}')
    plt.close(fig)


if __name__ == '__main__':
    main()
