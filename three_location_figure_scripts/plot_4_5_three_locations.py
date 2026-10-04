#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_common import (
    BASE_DEFAULT, OUTPUT_DEFAULT, LOCATIONS, DIMENSION_ORDER,
    COL_FOCAL, COL_COMPARISON, COL_PATHWAY, COL_SYNTHESIS,
    COL_INK, COL_MUTED, COL_GRID, COL_LIGHT, COL_BG,
    load_location_bundle, effect_record, correlation_record,
    draw_multi_location_forest, make_location_legend, add_significance_legend,
    padded_xlim, save_figure,
)


def collect(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gap_rows = []
    corr_rows = []
    quality_rows = []

    for slug, info in LOCATIONS.items():
        evidence = load_location_bundle(base, slug)['evidence']

        for row in evidence['reference_analysis_50pct']['dimension_tests']:
            gap_rows.append(effect_record(
                location_slug=slug,
                location_label=info['label'],
                metric='evidence_semantic_gap',
                dimension=row['dimension'],
                row=row,
            ))

        agg50 = [
            row for row in evidence['coverage_sensitivity']['aggregate_outcome_blocked']
            if np.isclose(float(row.get('coverage_threshold', np.nan)), 0.5)
            and row.get('metric') == 'evidence_semantic_gap_source_balanced'
        ]
        if len(agg50) != 1:
            raise RuntimeError(
                f'Expected one 50% aggregate evidence-gap result for {slug}; found {len(agg50)}'
            )
        gap_rows.append(effect_record(
            location_slug=slug,
            location_label=info['label'],
            metric='evidence_semantic_gap',
            dimension='Overall',
            row=agg50[0],
            aggregate=True,
        ))

        candidates = [
            row for row in evidence['pipeline']['blocked_correlations']
            if row.get('metric_x') == 'evidence_semantic_gap_source_balanced'
            and row.get('metric_y') == 'dense_distance_subject_normalized'
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f'Expected one evidence-gap -> answer-displacement blocked correlation for {slug}; found {len(candidates)}'
            )
        corr_rows.append(correlation_record(
            location_slug=slug,
            location_label=info['label'],
            metric='answer_displacement_vs_evidence_alignment',
            row=candidates[0],
            flip_sign=True,  # stored metric is gap; figure reports alignment
        ))

        cq = evidence['collection_quality']
        coverage_diag = [
            row for row in cq.get('metric_vs_coverage', [])
            if row.get('metric') == 'evidence_alignment_source_balanced'
            and row.get('predictor') == 'fetch_coverage'
        ]
        rho = float(coverage_diag[0]['spearman_rho']) if coverage_diag else np.nan
        p_cov = float(coverage_diag[0].get('p_holm', np.nan)) if coverage_diag else np.nan
        quality_rows.append({
            'location_slug': slug,
            'location': info['label'],
            'n_queries': int(cq.get('n_queries', 0)),
            'reference_eligible_queries': int(cq.get('reference_eligible_queries', 0)),
            'mean_fetch_coverage': float(cq.get('mean_fetch_coverage', np.nan)),
            'alignment_vs_coverage_rho': rho,
            'alignment_vs_coverage_p_holm': p_cov,
        })

    return pd.DataFrame(gap_rows), pd.DataFrame(corr_rows), pd.DataFrame(quality_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    args = ap.parse_args()

    base = args.base_dir.absolute()
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)

    gaps, corrs, quality = collect(base)
    gaps.to_csv(out / 'figure_4_5_grounding_3loc_stats_gap.csv', index=False)
    corrs.to_csv(out / 'figure_4_5_grounding_3loc_stats_correlation.csv', index=False)
    quality.to_csv(out / 'figure_4_5_grounding_3loc_coverage.csv', index=False)

    rows = DIMENSION_ORDER + ['Overall']

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.6, 6.6),
        facecolor=COL_BG,
        gridspec_kw={'width_ratios': [2.45, 1.0], 'wspace': 0.28},
    )

    gap_xlim = padded_xlim(
        list(gaps['ci_low']) + list(gaps['ci_high']) + list(gaps['effect']),
        zero=True,
        min_pad=0.012,
    )
    draw_multi_location_forest(
        axes[0], gaps,
        row_order=rows,
        x_label='Focal - comparison evidence semantic gap\n(lower gap = stronger alignment to cited evidence)',
        title="ALIGNMENT WITH EACH CONDITION'S OWN EVIDENCE",
        title_color=COL_PATHWAY,
        xlim=gap_xlim,
        left_label='FOCAL MORE ALIGNED',
        right_label='COMPARISON MORE ALIGNED',
        show_ylabels=True,
    )

    leg1 = make_location_legend(axes[0], anchor=(0.0, 1.145), ncol=3)
    axes[0].add_artist(leg1)

    # Right panel: per-location correlation between answer displacement and alignment.
    ax = axes[1]
    vals = list(corrs['ci_low']) + list(corrs['ci_high']) + list(corrs['effect'])
    corr_xlim = padded_xlim(vals, zero=True, min_pad=0.05)
    corr_xlim = (min(corr_xlim[0], -0.08), max(corr_xlim[1], 0.58))
    ax.axvspan(0, corr_xlim[1], color=COL_PATHWAY, alpha=0.045, zorder=0)
    ax.axvline(0, color=COL_GRID, linewidth=1.2, zorder=1)

    order = ['dallas', 'ny', 'la']
    y = {slug: 2 - i for i, slug in enumerate(order)}
    for _, rec in corrs.iterrows():
        slug = rec['location_slug']
        info = LOCATIONS[slug]
        yy = y[slug]
        sig = bool(rec['significant'])
        ax.hlines(
            yy, rec['ci_low'], rec['ci_high'],
            color=info['color'],
            linewidth=4.0 if sig else 2.0,
            alpha=1.0 if sig else 0.62,
            zorder=2,
        )
        ax.scatter(
            [rec['effect']], [yy],
            s=110,
            marker=info['marker'],
            facecolor=info['color'] if sig else 'white',
            edgecolor=info['color'],
            linewidth=1.6,
            zorder=3,
        )
        ptxt = '< .001' if rec['p_display'] < .001 else f"= {rec['p_display']:.3f}"
        ax.text(
            rec['ci_high'] + (corr_xlim[1]-corr_xlim[0])*0.025,
            yy,
            rf"$r={rec['effect']:.3f}$ · $p_{{Holm}} {ptxt}$",
            ha='left', va='center', fontsize=8.6,
            color=COL_INK if sig else COL_MUTED,
        )

    ax.set_yticks([y[s] for s in order])
    ax.set_yticklabels([LOCATIONS[s]['label'] for s in order], fontsize=9.7, color=COL_INK)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=8.5, colors=COL_MUTED)
    ax.set_xlim(*corr_xlim)
    ax.set_ylim(-0.65, 2.7)
    ax.set_xlabel(
        'Blocked correlation: answer displacement ↔ evidence alignment',
        fontsize=9.0, color=COL_INK, labelpad=9,
    )
    ax.set_title(
        'DIVERGENCE AND GROUNDING',
        loc='left', fontsize=10.7, fontweight='bold', color=COL_SYNTHESIS, pad=16,
    )
    ax.grid(axis='x', color=COL_GRID, linewidth=0.65, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compact coverage note with all three locations.
    qlines = []
    for slug in order:
        q = quality.loc[quality['location_slug'] == slug].iloc[0]
        rho = q['alignment_vs_coverage_rho']
        qlines.append(
            f"{LOCATIONS[slug]['label']}: {int(q['reference_eligible_queries'])}/{int(q['n_queries'])} eligible, "
            f"coverage {q['mean_fetch_coverage']:.0%}, ρcov={rho:.03f}"
        )
    eligibility_line = ' · '.join([
        f"{row['location']}: {int(row['reference_eligible_queries'])}/{int(row['n_queries'])}"
        for _, row in quality.iterrows()
    ])

    add_significance_legend(axes[0], anchor=(1.0, 1.145))

    fig.suptitle(
        'Semantic divergence is not accompanied by weaker evidentiary alignment',
        x=0.075, y=0.985, ha='left',
        fontsize=15.0, fontweight='bold', color=COL_INK,
    )
    fig.text(
        0.075, 0.946,
        'Dallas, New York, and Los Angeles · source-balanced alignment at the 50% recovered-source threshold',
        ha='left', va='top', fontsize=9.2, color=COL_MUTED,
    )
    fig.text(
        0.075, 0.026,
        'Semantic alignment is a similarity measure, not a factual-support or claim-entailment test. '
        'Filled dimension markers use Holm-corrected significance; Overall uses the single aggregate test. '
        f'Eligible at 50% recovery: {eligibility_line}.',
        ha='left', va='bottom', fontsize=7.9, color=COL_MUTED,
    )

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.18, top=0.82)
    png, pdf = save_figure(fig, out, 'figure_4_5_grounding_3loc')
    print(f'PNG: {png}')
    print(f'PDF: {pdf}')
    plt.close(fig)


if __name__ == '__main__':
    main()
