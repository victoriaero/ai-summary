#!/usr/bin/env python3
"""Figure 4.2 pooled across Dallas, New York, and Los Angeles.

Location repetitions are NEVER treated as independent outcomes. For inference,
we first compute each focal-comparison effect within outcome x location, then
average the three location repetitions within outcome, and only then bootstrap /
test across outcomes. Thus every displayed effect is a three-location summary
with an outcome-level 95% CI.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps
from pooled_stats import (
    BASE_DEFAULT, OUTPUT_DEFAULT, DIMENSIONS,
    read_three_location_csv, pooled_dimension_analysis,
    pooled_group_descriptives,
)

ap = argparse.ArgumentParser()
ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
ap.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
args = ap.parse_args()
BASE = args.base_dir.absolute()
OUTPUT_DIR = args.output_dir.absolute(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STEM = 'figure_4_2_evidentiary_pathway_pooled3'

# Raw three-location observations.
links = read_three_location_csv(BASE, 'link_count_analysis', 'clean_aio_link_data.csv')
overlap_raw = read_three_location_csv(BASE, 'source_overlap_analysis', 'source_overlap_vs_generic.csv')

# Primary pooled tests. Citation counts follow the paper's Wilcoxon convention;
# source-set displacement uses paired sign-flip inference.
vol_tests, vol_agg, _ = pooled_dimension_analysis(
    links, 'n_links', 'citation_count_3loc', primary='wilcoxon',
)
src_tests, src_agg, _ = pooled_dimension_analysis(
    overlap_raw, 'url_jaccard_distance', 'url_jaccard_3loc', primary='signflip',
)

vol_tests.to_csv(OUTPUT_DIR / 'figure_4_2_pooled_volume_tests.csv', index=False)
src_tests.to_csv(OUTPUT_DIR / 'figure_4_2_pooled_source_tests.csv', index=False)
pd.DataFrame([vol_agg]).to_csv(OUTPUT_DIR / 'figure_4_2_pooled_volume_overall.csv', index=False)
pd.DataFrame([src_agg]).to_csv(OUTPUT_DIR / 'figure_4_2_pooled_source_overall.csv', index=False)

# Descriptive group means for glyph positions/wedges.
link_group, overlap_group = pooled_group_descriptives(BASE)
sources = dict(zip(link_group['group'], link_group['mean_links']))
overlap_by_group = dict(zip(overlap_group['group'], overlap_group['mean_url_overlap']))

# Group names by dimension/condition (stable across locations).
group_map = (
    links.loc[links['condition'].isin(['minority', 'majority']), ['dimension', 'condition', 'group']]
    .drop_duplicates()
)
if group_map.groupby(['dimension', 'condition']).size().max() != 1:
    raise RuntimeError('Group labels differ across locations.')
lookup = group_map.set_index(['dimension', 'condition'])['group'].to_dict()

rows = []
for dim in DIMENSIONS:
    v = vol_tests.loc[vol_tests['dimension'] == dim].iloc[0]
    s = src_tests.loc[src_tests['dimension'] == dim].iloc[0]
    minority = lookup[(dim, 'minority')]
    majority = lookup[(dim, 'majority')]
    rows.append({
        'label': ps.DIMENSION_LABELS[dim],
        'sources_majority': sources[majority],
        'sources_minority': sources[minority],
        'overlap_majority': overlap_by_group[majority],
        'overlap_minority': overlap_by_group[minority],
        'd_sources': v['mean_difference'],
        'd_sources_low': v['bootstrap_ci_low'],
        'd_sources_high': v['bootstrap_ci_high'],
        'sig_sources': bool(v['significant_holm']),
        # Statistical test is on distance; display overlap difference, hence sign flip.
        'd_overlap': -100 * s['mean_difference'],
        'd_overlap_low': -100 * s['bootstrap_ci_high'],
        'd_overlap_high': -100 * s['bootstrap_ci_low'],
        'sig_overlap': bool(s['significant_holm']),
    })

# Overall descriptive source means: average focal/comparison conditions across
# dimensions and locations; inferential effect uses outcome-level pooled test.
condition_city = (
    links.loc[links['condition'].isin(['minority','majority'])]
    .groupby(['location','condition'], as_index=False)['n_links'].mean()
)
cond_mean = condition_city.groupby('condition')['n_links'].mean()

ov_city = overlap_raw.copy()
ov_city['url_overlap'] = 1 - pd.to_numeric(ov_city['url_jaccard_distance'], errors='coerce')
ov_cond = (
    ov_city.loc[ov_city['condition'].isin(['minority','majority'])]
    .groupby(['location','condition'], as_index=False)['url_overlap'].mean()
    .groupby('condition')['url_overlap'].mean()
)

rows.append({
    'label': 'Overall',
    'sources_majority': float(cond_mean['majority']),
    'sources_minority': float(cond_mean['minority']),
    'overlap_majority': float(ov_cond['majority']),
    'overlap_minority': float(ov_cond['minority']),
    'd_sources': vol_agg['mean_difference'],
    'd_sources_low': vol_agg['bootstrap_ci_low'],
    'd_sources_high': vol_agg['bootstrap_ci_high'],
    'sig_sources': bool(vol_agg['p_wilcoxon_raw'] < .05),
    'd_overlap': -100 * src_agg['mean_difference'],
    'd_overlap_low': -100 * src_agg['bootstrap_ci_high'],
    'd_overlap_high': -100 * src_agg['bootstrap_ci_low'],
    'sig_overlap': bool(src_agg['p_permutation_raw'] < .05),
})

table = pd.DataFrame(rows).set_index('label')
GENERIC_SOURCES = float(sources['people'])
print('\nTHREE-LOCATION POOLED VALUES\n')
print(table.round(4).to_string())
print('\nRule: location-specific effects are averaged within outcome before inference.\n')

# ============================================================
# DRAW (same visual language as figures_v2)
# ============================================================
ps.use_paper_style()
FIG_W = getattr(ps, 'DOUBLE_COLUMN_WIDTH', getattr(ps, 'TEXT_WIDTH', 7.0))
FIG_H = 2.42
PLOT_LEFT = 1.48; PLOT_W = 3.55
PLOT_BOTTOM = 0.60; PLOT_H = 1.67
COLUMN_X = {'d_sources': 5.64, 'd_overlap': 6.48}
GENERIC_Y = 7.45
ROW_Y = [6, 5, 4, 3, 2, 1, -0.45]
YLIM = (-1.05, 8.05)
GLYPH = 9.4

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = ps.add_axes_in(fig, PLOT_LEFT, PLOT_BOTTOM, PLOT_W, PLOT_H)
# Dynamic x-range for the pooled descriptive means.
all_source_means = [GENERIC_SOURCES] + table['sources_majority'].tolist() + table['sources_minority'].tolist()
xmin = max(0, np.floor(min(all_source_means) - 1.0))
xmax = np.ceil(max(all_source_means) + 1.0)
ax.set_xlim(xmin, xmax)
xticks = np.arange(math.ceil(xmin/2)*2, xmax + 0.1, 2)
if len(xticks) < 3:
    xticks = np.linspace(xmin, xmax, 4)
ax.set_xticks(xticks)
ax.set_ylim(*YLIM); ax.set_yticks([])
ps.strip_axes(ax, keep=('bottom',))
ax.set_xlabel('Mean cited sources per answer')

# Generic reference.
ax.axvline(GENERIC_SOURCES, color=ps.GENERIC, linewidth=0.75, linestyle=(0,(2,2)), zorder=0)
ps.overlap_glyph(ax, GENERIC_SOURCES, GENERIC_Y, 1.0, 'generic', diameter=GLYPH)

# Dimension rows.
for (label, row), y in zip(table.iterrows(), ROW_Y):
    if label == 'Overall':
        ax.axhspan(y - 0.37, y + 0.37, color=ps.BAND, zorder=-2)
    # connector
    ax.plot([row['sources_majority'], row['sources_minority']], [y,y], color=ps.RULE, linewidth=0.7, zorder=0)
    ps.overlap_glyph(ax, row['sources_majority'], y, row['overlap_majority'], 'majority', diameter=GLYPH)
    ps.overlap_glyph(ax, row['sources_minority'], y, row['overlap_minority'], 'minority', diameter=GLYPH)

# Figure-coordinate labels/stat columns.
rows_trans = ps.fig_x_data_y(fig, ax)
label_x = ps.fig_x(fig, 0.04)
fig.text(label_x, GENERIC_Y, 'Generic (“people”)', transform=rows_trans, ha='left', va='center', fontsize=ps.FS_LABEL, color=ps.INK_SOFT)

for (label, row), y in zip(table.iterrows(), ROW_Y):
    fig.text(label_x, y, label, transform=rows_trans, ha='left', va='center', fontsize=ps.FS_LABEL,
             color=ps.INK, fontweight='bold' if label == 'Overall' else 'normal')
    for col, decimals in [('d_sources',1), ('d_overlap',1)]:
        sig = row['sig_sources'] if col == 'd_sources' else row['sig_overlap']
        lo, hi = row[f'{col}_low'], row[f'{col}_high']
        # Two-line estimate + 95% CI. This is the main change from the Dallas-only figure.
        text = (
            f"{ps.format_signed(row[col], decimals)}\n"
            f"[{ps.format_signed(lo, decimals)}, {ps.format_signed(hi, decimals)}]"
        )
        fig.text(ps.fig_x(fig, COLUMN_X[col]), y, text, transform=rows_trans,
                 ha='center', va='center', fontsize=5.45, linespacing=1.05,
                 fontweight='bold' if sig else 'normal',
                 color=ps.INK if sig else ps.MUTED)

# Separators.
for y in [(GENERIC_Y + ROW_Y[0])/2, (ROW_Y[-2] + ROW_Y[-1])/2]:
    fig.add_artist(Line2D([label_x, ps.fig_x(fig, FIG_W - 0.06)], [y,y], transform=rows_trans,
                          color=ps.RULE, linewidth=0.5))

# Statistical headers.
span_left = COLUMN_X['d_sources'] - 0.34; span_right = COLUMN_X['d_overlap'] + 0.34
span_centre = (span_left + span_right)/2
fig.text(ps.fig_x(fig, span_centre), GENERIC_Y, 'Three-location focal − comparison',
         transform=ps.offset(rows_trans, fig, y_pt=6.5), ha='center', va='center', fontsize=ps.FS_NOTE, color=ps.INK_SOFT)
fig.add_artist(Line2D([ps.fig_x(fig, span_left), ps.fig_x(fig, span_right)], [GENERIC_Y, GENERIC_Y],
                      transform=ps.offset(rows_trans, fig, y_pt=1.5), color=ps.RULE, linewidth=0.5))
for x, text in [(COLUMN_X['d_sources'], 'sources\nmean [95% CI]'), (COLUMN_X['d_overlap'], 'overlap (pp)\nmean [95% CI]')]:
    fig.text(ps.fig_x(fig, x), GENERIC_Y, text, transform=ps.offset(rows_trans, fig, y_pt=-7.0),
             ha='center', va='center', fontsize=5.6, color=ps.INK_SOFT, linespacing=1.0)

# Legend/key.
key = ps.add_axes_in(fig, 0.0, 0.0, FIG_W, 0.16); key.set_xlim(0,FIG_W); key.set_ylim(0,1); key.axis('off')
KEY_START = (FIG_W - 4.75)/2 + 0.22
for x, condition, share, text in [
    (KEY_START, 'majority', 0.0, 'Majority-marked'),
    (KEY_START+1.60, 'minority', 0.0, 'Minority-marked'),
    (KEY_START+3.20, 'generic', 0.3, 'Shared with generic'),
]:
    ps.overlap_glyph(key, x, 0.5, share, condition, diameter=7.2)
    key.text(x+0.13, 0.5, text, ha='left', va='center', fontsize=ps.FS_NOTE, color=ps.INK_SOFT)

fig.text(0.5, 0.988, 'Dallas + New York + Los Angeles; location repetitions averaged within outcome before inference',
         transform=fig.transFigure, ha='center', va='top', fontsize=5.8, color=ps.MUTED)

ps.save_to(fig, OUTPUT_DIR, OUTPUT_STEM)
