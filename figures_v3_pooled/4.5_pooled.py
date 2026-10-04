#!/usr/bin/env python3
"""Figure 4.5 pooled across Dallas, New York, and Los Angeles.

Evidence-eligible location repetitions are averaged for the same explicit
condition and outcome before visualization/inference. The blocked association
is then computed across these three-location condition means and clustered by
outcome. Evidence availability can differ by location, so `n_locations` is
retained and exported for auditing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps
from pooled_stats import (
    BASE_DEFAULT, OUTPUT_DEFAULT, ensure_pipeline_cache,
    read_three_location_csv, pooled_dimension_analysis, evidence_eligible,
)

ap = argparse.ArgumentParser()
ap.add_argument('--base-dir', type=Path, default=BASE_DEFAULT)
ap.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
ap.add_argument('--force-recompute', action='store_true')
args = ap.parse_args()
BASE = args.base_dir.absolute()
OUTPUT_DIR = args.output_dir.absolute(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STEM = 'figure_4_5_grounding_pooled3'
GAP = 'evidence_semantic_gap_source_balanced'
ANSWER = 'dense_distance_subject_normalized'

# Pooled focal-comparison grounding contrasts (used by the Results section even
# though the compact figure emphasizes the continuous coupling result).
query_evidence = read_three_location_csv(
    BASE, 'evidence', 'query_evidence_alignment_metrics_v2.csv'
)
gap_tests, gap_agg, gap_location_effects = pooled_dimension_analysis(
    query_evidence,
    GAP,
    'evidence_gap_3loc',
    eligibility=evidence_eligible,
    primary='signflip',
)
gap_tests.to_csv(OUTPUT_DIR / 'figure_4_5_pooled_gap_dimension_tests.csv', index=False)
pd.DataFrame([gap_agg]).to_csv(OUTPUT_DIR / 'figure_4_5_pooled_gap_overall.csv', index=False)
gap_location_effects.to_csv(OUTPUT_DIR / 'figure_4_5_location_outcome_gap_effects.csv', index=False)

corr, maps = ensure_pipeline_cache(BASE, OUTPUT_DIR, force=args.force_recompute)
data = maps['evidence'].copy()
# Match the pipeline's rule: keep outcomes with at least 3 eligible explicit conditions.
data = data[data.groupby('outcome_id')['outcome_id'].transform('size') >= 3].copy()

candidates = corr.loc[
    (corr['metric_x'] == GAP)
    & (corr['metric_y'] == ANSWER)
]
if len(candidates) != 1:
    raise RuntimeError(f'Expected one pooled evidence-gap -> answer correlation; found {len(candidates)}.')
reported = candidates.iloc[0].to_dict()

# Report alignment rather than gap: stronger alignment = smaller gap.
association = {
    'r': -float(reported['blocked_rank_correlation']),
    'ci_low': -float(reported['cluster_bootstrap_ci_high']),
    'ci_high': -float(reported['cluster_bootstrap_ci_low']),
    'p_holm': float(reported['p_blocked_permutation_holm']),
}

print('\nTHREE-LOCATION POOLED GROUNDING ASSOCIATION\n')
print(
    f"alignment r = {association['r']:.4f} "
    f"[{association['ci_low']:.4f}, {association['ci_high']:.4f}], "
    f"p_Holm = {association['p_holm']:.5g}, n={int(reported['n_observations'])}"
)
print('Eligible-city counts among pooled condition means:')
print(data['n_locations'].value_counts().sort_index().to_string())
print()
data.to_csv(OUTPUT_DIR / 'figure_4_5_pooled_condition_means.csv', index=False)

# ============================================================
# WITHIN-OUTCOME CENTERING
# ============================================================

data["alignment"] = (
    1 - data[GAP]
)


for column, name in [
    (ANSWER, "x"),
    ("alignment", "y"),
]:

    data[name] = (
        data[column]
        - data.groupby("outcome_id")[column]
        .transform("mean")
    )


# ============================================================
# DISPLAY TREND
# ============================================================

X_GRID = np.linspace(
    -0.23,
    0.24,
    160,
)


def ols_line(frame):

    slope, intercept = np.polyfit(
        frame["x"],
        frame["y"],
        1,
    )

    return (
        intercept
        + slope * X_GRID
    )


# ------------------------------------------------------------
# Outcome-cluster bootstrap
# ------------------------------------------------------------

rng = np.random.default_rng(
    ps.SEED
)

outcome_ids = (
    data["outcome_id"]
    .unique()
)

by_outcome = {
    key: group
    for key, group
    in data.groupby("outcome_id")
}


boot_lines = np.array(
    [

        ols_line(
            pd.concat(
                [
                    by_outcome[key]
                    for key in rng.choice(
                        outcome_ids,
                        len(outcome_ids),
                    )
                ]
            )
        )

        for _ in range(4000)
    ]
)


trend = ols_line(
    data
)


band_low, band_high = np.percentile(
    boot_lines,
    [
        2.5,
        97.5,
    ],
    axis=0,
)


# ============================================================
# GROUP MEANS
# ============================================================

means = (
    data.groupby("condition")[
        ["x", "y"]
    ]
    .mean()
)


# ============================================================
# PRINT VALUES USED IN FIGURE
# ============================================================

print(
    "\nVALUES USED IN FIGURE\n"
)


print(
    f"Blocked rank correlation "
    f"(answer displacement vs alignment): "
    f"r = {association['r']:.4f} "
    f"[{association['ci_low']:.4f}, "
    f"{association['ci_high']:.4f}], "
    f"p_Holm = {association['p_holm']:.4g}, "
    f"n = {len(data)} "
    f"in {data['outcome_id'].nunique()} outcomes"
)


print(
    f"Pearson r of centred values "
    f"(display): "
    f"{np.corrcoef(data['x'], data['y'])[0, 1]:.4f}"
)


print(
    "\nWithin-outcome means by condition:"
)


print(
    means
    .round(4)
    .to_string()
)


# ============================================================
# DRAW
# ============================================================

ps.use_paper_style()


# ============================================================
# FIGURE GEOMETRY
# ============================================================

FIG_W = ps.COLUMN_WIDTH
FIG_H = 3.12


# ------------------------------------------------------------
# Main scatter panel
# ------------------------------------------------------------

MAIN_LEFT = 0.56
MAIN_BOTTOM = 0.48

MAIN_W = 2.30
MAIN_H = 2.02


# ------------------------------------------------------------
# Marginal panels
#
# Deliberately thinner than before so that they read as
# supporting information rather than separate plots.
# ------------------------------------------------------------

MARGIN_TOP = 0.26
MARGIN_RIGHT = 0.28

GAP_TOP = 0.035
GAP_RIGHT = 0.035


XLIM = (
    -0.245,
    0.255,
)

YLIM = (
    -0.205,
    0.135,
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        FIG_W,
        FIG_H,
    )
)


ax = ps.add_axes_in(
    fig,
    MAIN_LEFT,
    MAIN_BOTTOM,
    MAIN_W,
    MAIN_H,
)


ax_top = ps.add_axes_in(
    fig,
    MAIN_LEFT,
    MAIN_BOTTOM
    + MAIN_H
    + GAP_TOP,
    MAIN_W,
    MARGIN_TOP,
)


ax_right = ps.add_axes_in(
    fig,
    MAIN_LEFT
    + MAIN_W
    + GAP_RIGHT,
    MAIN_BOTTOM,
    MARGIN_RIGHT,
    MAIN_H,
)


# ============================================================
# MAIN AXIS
# ============================================================

ax.set_xlim(
    *XLIM
)

ax.set_ylim(
    *YLIM
)


ax.set_xticks(
    [
        -0.2,
        -0.1,
        0,
        0.1,
        0.2,
    ]
)


ax.set_yticks(
    [
        -0.2,
        -0.1,
        0,
        0.1,
    ]
)


# ------------------------------------------------------------
# Signed tick labels
# ------------------------------------------------------------

def signed_tick(value, _):

    if np.isclose(
        value,
        0,
    ):
        return "0"

    return (
        f"{value:+.1f}"
        .replace(
            "-",
            "−",
        )
    )


ax.xaxis.set_major_formatter(
    plt.FuncFormatter(
        signed_tick
    )
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        signed_tick
    )
)


ps.strip_axes(
    ax,
    keep=(
        "bottom",
        "left",
    ),
)


# ============================================================
# ZERO REFERENCE LINES
#
# Lighter than before so that they do not compete with the
# regression line.
# ============================================================

ax.axvline(
    0,
    color=ps.RULE,
    linewidth=0.55,
    zorder=0,
)


ax.axhline(
    0,
    color=ps.RULE,
    linewidth=0.55,
    zorder=0,
)


# ============================================================
# AXIS LABELS
#
# Shorter wording than the previous version.
# ============================================================

ax.set_xlabel(
    "Answer distance from generic\n"
    "(within-outcome centered)",
    linespacing=1.10,
)


ax.set_ylabel(
    "Alignment with cited evidence\n"
    "(within-outcome centered)",
    linespacing=1.10,
)


# ============================================================
# TREND BAND
# ============================================================

ax.fill_between(
    X_GRID,
    band_low,
    band_high,

    color=ps.GENERIC_TINT,

    linewidth=0,

    alpha=0.75,

    zorder=1,
)


# ============================================================
# SCATTER
#
# Slightly larger points with a little transparency.
# ============================================================

for condition in [
    "majority",
    "minority",
]:

    subset = data[
        data["condition"]
        == condition
    ]

    ax.scatter(
        subset["x"],
        subset["y"],

        s=10.5,

        facecolor=(
            ps.CONDITION[condition]["color"]
        ),

        edgecolor=ps.WHITE,

        linewidth=0.35,

        alpha=0.88,

        zorder=2,
    )


# ============================================================
# DISPLAY TREND
# ============================================================

ax.plot(
    X_GRID,
    trend,

    color=ps.INK,

    linewidth=1.25,

    zorder=3,
)


# ============================================================
# MARGINAL DENSITIES
# ============================================================

def density(values, grid):

    return gaussian_kde(
        values
    )(
        grid
    )


x_grid = np.linspace(
    *XLIM,
    300,
)

y_grid = np.linspace(
    *YLIM,
    300,
)


for condition in [
    "majority",
    "minority",
]:

    colors = (
        ps.CONDITION[
            condition
        ]
    )

    subset = data[
        data["condition"]
        == condition
    ]


    # --------------------------------------------------------
    # Top marginal
    # --------------------------------------------------------

    dx = density(
        subset["x"],
        x_grid,
    )


    ax_top.fill_between(
        x_grid,
        dx,

        color=colors["tint"],

        alpha=0.48,

        linewidth=0,

        zorder=1,
    )


    ax_top.plot(
        x_grid,
        dx,

        color=colors["color"],

        linewidth=0.85,

        zorder=2,
    )


    ax_top.axvline(
        means.loc[
            condition,
            "x",
        ],

        ymax=0.88,

        color=colors["color"],

        linewidth=0.75,

        zorder=3,
    )


    # --------------------------------------------------------
    # Right marginal
    # --------------------------------------------------------

    dy = density(
        subset["y"],
        y_grid,
    )


    ax_right.fill_betweenx(
        y_grid,
        dy,

        color=colors["tint"],

        alpha=0.48,

        linewidth=0,

        zorder=1,
    )


    ax_right.plot(
        dy,
        y_grid,

        color=colors["color"],

        linewidth=0.85,

        zorder=2,
    )


    ax_right.axhline(
        means.loc[
            condition,
            "y",
        ],

        xmax=0.88,

        color=colors["color"],

        linewidth=0.75,

        zorder=3,
    )


# ============================================================
# FORMAT MARGINAL PANELS
# ============================================================

ax_top.set_xlim(
    *XLIM
)

ax_top.set_ylim(
    0,
    None,
)


ax_right.set_ylim(
    *YLIM
)

ax_right.set_xlim(
    0,
    None,
)


for marginal in (
    ax_top,
    ax_right,
):

    marginal.set_xticks([])
    marginal.set_yticks([])

    ps.strip_axes(
        marginal
    )


# ============================================================
# LEGEND
#
# Compact vertical legend in the relatively empty upper-left
# region of the scatter panel.
# ============================================================

LEGEND_X = 0.035
LEGEND_Y = 0.965
LEGEND_GAP = 0.073


legend_items = [
    (
        "majority",
        "Majority-marked",
    ),
    (
        "minority",
        "Minority-marked",
    ),
]


for i, (
    condition,
    text,
) in enumerate(
    legend_items
):

    y = (
        LEGEND_Y
        - i * LEGEND_GAP
    )


    ax.scatter(
        LEGEND_X,
        y,

        s=15,

        transform=ax.transAxes,

        facecolor=(
            ps.CONDITION[
                condition
            ]["color"]
        ),

        edgecolor=ps.WHITE,

        linewidth=0.35,

        zorder=5,
    )


    ax.text(
        LEGEND_X,
        y,

        text,

        transform=ps.offset(
            ax.transAxes,
            fig,
            5.2,
            0,
        ),

        ha="left",
        va="center",

        fontsize=ps.FS_NOTE,

        color=ps.INK_SOFT,

        zorder=5,
    )



# ============================================================
# THREE-LOCATION INFERENTIAL ANNOTATION
# ============================================================

p_txt = (
    r"$p_{\mathrm{Holm}} < .001$"
    if association["p_holm"] < .001
    else rf"$p_{{\mathrm{{Holm}}}} = {association['p_holm']:.3f}$"
)

fig.text(
    0.5,
    0.972,
    (
        rf"three-location pooled $r_{{\mathrm{{blocked}}}} = {association['r']:.2f}$, "
        rf"95% CI [{association['ci_low']:.2f}, {association['ci_high']:.2f}], "
        + p_txt
    ),
    ha="center",
    va="top",
    fontsize=ps.FS_NOTE,
    color=ps.MUTED,
)

# ============================================================
# SAVE
# ============================================================

ps.save_to(
    fig,
    OUTPUT_DIR,
    OUTPUT_STEM,
)