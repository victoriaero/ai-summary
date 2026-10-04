"""
Figure 2 - identity marking changes the semantic content of AI Overviews.

Single-column semantic-signature matrix.

Each row is one social dimension (plus the outcome-level aggregate). The
three aligned columns show:

    1. Full-answer semantic displacement from the generic answer
       (primary, label-normalized metric).

    2. Generic content lost:
       generic-answer content less retained by the focal condition.

    3. New content added:
       content introduced by the focal condition that is not represented
       in the generic answer.

All columns show focal - comparison paired differences on one common scale.
Positive values indicate a larger effect for the focal condition;
negative values indicate a larger effect for the comparison condition.

Encoding, in the shared palette:
    cell wash   Holm-significant contrast, orange = focal larger,
                blue = comparison larger; grey = not significant
    capsule     95% paired bootstrap confidence interval
    marker      point estimate; filled = Holm-significant, open = n.s.;
                outline colour = direction (orange focal, blue comparison)
    diamond     primary full-answer measure; circles = sentence-level

The Overall row is the single prespecified outcome-level aggregate,
so it is not part of the Holm family.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ============================================================
# LOCAL STYLE
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import paper_style as ps


# ============================================================
# LOAD RESULTS
# ============================================================

SEMANTIC_RESULTS = (
    ps.RESULTS_DIR
    / "semantic_embedding_analysis_dallas"
    / "all_results.json"
)

OUTPUT_STEM = "figure_4_3_semantic_shift_anatomy"


semantic = json.loads(
    SEMANTIC_RESULTS.read_text(
        encoding="utf-8"
    )
)


if (
    semantic["metadata"]["primary_metric"]
    != "dense_distance_subject_normalized"
):
    raise RuntimeError(
        "Unexpected primary semantic metric."
    )


# ============================================================
# HELPERS
# ============================================================

def tests(rows):
    """
    Convert a list of dimension-level test dictionaries into a
    consistently ordered dataframe.
    """
    return (
        pd.DataFrame(rows)
        .set_index("dimension")
        .loc[ps.DIMENSIONS]
    )


# ============================================================
# DIMENSION-LEVEL RESULTS
# ============================================================

full = tests(
    semantic[
        "primary_analysis"
    ][
        "dimension_tests"
    ]
)


lost = tests(
    semantic[
        "sentence_level"
    ][
        "generic_coverage_dimension_tests"
    ]
)


added = tests(
    semantic[
        "sentence_level"
    ][
        "group_novelty_dimension_tests"
    ]
)


# ============================================================
# VALIDATE EXPECTED FIELDS
# ============================================================

required = {
    "mean_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "significant_permutation_holm",
}


for name, frame in [
    ("full", full),
    ("lost", lost),
    ("added", added),
]:

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{name} is missing fields: "
            f"{sorted(missing)}"
        )


# ============================================================
# TABLE USED BY FIGURE
# ============================================================

table = pd.DataFrame(
    {
        # ----------------------------------------------------
        # Primary full-answer displacement
        # ----------------------------------------------------
        "full": (
            full["mean_difference"]
        ),

        "full_low": (
            full["bootstrap_ci_low"]
        ),

        "full_high": (
            full["bootstrap_ci_high"]
        ),

        "full_sig": (
            full[
                "significant_permutation_holm"
            ]
            .astype(bool)
        ),

        # ----------------------------------------------------
        # Generic-content displacement
        # ----------------------------------------------------
        "lost": (
            lost["mean_difference"]
        ),

        "lost_low": (
            lost["bootstrap_ci_low"]
        ),

        "lost_high": (
            lost["bootstrap_ci_high"]
        ),

        "lost_sig": (
            lost[
                "significant_permutation_holm"
            ]
            .astype(bool)
        ),

        # ----------------------------------------------------
        # Group-specific novelty
        # ----------------------------------------------------
        "added": (
            added["mean_difference"]
        ),

        "added_low": (
            added["bootstrap_ci_low"]
        ),

        "added_high": (
            added["bootstrap_ci_high"]
        ),

        "added_sig": (
            added[
                "significant_permutation_holm"
            ]
            .astype(bool)
        ),
    }
)


# ============================================================
# AGGREGATES
# ============================================================

aggregates = {

    name: semantic[section][key]

    for name, (
        section,
        key,
    ) in {

        "full": (
            "primary_analysis",
            "aggregate_minority_vs_majority",
        ),

        "lost": (
            "sentence_level",
            "aggregate_generic_coverage",
        ),

        "added": (
            "sentence_level",
            "aggregate_group_novelty",
        ),

    }.items()
}


# ------------------------------------------------------------
# Overall row: one prespecified test per measure, so its
# significance is the raw permutation p, not Holm-corrected.
# ------------------------------------------------------------

overall = {}

for name, agg in aggregates.items():

    overall[name] = agg["mean_difference"]
    overall[f"{name}_low"] = agg["bootstrap_ci_low"]
    overall[f"{name}_high"] = agg["bootstrap_ci_high"]
    overall[f"{name}_sig"] = agg["p_permutation_raw"] < 0.05


# ============================================================
# PRINT VALUES USED IN FIGURE
# ============================================================

print(
    "\nVALUES USED IN FIGURE\n"
)


print(
    pd.concat(
        [
            table,
            pd.DataFrame(
                [overall],
                index=["Overall"],
            ),
        ]
    )
    .round(4)
    .to_string()
)


print(
    "\nOverall row = outcome-level aggregates "
    "(raw permutation p):"
)


for name, agg in aggregates.items():

    print(
        f"  {name:5s} "
        f"{agg['mean_difference']:+.4f} "
        f"[{agg['bootstrap_ci_low']:.4f}, "
        f"{agg['bootstrap_ci_high']:.4f}] "
        f"p = {agg['p_permutation_raw']:.2g}"
    )


# ============================================================
# DRAW
# ============================================================

ps.use_paper_style()


# ============================================================
# COLOURS
#
# Everything is derived from the shared palette: orange means
# the focal (minority-marked) condition is larger, blue means
# the comparison (majority-marked) condition is larger.
# ============================================================

def mix(color, share):
    """`color` blended with white; share = 1 returns the colour."""

    rgb = np.array(
        to_rgb(color)
    )

    return tuple(
        share * rgb
        + (1 - share) * np.ones(3)
    )


DIRECTION_COLOR = {
    "focal": ps.FOCAL,
    "comparison": ps.COMPARISON,
}

# Cell wash behind each estimate.
WASH = {
    "focal": mix(ps.FOCAL, 0.22),
    "comparison": mix(ps.COMPARISON, 0.22),
    "ns": ps.BAND,
}

# Confidence-interval capsule.
CAPSULE = {
    "focal": mix(ps.FOCAL, 0.55),
    "comparison": mix(ps.COMPARISON, 0.55),
    "ns": mix(ps.GENERIC, 0.40),
}


def direction(value):

    return (
        "focal"
        if value >= 0
        else "comparison"
    )


# ============================================================
# FIGURE GEOMETRY (inches)
# ============================================================

FIG_W = ps.COLUMN_WIDTH
FIG_H = 2.56


# ------------------------------------------------------------
# Left column for labels, three aligned measure columns
# ------------------------------------------------------------

LABEL_W = 0.86
PANEL_GAP = 0.065
RIGHT_MARGIN = 0.035

PANEL_W = (
    FIG_W
    - LABEL_W
    - RIGHT_MARGIN
    - 2 * PANEL_GAP
) / 3

PANEL_BOTTOM = 0.37
PANEL_H = 1.70


# ------------------------------------------------------------
# Common x scale, fitted to the data rather than symmetric:
# almost every effect is positive, so the intervals get the
# horizontal resolution. Equal margins outside -.1 and +.2.
# ------------------------------------------------------------

XLIM = (-0.135, 0.235)
XTICKS = [-0.1, 0.0, 0.1, 0.2]


# ------------------------------------------------------------
# Rows: six dimensions, a gap, then the aggregate
# ------------------------------------------------------------

DIM_Y = (
    np.arange(len(ps.DIMENSIONS), 0, -1)
    + 0.45
)

OVERALL_Y = 0.0

YLIM = (
    -0.52,
    DIM_Y[0] + 0.52,
)

CELL_H = 0.84      # share of a row; the rest is a white gap


# ------------------------------------------------------------
# Measures, left to right
# ------------------------------------------------------------

MEASURES = [

    {
        "key": "full",
        "title": "FULL ANSWER",
        "subtitle": "distance",
        "marker": "D",
        "size": 4.5,
    },

    {
        "key": "lost",
        "title": "GENERIC",
        "subtitle": "content lost",
        "marker": "o",
        "size": 4.6,
    },

    {
        "key": "added",
        "title": "NEW",
        "subtitle": "content added",
        "marker": "o",
        "size": 4.6,
    },
]


ROWS = [
    (
        ps.DIMENSION_LABELS[dimension],
        y,
        table.loc[dimension],
        False,
    )
    for dimension, y in zip(ps.DIMENSIONS, DIM_Y)
]

ROWS.append(
    (
        "Overall",
        OVERALL_Y,
        overall,
        True,
    )
)


# Every interval must fit the common scale.
bounds = [
    row[f"{measure['key']}_{end}"]
    for _, _, row, _ in ROWS
    for measure in MEASURES
    for end in ("low", "high")
]

if min(bounds) < XLIM[0] or max(bounds) > XLIM[1]:
    raise RuntimeError(
        f"XLIM {XLIM} does not cover "
        f"[{min(bounds):.3f}, {max(bounds):.3f}]"
    )


# ============================================================
# DRAWING PRIMITIVES
# ============================================================

def effect_tick(value, _):

    if np.isclose(value, 0):
        return "0"

    return (
        f"{value:+.1f}"
        .replace("-0.", "−.")
        .replace("+0.", "+.")
    )


def cell(ax, y, color):
    """A full-width cell for one row: x in axes fraction, y in data."""

    ax.add_patch(
        Rectangle(
            (0, y - CELL_H / 2),
            1,
            CELL_H,
            transform=ax.get_yaxis_transform(),
            facecolor=color,
            edgecolor="none",
            zorder=0,
        )
    )


def capsule(ax, low, high, y, color, width_pt):
    """
    Confidence interval as a rounded bar whose visible ends sit
    exactly on the interval bounds: the round caps are pulled in
    by their radius, converted from printed points to data units.
    """

    radius = (
        (width_pt / 2)
        / (PANEL_W * 72)
        * (XLIM[1] - XLIM[0])
    )

    start = low + radius
    end = high - radius

    if end < start:
        start = end = (low + high) / 2

    ax.plot(
        [start, end],
        [y, y],
        color=color,
        linewidth=width_pt,
        solid_capstyle="round",
        zorder=2,
    )


def estimate(ax, x, y, marker, size, significant):
    """Point estimate: filled when significant, outline = direction."""

    color = DIRECTION_COLOR[direction(x)]

    ax.plot(
        [x],
        [y],
        marker=marker,
        markersize=size,
        markerfacecolor=color if significant else ps.WHITE,
        markeredgecolor=ps.WHITE if significant else color,
        markeredgewidth=0.8 if significant else 1.0,
        linestyle="none",
        zorder=4,
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


axes = [
    ps.add_axes_in(
        fig,
        LABEL_W + i * (PANEL_W + PANEL_GAP),
        PANEL_BOTTOM,
        PANEL_W,
        PANEL_H,
    )
    for i in range(len(MEASURES))
]


# ============================================================
# COLUMNS
# ============================================================

for ax, measure in zip(axes, MEASURES):

    key = measure["key"]

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(effect_tick)
    )
    ax.set_yticks([])

    ax.tick_params(
        axis="x",
        length=0,
        pad=2.5,
        labelsize=ps.FS_NOTE - 0.5,
        labelcolor=ps.MUTED,
    )

    ps.strip_axes(ax)


    # --------------------------------------------------------
    # Grid: white hairlines cut through the cells, and a
    # solid zero reference on top of them
    # --------------------------------------------------------

    for tick in XTICKS:

        if np.isclose(tick, 0):
            continue

        ax.axvline(
            tick,
            color=ps.WHITE,
            linewidth=0.6,
            zorder=1,
        )

    ax.axvline(
        0,
        color=ps.INK_SOFT,
        linewidth=0.6,
        zorder=1.5,
    )


    # --------------------------------------------------------
    # One cell, interval and estimate per row
    # --------------------------------------------------------

    for label, y, row, is_overall in ROWS:

        value = row[key]
        significant = bool(row[f"{key}_sig"])

        tone = (
            direction(value)
            if significant
            else "ns"
        )

        cell(
            ax,
            y,
            WASH[tone],
        )

        capsule(
            ax,
            row[f"{key}_low"],
            row[f"{key}_high"],
            y,
            CAPSULE[tone],
            width_pt=3.8 if is_overall else 3.2,
        )

        estimate(
            ax,
            value,
            y,
            measure["marker"],
            measure["size"] + (0.8 if is_overall else 0.0),
            significant,
        )


# ============================================================
# ROW LABELS AND THE OVERALL SEPARATOR
# ============================================================

rows_trans = ps.fig_x_data_y(
    fig,
    axes[0],
)

LABEL_X = ps.fig_x(fig, 0.04)


for label, y, _, is_overall in ROWS:

    fig.text(
        LABEL_X,
        y,
        label,
        transform=rows_trans,
        ha="left",
        va="center",
        fontsize=ps.FS_TICK,
        fontweight="bold" if is_overall else "normal",
        color=ps.INK,
    )


separator_y = (DIM_Y[-1] + OVERALL_Y) / 2

fig.add_artist(
    Line2D(
        [LABEL_X, ps.fig_x(fig, FIG_W - RIGHT_MARGIN)],
        [separator_y, separator_y],
        transform=rows_trans,
        color=ps.RULE,
        linewidth=0.5,
    )
)


# ============================================================
# HEADERS
#
# Baselines are fixed distances above the matrix, in points:
# spanner text, spanner rule, column title, column subtitle.
# ============================================================

SPANNER_PT = 27.0
RULE_PT = 23.5
TITLE_PT = 14.5
SUBTITLE_PT = 6.0

top = PANEL_BOTTOM + PANEL_H


for ax, measure in zip(axes, MEASURES):

    ax.text(
        0.5,
        1,
        measure["title"],
        transform=ps.offset(ax.transAxes, fig, 0, TITLE_PT),
        ha="center",
        va="baseline",
        fontsize=ps.FS_NOTE + 0.1,
        fontweight="bold",
        color=ps.INK,
    )

    ax.text(
        0.5,
        1,
        measure["subtitle"],
        transform=ps.offset(ax.transAxes, fig, 0, SUBTITLE_PT),
        ha="center",
        va="baseline",
        fontsize=ps.FS_NOTE - 0.4,
        color=ps.MUTED,
    )


def spanner(first, last, text):
    """Muted caption over a group of columns, with a hairline rule."""

    left = first.get_position().x0 + ps.fig_x(fig, 0.02)
    right = last.get_position().x1 - ps.fig_x(fig, 0.02)

    fig.add_artist(
        Line2D(
            [left, right],
            [ps.fig_y(fig, top + RULE_PT / 72)] * 2,
            transform=fig.transFigure,
            color=ps.RULE,
            linewidth=0.5,
        )
    )

    fig.text(
        (left + right) / 2,
        ps.fig_y(fig, top + SPANNER_PT / 72),
        text,
        ha="center",
        va="baseline",
        fontsize=ps.FS_NOTE - 0.7,
        color=ps.MUTED,
    )


spanner(axes[0], axes[0], "PRIMARY")
spanner(axes[1], axes[2], "SENTENCE-LEVEL MECHANISMS")


# ============================================================
# BOTTOM: ONE LINE UNDER THE TICK LABELS
#
# Direction guide at both ends, common axis label between.
# ============================================================

GUIDE_Y = PANEL_BOTTOM - 0.235

panels_left = LABEL_W
panels_right = LABEL_W + 3 * PANEL_W + 2 * PANEL_GAP


fig.text(
    ps.fig_x(fig, (panels_left + panels_right) / 2),
    ps.fig_y(fig, GUIDE_Y),
    "Δ focal − comparison",
    ha="center",
    va="center",
    fontsize=ps.FS_NOTE,
    color=ps.INK_SOFT,
)


def guide_marker(x, marker, color):

    fig.add_artist(
        Line2D(
            [ps.fig_x(fig, x)],
            [ps.fig_y(fig, GUIDE_Y)],
            transform=fig.transFigure,
            marker=marker,
            markersize=3.6,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="none",
        )
    )


guide_marker(panels_left + 0.04, "<", ps.COMPARISON)

fig.text(
    ps.fig_x(fig, panels_left + 0.085),
    ps.fig_y(fig, GUIDE_Y),
    "comparison larger",
    ha="left",
    va="center",
    fontsize=ps.FS_NOTE - 0.6,
    color=ps.MUTED,
)

guide_marker(panels_right - 0.04, ">", ps.FOCAL)

fig.text(
    ps.fig_x(fig, panels_right - 0.085),
    ps.fig_y(fig, GUIDE_Y),
    "focal larger",
    ha="right",
    va="center",
    fontsize=ps.FS_NOTE - 0.6,
    color=ps.MUTED,
)


# ------------------------------------------------------------
# Key, top left: a miniature cell + marker for each state,
# on the baselines of the column titles and subtitles.
# ------------------------------------------------------------

KEY_X = 0.04
SWATCH_W, SWATCH_H = 0.17, 0.095
X_HEIGHT_PT = 2.2          # centre of lowercase text above its baseline

for baseline_pt, tone, significant, text in [
    (TITLE_PT, "focal", True, "Holm-significant"),
    (SUBTITLE_PT, "ns", False, "not significant"),
]:

    line_y = top + (baseline_pt + X_HEIGHT_PT) / 72

    fig.add_artist(
        Rectangle(
            (
                ps.fig_x(fig, KEY_X),
                ps.fig_y(fig, line_y - SWATCH_H / 2),
            ),
            ps.fig_x(fig, SWATCH_W),
            ps.fig_y(fig, SWATCH_H),
            transform=fig.transFigure,
            facecolor=WASH[tone],
            edgecolor="none",
        )
    )

    fig.add_artist(
        Line2D(
            [ps.fig_x(fig, KEY_X + SWATCH_W / 2)],
            [ps.fig_y(fig, line_y)],
            transform=fig.transFigure,
            marker="o",
            markersize=4.0,
            markerfacecolor=ps.FOCAL if significant else ps.WHITE,
            markeredgecolor=ps.WHITE if significant else ps.FOCAL,
            markeredgewidth=0.8 if significant else 1.0,
            linestyle="none",
        )
    )

    fig.text(
        ps.fig_x(fig, KEY_X + SWATCH_W + 0.05),
        ps.fig_y(fig, line_y),
        text,
        ha="left",
        va="center",
        fontsize=ps.FS_NOTE - 0.5,
        color=ps.INK_SOFT,
    )


# ============================================================
# SAVE
# ============================================================

ps.save(
    fig,
    OUTPUT_STEM,
)
