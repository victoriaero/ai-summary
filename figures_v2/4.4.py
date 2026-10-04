"""
Figure 3 - source-pathway shifts track semantic shifts in the answer.

Full-width figure (figure*): a unit matrix of all 252 explicit-group
conditions. Each row is one matched outcome; its 12 conditions are ordered
by how far their AI Overview lies from the generic answer (label-normalized
semantic distance), closest on the left. Each glyph uses the encoding of
Figure 1: ring colour = majority- or minority-marked query, grey wedge =
URL Jaccard overlap between its cited sources and the generic query's.

Bottom: share of conditions at each position that share at least one cited
source with the generic query.

The prespecified association between source-set displacement and answer
displacement is summarized by the pooled blocked rank correlation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import transforms
from matplotlib.patches import Rectangle
from scipy.stats import rankdata


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import paper_style as ps


# ============================================================
# LOAD RESULTS
# ============================================================

EVIDENCE_DIR = (
    ps.RESULTS_DIR
    / "evidence_synthesis_analysis_dallas_v2"
)

MECHANISM_MAP = (
    EVIDENCE_DIR
    / "mechanism_map_v2.csv"
)

EVIDENCE_RESULTS = (
    EVIDENCE_DIR
    / "all_results_v2.json"
)

OUTPUT_STEM = (
    "figure_4_4_pathway_surface_coupling_v2"
)

SOURCE = "url_jaccard_distance"
ANSWER = "dense_distance_subject_normalized"


data = pd.read_csv(
    MECHANISM_MAP
)

evidence = json.loads(
    EVIDENCE_RESULTS.read_text(
        encoding="utf-8"
    )
)


if (
    len(data) != 252
    or data[[SOURCE, ANSWER]].isna().any().any()
):
    raise RuntimeError(
        "Expected 252 complete explicit-group conditions."
    )


(reported,) = [
    row
    for row in evidence["pipeline"]["blocked_correlations"]
    if (
        row["metric_x"] == SOURCE
        and row["metric_y"] == ANSWER
    )
]


# ============================================================
# REPRODUCE THE BLOCKED RANK CORRELATION
# ============================================================

def centered_ranks(series):

    ranks = rankdata(
        series,
        method="average",
    )

    return ranks - ranks.mean()


blocked_r = np.corrcoef(
    data.groupby("outcome_id")[SOURCE].transform(
        centered_ranks
    ),
    data.groupby("outcome_id")[ANSWER].transform(
        centered_ranks
    ),
)[0, 1]


if not np.isclose(
    blocked_r,
    reported["blocked_rank_correlation"],
    atol=1e-10,
):
    raise RuntimeError(
        f"Blocked correlation mismatch: "
        f"{blocked_r} vs "
        f"{reported['blocked_rank_correlation']}"
    )


# ------------------------------------------------------------
# Compact p-value label for the figure
# ------------------------------------------------------------

p_holm = reported["p_blocked_permutation_holm"]

if p_holm < 0.001:
    p_label = "pHolm < .001"
else:
    p_label = f"pHolm = {p_holm:.3f}"


# ============================================================
# MATRIX LAYOUT
# ============================================================

DOMAINS = {
    "Healthcare": "Healthcare",
    "Employment": "Employment",
    "Education": "Education",
    "Housing": "Housing",
    "Credit & Financial Services": "Credit & finance",
    "Criminal Justice": "Criminal justice",
    "Government Benefits": "Gov. benefits",
}


if set(data["domain"]) != set(DOMAINS):

    raise RuntimeError(
        "Unexpected domains in the mechanism map."
    )


data["share"] = (
    1 - data[SOURCE]
)

data["position"] = (
    data.groupby("outcome_id")[ANSWER]
    .rank(method="first")
    .astype(int)
)


DOMAIN_GAP = 0.45


outcomes = []

y = 0.0


for domain in DOMAINS:

    domain_outcomes = sorted(
        data.loc[
            data["domain"] == domain,
            "outcome",
        ].unique()
    )

    for outcome in domain_outcomes:

        outcomes.append(
            {
                "domain": domain,
                "outcome": outcome,
                "y": y,
            }
        )

        y -= 1

    y -= DOMAIN_GAP


outcomes = pd.DataFrame(
    outcomes
)


outcomes["outcome_id"] = (
    outcomes["domain"]
    + " :: "
    + outcomes["outcome"]
)


if set(outcomes["outcome_id"]) != set(data["outcome_id"]):

    raise RuntimeError(
        "Outcome identifiers do not match the mechanism map."
    )


# ============================================================
# SHARE OF CONDITIONS WITH GENERIC-SOURCE OVERLAP
# ============================================================

by_position = (
    data.groupby("position")["share"]
    .apply(
        lambda s: (s > 0).mean()
    )
)


# ============================================================
# PRINT VALUES USED IN FIGURE
# ============================================================

print(
    "\nVALUES USED IN FIGURE\n"
)


print(
    f"Blocked rank correlation: "
    f"r = {blocked_r:.4f} "
    f"[{reported['cluster_bootstrap_ci_low']:.4f}, "
    f"{reported['cluster_bootstrap_ci_high']:.4f}], "
    f"p_Holm = {p_holm:.4g}"
)


print(
    "\nShare of conditions sharing >= 1 source "
    "with the generic query, by position:"
)


print(
    by_position
    .round(3)
    .to_string()
)


# ============================================================
# DRAW
# ============================================================

ps.use_paper_style()


FIG_W = ps.TEXT_WIDTH


# ============================================================
# LAYOUT
# ============================================================

ROW_PITCH = 0.089

Y_TOP = 0.55
Y_BOTTOM = outcomes["y"].min() - 0.55

MATRIX_H = (
    Y_TOP - Y_BOTTOM
) * ROW_PITCH


# ------------------------------------------------------------
# Left labels
# ------------------------------------------------------------

DOMAIN_X = 0.04
OUTCOME_X = 0.98


# ------------------------------------------------------------
# Matrix
#
# CHANGED:
# The former right-hand rho panel has been removed.
# The central matrix now occupies most of the remaining width.
# ------------------------------------------------------------

MATRIX_LEFT = 2.42
MATRIX_W = FIG_W - MATRIX_LEFT - 0.22


# ------------------------------------------------------------
# Bottom bar chart
# ------------------------------------------------------------

BARS_BOTTOM = 0.20
BARS_H = 0.36


MATRIX_BOTTOM = (
    BARS_BOTTOM
    + BARS_H
    + 0.10
)


FIG_H = (
    MATRIX_BOTTOM
    + MATRIX_H
    + 0.50
)


GLYPH = 5.3

XLIM = (
    0.5,
    12.5,
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
    MATRIX_LEFT,
    MATRIX_BOTTOM,
    MATRIX_W,
    MATRIX_H,
)


ax_bars = ps.add_axes_in(
    fig,
    MATRIX_LEFT,
    BARS_BOTTOM,
    MATRIX_W,
    BARS_H,
)


# ============================================================
# MAIN MATRIX AXIS
# ============================================================

ax.set_xlim(
    *XLIM
)

ax_bars.set_xlim(
    *XLIM
)

ax.set_ylim(
    Y_BOTTOM,
    Y_TOP,
)


ax.set_xticks([])
ax.set_yticks([])


ps.strip_axes(
    ax
)


# ============================================================
# DOMAIN BANDS AND ROW LABELS
# ============================================================

rows_trans = ps.fig_x_data_y(
    fig,
    ax,
)


for index, domain in enumerate(DOMAINS):

    ys = outcomes.loc[
        outcomes["domain"] == domain,
        "y",
    ]


    # --------------------------------------------------------
    # Alternating background bands
    # --------------------------------------------------------

    if index % 2 == 0:

        fig.add_artist(
            Rectangle(
                (
                    ps.fig_x(fig, 0.0),
                    ys.min() - 0.5,
                ),

                ps.fig_x(
                    fig,
                    FIG_W - 0.03,
                ),

                ys.max()
                - ys.min()
                + 1.0,

                transform=rows_trans,

                facecolor=ps.BAND,
                edgecolor="none",

                zorder=-10,
            )
        )


    # --------------------------------------------------------
    # Domain name
    # --------------------------------------------------------

    fig.text(
        ps.fig_x(
            fig,
            DOMAIN_X,
        ),

        ys.mean(),

        DOMAINS[domain],

        transform=rows_trans,

        ha="left",
        va="center",

        fontsize=ps.FS_TICK,
        fontweight="bold",

        color=ps.INK,
    )


# ------------------------------------------------------------
# Outcome labels
# ------------------------------------------------------------

for _, row in outcomes.iterrows():

    fig.text(
        ps.fig_x(
            fig,
            OUTCOME_X,
        ),

        row["y"],

        row["outcome"],

        transform=rows_trans,

        ha="left",
        va="center",

        fontsize=ps.FS_NOTE,
        color=ps.INK_SOFT,
    )


# ============================================================
# GLYPHS
# ============================================================

y_of = (
    outcomes
    .set_index("outcome_id")["y"]
)


for _, row in data.iterrows():

    ps.overlap_glyph(
        ax,

        row["position"],

        y_of[
            row["outcome_id"]
        ],

        row["share"],

        row["condition"],

        diameter=GLYPH,
        ring=0.75,
    )


# ============================================================
# MATRIX HEADER
# ============================================================

HEADER_PT = 20.0
STAT_PT = 11.5
SUBHEADER_PT = 3.5


# ------------------------------------------------------------
# Main title
# ------------------------------------------------------------

ax.text(
    0.5,
    1,

    "Answer distance from the generic answer, "
    "ranked within outcome",

    transform=ps.offset(
        ax.transAxes,
        fig,
        0,
        HEADER_PT,
    ),

    ha="center",
    va="baseline",

    fontsize=ps.FS_NOTE,
    color=ps.INK_SOFT,
)


# ------------------------------------------------------------
# Aggregate association
#
# Replaces the former right-hand rho panel.
# ------------------------------------------------------------

ax.text(
    0.5,
    1,

    f"blocked r = {blocked_r:.2f}, {p_label}",

    transform=ps.offset(
        ax.transAxes,
        fig,
        0,
        STAT_PT,
    ),

    ha="center",
    va="baseline",

    fontsize=ps.FS_NOTE - 0.4,
    color=ps.MUTED,
)


# ------------------------------------------------------------
# Closest / farthest labels
# ------------------------------------------------------------

top_trans = ps.offset(
    transforms.blended_transform_factory(
        ax.transData,
        ax.transAxes,
    ),
    fig,
    0,
    SUBHEADER_PT,
)


ax.text(
    1,
    1,

    "← closest",

    transform=top_trans,

    ha="center",
    va="baseline",

    fontsize=ps.FS_NOTE,
    color=ps.MUTED,
)


ax.text(
    12,
    1,

    "farthest →",

    transform=top_trans,

    ha="center",
    va="baseline",

    fontsize=ps.FS_NOTE,
    color=ps.MUTED,
)


# ============================================================
# BOTTOM PANEL
# SHARE OF CONDITIONS KEEPING A GENERIC SOURCE
# ============================================================

ax_bars.bar(
    by_position.index,
    by_position.values,

    width=0.56,

    color=ps.GENERIC,
    edgecolor="none",

    zorder=2,
)


ax_bars.set_ylim(
    0,
    0.62,
)


ax_bars.set_yticks(
    [
        0,
        0.5,
    ]
)


ax_bars.set_yticklabels(
    [
        "0",
        "50%",
    ]
)


ax_bars.set_xticks(
    range(
        1,
        13,
    )
)


ax_bars.tick_params(
    axis="x",

    length=0,

    pad=1.5,

    labelsize=ps.FS_NOTE - 0.5,

    labelcolor=ps.MUTED,
)


ps.strip_axes(
    ax_bars,
    keep=(
        "left",
        "bottom",
    ),
)


ax_bars.axhline(
    0.5,

    color=ps.GRID,

    linewidth=0.45,

    zorder=0,
)


# ------------------------------------------------------------
# Shortened bottom label
# ------------------------------------------------------------

fig.text(
    ps.fig_x(
        fig,
        MATRIX_LEFT - 0.30,
    ),

    ps.fig_y(
        fig,
        BARS_BOTTOM
        + BARS_H / 2,
    ),

    "Conditions sharing ≥1 source\n"
    "with generic query",

    ha="right",
    va="center",

    fontsize=ps.FS_NOTE,
    color=ps.INK_SOFT,

    linespacing=1.15,
)


# ============================================================
# LEGEND
# Vertical list in upper-left corner
# ============================================================

KEY_LEFT = 0.0
KEY_BOTTOM = FIG_H - 0.47

KEY_W = 2.35
KEY_H = 0.43


key = ps.add_axes_in(
    fig,
    KEY_LEFT,
    KEY_BOTTOM,
    KEY_W,
    KEY_H,
)


key.set_xlim(
    0,
    KEY_W,
)

key.set_ylim(
    0,
    1,
)

key.axis(
    "off"
)


# ------------------------------------------------------------
# Same x-position for all glyphs
# ------------------------------------------------------------

KEY_X = 0.09


KEY_ITEMS = [

    (
        0.78,
        "majority",
        0.0,
        "Majority-marked query",
    ),

    (
        0.48,
        "minority",
        0.0,
        "Minority-marked query",
    ),

    (
        0.18,
        "generic",
        0.3,
        "Sources shared with the generic query",
    ),
]


for (
    y,
    condition,
    share,
    text,
) in KEY_ITEMS:

    ps.overlap_glyph(
        key,

        KEY_X,
        y,

        share,

        condition,

        diameter=6.4,
        ring=0.8,
    )


    key.text(
        KEY_X + 0.08,
        y,

        text,

        ha="left",
        va="center",

        fontsize=ps.FS_NOTE,
        color=ps.INK_SOFT,
    )


# ============================================================
# SAVE
# ============================================================

ps.save(
    fig,
    OUTPUT_STEM,
)