"""
Figure 1 - identity marking changes how many sources are cited and which.

Double-column figure. Each row joins the majority-marked (blue) and the
minority-marked (orange) query of one social dimension. A glyph's position is
the mean number of cited sources per answer (evidence volume); its grey wedge
is the mean URL Jaccard overlap between its source set and the generic
query's (evidence composition). The generic query sits on top as a full grey
disc, and its volume is carried down as a reference line.

The two columns on the right are the paired focal - comparison differences:
cited sources (Wilcoxon) and overlap in percentage points (sign-flip test on
Jaccard distance); bold values are significant after Holm correction across
the six dimensions, and the Overall row is the single prespecified aggregate.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# IMPORT LOCAL STYLE
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as ps


# ============================================================
# LOAD RESULTS
# ============================================================

LINK_RESULTS = (
    ps.RESULTS_DIR
    / "link_count_analysis_dallas"
    / "all_results.json"
)

OVERLAP_RESULTS = (
    ps.RESULTS_DIR
    / "source_overlap_analysis_dallas"
    / "all_results.json"
)

OUTPUT_STEM = "figure_4_2_evidentiary_pathway"


link = json.loads(
    LINK_RESULTS.read_text(encoding="utf-8")
)

overlap = json.loads(
    OVERLAP_RESULTS.read_text(encoding="utf-8")
)


# ============================================================
# DESCRIPTIVE RESULTS
# ============================================================

sources = {
    row["group"]: row["mean_links"]
    for row in link["descriptive_results"]["by_group"]
}

jaccard = {
    row["group"]: row["mean_url_jaccard"]
    for row in overlap["descriptive"]["by_group"]
}


# ============================================================
# STATISTICAL TESTS
# ============================================================

volume_tests = pd.DataFrame(
    link["statistical_tests"]["minority_vs_majority_by_dimension"]
).set_index("dimension")


composition_tests = pd.DataFrame(
    overlap["minority_vs_majority"]["url_jaccard_distance"]
).set_index("dimension")


volume_agg = (
    link["statistical_tests"]["aggregate_minority_vs_majority"]
)


(composition_agg,) = [
    row
    for row in overlap["minority_vs_majority"]["aggregate"]
    if row["metric"] == "url_jaccard_distance"
]


# ============================================================
# BUILD TABLE USED BY FIGURE
# ============================================================

rows = []


for dimension in ps.DIMENSIONS:

    volume = volume_tests.loc[dimension]
    composition = composition_tests.loc[dimension]

    minority = volume["minority_group"]
    majority = volume["majority_group"]

    if (
        minority,
        majority,
    ) != (
        composition["minority_group"],
        composition["majority_group"],
    ):
        raise RuntimeError(
            f"Group mismatch for {dimension}."
        )

    rows.append(
        {
            "label": ps.DIMENSION_LABELS[dimension],

            "sources_majority": sources[majority],
            "sources_minority": sources[minority],

            "overlap_majority": jaccard[majority],
            "overlap_minority": jaccard[minority],

            "d_sources": volume["mean_difference"],
            "sig_sources": bool(
                volume["significant_holm"]
            ),

            # The statistical test is based on Jaccard distance.
            # Because overlap = 1 - distance, the difference
            # changes sign when expressed as overlap.
            "d_overlap": (
                -100 * composition["mean_difference"]
            ),

            "sig_overlap": bool(
                composition["significant_permutation_holm"]
            ),
        }
    )


# ------------------------------------------------------------
# Overall row
# ------------------------------------------------------------

rows.append(
    {
        "label": "Overall",

        "sources_majority": volume_agg["majority_mean"],
        "sources_minority": volume_agg["minority_mean"],

        "overlap_majority": (
            1 - composition_agg["majority_mean"]
        ),
        "overlap_minority": (
            1 - composition_agg["minority_mean"]
        ),

        "d_sources": volume_agg["mean_difference"],

        "sig_sources": (
            volume_agg["p_value"] < 0.05
        ),

        "d_overlap": (
            -100 * composition_agg["mean_difference"]
        ),

        "sig_overlap": (
            composition_agg["p_permutation_raw"] < 0.05
        ),
    }
)


table = pd.DataFrame(rows).set_index("label")


GENERIC_SOURCES = sources["people"]


# ============================================================
# PRINT VALUES USED IN FIGURE
# ============================================================

print("\nVALUES USED IN FIGURE\n")

print(
    f"Generic query: "
    f"{GENERIC_SOURCES:.3f} cited sources\n"
)

print(
    table.round(4).to_string()
)


# ============================================================
# DRAW
# DOUBLE-COLUMN / LANDSCAPE-STYLE VERSION
# ============================================================

ps.use_paper_style()


# ------------------------------------------------------------
# Figure dimensions
# ------------------------------------------------------------

# Prefer a width defined by paper_style.py if available.
# Otherwise use 7 inches, a common full-text / double-column width.
FIG_W = getattr(
    ps,
    "DOUBLE_COLUMN_WIDTH",
    getattr(ps, "TEXT_WIDTH", 7.0),
)

FIG_H = 2.30


# ------------------------------------------------------------
# Main plot geometry
#
# Layout:
#
#   labels       quantitative panel             statistics
#   |-----|--------------------------------|----------------|
#
# ------------------------------------------------------------

PLOT_LEFT = 1.48
PLOT_W = 3.55

PLOT_BOTTOM = 0.53
PLOT_H = 1.63


# Centres of the two numerical columns, in figure inches.
COLUMN_X = {
    "d_sources": 5.65,
    "d_overlap": 6.48,
}


# ------------------------------------------------------------
# Vertical positions
# ------------------------------------------------------------

GENERIC_Y = 7.45

ROW_Y = [
    6,
    5,
    4,
    3,
    2,
    1,
    -0.45,
]

YLIM = (
    -1.05,
    8.05,
)


# Glyph diameter
GLYPH = 9.4


# ============================================================
# CREATE FIGURE AND MAIN AXIS
# ============================================================

fig = plt.figure(
    figsize=(FIG_W, FIG_H)
)

ax = ps.add_axes_in(
    fig,
    PLOT_LEFT,
    PLOT_BOTTOM,
    PLOT_W,
    PLOT_H,
)


# ------------------------------------------------------------
# Main x/y axes
# ------------------------------------------------------------

ax.set_xlim(
    3.6,
    10.6,
)

ax.set_ylim(
    *YLIM
)

ax.set_xticks(
    [4, 6, 8, 10]
)

ax.set_yticks([])


ps.strip_axes(
    ax,
    keep=("bottom",),
)


ax.set_xlabel(
    "Mean cited sources per answer"
)


# ------------------------------------------------------------
# Vertical grid lines
# ------------------------------------------------------------

for tick in [4, 6, 8, 10]:

    ax.axvline(
        tick,
        color=ps.GRID,
        linewidth=0.45,
        zorder=0,
    )


# ============================================================
# GENERIC QUERY
# ============================================================

# Vertical reference line at the generic-query source count.
ax.plot(
    [
        GENERIC_SOURCES,
        GENERIC_SOURCES,
    ],
    [
        YLIM[0],
        GENERIC_Y,
    ],
    color=ps.GENERIC,
    linewidth=0.7,
    zorder=1,
)


# Generic query glyph:
# full grey disc = complete overlap with itself.
ps.overlap_glyph(
    ax,
    GENERIC_SOURCES,
    GENERIC_Y,
    1.0,
    "generic",
    diameter=GLYPH,
)


# ============================================================
# DIMENSION ROWS
# ============================================================

for (label, row), y in zip(
    table.iterrows(),
    ROW_Y,
):

    # --------------------------------------------------------
    # Horizontal connector:
    # majority-marked -> minority-marked
    # --------------------------------------------------------

    ax.plot(
        [
            row["sources_majority"],
            row["sources_minority"],
        ],
        [
            y,
            y,
        ],
        color=ps.RULE,
        linewidth=1.3,
        solid_capstyle="butt",
        zorder=2,
    )


    # --------------------------------------------------------
    # Majority-marked glyph
    # --------------------------------------------------------

    ps.overlap_glyph(
        ax,
        row["sources_majority"],
        y,
        row["overlap_majority"],
        "majority",
        diameter=GLYPH,
    )


    # --------------------------------------------------------
    # Minority-marked glyph
    # --------------------------------------------------------

    ps.overlap_glyph(
        ax,
        row["sources_minority"],
        y,
        row["overlap_minority"],
        "minority",
        diameter=GLYPH,
    )


# ============================================================
# LABELS + DIFFERENCE COLUMNS
# ============================================================

# Transformation that combines:
# figure coordinates for x
# data coordinates for y
rows_trans = ps.fig_x_data_y(
    fig,
    ax,
)


# Left edge for row labels
label_x = ps.fig_x(
    fig,
    0.06,
)


# ------------------------------------------------------------
# Generic-query label
# ------------------------------------------------------------

fig.text(
    label_x,
    GENERIC_Y,
    "Generic (“people”)",
    transform=rows_trans,
    ha="left",
    va="center",
    fontsize=ps.FS_LABEL,
    color=ps.INK_SOFT,
)


# ------------------------------------------------------------
# Dimension labels + differences
# ------------------------------------------------------------

for (label, row), y in zip(
    table.iterrows(),
    ROW_Y,
):

    # Dimension name
    fig.text(
        label_x,
        y,
        label,
        transform=rows_trans,
        ha="left",
        va="center",
        fontsize=ps.FS_LABEL,
        color=ps.INK,
        fontweight=(
            "bold"
            if label == "Overall"
            else "normal"
        ),
    )


    # --------------------------------------------------------
    # Difference values
    # --------------------------------------------------------

    for column, decimals in [
        ("d_sources", 1),
        ("d_overlap", 1),
    ]:

        if column == "d_sources":

            significant = row[
                "sig_sources"
            ]

        else:

            significant = row[
                "sig_overlap"
            ]


        fig.text(
            ps.fig_x(
                fig,
                COLUMN_X[column],
            ),
            y,

            ps.format_signed(
                row[column],
                decimals,
            ),

            transform=rows_trans,

            ha="center",
            va="center",

            fontsize=ps.FS_TICK,

            fontweight=(
                "bold"
                if significant
                else "normal"
            ),

            color=(
                ps.INK
                if significant
                else ps.MUTED
            ),
        )


# ============================================================
# HORIZONTAL SEPARATORS
# ============================================================

separator_y = [
    # Between generic row and first dimension
    (GENERIC_Y + ROW_Y[0]) / 2,

    # Between final dimension and Overall
    (ROW_Y[-2] + ROW_Y[-1]) / 2,
]


for y in separator_y:

    fig.add_artist(
        Line2D(
            [
                label_x,
                ps.fig_x(
                    fig,
                    FIG_W - 0.06,
                ),
            ],
            [
                y,
                y,
            ],
            transform=rows_trans,
            color=ps.RULE,
            linewidth=0.5,
        )
    )


# ============================================================
# DIFFERENCE-COLUMN HEADER
# ============================================================

# Wider spanner because this is now a double-column figure.
span_left = (
    COLUMN_X["d_sources"]
    - 0.34
)

span_right = (
    COLUMN_X["d_overlap"]
    + 0.34
)

span_centre = (
    span_left
    + span_right
) / 2


# ------------------------------------------------------------
# Spanner title
# ------------------------------------------------------------

fig.text(
    ps.fig_x(
        fig,
        span_centre,
    ),

    GENERIC_Y,

    "Focal − comparison",

    transform=ps.offset(
        rows_trans,
        fig,
        y_pt=4.5,
    ),

    ha="center",
    va="center",

    fontsize=ps.FS_NOTE,
    color=ps.INK_SOFT,
)


# ------------------------------------------------------------
# Horizontal line underneath spanner
# ------------------------------------------------------------

fig.add_artist(
    Line2D(
        [
            ps.fig_x(
                fig,
                span_left,
            ),

            ps.fig_x(
                fig,
                span_right,
            ),
        ],

        [
            GENERIC_Y,
            GENERIC_Y,
        ],

        transform=ps.offset(
            rows_trans,
            fig,
            y_pt=-0.5,
        ),

        color=ps.RULE,
        linewidth=0.5,
    )
)


# ------------------------------------------------------------
# Column headings
# ------------------------------------------------------------

for column, text in [
    ("d_sources", "sources"),
    ("d_overlap", "overlap"),
]:

    fig.text(
        ps.fig_x(
            fig,
            COLUMN_X[column],
        ),

        GENERIC_Y,

        text,

        transform=ps.offset(
            rows_trans,
            fig,
            y_pt=-5.0,
        ),

        ha="center",
        va="center",

        fontsize=ps.FS_NOTE,
        color=ps.INK_SOFT,
    )


# ============================================================
# LEGEND / KEY
# ============================================================

key = ps.add_axes_in(
    fig,
    0.0,
    0.0,
    FIG_W,
    0.15,
)

key.set_xlim(0, FIG_W)
key.set_ylim(0, 1)
key.axis("off")


# ------------------------------------------------------------
# Center legend as a single visual group
# ------------------------------------------------------------

KEY_REL_X = [
    0.00,
    1.60,
    3.20,
]

KEY_GROUP_W = 4.75

# Base geometric centering
KEY_START = (FIG_W - KEY_GROUP_W) / 2

# Optical correction:
# the long text on the right makes the legend look left-shifted.
KEY_CENTER_OFFSET = 0.22

KEY_START += KEY_CENTER_OFFSET


KEY_ITEMS = [
    (
        KEY_START + KEY_REL_X[0],
        "majority",
        0.0,
        "Majority-marked",
    ),
    (
        KEY_START + KEY_REL_X[1],
        "minority",
        0.0,
        "Minority-marked",
    ),
    (
        KEY_START + KEY_REL_X[2],
        "generic",
        0.3,
        "Shared with generic",
    ),
]


for x, condition, share, text in KEY_ITEMS:

    ps.overlap_glyph(
        key,
        x,
        0.5,
        share,
        condition,
        diameter=7.2,
    )

    key.text(
        x + 0.13,
        0.5,
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