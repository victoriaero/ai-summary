from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/scratch/victoria.estanislau/ai-summary")
RESULTS_DIR = BASE_DIR / "results"

EVIDENCE_DIR = (
    RESULTS_DIR
    / "evidence_synthesis_analysis_dallas_v2"
)

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PNG = (
    FIGURE_DIR
    / "figure_4_5_grounding.png"
)

OUTPUT_PDF = (
    FIGURE_DIR
    / "figure_4_5_grounding.pdf"
)


# ============================================================
# DISCOVER RESULTS FILE
# ============================================================

def try_load_json(path):

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            obj = json.load(f)

        if isinstance(
            obj,
            dict,
        ):

            return obj

    except Exception:

        pass

    return None


def is_evidence_results(obj):

    return (
        "metadata" in obj
        and
        "collection_quality" in obj
        and
        "reference_analysis_50pct" in obj
        and
        "coverage_sensitivity" in obj
        and
        "pipeline" in obj
        and
        obj.get(
            "metadata",
            {}
        ).get(
            "primary_metric"
        )
        ==
        "evidence_semantic_gap_source_balanced"
    )


def find_results_file(
    directory
):

    if not directory.exists():

        raise FileNotFoundError(
            f"\nDirectory does not exist:\n"
            f"{directory}"
        )


    candidates = []


    preferred_names = [

        "all_results.json",

        "all_results_v2.json",

        "results.json",

        "analysis_results.json",

        "summary.json",

        "all_results.txt",

        "results.txt",
    ]


    for name in preferred_names:

        path = (
            directory
            /
            name
        )

        if path.exists():

            candidates.append(
                path
            )


    candidates.extend(
        sorted(
            directory.rglob(
                "*.json"
            )
        )
    )


    candidates.extend(
        sorted(
            directory.rglob(
                "*.txt"
            )
        )
    )


    seen = set()


    for path in candidates:

        key = str(
            path.resolve()
        )


        if key in seen:

            continue


        seen.add(
            key
        )


        obj = try_load_json(
            path
        )


        if (
            obj is not None
            and
            is_evidence_results(
                obj
            )
        ):

            print(
                "Found evidence-synthesis results:"
            )

            print(
                f"  {path}"
            )

            return (
                path,
                obj,
            )


    raise RuntimeError(
        "\nCould not identify the evidence-synthesis "
        f"results file inside:\n{directory}"
    )


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 80)
print("LOCATING EVIDENCE-SYNTHESIS RESULTS")
print("=" * 80)
print()


RESULTS_FILE, data = (
    find_results_file(
        EVIDENCE_DIR
    )
)


# ============================================================
# PRIMARY DIMENSION RESULTS
# ============================================================

ref = pd.DataFrame(

    data[
        "reference_analysis_50pct"
    ][
        "dimension_tests"
    ]
)


ORDER = [

    "Race",

    "Ethnicity",

    "Gender",

    "Disability",

    "Sexual Orientation",

    "Gender Identity",
]


missing = (
    set(
        ORDER
    )
    -
    set(
        ref[
            "dimension"
        ]
    )
)


if missing:

    raise RuntimeError(
        "Missing dimensions in reference analysis: "
        f"{sorted(missing)}"
    )


ref = (

    ref
    .set_index(
        "dimension"
    )
    .loc[
        ORDER
    ]
    .reset_index()
)


# ============================================================
# AGGREGATE 50% RESULT
# ============================================================

agg = pd.DataFrame(

    data[
        "coverage_sensitivity"
    ][
        "aggregate_outcome_blocked"
    ]
)


agg50_rows = agg.loc[

    np.isclose(

        agg[
            "coverage_threshold"
        ]
        .astype(float),

        0.5,
    )
]


if len(
    agg50_rows
) != 1:

    raise RuntimeError(
        "Could not uniquely identify the aggregate "
        "50% coverage-threshold result."
    )


agg50 = (
    agg50_rows.iloc[0]
)


# ============================================================
# BLOCKED ASSOCIATION:
# EVIDENCE ALIGNMENT <-> ANSWER DISPLACEMENT
#
# The stored metric is semantic GAP:
#   larger gap = weaker alignment
#
# To make the figure intuitive, flip the sign:
#   positive = more answer displacement is associated with
#              stronger evidence alignment.
# ============================================================

blocked_candidates = [

    row

    for row
    in data[
        "pipeline"
    ][
        "blocked_correlations"
    ]

    if (
        row.get(
            "metric_x"
        )
        ==
        "evidence_semantic_gap_source_balanced"

        and

        row.get(
            "metric_y"
        )
        ==
        "dense_distance_subject_normalized"
    )
]


if len(
    blocked_candidates
) != 1:

    raise RuntimeError(
        "Could not uniquely identify the evidence-gap "
        "-> answer-displacement blocked association."
    )


blocked = (
    blocked_candidates[
        0
    ]
)


alignment_r = (
    -
    float(
        blocked[
            "blocked_rank_correlation"
        ]
    )
)


alignment_ci_low = (
    -
    float(
        blocked[
            "cluster_bootstrap_ci_high"
        ]
    )
)


alignment_ci_high = (
    -
    float(
        blocked[
            "cluster_bootstrap_ci_low"
        ]
    )
)


alignment_n = int(
    blocked[
        "n_observations"
    ]
)


alignment_p = float(
    blocked[
        "p_blocked_permutation_holm"
    ]
)


# ============================================================
# COVERAGE DIAGNOSTIC
# ============================================================

coverage_candidates = [

    row

    for row
    in data[
        "collection_quality"
    ][
        "metric_vs_coverage"
    ]

    if (
        row.get(
            "metric"
        )
        ==
        "evidence_alignment_source_balanced"

        and

        row.get(
            "predictor"
        )
        ==
        "fetch_coverage"
    )
]


if len(
    coverage_candidates
) != 1:

    raise RuntimeError(
        "Could not identify the evidence-alignment "
        "vs fetch-coverage diagnostic."
    )


coverage_metric = (
    coverage_candidates[
        0
    ]
)


coverage_rho = float(
    coverage_metric[
        "spearman_rho"
    ]
)


reference_eligible = int(

    data[
        "collection_quality"
    ][
        "reference_eligible_queries"
    ]
)


total_queries = int(

    data[
        "collection_quality"
    ][
        "n_queries"
    ]
)


mean_fetch_coverage = float(

    data[
        "collection_quality"
    ][
        "mean_fetch_coverage"
    ]
)


# ============================================================
# PAPER PALETTE
# ============================================================

COL_FOCAL = "#D55E00"

COL_COMPARISON = "#0072B2"

COL_EVIDENCE = "#009E73"

COL_SYNTHESIS = "#CC79A7"

COL_INK = "#253142"

COL_MUTED = "#667085"

COL_GRID = "#DDE3EA"

COL_LIGHT = "#F5F7FA"

COL_BG = "#FFFFFF"


# ============================================================
# BUILD FOREST TABLE
# ============================================================

rows = []


for _, row in (
    ref.iterrows()
):

    rows.append({

        "label":
            row[
                "dimension"
            ],

        "delta":
            float(
                row[
                    "mean_difference"
                ]
            ),

        "ci_low":
            float(
                row[
                    "bootstrap_ci_low"
                ]
            ),

        "ci_high":
            float(
                row[
                    "bootstrap_ci_high"
                ]
            ),

        "sig_holm":
            bool(
                row[
                    "significant_permutation_holm"
                ]
            ),

        "overall":
            False,
    })


rows.append({

    "label":
        "Overall",

    "delta":
        float(
            agg50[
                "mean_difference"
            ]
        ),

    "ci_low":
        float(
            agg50[
                "bootstrap_ci_low"
            ]
        ),

    "ci_high":
        float(
            agg50[
                "bootstrap_ci_high"
            ]
        ),

    "sig_holm":
        False,

    "overall":
        True,
})


plot_df = pd.DataFrame(
    rows
)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(

    figsize=(
        12.1,
        6.35,
    ),

    facecolor=
        COL_BG,
)


gs = GridSpec(

    1,

    2,

    width_ratios=[
        2.65,
        1.0,
    ],

    wspace=0.26,
)


ax = fig.add_subplot(
    gs[
        0,
        0,
    ]
)


ax_r = fig.add_subplot(
    gs[
        0,
        1,
    ]
)


for axis in [
    ax,
    ax_r,
]:

    axis.set_facecolor(
        COL_BG
    )


# ============================================================
# LEFT PANEL
# ============================================================

y = (
    np.arange(
        len(
            plot_df
        )
    )[::-1]
)


xlim = (
    -0.135,
    0.105,
)


# Very light directional backgrounds
ax.axvspan(

    xlim[
        0
    ],

    0,

    color=
        COL_FOCAL,

    alpha=0.035,

    zorder=0,
)


ax.axvspan(

    0,

    xlim[
        1
    ],

    color=
        COL_COMPARISON,

    alpha=0.025,

    zorder=0,
)


ax.axvline(

    0,

    color=
        COL_GRID,

    linewidth=1.2,

    zorder=1,
)


# Overall row band
overall_index = (
    plot_df.index[
        plot_df[
            "overall"
        ]
    ][0]
)


overall_y = (
    y[
        overall_index
    ]
)


ax.axhspan(

    overall_y
    -
    0.42,

    overall_y
    +
    0.42,

    color=
        COL_LIGHT,

    zorder=0,
)


for idx, row in (
    plot_df.iterrows()
):

    yy = (
        y[
            idx
        ]
    )


    delta = (
        row[
            "delta"
        ]
    )


    ci_low = (
        row[
            "ci_low"
        ]
    )


    ci_high = (
        row[
            "ci_high"
        ]
    )


    base = (

        COL_FOCAL

        if delta < 0

        else COL_COMPARISON
    )


    if row[
        "overall"
    ]:

        line_color = (
            base
        )

        line_width = (
            3.0
        )

        alpha = 1.0

        marker = "D"

        marker_size = 105


    else:

        line_color = (

            base

            if row[
                "sig_holm"
            ]

            else COL_MUTED
        )


        line_width = (

            2.5

            if row[
                "sig_holm"
            ]

            else 1.8
        )


        alpha = (

            1.0

            if row[
                "sig_holm"
            ]

            else 0.75
        )


        marker = "o"

        marker_size = 70


    ax.hlines(

        yy,

        ci_low,

        ci_high,

        color=
            line_color,

        linewidth=
            line_width,

        alpha=
            alpha,

        zorder=2,
    )


    ax.scatter(

        [
            delta
        ],

        [
            yy
        ],

        s=
            marker_size,

        marker=
            marker,

        facecolor=
            base,

        edgecolor=
            "white",

        linewidth=
            0.8,

        zorder=3,
    )


    label = (
        row[
            "label"
        ]
        +
        (
            " *"

            if row[
                "sig_holm"
            ]

            else ""
        )
    )


    ax.text(

        xlim[
            0
        ]
        -
        0.006,

        yy,

        label,

        ha="right",

        va="center",

        fontsize=10.5,

        fontweight=(

            "bold"

            if row[
                "overall"
            ]

            else "normal"
        ),

        color=
            COL_INK,
    )


# Direction labels
ax.text(

    xlim[
        0
    ]
    +
    0.002,

    y[
        0
    ]
    +
    0.76,

    "FOCAL MORE ALIGNED",

    ha="left",

    va="bottom",

    fontsize=9.3,

    fontweight="bold",

    color=
        COL_FOCAL,
)


ax.text(

    xlim[
        1
    ]
    -
    0.002,

    y[
        0
    ]
    +
    0.76,

    "COMPARISON MORE ALIGNED",

    ha="right",

    va="bottom",

    fontsize=9.3,

    fontweight="bold",

    color=
        COL_COMPARISON,
)


ax.set_xlim(
    *xlim
)


ax.set_ylim(

    -0.75,

    y[
        0
    ]
    +
    1.05,
)


ax.set_yticks(
    []
)


ax.set_xticks(
    [
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
    ]
)


ax.tick_params(

    axis="x",

    labelsize=8.8,

    colors=
        COL_MUTED,
)


ax.set_xlabel(

    (
        "Focal - comparison evidence semantic gap\n"
        "(lower gap = stronger alignment to cited evidence)"
    ),

    fontsize=9.7,

    color=
        COL_INK,

    labelpad=10,
)


for spine in (
    ax.spines.values()
):

    spine.set_visible(
        False
    )


ax.set_title(

    "ALIGNMENT WITH EACH CONDITION'S OWN EVIDENCE",

    loc="left",

    fontsize=10.6,

    fontweight="bold",

    color=
        COL_EVIDENCE,

    pad=23,
)


ax.text(

    0.0,

    1.018,

    (
        "50% recovered-source threshold · "
        "* Holm-corrected dimension effect"
    ),

    transform=
        ax.transAxes,

    ha="left",

    va="bottom",

    fontsize=8.7,

    color=
        COL_MUTED,
)


# ============================================================
# RIGHT PANEL
# ============================================================

ax_r.axvline(

    0,

    color=
        COL_GRID,

    linewidth=1.2,

    zorder=1,
)


ax_r.axvspan(

    0,

    0.58,

    color=
        COL_EVIDENCE,

    alpha=0.045,

    zorder=0,
)


gauge_y = 0.53


ax_r.hlines(

    gauge_y,

    alignment_ci_low,

    alignment_ci_high,

    color=
        COL_EVIDENCE,

    linewidth=5.0,

    zorder=2,
)


ax_r.scatter(

    [
        alignment_r
    ],

    [
        gauge_y
    ],

    s=150,

    marker="D",

    facecolor=
        COL_SYNTHESIS,

    edgecolor=
        "white",

    linewidth=1.0,

    zorder=3,
)


ax_r.text(

    0.50,

    0.91,

    (
        "ANSWER DIVERGENCE\n"
        "TRACKS STRONGER ALIGNMENT"
    ),

    transform=
        ax_r.transAxes,

    ha="center",

    va="top",

    fontsize=10.5,

    fontweight="bold",

    color=
        COL_SYNTHESIS,

    linespacing=1.12,
)


ax_r.text(

    0.50,

    0.79,

    (
        "with the evidence surfaced\n"
        "for that same condition"
    ),

    transform=
        ax_r.transAxes,

    ha="center",

    va="top",

    fontsize=8.9,

    color=
        COL_INK,

    linespacing=1.15,
)


ax_r.text(

    alignment_r,

    gauge_y
    +
    0.11,

    rf"$r_{{blocked}}={alignment_r:.3f}$",

    ha="center",

    va="bottom",

    fontsize=11.4,

    fontweight="bold",

    color=
        COL_INK,
)


if alignment_p < 0.001:

    p_label = (
        r"$p_{\mathrm{Holm}}<.001$"
    )

else:

    p_label = (
        rf"$p_{{\mathrm{{Holm}}}}={alignment_p:.3f}$"
    )


ax_r.text(

    alignment_r,

    gauge_y
    -
    0.105,

    (
        rf"95% CI [{alignment_ci_low:.3f}, "
        rf"{alignment_ci_high:.3f}]"
        "\n"
        rf"$n={alignment_n}$ · "
        +
        p_label
    ),

    ha="center",

    va="top",

    fontsize=8.4,

    color=
        COL_MUTED,

    linespacing=1.35,
)


ax_r.text(

    0.02,

    0.33,

    "weaker alignment",

    transform=
        ax_r.transAxes,

    ha="left",

    va="center",

    fontsize=8.4,

    color=
        COL_MUTED,
)


ax_r.text(

    0.98,

    0.33,

    "stronger alignment",

    transform=
        ax_r.transAxes,

    ha="right",

    va="center",

    fontsize=8.4,

    color=
        COL_EVIDENCE,
)


ax_r.text(

    0.50,

    0.155,

    (
        f"{reference_eligible}/{total_queries} eligible at "
        f"50% recovery · mean coverage "
        f"{mean_fetch_coverage:.0%}\n"
        rf"alignment vs. coverage $\rho={coverage_rho:.03f}$ "
        "(ns)"
    ),

    transform=
        ax_r.transAxes,

    ha="center",

    va="center",

    fontsize=8.1,

    color=
        COL_MUTED,

    linespacing=1.35,
)


ax_r.set_xlim(
    -0.55,
    0.55,
)


ax_r.set_ylim(
    0,
    1,
)


ax_r.set_yticks(
    []
)


ax_r.set_xticks(
    [
        -0.5,
        0.0,
        0.5,
    ]
)


ax_r.tick_params(

    axis="x",

    labelsize=8.5,

    colors=
        COL_MUTED,
)


ax_r.set_xlabel(

    (
        "Association between answer displacement\n"
        "and evidence alignment"
    ),

    fontsize=9.2,

    color=
        COL_INK,

    labelpad=9,
)


for spine in (
    ax_r.spines.values()
):

    spine.set_visible(
        False
    )


ax_r.set_title(

    "DIVERGENCE AND GROUNDING",

    loc="left",

    fontsize=10.6,

    fontweight="bold",

    color=
        COL_SYNTHESIS,

    pad=23,
)


# ============================================================
# GLOBAL TITLE / FOOTER
# ============================================================

fig.suptitle(

    (
        "Semantic divergence is not accompanied by "
        "weaker evidentiary alignment"
    ),

    x=0.07,

    y=0.975,

    ha="left",

    fontsize=15.0,

    fontweight="bold",

    color=
        COL_INK,
)


fig.text(

    0.07,

    0.935,

    (
        "Dallas · source-balanced semantic alignment using "
        "query-relevant passages from recovered cited sources"
    ),

    ha="left",

    va="top",

    fontsize=9.4,

    color=
        COL_MUTED,
)


fig.text(

    0.07,

    0.025,

    (
        "Semantic alignment is a similarity measure, not a test "
        "of factual support or claim entailment. Overall uses the "
        "prespecified outcome-level aggregate at the 50% threshold."
    ),

    ha="left",

    va="bottom",

    fontsize=8.1,

    color=
        COL_MUTED,
)


fig.subplots_adjust(

    left=0.16,

    right=0.97,

    bottom=0.16,

    top=0.84,

    wspace=0.26,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(

    OUTPUT_PNG,

    dpi=350,

    bbox_inches="tight",

    facecolor=
        COL_BG,
)


fig.savefig(

    OUTPUT_PDF,

    bbox_inches="tight",

    facecolor=
        COL_BG,
)


print()
print("=" * 80)
print("FIGURE SAVED")
print("=" * 80)
print()

print(f"PNG:\n{OUTPUT_PNG}\n")
print(f"PDF:\n{OUTPUT_PDF}\n")

plt.show()
