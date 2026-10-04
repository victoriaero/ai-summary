from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/scratch/victoria.estanislau/ai-summary"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

LINK_DIR = (
    RESULTS_DIR
    / "link_count_analysis_dallas"
)

OVERLAP_DIR = (
    RESULTS_DIR
    / "source_overlap_analysis_dallas"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "/scratch/victoria.estanislau/ai-summary/figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PNG = (
    FIGURE_DIR
    / "figure_4_2_evidentiary_pathway.png"
)

OUTPUT_PDF = (
    FIGURE_DIR
    / "figure_4_2_evidentiary_pathway.pdf"
)


# ============================================================
# JSON DISCOVERY
#
# We do not assume an exact filename.
# Instead, find the JSON/TXT file containing the expected keys.
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


def find_result_file(
    directory,
    validator,
    description,
):

    if not directory.exists():

        raise FileNotFoundError(
            f"\nDirectory does not exist:\n"
            f"{directory}"
        )


    candidates = []


    # Prefer likely result filenames first.
    preferred_names = [

        "all_results.json",
        "results.json",
        "analysis_results.json",
        "summary.json",
        "all_results.txt",
        "results.txt",
    ]


    for name in preferred_names:

        path = directory / name

        if path.exists():
            candidates.append(
                path
            )


    # Then search recursively.
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


    # Remove duplicates while preserving order.
    seen = set()

    unique_candidates = []

    for path in candidates:

        resolved = str(
            path.resolve()
        )

        if resolved in seen:
            continue

        seen.add(
            resolved
        )

        unique_candidates.append(
            path
        )


    for path in unique_candidates:

        obj = try_load_json(
            path
        )

        if obj is None:
            continue


        try:

            if validator(
                obj
            ):

                print(
                    f"Found {description}:"
                )

                print(
                    f"  {path}"
                )

                return (
                    path,
                    obj,
                )

        except Exception:
            continue


    raise RuntimeError(
        f"\nCould not identify {description} "
        f"inside:\n{directory}\n\n"
        f"Checked {len(unique_candidates)} JSON/TXT files."
    )


# ============================================================
# RESULT VALIDATORS
# ============================================================

def is_link_count_results(
    obj
):

    return (

        "statistical_tests"
        in obj

        and

        "minority_vs_majority_by_dimension"
        in obj[
            "statistical_tests"
        ]

        and

        "aggregate_minority_vs_majority"
        in obj[
            "statistical_tests"
        ]
    )


def is_source_overlap_results(
    obj
):

    return (

        "minority_vs_majority"
        in obj

        and

        "url_jaccard_distance"
        in obj[
            "minority_vs_majority"
        ]

        and

        "aggregate"
        in obj[
            "minority_vs_majority"
        ]
    )


# ============================================================
# LOAD RESULTS
# ============================================================

print()
print(
    "=" * 80
)

print(
    "LOCATING RESULT FILES"
)

print(
    "=" * 80
)

print()


link_path, link = (
    find_result_file(

        LINK_DIR,

        is_link_count_results,

        "link-count results",
    )
)


print()


overlap_path, overlap = (
    find_result_file(

        OVERLAP_DIR,

        is_source_overlap_results,

        "source-overlap results",
    )
)


# ============================================================
# EXTRACT TABLES
# ============================================================

link_df = pd.DataFrame(

    link[
        "statistical_tests"
    ][
        "minority_vs_majority_by_dimension"
    ]
)


overlap_df = pd.DataFrame(

    overlap[
        "minority_vs_majority"
    ][
        "url_jaccard_distance"
    ]
)


# ============================================================
# PAPER ORDER
# ============================================================

ORDER = [

    "Race",

    "Ethnicity",

    "Gender",

    "Disability",

    "Sexual Orientation",

    "Gender Identity",
]


missing_link_dimensions = (

    set(
        ORDER
    )
    -
    set(
        link_df[
            "dimension"
        ]
    )
)


missing_overlap_dimensions = (

    set(
        ORDER
    )
    -
    set(
        overlap_df[
            "dimension"
        ]
    )
)


if missing_link_dimensions:

    raise RuntimeError(
        "Missing dimensions in link-count results: "
        f"{sorted(missing_link_dimensions)}"
    )


if missing_overlap_dimensions:

    raise RuntimeError(
        "Missing dimensions in source-overlap results: "
        f"{sorted(missing_overlap_dimensions)}"
    )


link_df = (

    link_df
    .set_index(
        "dimension"
    )
    .loc[
        ORDER
    ]
    .reset_index()
)


overlap_df = (

    overlap_df
    .set_index(
        "dimension"
    )
    .loc[
        ORDER
    ]
    .reset_index()
)


# ============================================================
# AGGREGATE RESULTS
# ============================================================

link_agg = (

    link[
        "statistical_tests"
    ][
        "aggregate_minority_vs_majority"
    ]
)


overlap_aggregate_candidates = [

    result

    for result
    in overlap[
        "minority_vs_majority"
    ][
        "aggregate"
    ]

    if (
        result.get(
            "metric"
        )
        ==
        "url_jaccard_distance"
    )
]


if (
    len(
        overlap_aggregate_candidates
    )
    != 1
):

    raise RuntimeError(
        "Could not uniquely identify aggregate "
        "URL Jaccard-distance result."
    )


overlap_agg = (
    overlap_aggregate_candidates[
        0
    ]
)


# ============================================================
# PAPER PALETTE
# ============================================================

# Conditions
COL_FOCAL = "#D55E00"
COL_COMPARISON = "#0072B2"
COL_GENERIC = "#6B7280"

# Analytical layers
COL_PATHWAY = "#009E73"
COL_SYNTHESIS = "#CC79A7"

# Supporting colors
COL_INK = "#253142"
COL_MUTED = "#667085"
COL_GRID = "#DDE3EA"
COL_BG = "#FFFFFF"
COL_LIGHT = "#F5F7FA"


# ============================================================
# HELPERS
# ============================================================

def hex_to_rgb(
    hexcolor
):

    h = hexcolor.lstrip(
        "#"
    )

    return np.array(
        [
            int(
                h[i:i + 2],
                16,
            )

            for i
            in (
                0,
                2,
                4,
            )
        ]
    ) / 255.0


def blend_with_white(
    hexcolor,
    strength,
):

    color = hex_to_rgb(
        hexcolor
    )

    return tuple(
        1
        -
        strength
        *
        (
            1
            -
            color
        )
    )


def fmt_p(
    p
):

    if p < 0.001:
        return "p < .001"

    return (
        f"p = {p:.3f}"
        .replace(
            "0.",
            ".",
        )
    )


# ============================================================
# BUILD FIGURE DATA
# ============================================================

rows = []


for dimension in ORDER:

    link_row = (
        link_df[
            link_df[
                "dimension"
            ]
            ==
            dimension
        ]
        .iloc[0]
    )


    overlap_row = (
        overlap_df[
            overlap_df[
                "dimension"
            ]
            ==
            dimension
        ]
        .iloc[0]
    )


    rows.append({

        "dimension":
            dimension,

        # --------------------------------------------
        # Evidence volume
        # --------------------------------------------

        "vol_delta":
            float(
                link_row[
                    "mean_difference"
                ]
            ),

        "vol_rb":
            abs(
                float(
                    link_row[
                        "rank_biserial"
                    ]
                )
            ),

        "vol_p":
            float(
                link_row[
                    "p_holm"
                ]
            ),

        "vol_sig":
            bool(
                link_row[
                    "significant_holm"
                ]
            ),

        # --------------------------------------------
        # Source-set displacement
        # --------------------------------------------

        "path_delta":
            float(
                overlap_row[
                    "mean_difference"
                ]
            ),

        "path_rb":
            abs(
                float(
                    overlap_row[
                        "rank_biserial"
                    ]
                )
            ),

        "path_p":
            float(
                overlap_row[
                    "p_permutation_holm"
                ]
            ),

        "path_sig":
            bool(
                overlap_row[
                    "significant_permutation_holm"
                ]
            ),
    })


# ============================================================
# OVERALL ROW
# ============================================================

rows.append({

    "dimension":
        "Overall",

    "vol_delta":
        float(
            link_agg[
                "mean_difference"
            ]
        ),

    "vol_rb":
        abs(
            float(
                link_agg[
                    "rank_biserial"
                ]
            )
        ),

    "vol_p":
        float(
            link_agg[
                "p_value"
            ]
        ),

    "vol_sig":
        bool(
            link_agg[
                "significant"
            ]
        ),

    "path_delta":
        float(
            overlap_agg[
                "mean_difference"
            ]
        ),

    "path_rb":
        abs(
            float(
                overlap_agg[
                    "rank_biserial"
                ]
            )
        ),

    "path_p":
        float(
            overlap_agg[
                "p_permutation_raw"
            ]
        ),

    # The aggregate is one prespecified test,
    # so it is not part of the six-dimension Holm family.
    "path_sig":
        (
            float(
                overlap_agg[
                    "p_permutation_raw"
                ]
            )
            <
            0.05
        ),
})


df = pd.DataFrame(
    rows
)


# ============================================================
# PRINT DATA USED IN FIGURE
# ============================================================

print()
print(
    "=" * 80
)

print(
    "VALUES USED IN FIGURE"
)

print(
    "=" * 80
)

print()


print(

    df[
        [
            "dimension",
            "vol_delta",
            "vol_p",
            "vol_sig",
            "path_delta",
            "path_p",
            "path_sig",
        ]
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# DRAW FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10.7,
        7.0,
    )
)


fig.patch.set_facecolor(
    COL_BG
)

ax.set_facecolor(
    COL_BG
)


n_rows = len(
    df
)


y_positions = (
    np.arange(
        n_rows
    )[::-1]
)


box_w = 0.78

box_h = 0.70


# ============================================================
# OVERALL BACKGROUND
# ============================================================

overall_y = (
    y_positions[
        -1
    ]
)


ax.add_patch(

    FancyBboxPatch(

        (
            -0.58,
            overall_y
            -
            0.45,
        ),

        2.16,

        0.90,

        boxstyle=(
            "round,"
            "pad=0.02,"
            "rounding_size=0.04"
        ),

        facecolor=
            COL_LIGHT,

        edgecolor=
            "none",

        zorder=0,
    )
)


# ============================================================
# CELLS
# ============================================================

for idx, row in (
    df.iterrows()
):

    y = y_positions[
        idx
    ]


    metrics = [

        (
            0.0,
            COL_FOCAL,
            row[
                "vol_delta"
            ],
            row[
                "vol_rb"
            ],
            row[
                "vol_p"
            ],
            row[
                "vol_sig"
            ],
            f"+{row['vol_delta']:.2f}",
        ),

        (
            1.0,
            COL_PATHWAY,
            row[
                "path_delta"
            ],
            row[
                "path_rb"
            ],
            row[
                "path_p"
            ],
            row[
                "path_sig"
            ],
            f"+{row['path_delta']:.3f}",
        ),
    ]


    for (
        x,
        base_color,
        delta,
        rb,
        p,
        significant,
        main_text,
    ) in metrics:

        # --------------------------------------------
        # Saturation encodes rank-biserial magnitude
        # --------------------------------------------

        strength = (
            0.18
            +
            0.60
            *
            min(
                max(
                    rb,
                    0,
                ),
                1,
            )
        )


        face_color = (
            blend_with_white(
                base_color,
                strength,
            )
        )


        edge_color = (
            base_color
            if significant
            else COL_GRID
        )


        line_width = (
            2.4
            if significant
            else 1.1
        )


        ax.add_patch(

            FancyBboxPatch(

                (
                    x
                    -
                    box_w
                    /
                    2,

                    y
                    -
                    box_h
                    /
                    2,
                ),

                box_w,

                box_h,

                boxstyle=(
                    "round,"
                    "pad=0.02,"
                    "rounding_size=0.055"
                ),

                facecolor=
                    face_color,

                edgecolor=
                    edge_color,

                linewidth=
                    line_width,

                zorder=2,
            )
        )


        # --------------------------------------------
        # Main effect value
        # --------------------------------------------

        ax.text(

            x,

            y
            +
            0.10,

            main_text,

            ha="center",

            va="center",

            fontsize=13.6,

            fontweight="bold",

            color=
                COL_INK,

            zorder=3,
        )


        # --------------------------------------------
        # p-value
        # --------------------------------------------

        p_text = (
            fmt_p(
                p
            )
        )


        if significant:

            p_text += (
                "  ●"
            )


        ax.text(

            x,

            y
            -
            0.15,

            p_text,

            ha="center",

            va="center",

            fontsize=8.7,

            color=(
                COL_INK
                if significant
                else COL_MUTED
            ),

            zorder=3,
        )


    # ================================================
    # ROW LABEL
    # ================================================

    ax.text(

        -0.54,

        y,

        row[
            "dimension"
        ],

        ha="right",

        va="center",

        fontsize=10.8,

        fontweight=(
            "bold"
            if (
                row[
                    "dimension"
                ]
                ==
                "Overall"
            )
            else "normal"
        ),

        color=
            COL_INK,
    )


# ============================================================
# HEADERS
# ============================================================

top_y = (
    y_positions[
        0
    ]
)


# ------------------------------------------------------------
# Evidence volume
# ------------------------------------------------------------

ax.text(

    0.0,

    top_y
    +
    0.98,

    "EVIDENCE VOLUME",

    ha="center",

    va="bottom",

    fontsize=10.7,

    fontweight="bold",

    color=
        COL_FOCAL,
)


ax.text(

    0.0,

    top_y
    +
    0.58,

    (
        "Focal − comparison\n"
        "mean cited sources"
    ),

    ha="center",

    va="bottom",

    fontsize=9.1,

    linespacing=1.15,

    color=
        COL_MUTED,
)


# ------------------------------------------------------------
# Source displacement
# ------------------------------------------------------------

ax.text(

    1.0,

    top_y
    +
    0.98,

    "SOURCE-SET DISPLACEMENT",

    ha="center",

    va="bottom",

    fontsize=10.7,

    fontweight="bold",

    color=
        COL_PATHWAY,
)


ax.text(

    1.0,

    top_y
    +
    0.58,

    (
        "Focal − comparison Jaccard distance\n"
        "from the matched generic source set"
    ),

    ha="center",

    va="bottom",

    fontsize=9.1,

    linespacing=1.15,

    color=
        COL_MUTED,
)


# ============================================================
# TITLE
# ============================================================

ax.text(

    -0.57,

    top_y
    +
    2.08,

    (
        ""
        ""
    ),

    ha="left",

    va="bottom",

    fontsize=15.2,

    fontweight="bold",

    color=
        COL_INK,
)


ax.text(

    -0.57,

    top_y
    +
    1.72,

    (
        " "
        ""
    ),

    ha="left",

    va="bottom",

    fontsize=9.8,

    color=
        COL_MUTED,
)


# ============================================================
# FOOTNOTES
# ============================================================

ax.text(

    -0.57,

    -0.93,

    (
        "Saturation encodes paired rank-biserial effect magnitude.  "
        "● = significant after Holm correction within the six dimensions."
    ),

    ha="left",

    va="top",

    fontsize=8.6,

    color=
        COL_MUTED,
)


ax.text(

    -0.57,

    -1.25,

    (
        "Overall: outcome-level aggregate paired contrast "
        "(uncorrected because it is a single prespecified aggregate test)."
    ),

    ha="left",

    va="top",

    fontsize=8.6,

    color=
        COL_MUTED,
)


# ============================================================
# FINAL LAYOUT
# ============================================================

ax.set_xlim(
    -0.63,
    1.63,
)


ax.set_ylim(
    -1.42,
    top_y
    +
    2.38,
)


ax.axis(
    "off"
)


plt.tight_layout()


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
print(
    "=" * 80
)

print(
    "FIGURE SAVED"
)

print(
    "=" * 80
)

print()


print(
    f"PNG:\n"
    f"{OUTPUT_PNG}"
)


print()


print(
    f"PDF:\n"
    f"{OUTPUT_PDF}"
)


plt.show()