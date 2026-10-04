from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/scratch/victoria.estanislau/ai-summary")
RESULTS_DIR = BASE_DIR / "results"

EVIDENCE_DIR = RESULTS_DIR / "evidence_synthesis_analysis_dallas_v2"
OVERLAP_DIR = RESULTS_DIR / "source_overlap_analysis_dallas"
SEMANTIC_DIR = RESULTS_DIR / "semantic_embedding_analysis_dallas"

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = FIGURE_DIR / "figure_4_4_pathway_surface_coupling_v2.png"
OUTPUT_PDF = FIGURE_DIR / "figure_4_4_pathway_surface_coupling_v2.pdf"


# ============================================================
# RESULT DISCOVERY
# ============================================================

def try_load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def find_result_file(directory, validator, description):
    if not directory.exists():
        raise FileNotFoundError(
            f"\nDirectory does not exist:\n{directory}"
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
        p = directory / name
        if p.exists():
            candidates.append(p)

    candidates.extend(sorted(directory.rglob("*.json")))
    candidates.extend(sorted(directory.rglob("*.txt")))

    seen = set()
    unique_candidates = []

    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(p)

    for p in unique_candidates:
        obj = try_load_json(p)
        if obj is None:
            continue
        try:
            if validator(obj):
                print(f"Found {description}:")
                print(f"  {p}")
                return p, obj
        except Exception:
            continue

    raise RuntimeError(
        f"\nCould not identify {description} inside:\n"
        f"{directory}\n"
        f"Checked {len(unique_candidates)} JSON/TXT files."
    )


def is_evidence_results(obj):
    return (
        "pipeline" in obj
        and "blocked_correlations" in obj["pipeline"]
    )


def is_overlap_results(obj):
    return (
        "all_pairwise_observations" in obj
        and "minority_vs_majority" in obj
    )


def is_semantic_results(obj):
    return (
        "group_vs_generic_observations" in obj
        and "primary_analysis" in obj
        and obj.get("metadata", {}).get("primary_metric")
        == "dense_distance_subject_normalized"
    )


# ============================================================
# LOAD RESULTS
# ============================================================

print("\n" + "=" * 80)
print("LOCATING RESULT FILES")
print("=" * 80 + "\n")

_, evidence = find_result_file(
    EVIDENCE_DIR,
    is_evidence_results,
    "evidence-synthesis V2 results",
)

print()

_, overlap = find_result_file(
    OVERLAP_DIR,
    is_overlap_results,
    "source-overlap results",
)

print()

_, semantic = find_result_file(
    SEMANTIC_DIR,
    is_semantic_results,
    "semantic-displacement results",
)


# ============================================================
# MERGE OBSERVATION-LEVEL RESULTS
# ============================================================

source_obs = pd.DataFrame(
    overlap["all_pairwise_observations"]
)

answer_obs = pd.DataFrame(
    semantic["group_vs_generic_observations"]
)

merge_keys = [
    "outcome_id",
    "dimension",
    "condition",
    "group",
]

df = source_obs.merge(
    answer_obs,
    on=merge_keys,
    how="inner",
    validate="one_to_one",
)

if len(df) != 252:
    raise RuntimeError(
        f"Expected 252 explicit-group observations; got {len(df)}."
    )


# ============================================================
# PRIMARY BLOCKED ASSOCIATION
# ============================================================

blocked_candidates = [
    row
    for row in evidence["pipeline"]["blocked_correlations"]
    if (
        row.get("metric_x") == "url_jaccard_distance"
        and row.get("metric_y") == "dense_distance_subject_normalized"
    )
]

if len(blocked_candidates) != 1:
    raise RuntimeError(
        "Could not uniquely identify the URL-displacement -> "
        "answer-displacement blocked correlation."
    )

blocked = blocked_candidates[0]

blocked_r = float(
    blocked["blocked_rank_correlation"]
)

blocked_ci_low = float(
    blocked["cluster_bootstrap_ci_low"]
)

blocked_ci_high = float(
    blocked["cluster_bootstrap_ci_high"]
)

blocked_p = float(
    blocked["p_blocked_permutation_holm"]
)


# ============================================================
# REPRODUCE THE BLOCKED-RANK TRANSFORMATION
#
# We rank the 12 explicit conditions separately within each
# outcome, then center them. The resulting correlation must
# exactly reproduce the reported blocked rank correlation.
# ============================================================

df["source_rank"] = (
    df.groupby("outcome_id")["url_jaccard_distance"]
      .rank(method="average")
)

df["answer_rank"] = (
    df.groupby("outcome_id")["dense_distance_subject_normalized"]
      .rank(method="average")
)

df["source_percentile"] = (
    df["source_rank"] - 1.0
) / 11.0

df["answer_percentile"] = (
    df["answer_rank"] - 1.0
) / 11.0

df["source_rank_centered"] = (
    df["source_rank"]
    - df.groupby("outcome_id")["source_rank"].transform("mean")
)

df["answer_rank_centered"] = (
    df["answer_rank"]
    - df.groupby("outcome_id")["answer_rank"].transform("mean")
)

reproduced_r = np.corrcoef(
    df["source_rank_centered"],
    df["answer_rank_centered"],
)[0, 1]

if not np.isclose(
    reproduced_r,
    blocked_r,
    atol=1e-10,
):
    raise RuntimeError(
        "Blocked correlation mismatch:\n"
        f"  reproduced = {reproduced_r}\n"
        f"  reported   = {blocked_r}"
    )

print()
print(
    f"Reproduced blocked rank correlation: "
    f"{reproduced_r:.10f}"
)


# ============================================================
# VISUAL SUMMARY
#
# The raw Jaccard distance is heavily concentrated near 1.
# Plotting 252 raw points therefore creates severe overplotting.
#
# Instead, the figure preserves the within-outcome ranking used
# by the primary test and summarizes observations sharing the
# same source-rank level.
#
# X: relative source-displacement rank within outcome
# Y: mean relative answer-displacement rank at that X level
# Point size: number of observations at the source-rank level
#
# This aggregation is descriptive only. Statistical inference
# remains the prespecified blocked correlation on all N=252.
# ============================================================

curve = (
    df.groupby(
        "source_percentile",
        as_index=False,
    )
    .agg(
        mean_answer_percentile=(
            "answer_percentile",
            "mean",
        ),
        n=(
            "answer_percentile",
            "size",
        ),
    )
    .sort_values(
        "source_percentile"
    )
)


# ============================================================
# PAPER PALETTE
# ============================================================

COL_PATHWAY = "#009E73"
COL_SYNTHESIS = "#CC79A7"
COL_INK = "#253142"
COL_MUTED = "#667085"
COL_GRID = "#DDE3EA"
COL_LIGHT = "#F5F7FA"
COL_BG = "#FFFFFF"

cmap = LinearSegmentedColormap.from_list(
    "pathway_to_surface",
    [
        COL_PATHWAY,
        COL_SYNTHESIS,
    ],
)


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8.5,
        6.4,
    )
)

fig.patch.set_facecolor(
    COL_BG
)

ax.set_facecolor(
    COL_BG
)


# Perfect rank coupling reference
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.2,
    color=COL_GRID,
    zorder=1,
)


# Subtle guides
for tick in [
    0.25,
    0.50,
    0.75,
]:
    ax.axhline(
        tick,
        color=COL_GRID,
        linewidth=0.7,
        alpha=0.55,
        zorder=0,
    )

    ax.axvline(
        tick,
        color=COL_GRID,
        linewidth=0.7,
        alpha=0.55,
        zorder=0,
    )


# Response curve
ax.plot(
    curve["source_percentile"],
    curve["mean_answer_percentile"],
    color=COL_INK,
    linewidth=2.1,
    alpha=0.9,
    zorder=2,
)


# Point size represents the amount of data at that rank level.
sizes = (
    36
    +
    4.5
    *
    curve["n"].to_numpy()
)


colors = cmap(
    curve["source_percentile"]
    .to_numpy()
)


ax.scatter(
    curve["source_percentile"],
    curve["mean_answer_percentile"],
    s=sizes,
    c=colors,
    edgecolors="white",
    linewidths=1.0,
    zorder=3,
)


# ============================================================
# STATISTIC CARD
# ============================================================

if blocked_p < 0.001:
    p_text = r"$p_{\mathrm{Holm}}<.001$"
else:
    p_text = (
        rf"$p_{{\mathrm{{Holm}}}}={blocked_p:.3f}$"
    )


stat_text = (
    rf"$r_{{blocked}}={blocked_r:.3f}$"
    + "\n"
    + rf"95\% CI [{blocked_ci_low:.3f}, {blocked_ci_high:.3f}]"
    + "\n"
    + p_text
)


ax.text(
    0.055,
    0.945,
    stat_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10.5,
    color=COL_INK,
    linespacing=1.35,
    bbox=dict(
        boxstyle="round,pad=0.55",
        facecolor=COL_LIGHT,
        edgecolor=COL_GRID,
        linewidth=1.0,
    ),
    zorder=5,
)


# ============================================================
# AXES
# ============================================================

ax.set_xlim(
    -0.035,
    1.02,
)

ax.set_ylim(
    -0.035,
    1.02,
)


ax.set_xticks(
    [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    ]
)

ax.set_yticks(
    [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    ]
)


ax.set_xticklabels(
    [
        "closest",
        "25%",
        "middle",
        "75%",
        "farthest",
    ],
    fontsize=9,
    color=COL_MUTED,
)

ax.set_yticklabels(
    [
        "closest",
        "25%",
        "middle",
        "75%",
        "farthest",
    ],
    fontsize=9,
    color=COL_MUTED,
)


ax.set_xlabel(
    (
        "Relative source-set displacement from generic\n"
        "within the same outcome"
    ),
    fontsize=10.7,
    color=COL_PATHWAY,
    fontweight="bold",
    labelpad=12,
)


ax.set_ylabel(
    (
        "Relative answer displacement from generic\n"
        "within the same outcome"
    ),
    fontsize=10.7,
    color=COL_SYNTHESIS,
    fontweight="bold",
    labelpad=12,
)


ax.text(
    0.985,
    0.965,
    "pathway and surface\nmove together",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9.2,
    color=COL_SYNTHESIS,
    fontweight="bold",
)


ax.text(
    0.985,
    0.055,
    (
        "Point size = number of conditions\n"
        "at each source-rank level"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.2,
    color=COL_MUTED,
)


# ============================================================
# STYLE
# ============================================================

for spine in (
    ax.spines.values()
):
    spine.set_visible(
        False
    )

ax.tick_params(
    axis="both",
    length=0,
)


fig.suptitle(
    (
        "Source-pathway shifts track semantic shifts "
        "in the final answer"
    ),
    x=0.12,
    y=0.975,
    ha="left",
    fontsize=15.2,
    fontweight="bold",
    color=COL_INK,
)


fig.text(
    0.12,
    0.925,
    (
        "Dallas · 252 explicit group conditions "
        "across 21 matched outcomes"
    ),
    ha="left",
    va="top",
    fontsize=9.6,
    color=COL_MUTED,
)


fig.text(
    0.12,
    0.025,
    (
        "Dots summarize within-outcome ranks for visualization; "
        "inference uses the prespecified blocked rank correlation "
        "on all 252 observations."
    ),
    ha="left",
    va="bottom",
    fontsize=8.4,
    color=COL_MUTED,
)


fig.subplots_adjust(
    left=0.15,
    right=0.96,
    bottom=0.16,
    top=0.86,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUTPUT_PNG,
    dpi=350,
    bbox_inches="tight",
    facecolor=COL_BG,
)


fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor=COL_BG,
)


print()
print("=" * 80)
print("FIGURE SAVED")
print("=" * 80)
print()

print(f"PNG:\n{OUTPUT_PNG}\n")
print(f"PDF:\n{OUTPUT_PDF}\n")

plt.show()
