from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/scratch/victoria.estanislau/ai-summary")

RESULTS_DIR = BASE_DIR / "results"
SEMANTIC_DIR = RESULTS_DIR / "semantic_embedding_analysis_dallas"

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = FIGURE_DIR / "figure_4_3_semantic_shift_anatomy.png"
OUTPUT_PDF = FIGURE_DIR / "figure_4_3_semantic_shift_anatomy.pdf"


# ============================================================
# JSON DISCOVERY
# ============================================================

def try_load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def is_semantic_results(obj):
    return (
        "primary_analysis" in obj
        and "sentence_level" in obj
        and "dimension_tests" in obj["primary_analysis"]
        and "aggregate_minority_vs_majority" in obj["primary_analysis"]
        and "generic_coverage_dimension_tests" in obj["sentence_level"]
        and "group_novelty_dimension_tests" in obj["sentence_level"]
        and "aggregate_generic_coverage" in obj["sentence_level"]
        and "aggregate_group_novelty" in obj["sentence_level"]
    )


def find_result_file(directory, validator, description):
    if not directory.exists():
        raise FileNotFoundError(
            f"\nDirectory does not exist:\n{directory}"
        )

    candidates = []

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
            candidates.append(path)

    candidates.extend(sorted(directory.rglob("*.json")))
    candidates.extend(sorted(directory.rglob("*.txt")))

    seen = set()
    unique_candidates = []

    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(path)

    for path in unique_candidates:
        obj = try_load_json(path)
        if obj is None:
            continue
        try:
            if validator(obj):
                print(f"Found {description}:")
                print(f"  {path}")
                return path, obj
        except Exception:
            continue

    raise RuntimeError(
        f"\nCould not identify {description} inside:\n{directory}\n"
        f"Checked {len(unique_candidates)} JSON/TXT files."
    )


# ============================================================
# LOAD RESULTS
# ============================================================

print("\n" + "=" * 80)
print("LOCATING SEMANTIC RESULTS")
print("=" * 80 + "\n")

semantic_path, data = find_result_file(
    SEMANTIC_DIR,
    is_semantic_results,
    "semantic-embedding results",
)


# ============================================================
# EXTRACT TABLES
# ============================================================

primary = pd.DataFrame(
    data["primary_analysis"]["dimension_tests"]
)

coverage = pd.DataFrame(
    data["sentence_level"]["generic_coverage_dimension_tests"]
)

novelty = pd.DataFrame(
    data["sentence_level"]["group_novelty_dimension_tests"]
)

primary_agg = data["primary_analysis"]["aggregate_minority_vs_majority"]
coverage_agg = data["sentence_level"]["aggregate_generic_coverage"]
novelty_agg = data["sentence_level"]["aggregate_group_novelty"]


# ============================================================
# ORDER
# ============================================================

ORDER = [
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
]

for name, df in [
    ("primary", primary),
    ("coverage", coverage),
    ("novelty", novelty),
]:
    missing = set(ORDER) - set(df["dimension"])
    if missing:
        raise RuntimeError(
            f"Missing dimensions in {name} results: {sorted(missing)}"
        )

primary = primary.set_index("dimension").loc[ORDER].reset_index()
coverage = coverage.set_index("dimension").loc[ORDER].reset_index()
novelty = novelty.set_index("dimension").loc[ORDER].reset_index()


# ============================================================
# HELPERS
# ============================================================

def get_ci_low(row):
    for col in [
        "bootstrap_ci_low",
        "ci_low",
        "ci_lower",
        "mean_difference_ci_low",
    ]:
        if col in row.index:
            return float(row[col])
    return np.nan


def get_ci_high(row):
    for col in [
        "bootstrap_ci_high",
        "ci_high",
        "ci_upper",
        "mean_difference_ci_high",
    ]:
        if col in row.index:
            return float(row[col])
    return np.nan


def get_p(row):
    for col in [
        "p_permutation_holm",
        "p_holm",
        "p_value_holm",
        "p_adjusted",
    ]:
        if col in row.index:
            return float(row[col])
    return np.nan


def get_sig(row):
    for col in [
        "significant_permutation_holm",
        "significant_holm",
        "significant",
    ]:
        if col in row.index:
            return bool(row[col])
    return False


def fmt_delta(x):
    return f"{x:+.3f}"


# ============================================================
# BUILD FIGURE DATA
# ============================================================

rows = []

for dim in ORDER:
    p = primary[primary["dimension"] == dim].iloc[0]
    c = coverage[coverage["dimension"] == dim].iloc[0]
    n = novelty[novelty["dimension"] == dim].iloc[0]

    rows.append({
        "dimension": dim,

        # Primary analysis: full-answer displacement
        "answer_delta": float(p["mean_difference"]),
        "answer_ci_low": get_ci_low(p),
        "answer_ci_high": get_ci_high(p),
        "answer_rb": float(p["rank_biserial"]),
        "answer_p": get_p(p),
        "answer_sig": get_sig(p),

        # Mechanism 1: generic-content displacement
        "coverage_delta": float(c["mean_difference"]),
        "coverage_rb": float(c["rank_biserial"]),
        "coverage_p": get_p(c),
        "coverage_sig": get_sig(c),

        # Mechanism 2: group-specific novelty
        "novelty_delta": float(n["mean_difference"]),
        "novelty_rb": float(n["rank_biserial"]),
        "novelty_p": get_p(n),
        "novelty_sig": get_sig(n),
    })

rows.append({
    "dimension": "Overall",

    "answer_delta": float(primary_agg["mean_difference"]),
    "answer_ci_low": np.nan,
    "answer_ci_high": np.nan,
    "answer_rb": float(primary_agg["rank_biserial"]),
    "answer_p": float(primary_agg["p_permutation_raw"]),
    "answer_sig": float(primary_agg["p_permutation_raw"]) < 0.05,

    "coverage_delta": float(coverage_agg["mean_difference"]),
    "coverage_rb": float(coverage_agg["rank_biserial"]),
    "coverage_p": float(coverage_agg["p_permutation_raw"]),
    "coverage_sig": float(coverage_agg["p_permutation_raw"]) < 0.05,

    "novelty_delta": float(novelty_agg["mean_difference"]),
    "novelty_rb": float(novelty_agg["rank_biserial"]),
    "novelty_p": float(novelty_agg["p_permutation_raw"]),
    "novelty_sig": float(novelty_agg["p_permutation_raw"]) < 0.05,
})

df = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("VALUES USED IN FIGURE")
print("=" * 80 + "\n")

print(
    df[
        [
            "dimension",
            "answer_delta",
            "coverage_delta",
            "novelty_delta",
            "answer_sig",
            "coverage_sig",
            "novelty_sig",
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# PALETTE
# ============================================================

# Condition palette
COL_FOCAL = "#D55E00"
COL_COMPARISON = "#0072B2"
COL_GENERIC = "#6B7280"

# Layer palette
COL_PATHWAY = "#009E73"
COL_SYNTHESIS = "#CC79A7"

# Semantic-specific shades
COL_ANSWER = "#CC79A7"    # primary semantic result
COL_COVERAGE = "#8F6BB8"  # generic-content displacement
COL_NOVELTY = "#B65D8B"   # group-specific novelty

# Supporting
COL_INK = "#253142"
COL_MUTED = "#667085"
COL_GRID = "#DDE3EA"
COL_BG = "#FFFFFF"
COL_LIGHT = "#F5F7FA"


# ============================================================
# COLOR HELPERS
# ============================================================

def hex_to_rgb(hexcolor):
    h = hexcolor.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def blend_with_white(hexcolor, strength):
    c = hex_to_rgb(hexcolor)
    return tuple(1 - strength * (1 - c))


def effect_fill(base_color, rb):
    strength = 0.18 + 0.60 * min(abs(rb), 1.0)
    return blend_with_white(base_color, strength)


def signed_color(delta, positive_color):
    return positive_color if delta >= 0 else COL_COMPARISON


# ============================================================
# FIGURE LAYOUT
# ============================================================

fig = plt.figure(figsize=(13.4, 7.4), facecolor=COL_BG)
gs = GridSpec(
    1,
    3,
    width_ratios=[2.7, 1.15, 1.15],
    wspace=0.18,
)

ax_main = fig.add_subplot(gs[0, 0])
ax_cov = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_nov = fig.add_subplot(gs[0, 2], sharey=ax_main)

for ax in [ax_main, ax_cov, ax_nov]:
    ax.set_facecolor(COL_BG)

n = len(df)
y_positions = np.arange(n)[::-1]
overall_y = y_positions[-1]

# Overall-row background band across all panels
for ax, x0, width in [
    (ax_main, -0.25, 1.50),
    (ax_cov, -0.25, 1.50),
    (ax_nov, -0.25, 1.50),
]:
    ax.add_patch(
        FancyBboxPatch(
            (x0, overall_y - 0.45),
            width,
            0.90,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=COL_LIGHT,
            edgecolor="none",
            zorder=0,
            transform=ax.transData,
        )
    )


# ============================================================
# PANEL 1: PRIMARY FOREST PLOT
# ============================================================

answer_min = np.nanmin([
    df["answer_delta"].min(),
    df["answer_ci_low"].dropna().min() if df["answer_ci_low"].notna().any() else df["answer_delta"].min(),
])
answer_max = np.nanmax([
    df["answer_delta"].max(),
    df["answer_ci_high"].dropna().max() if df["answer_ci_high"].notna().any() else df["answer_delta"].max(),
])

pad = max(0.015, (answer_max - answer_min) * 0.18)
x_left = min(-0.02, answer_min - pad)
x_right = answer_max + pad

ax_main.axvline(0, color=COL_GRID, lw=1.4, zorder=1)

for idx, row in df.iterrows():
    y = y_positions[idx]

    delta = row["answer_delta"]
    ci_low = row["answer_ci_low"]
    ci_high = row["answer_ci_high"]
    sig = row["answer_sig"]
    rb = row["answer_rb"]

    base_color = signed_color(delta, COL_ANSWER)
    face = effect_fill(base_color, rb)

    if np.isfinite(ci_low) and np.isfinite(ci_high):
        ax_main.hlines(
            y,
            ci_low,
            ci_high,
            color=base_color if sig else COL_MUTED,
            lw=2.6 if sig else 1.7,
            zorder=2,
            alpha=1.0 if sig else 0.7,
        )

    ax_main.scatter(
        [delta],
        [y],
        s=110,
        color=face,
        edgecolor=base_color if sig else COL_MUTED,
        linewidth=2.2 if sig else 1.2,
        zorder=3,
    )

    if sig:
        ax_main.text(
            delta + (x_right - x_left) * 0.015,
            y,
            "●",
            va="center",
            ha="left",
            fontsize=9,
            color=base_color,
            zorder=4,
        )

    ax_main.text(
        x_left - (x_right - x_left) * 0.03,
        y,
        row["dimension"],
        ha="right",
        va="center",
        fontsize=10.8,
        fontweight="bold" if row["dimension"] == "Overall" else "normal",
        color=COL_INK,
    )

ax_main.set_xlim(x_left, x_right)
ax_main.set_ylim(-1.0, y_positions[0] + 1.0)
ax_main.set_yticks([])
ax_main.tick_params(axis="x", labelsize=9, colors=COL_MUTED)

for spine in ax_main.spines.values():
    spine.set_visible(False)

ax_main.set_title(
    "FULL-ANSWER DISPLACEMENT",
    fontsize=10.8,
    fontweight="bold",
    color=COL_ANSWER,
    pad=26,
)

ax_main.text(
    0.5,
    1.02,
    "",
    transform=ax_main.transAxes,
    ha="center",
    va="bottom",
    fontsize=9.0,
    color=COL_MUTED,
)

ax_main.text(
    x_left,
    y_positions[0] + 0.72,
    "comparison",
    ha="left",
    va="bottom",
    fontsize=8.5,
    color=COL_COMPARISON,
)

ax_main.text(
    x_right,
    y_positions[0] + 0.72,
    "focal",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color=COL_ANSWER,
)


# ============================================================
# PANEL 2: GENERIC-CONTENT DISPLACEMENT
# ============================================================

cov_abs = max(abs(df["coverage_delta"]).max(), 0.02)
cov_lim = cov_abs * 1.28

ax_cov.axvline(0, color=COL_GRID, lw=1.4, zorder=1)

for idx, row in df.iterrows():
    y = y_positions[idx]
    delta = row["coverage_delta"]
    rb = row["coverage_rb"]
    sig = row["coverage_sig"]

    base_color = signed_color(delta, COL_COVERAGE)
    face = effect_fill(base_color, rb)

    ax_cov.hlines(y, 0, delta, color=base_color, lw=6.0, zorder=2, alpha=0.95)
    ax_cov.scatter(
        [delta], [y],
        s=55,
        color=face,
        edgecolor=base_color if sig else COL_MUTED,
        linewidth=1.8 if sig else 1.0,
        zorder=3,
    )

    if sig:
        ax_cov.text(
            delta + np.sign(delta if delta != 0 else 1) * cov_lim * 0.05,
            y,
            "●",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=8,
            color=base_color,
            zorder=4,
        )

ax_cov.set_xlim(-cov_lim, cov_lim)
ax_cov.set_yticks([])
ax_cov.tick_params(axis="x", labelsize=9, colors=COL_MUTED)

for spine in ax_cov.spines.values():
    spine.set_visible(False)

ax_cov.set_title(
    "GENERIC-CONTENT DISPLACEMENT",
    fontsize=10.3,
    fontweight="bold",
    color=COL_COVERAGE,
    pad=26,
)

ax_cov.text(
    0.5,
    1.02,
    "Sentence-level loss of content\nshared with the generic answer",
    transform=ax_cov.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.8,
    color=COL_MUTED,
)


# ============================================================
# PANEL 3: GROUP-SPECIFIC NOVELTY
# ============================================================

nov_abs = max(abs(df["novelty_delta"]).max(), 0.02)
nov_lim = nov_abs * 1.28

ax_nov.axvline(0, color=COL_GRID, lw=1.4, zorder=1)

for idx, row in df.iterrows():
    y = y_positions[idx]
    delta = row["novelty_delta"]
    rb = row["novelty_rb"]
    sig = row["novelty_sig"]

    base_color = signed_color(delta, COL_NOVELTY)
    face = effect_fill(base_color, rb)

    ax_nov.hlines(y, 0, delta, color=base_color, lw=6.0, zorder=2, alpha=0.95)
    ax_nov.scatter(
        [delta], [y],
        s=55,
        color=face,
        edgecolor=base_color if sig else COL_MUTED,
        linewidth=1.8 if sig else 1.0,
        zorder=3,
    )

    if sig:
        ax_nov.text(
            delta + np.sign(delta if delta != 0 else 1) * nov_lim * 0.05,
            y,
            "●",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=8,
            color=base_color,
            zorder=4,
        )

ax_nov.set_xlim(-nov_lim, nov_lim)
ax_nov.set_yticks([])
ax_nov.tick_params(axis="x", labelsize=9, colors=COL_MUTED)

for spine in ax_nov.spines.values():
    spine.set_visible(False)

ax_nov.set_title(
    "GROUP-SPECIFIC NOVELTY",
    fontsize=10.3,
    fontweight="bold",
    color=COL_NOVELTY,
    pad=26,
)

ax_nov.text(
    0.5,
    1.02,
    "Additional sentence-level content\nnot represented in the generic answer",
    transform=ax_nov.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.8,
    color=COL_MUTED,
)


# ============================================================
# GLOBAL TITLE / FOOTNOTES
# ============================================================

fig.text(
    0.07,
    0.965,
    "",
    ha="left",
    va="top",
    fontsize=15.4,
    fontweight="bold",
    color=COL_INK,
)

fig.text(
    0.07,
    0.935,
    "",
    ha="left",
    va="top",
    fontsize=9.7,
    color=COL_MUTED,
)

fig.text(
    0.07,
    0.06,
    "Left: primary semantic result. Right: mechanism analyses showing whether focal answers differ by departing from generic content, adding group-specific content, or both.",
    ha="left",
    va="bottom",
    fontsize=8.6,
    color=COL_MUTED,
)

fig.text(
    0.07,
    0.038,
    "Saturation encodes paired rank-biserial effect magnitude. Filled significance markers indicate Holm-corrected effects within the six dimensions. Blue denotes negative focal − comparison effects.",
    ha="left",
    va="bottom",
    fontsize=8.6,
    color=COL_MUTED,
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout(rect=[0.08, 0.11, 0.99, 0.90])

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

print("\n" + "=" * 80)
print("FIGURE SAVED")
print("=" * 80 + "\n")

print(f"PNG:\n{OUTPUT_PNG}\n")
print(f"PDF:\n{OUTPUT_PDF}\n")

plt.show()