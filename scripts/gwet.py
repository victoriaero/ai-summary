import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from scipy.stats import t

# ============================================================
# FILES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "artifacts" / "annotation_inputs"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "annotation_results"

FILE_1 = INPUT_DIR / "outcomes_annotator1.csv"
FILE_2 = INPUT_DIR / "outcomes_annotator2.csv"

SCORE_COL = (
    "How relevant is this outcome for studying how AI-generated information "
    "may differ across social groups within this domain? (1-5)"
)

CONCERN_COL = "Is there a cross-group comparability concern?"

# If True, any outcome marked as having a comparability concern
# by at least one annotator is excluded from the Top 3.
EXCLUDE_ANY_CONCERN = False


# ============================================================
# PARSING
# ============================================================

def parse_stars(x):
    """
    Converts:
        ★★★★★ -> 5
        ★★★☆☆ -> 3
        ☆☆☆☆☆ -> 0

    Also accepts numeric values if the spreadsheet is later changed.
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float)):
        return float(x)

    x = str(x).strip()

    if "★" in x or "☆" in x:
        return x.count("★")

    try:
        return float(x)
    except ValueError:
        return np.nan


def parse_bool(x):
    """
    Treats blank as False/No.
    """
    if pd.isna(x):
        return False

    x = str(x).strip().lower()

    return x in {
        "yes", "true", "1", "y",
        "x", "✓", "checked"
    }


# ============================================================
# GWET AC1 / AC2
# Implementation following the irrCAC formulation
# ============================================================

def gwet_ac(ratings, categories, weights="unweighted", conf_level=0.95):

    ratings = np.asarray(ratings, dtype=float)
    categories = np.asarray(categories, dtype=float)

    n, r = ratings.shape
    q = len(categories)

    if weights == "unweighted":
        W = np.eye(q)

    elif weights == "quadratic":
        idx = np.arange(q)
        W = 1 - (
            (idx[:, None] - idx[None, :]) / (q - 1)
        ) ** 2

    else:
        raise ValueError("weights must be 'unweighted' or 'quadratic'")

    # Subject × category count matrix
    agree = np.zeros((n, q))

    for k, cat in enumerate(categories):
        agree[:, k] = np.nansum(ratings == cat, axis=1)

    agree_w = (W @ agree.T).T

    ri = agree.sum(axis=1)

    sum_q = (
        agree * (agree_w - 1)
    ).sum(axis=1)

    valid = ri >= 2
    n_valid = valid.sum()

    # Observed weighted agreement
    pa = np.mean(
        sum_q[valid] /
        (ri[valid] * (ri[valid] - 1))
    )

    # Marginal category probabilities
    pi_vec = np.mean(
        agree / ri[:, None],
        axis=0
    )

    # Chance agreement
    pe = (
        W.sum()
        * np.sum(pi_vec * (1 - pi_vec))
        / (q * (q - 1))
    )

    coeff = (pa - pe) / (1 - pe)

    # Standard error following irrCAC formulation
    den = ri * (ri - 1)
    den_safe = den.copy()
    den_safe[den_safe == 0] = -1

    pa_i = sum_q / den_safe
    pe_r2 = pe * valid.astype(float)

    ac_i = (
        (n / n_valid)
        * (pa_i - pe_r2)
        / (1 - pe)
    )

    pe_i = (
        W.sum() / (q * (q - 1))
    ) * (
        (agree @ (1 - pi_vec)) / ri
    )

    ac_ix = (
        ac_i
        - 2 * (1 - coeff)
        * (pe_i - pe)
        / (1 - pe)
    )

    variance = (
        1 / (n * (n - 1))
    ) * np.sum(
        (ac_ix - coeff) ** 2
    )

    se = np.sqrt(variance)

    alpha = 1 - conf_level
    crit = t.ppf(
        1 - alpha / 2,
        df=n - 1
    )

    ci_low = coeff - crit * se
    ci_high = min(
        1,
        coeff + crit * se
    )

    return {
        "coefficient": coeff,
        "pa": pa,
        "pe": pe,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ============================================================
# LOAD
# ============================================================

a1 = pd.read_csv(FILE_1)
a2 = pd.read_csv(FILE_2)

a1["score"] = a1[SCORE_COL].apply(parse_stars)
a2["score"] = a2[SCORE_COL].apply(parse_stars)

a1["concern"] = a1[CONCERN_COL].apply(parse_bool)
a2["concern"] = a2[CONCERN_COL].apply(parse_bool)


# ============================================================
# MATCH BY DOMAIN + OUTCOME
# Do NOT rely on row order
# ============================================================

merged = a1[
    ["Domain", "Outcome", "score", "concern"]
].merge(
    a2[
        ["Domain", "Outcome", "score", "concern"]
    ],
    on=["Domain", "Outcome"],
    suffixes=("_1", "_2"),
    validate="one_to_one"
)

print(f"Matched outcomes: {len(merged)}")


# ============================================================
# ORDINAL AGREEMENT: 0–5 STARS
# ============================================================

valid_scores = merged.dropna(
    subset=["score_1", "score_2"]
).copy()

score_matrix = valid_scores[
    ["score_1", "score_2"]
].to_numpy()

# Use full possible scale 0–5
ac2 = gwet_ac(
    score_matrix,
    categories=[0, 1, 2, 3, 4, 5],
    weights="quadratic"
)

weighted_kappa = cohen_kappa_score(
    valid_scores["score_1"],
    valid_scores["score_2"],
    labels=[0, 1, 2, 3, 4, 5],
    weights="quadratic"
)

spearman_r, spearman_p = spearmanr(
    valid_scores["score_1"],
    valid_scores["score_2"]
)

difference = (
    valid_scores["score_1"]
    - valid_scores["score_2"]
).abs()

exact_agreement = (
    valid_scores["score_1"]
    == valid_scores["score_2"]
).mean()

within_one = (
    difference <= 1
).mean()

mae = difference.mean()


# ============================================================
# BINARY AGREEMENT: COMPARABILITY CONCERN
# ============================================================

binary_matrix = merged[
    ["concern_1", "concern_2"]
].astype(int).to_numpy()

ac1 = gwet_ac(
    binary_matrix,
    categories=[0, 1],
    weights="unweighted"
)

binary_raw_agreement = (
    merged["concern_1"]
    == merged["concern_2"]
).mean()


# ============================================================
# PRINT AGREEMENT REPORT
# ============================================================

print("\n" + "=" * 70)
print("INTER-ANNOTATOR AGREEMENT")
print("=" * 70)

print("\nRELEVANCE SCORE (0–5)")
print("-" * 70)

print(
    f"Gwet's AC2 (quadratic): "
    f"{ac2['coefficient']:.3f}"
)

print(
    f"95% CI: "
    f"[{ac2['ci_low']:.3f}, "
    f"{ac2['ci_high']:.3f}]"
)

print(
    f"Weighted observed agreement: "
    f"{ac2['pa']:.3f}"
)

print(
    f"Quadratic Cohen's kappa: "
    f"{weighted_kappa:.3f}"
)

print(
    f"Spearman rho: "
    f"{spearman_r:.3f} "
    f"(p={spearman_p:.4f})"
)

print(
    f"Exact agreement: "
    f"{exact_agreement:.1%}"
)

print(
    f"Agreement within ±1: "
    f"{within_one:.1%}"
)

print(
    f"Mean absolute difference: "
    f"{mae:.2f} stars"
)


print("\nCOMPARABILITY CONCERN (YES/NO)")
print("-" * 70)

print(
    f"Gwet's AC1: "
    f"{ac1['coefficient']:.3f}"
)

print(
    f"95% CI: "
    f"[{ac1['ci_low']:.3f}, "
    f"{ac1['ci_high']:.3f}]"
)

print(
    f"Raw agreement: "
    f"{binary_raw_agreement:.1%}"
)


# ============================================================
# CREATE CONSENSUS / RANKING VARIABLES
# ============================================================

merged["mean_score"] = (
    merged["score_1"]
    + merged["score_2"]
) / 2

merged["min_score"] = merged[
    ["score_1", "score_2"]
].min(axis=1)

merged["abs_difference"] = (
    merged["score_1"]
    - merged["score_2"]
).abs()

merged["any_comparability_concern"] = (
    merged["concern_1"]
    | merged["concern_2"]
)

merged["both_comparability_concern"] = (
    merged["concern_1"]
    & merged["concern_2"]
)


# ============================================================
# SHOW COMPARABILITY FLAGS
# ============================================================

print("\n" + "=" * 70)
print("OUTCOMES WITH COMPARABILITY CONCERNS")
print("=" * 70)

concerns = merged[
    merged["any_comparability_concern"]
][
    [
        "Domain",
        "Outcome",
        "score_1",
        "score_2",
        "concern_1",
        "concern_2"
    ]
]

if len(concerns) == 0:
    print("None.")
else:
    print(concerns.to_string(index=False))


# ============================================================
# TOP 3 PER DOMAIN
# ============================================================

print("\n" + "=" * 70)
print("TOP 3 OUTCOMES PER DOMAIN")
print("=" * 70)

selected_rows = []

for domain, group in merged.groupby(
    "Domain",
    sort=False
):

    g = group.copy()

    if EXCLUDE_ANY_CONCERN:
        g = g[
            ~g["any_comparability_concern"]
        ]

    # Ranking:
    # 1. Highest mean score
    # 2. Highest minimum annotator score
    # 3. Smallest disagreement
    # 4. Alphabetical only for deterministic final tie-break
    g = g.sort_values(
        by=[
            "mean_score",
            "min_score",
            "abs_difference",
            "Outcome"
        ],
        ascending=[
            False,
            False,
            True,
            True
        ]
    )

    top3 = g.head(3).copy()
    top3["rank"] = range(1, len(top3) + 1)

    selected_rows.append(top3)

    print(f"\n{domain}")

    for _, row in top3.iterrows():

        concern_text = (
            " [COMPARABILITY FLAG]"
            if row["any_comparability_concern"]
            else ""
        )

        print(
            f"{int(row['rank'])}. "
            f"{row['Outcome']} "
            f"(A1={row['score_1']:.0f}, "
            f"A2={row['score_2']:.0f}, "
            f"mean={row['mean_score']:.2f})"
            f"{concern_text}"
        )


# ============================================================
# BIGGEST DISAGREEMENTS
# ============================================================

print("\n" + "=" * 70)
print("LARGEST RATING DISAGREEMENTS")
print("=" * 70)

disagreements = merged.sort_values(
    "abs_difference",
    ascending=False
)[
    [
        "Domain",
        "Outcome",
        "score_1",
        "score_2",
        "abs_difference"
    ]
]

print(
    disagreements.head(15).to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

selected = pd.concat(
    selected_rows,
    ignore_index=True
)

merged.to_csv(
    OUTPUT_DIR / "annotator_agreement_full.csv",
    index=False
)

selected.to_csv(
    OUTPUT_DIR / "selected_top3_outcomes.csv",
    index=False
)

print("\nSaved:")
print(f" - {OUTPUT_DIR / 'annotator_agreement_full.csv'}")
print(f" - {OUTPUT_DIR / 'selected_top3_outcomes.csv'}")
