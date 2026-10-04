from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
)

from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    "/scratch/victoria.estanislau/ai-summary"
)

COLLECTION_DIR = (
    BASE_DIR
    / "annotations"
    / "v1_dallas"
    / "google_aio_collection"
)

INPUT_FILE = (
    COLLECTION_DIR
    / "consistency_report.csv"
)

RESULTS_ROOT = (
    BASE_DIR
    / "results"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "link_count_analysis_dallas"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ALPHA = 0.05


# ============================================================
# JSON HELPERS
# ============================================================

def dataframe_to_records(df):
    """
    Convert a DataFrame to JSON-safe records.
    NaN becomes null.
    """
    if df is None or len(df) == 0:
        return []

    return json.loads(
        df.to_json(
            orient="records"
        )
    )


def json_safe(value):
    """
    Convert numpy/pandas values into regular
    Python values that json.dump can serialize.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
        )
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        )
    ):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        (
            np.bool_,
        )
    ):
        return bool(value)

    if isinstance(
        value,
        Path
    ):
        return str(value)

    if isinstance(
        value,
        dict
    ):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        list
    ):
        return [
            json_safe(v)
            for v in value
        ]

    return value


# ============================================================
# LOAD
# ============================================================

raw_df = pd.read_csv(
    INPUT_FILE
)

print("=" * 80)
print("GOOGLE AI OVERVIEW — LINK COUNT ANALYSIS")
print("=" * 80)

print(f"\nInput:")
print(INPUT_FILE)

print(f"\nResults directory:")
print(OUTPUT_DIR)

print(
    f"\nRaw observations: "
    f"{len(raw_df)}"
)


# ============================================================
# CLEANING
# ============================================================

errors = raw_df[
    raw_df["status"] == "ERROR"
].copy()

if len(errors):

    print(
        f"\nExcluding {len(errors)} "
        f"ERROR observation(s):"
    )

    for file in errors["file"]:
        print(f"  - {file}")


df = raw_df[
    raw_df["status"] != "ERROR"
].copy()


# AIO presence is inferred from collected text.
df["aio_present"] = (
    df["aio_chars"] > 0
)


# Same Domain + Outcome across groups = matched unit.
df["outcome_id"] = (
    df["domain"].astype(str)
    + " :: "
    + df["outcome"].astype(str)
)


print(
    f"\nUsable observations: "
    f"{len(df)}"
)

print(
    f"AIO present: "
    f"{df['aio_present'].sum()} / "
    f"{len(df)} "
    f"({df['aio_present'].mean():.1%})"
)


# ============================================================
# LINK ANALYSIS ONLY WHERE AIO EXISTS
# ============================================================

links = df[
    df["aio_present"]
].copy()

zero_link_count = int(
    (links["n_links"] == 0).sum()
)

zero_link_rate = (
    (links["n_links"] == 0).mean()
)

print(
    f"\nObservations used for "
    f"link-count analysis: "
    f"{len(links)}"
)

print(
    f"AIOs with zero links: "
    f"{zero_link_count} "
    f"({zero_link_rate:.1%})"
)


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_mean_ci(
    values,
    n_boot=10000,
    random_state=42,
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        ~np.isnan(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        random_state
    )

    boot = np.empty(
        n_boot
    )

    for i in range(
        n_boot
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot[i] = sample.mean()

    return (
        np.percentile(
            boot,
            2.5
        ),
        np.percentile(
            boot,
            97.5
        ),
    )


# ============================================================
# GENERIC SUMMARY
# ============================================================

def summarize(
    data,
    group_cols,
):

    rows = []

    grouped = data.groupby(
        group_cols,
        dropna=False,
        sort=False,
    )

    for keys, g in grouped:

        if not isinstance(
            keys,
            tuple
        ):
            keys = (keys,)

        values = (
            g["n_links"]
            .to_numpy()
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                values
            )
        )

        row = dict(
            zip(
                group_cols,
                keys
            )
        )

        row.update({
            "n":
                len(g),

            "mean_links":
                values.mean(),

            "median_links":
                np.median(values),

            "sd_links":
                (
                    values.std(
                        ddof=1
                    )
                    if len(values) > 1
                    else np.nan
                ),

            "min_links":
                values.min(),

            "max_links":
                values.max(),

            "zero_link_n":
                np.sum(
                    values == 0
                ),

            "zero_link_rate":
                np.mean(
                    values == 0
                ),

            "mean_ci_low":
                ci_low,

            "mean_ci_high":
                ci_high,

            "mean_aio_chars":
                g[
                    "aio_chars"
                ].mean(),
        })

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# TABLE 1 — LINKS BY GROUP
# ============================================================

group_table = summarize(
    links,
    [
        "dimension",
        "condition",
        "group",
    ]
)

group_table = (
    group_table
    .sort_values(
        "mean_links",
        ascending=False,
    )
)

print()
print("=" * 80)
print("TABLE 1 — LINKS BY GROUP")
print("=" * 80)

print(
    group_table[
        [
            "dimension",
            "condition",
            "group",
            "n",
            "mean_links",
            "median_links",
            "sd_links",
            "zero_link_n",
            "zero_link_rate",
            "mean_ci_low",
            "mean_ci_high",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)

group_table.to_csv(
    OUTPUT_DIR
    / "links_by_group.csv",
    index=False,
)


# ============================================================
# TABLE 2 — LINKS BY DOMAIN
# ============================================================

domain_table = summarize(
    links,
    ["domain"]
)

domain_table = (
    domain_table
    .sort_values(
        "mean_links",
        ascending=False,
    )
)

print()
print("=" * 80)
print("TABLE 2 — LINKS BY DOMAIN")
print("=" * 80)

print(
    domain_table[
        [
            "domain",
            "n",
            "mean_links",
            "median_links",
            "sd_links",
            "zero_link_rate",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)

domain_table.to_csv(
    OUTPUT_DIR
    / "links_by_domain.csv",
    index=False,
)


# ============================================================
# TABLE 3 — LINKS BY OUTCOME
# ============================================================

outcome_table = summarize(
    links,
    [
        "domain",
        "outcome",
    ]
)

outcome_table = (
    outcome_table
    .sort_values(
        "mean_links",
        ascending=False,
    )
)

print()
print("=" * 80)
print("TABLE 3 — LINKS BY OUTCOME")
print("=" * 80)

print(
    outcome_table[
        [
            "domain",
            "outcome",
            "n",
            "mean_links",
            "median_links",
            "zero_link_rate",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)

outcome_table.to_csv(
    OUTPUT_DIR
    / "links_by_outcome.csv",
    index=False,
)


# ============================================================
# TABLE 4 — GROUP × DOMAIN
# ============================================================

group_domain = (
    links
    .pivot_table(
        index="group",
        columns="domain",
        values="n_links",
        aggfunc="mean",
    )
)

print()
print("=" * 80)
print("TABLE 4 — MEAN LINKS: GROUP × DOMAIN")
print("=" * 80)

print(
    group_domain
    .round(2)
    .to_string()
)

group_domain.to_csv(
    OUTPUT_DIR
    / "group_by_domain_mean_links.csv"
)


# JSON-friendly version
group_domain_json = (
    group_domain
    .reset_index()
)


# ============================================================
# TABLE 5 — AIO PRESENCE BY GROUP
# ============================================================

aio_presence = (
    df
    .groupby(
        [
            "dimension",
            "condition",
            "group",
        ],
        dropna=False,
    )
    .agg(
        n=(
            "aio_present",
            "size"
        ),
        aio_present=(
            "aio_present",
            "sum"
        ),
    )
    .reset_index()
)

aio_presence[
    "aio_rate"
] = (
    aio_presence[
        "aio_present"
    ]
    / aio_presence["n"]
)

print()
print("=" * 80)
print("TABLE 5 — AI OVERVIEW PRESENCE BY GROUP")
print("=" * 80)

print(
    aio_presence
    .round(3)
    .to_string(
        index=False
    )
)

aio_presence.to_csv(
    OUTPUT_DIR
    / "aio_presence_by_group.csv",
    index=False,
)


# ============================================================
# MATCHED GROUP × OUTCOME MATRIX
# ============================================================

pivot = (
    links
    .pivot_table(
        index="outcome_id",
        columns="group",
        values="n_links",
        aggfunc="first",
    )
)

complete = (
    pivot.dropna()
)


# ============================================================
# FRIEDMAN TEST
# ============================================================

print()
print("=" * 80)
print("GLOBAL GROUP TEST")
print("=" * 80)

print(
    "\nComplete outcomes available "
    f"for all groups: {len(complete)}"
)


friedman_results = {
    "n_complete_outcomes":
        len(complete),

    "n_groups":
        complete.shape[1],

    "statistic":
        None,

    "p_value":
        None,

    "significant":
        None,
}


if (
    len(complete) >= 2
    and complete.shape[1] >= 3
):

    stat, p = (
        friedmanchisquare(
            *[
                complete[col].values
                for col
                in complete.columns
            ]
        )
    )

    friedman_results[
        "statistic"
    ] = float(stat)

    friedman_results[
        "p_value"
    ] = float(p)

    friedman_results[
        "significant"
    ] = bool(
        p < ALPHA
    )

    print(
        f"\nFriedman chi-square = "
        f"{stat:.3f}"
    )

    print(
        f"p = {p:.6g}"
    )

    if p < ALPHA:

        print(
            "\n→ Significant global "
            "group effect."
        )

    else:

        print(
            "\n→ No significant global "
            "group effect."
        )


pd.DataFrame(
    [friedman_results]
).to_csv(
    OUTPUT_DIR
    / "global_friedman_test.csv",
    index=False,
)


# ============================================================
# PAIRED RANK-BISERIAL
# ============================================================

def paired_rank_biserial(
    x,
    y
):

    d = (
        np.asarray(x)
        - np.asarray(y)
    )

    d = d[
        d != 0
    ]

    if len(d) == 0:
        return 0.0

    ranks = (
        pd.Series(
            np.abs(d)
        )
        .rank(
            method="average"
        )
        .to_numpy()
    )

    positive = (
        ranks[
            d > 0
        ]
        .sum()
    )

    negative = (
        ranks[
            d < 0
        ]
        .sum()
    )

    denominator = (
        positive
        + negative
    )

    if denominator == 0:
        return 0.0

    return (
        positive
        - negative
    ) / denominator


# ============================================================
# ALL PAIRWISE GROUP COMPARISONS
# ============================================================

groups = list(
    pivot.columns
)

pairwise_results = []

for i in range(
    len(groups)
):

    for j in range(
        i + 1,
        len(groups)
    ):

        g1 = groups[i]
        g2 = groups[j]

        pair = (
            pivot[
                [g1, g2]
            ]
            .dropna()
        )

        if len(pair) < 5:
            continue

        x = (
            pair[g1]
            .values
        )

        y = (
            pair[g2]
            .values
        )

        diff = x - y

        if np.all(
            diff == 0
        ):

            statistic = 0
            p_value = 1.0

        else:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                result = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                )

            statistic = (
                result.statistic
            )

            p_value = (
                result.pvalue
            )

        pairwise_results.append({
            "group_1":
                g1,

            "group_2":
                g2,

            "n_matched":
                len(pair),

            "mean_group_1":
                x.mean(),

            "mean_group_2":
                y.mean(),

            "mean_difference":
                diff.mean(),

            "median_difference":
                np.median(diff),

            "rank_biserial":
                paired_rank_biserial(
                    x,
                    y
                ),

            "wilcoxon_W":
                statistic,

            "p_raw":
                p_value,
        })


pairwise = pd.DataFrame(
    pairwise_results
)


if len(pairwise):

    reject, p_adj, _, _ = (
        multipletests(
            pairwise[
                "p_raw"
            ],
            alpha=ALPHA,
            method="holm",
        )
    )

    pairwise[
        "p_holm"
    ] = p_adj

    pairwise[
        "significant_holm"
    ] = reject

    pairwise = (
        pairwise
        .sort_values(
            [
                "significant_holm",
                "p_holm",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


print()
print("=" * 80)
print("PAIRWISE GROUP COMPARISONS")
print("=" * 80)

if len(pairwise):

    significant = pairwise[
        pairwise[
            "significant_holm"
        ]
    ]

else:

    significant = pd.DataFrame()


if significant.empty:

    print(
        "\nNo pairwise comparison "
        "survives Holm correction."
    )

else:

    print(
        "\nSignificant comparisons "
        "after Holm correction:\n"
    )

    print(
        significant[
            [
                "group_1",
                "group_2",
                "n_matched",
                "mean_group_1",
                "mean_group_2",
                "mean_difference",
                "rank_biserial",
                "p_holm",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


pairwise.to_csv(
    OUTPUT_DIR
    / "pairwise_group_wilcoxon.csv",
    index=False,
)


# ============================================================
# MINORITY vs MAJORITY BY DIMENSION
# ============================================================

print()
print("=" * 80)
print("MINORITY vs MAJORITY WITHIN EACH DIMENSION")
print("=" * 80)


social = df[
    df["condition"].isin(
        [
            "minority",
            "majority",
        ]
    )
    &
    df["aio_present"]
].copy()


dimension_tests = []

for dimension in sorted(
    social[
        "dimension"
    ].unique()
):

    subset = social[
        social[
            "dimension"
        ]
        == dimension
    ]

    minority_groups = (
        subset.loc[
            subset[
                "condition"
            ] == "minority",
            "group",
        ]
        .unique()
    )

    majority_groups = (
        subset.loc[
            subset[
                "condition"
            ] == "majority",
            "group",
        ]
        .unique()
    )

    if (
        len(minority_groups) != 1
        or len(majority_groups) != 1
    ):
        continue

    minority = (
        minority_groups[0]
    )

    majority = (
        majority_groups[0]
    )

    pair = (
        subset[
            [
                "outcome_id",
                "group",
                "n_links",
            ]
        ]
        .pivot(
            index="outcome_id",
            columns="group",
            values="n_links",
        )
        .dropna(
            subset=[
                minority,
                majority,
            ]
        )
    )

    if len(pair) == 0:
        continue

    x = (
        pair[
            minority
        ]
        .values
    )

    y = (
        pair[
            majority
        ]
        .values
    )

    diff = x - y

    if np.all(
        diff == 0
    ):

        W = 0
        p_value = 1.0

    else:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            result = wilcoxon(
                x,
                y,
                alternative="two-sided",
            )

        W = result.statistic
        p_value = result.pvalue

    dimension_tests.append({
        "dimension":
            dimension,

        "minority_group":
            minority,

        "majority_group":
            majority,

        "n_outcomes":
            len(pair),

        "minority_mean":
            x.mean(),

        "majority_mean":
            y.mean(),

        "mean_difference":
            diff.mean(),

        "median_difference":
            np.median(diff),

        "rank_biserial":
            paired_rank_biserial(
                x,
                y
            ),

        "wilcoxon_W":
            W,

        "p_raw":
            p_value,
    })


dimension_tests = pd.DataFrame(
    dimension_tests
)


if len(dimension_tests):

    reject, p_adj, _, _ = (
        multipletests(
            dimension_tests[
                "p_raw"
            ],
            method="holm",
            alpha=ALPHA,
        )
    )

    dimension_tests[
        "p_holm"
    ] = p_adj

    dimension_tests[
        "significant_holm"
    ] = reject


print(
    dimension_tests
    .round(4)
    .to_string(
        index=False
    )
)

dimension_tests.to_csv(
    OUTPUT_DIR
    / "minority_vs_majority_by_dimension.csv",
    index=False,
)


# ============================================================
# AGGREGATED MINORITY vs MAJORITY
# ============================================================

condition_means = (
    social
    .groupby(
        [
            "outcome_id",
            "condition",
        ]
    )[
        "n_links"
    ]
    .mean()
    .unstack()
    .dropna()
)


aggregate_result = {}


print()
print("=" * 80)
print("AGGREGATED MINORITY vs MAJORITY")
print("=" * 80)


if (
    "minority"
    in condition_means.columns
    and
    "majority"
    in condition_means.columns
):

    x = (
        condition_means[
            "minority"
        ]
        .values
    )

    y = (
        condition_means[
            "majority"
        ]
        .values
    )

    diff = x - y

    result = wilcoxon(
        x,
        y,
        alternative="two-sided",
    )

    effect = (
        paired_rank_biserial(
            x,
            y
        )
    )

    aggregate_result = {
        "n_outcomes":
            len(x),

        "minority_mean":
            x.mean(),

        "majority_mean":
            y.mean(),

        "mean_difference":
            diff.mean(),

        "median_difference":
            np.median(diff),

        "wilcoxon_W":
            result.statistic,

        "p_value":
            result.pvalue,

        "rank_biserial":
            effect,

        "significant":
            bool(
                result.pvalue
                < ALPHA
            ),
    }

    print(
        f"\nMinority conditions mean: "
        f"{x.mean():.2f}"
    )

    print(
        f"Majority conditions mean: "
        f"{y.mean():.2f}"
    )

    print(
        f"Mean paired difference: "
        f"{diff.mean():.2f}"
    )

    print(
        f"Wilcoxon p: "
        f"{result.pvalue:.6g}"
    )

    print(
        f"Rank-biserial effect: "
        f"{effect:.3f}"
    )


if aggregate_result:

    pd.DataFrame(
        [aggregate_result]
    ).to_csv(
        OUTPUT_DIR
        / "aggregate_minority_vs_majority.csv",
        index=False,
    )


# ============================================================
# EXPLICIT GROUPS vs GENERIC CONTROL
# ============================================================

print()
print("=" * 80)
print("EXPLICIT GROUPS vs GENERIC CONTROL")
print("=" * 80)


control_name = "people"

control_tests = []


if control_name in pivot.columns:

    for group in pivot.columns:

        if group == control_name:
            continue

        pair = (
            pivot[
                [
                    group,
                    control_name,
                ]
            ]
            .dropna()
        )

        x = (
            pair[
                group
            ]
            .values
        )

        y = (
            pair[
                control_name
            ]
            .values
        )

        diff = x - y

        if np.all(
            diff == 0
        ):

            W = 0
            p_value = 1.0

        else:

            result = wilcoxon(
                x,
                y,
                alternative="two-sided",
            )

            W = (
                result.statistic
            )

            p_value = (
                result.pvalue
            )

        control_tests.append({
            "group":
                group,

            "n_matched":
                len(pair),

            "group_mean":
                x.mean(),

            "control_mean":
                y.mean(),

            "mean_difference":
                diff.mean(),

            "median_difference":
                np.median(diff),

            "rank_biserial":
                paired_rank_biserial(
                    x,
                    y
                ),

            "wilcoxon_W":
                W,

            "p_raw":
                p_value,
        })


control_tests = pd.DataFrame(
    control_tests
)


if len(control_tests):

    reject, p_adj, _, _ = (
        multipletests(
            control_tests[
                "p_raw"
            ],
            method="holm",
            alpha=ALPHA,
        )
    )

    control_tests[
        "p_holm"
    ] = p_adj

    control_tests[
        "significant_holm"
    ] = reject

    control_tests = (
        control_tests
        .sort_values(
            "p_holm"
        )
    )


print(
    control_tests
    .round(4)
    .to_string(
        index=False
    )
)


control_tests.to_csv(
    OUTPUT_DIR
    / "explicit_groups_vs_people.csv",
    index=False,
)


# ============================================================
# HIGHEST / LOWEST COUNTS
# ============================================================

cols = [
    "group",
    "domain",
    "outcome",
    "n_links",
]


highest = (
    links
    .sort_values(
        "n_links",
        ascending=False,
    )
    [cols]
    .head(15)
)

lowest = (
    links
    .sort_values(
        "n_links",
        ascending=True,
    )
    [cols]
    .head(15)
)


highest.to_csv(
    OUTPUT_DIR
    / "highest_link_counts.csv",
    index=False,
)

lowest.to_csv(
    OUTPUT_DIR
    / "lowest_link_counts.csv",
    index=False,
)


# ============================================================
# CLEAN ANALYTIC DATA
# ============================================================

links.to_csv(
    OUTPUT_DIR
    / "clean_aio_link_data.csv",
    index=False,
)


# ============================================================
# BUILD ONE LARGE JSON WITH EVERYTHING
# ============================================================

all_results = {

    # --------------------------------------------------------
    # Run metadata
    # --------------------------------------------------------

    "metadata": {
        "analysis":
            "Google AI Overview link count analysis",

        "location":
            "Dallas, Texas",

        "collection_directory":
            str(COLLECTION_DIR),

        "input_file":
            str(INPUT_FILE),

        "results_directory":
            str(OUTPUT_DIR),

        "alpha":
            ALPHA,
    },


    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    "dataset_summary": {
        "raw_observations":
            len(raw_df),

        "excluded_errors":
            len(errors),

        "usable_observations":
            len(df),

        "aio_present_n":
            int(
                df[
                    "aio_present"
                ].sum()
            ),

        "aio_present_rate":
            float(
                df[
                    "aio_present"
                ].mean()
            ),

        "aio_absent_n":
            int(
                (
                    ~df[
                        "aio_present"
                    ]
                ).sum()
            ),

        "link_analysis_n":
            len(links),

        "zero_link_aio_n":
            zero_link_count,

        "zero_link_aio_rate":
            float(
                zero_link_rate
            ),

        "mean_links_overall":
            float(
                links[
                    "n_links"
                ].mean()
            ),

        "median_links_overall":
            float(
                links[
                    "n_links"
                ].median()
            ),

        "sd_links_overall":
            float(
                links[
                    "n_links"
                ].std(
                    ddof=1
                )
            ),

        "min_links":
            int(
                links[
                    "n_links"
                ].min()
            ),

        "max_links":
            int(
                links[
                    "n_links"
                ].max()
            ),
    },


    # --------------------------------------------------------
    # Collection errors excluded
    # --------------------------------------------------------

    "excluded_error_observations":
        dataframe_to_records(
            errors
        ),


    # --------------------------------------------------------
    # Descriptive tables
    # --------------------------------------------------------

    "descriptive_results": {

        "by_group":
            dataframe_to_records(
                group_table
            ),

        "by_domain":
            dataframe_to_records(
                domain_table
            ),

        "by_outcome":
            dataframe_to_records(
                outcome_table
            ),

        "group_by_domain":
            dataframe_to_records(
                group_domain_json
            ),

        "aio_presence_by_group":
            dataframe_to_records(
                aio_presence
            ),
    },


    # --------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------

    "statistical_tests": {

        "global_friedman":
            json_safe(
                friedman_results
            ),

        "pairwise_group_wilcoxon":
            dataframe_to_records(
                pairwise
            ),

        "minority_vs_majority_by_dimension":
            dataframe_to_records(
                dimension_tests
            ),

        "aggregate_minority_vs_majority":
            json_safe(
                aggregate_result
            ),

        "explicit_groups_vs_generic_people":
            dataframe_to_records(
                control_tests
            ),
    },


    # --------------------------------------------------------
    # Extremes
    # --------------------------------------------------------

    "extreme_observations": {

        "highest_link_counts":
            dataframe_to_records(
                highest
            ),

        "lowest_link_counts":
            dataframe_to_records(
                lowest
            ),
    },


    # --------------------------------------------------------
    # Matched matrix
    # Useful for later analyses
    # --------------------------------------------------------

    "matched_group_outcome_matrix":
        dataframe_to_records(
            pivot
            .reset_index()
        ),


    # --------------------------------------------------------
    # ALL usable observations
    # Includes NO-AIO observations too
    # --------------------------------------------------------

    "all_usable_observations":
        dataframe_to_records(
            df
        ),


    # --------------------------------------------------------
    # ALL observations used in link analysis
    # --------------------------------------------------------

    "link_analysis_observations":
        dataframe_to_records(
            links
        ),
}


# Make absolutely sure numpy types
# have become regular Python values.
all_results = json_safe(
    all_results
)


# ============================================================
# WRITE GIANT JSON
# ============================================================

JSON_OUTPUT = (
    OUTPUT_DIR
    / "all_results.json"
)

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


# ============================================================
# README
# ============================================================

readme = f"""
Google AI Overview — Link Count Analysis
========================================

Collection:
{COLLECTION_DIR}

Input:
{INPUT_FILE}

Location:
Dallas, Texas

Raw observations:
{len(raw_df)}

Usable observations after ERROR exclusion:
{len(df)}

AIO-present observations:
{len(links)}

Main combined JSON:
all_results.json

Files generated:
- all_results.json
- links_by_group.csv
- links_by_domain.csv
- links_by_outcome.csv
- group_by_domain_mean_links.csv
- aio_presence_by_group.csv
- global_friedman_test.csv
- pairwise_group_wilcoxon.csv
- minority_vs_majority_by_dimension.csv
- aggregate_minority_vs_majority.csv
- explicit_groups_vs_people.csv
- highest_link_counts.csv
- lowest_link_counts.csv
- clean_aio_link_data.csv
"""

(
    OUTPUT_DIR
    / "README.txt"
).write_text(
    readme.strip() + "\n",
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "\nAll results saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCombined JSON:"
)

print(
    JSON_OUTPUT
)

print(
    "\nGenerated files:"
)

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  - {file.name}"
    )