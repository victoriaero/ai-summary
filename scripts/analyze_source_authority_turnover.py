from __future__ import annotations

import argparse
import json
import warnings
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "results" / "condition_source_analysis_dallas"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "source_authority_turnover_dallas"
)
DEFAULT_VALIDATION_SAMPLE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "annotation_inputs"
    / "authority_taxonomy_validation_sample.csv"
)

DIMENSION_ORDER = (
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
)

AUTHORITY_CATEGORIES = (
    "government",
    "intergovernmental",
    "academic_research",
    "policy_research",
    "news_media",
    "advocacy_community",
    "professional_association",
    "healthcare_provider",
    "commercial_service",
    "reference",
    "social_platform",
    "nonprofit_general",
    "other",
)

COMPARISONS = (
    ("minority_majority", "minority", "majority"),
    ("minority_generic", "minority", "control"),
    ("majority_generic", "majority", "control"),
)

BOOTSTRAP_RESAMPLES = 10_000
CORRELATION_BOOTSTRAP_RESAMPLES = 2_000
RANDOM_SEED = 20260929

SOCIAL_PLATFORMS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "reddit.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}

REFERENCE_SOURCES = {
    "britannica.com",
    "dictionary.com",
    "encyclopedia.com",
    "investopedia.com",
    "wikipedia.org",
}

INTERGOVERNMENTAL_SOURCES = {
    "europa.eu",
    "iadb.org",
    "imf.org",
    "oecd.org",
    "un.org",
    "unesco.org",
    "who.int",
    "worldbank.org",
}

ACADEMIC_PUBLISHERS_AND_REPOSITORIES = {
    "academia.edu",
    "annualreviews.org",
    "cambridge.org",
    "ebsco.com",
    "elsevier.com",
    "emerald.com",
    "frontiersin.org",
    "jstor.org",
    "mdpi.com",
    "nature.com",
    "oup.com",
    "plos.org",
    "researchgate.net",
    "sagepub.com",
    "sciencedirect.com",
    "semanticscholar.org",
    "springer.com",
    "ssrn.com",
    "tandfonline.com",
    "wiley.com",
}

POLICY_RESEARCH_SOURCES = {
    "americanprogress.org",
    "bipartisanpolicy.org",
    "brookings.edu",
    "cbpp.org",
    "cepr.net",
    "cgap.org",
    "commonwealthfund.org",
    "healthaffairs.org",
    "heritage.org",
    "kff.org",
    "opportunityinsights.org",
    "pewresearch.org",
    "rand.org",
    "tcf.org",
    "urban.org",
}

NEWS_MEDIA_SOURCES = {
    "abcnews.go.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "bloomberg.com",
    "cbsnews.com",
    "cnn.com",
    "forbes.com",
    "foxnews.com",
    "latimes.com",
    "medium.com",
    "nbcnews.com",
    "npr.org",
    "nytimes.com",
    "pbs.org",
    "politico.com",
    "reuters.com",
    "theconversation.com",
    "time.com",
    "usatoday.com",
    "washingtonpost.com",
}

ADVOCACY_SOURCES = {
    "aclu.org",
    "endhomelessness.org",
    "glaad.org",
    "hrc.org",
    "justiceinaging.org",
    "mapresearch.org",
    "nationalhomeless.org",
    "nlihc.org",
    "nwlc.org",
    "outandequal.org",
    "prisonpolicy.org",
    "sentencingproject.org",
    "transequality.org",
    "unidosus.org",
}

PROFESSIONAL_ASSOCIATIONS = {
    "americanbar.org",
    "ama-assn.org",
    "apa.org",
    "naspa.org",
}

POLICY_TOKENS = (
    "policy",
    "research",
    "institute",
    "insights",
)

ADVOCACY_TOKENS = (
    "advocacy",
    "civilrights",
    "disabilityrights",
    "equality",
    "equalrights",
    "homeless",
    "humanrights",
    "justice",
    "lgbt",
    "transgender",
)

PROFESSIONAL_TOKENS = (
    "association",
    "academy",
    "society",
)

HEALTHCARE_TOKENS = (
    "clinic",
    "healthcare",
    "hospital",
    "medicalcenter",
)


def hostname_matches(hostname: str, domains: set[str]) -> bool:
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in domains
    )


def institutional_sector(hostname: str, category: str) -> str:
    if category == "government":
        return "public"
    if category == "intergovernmental":
        return "intergovernmental"
    if hostname.endswith(".edu") or ".edu." in hostname or ".ac." in hostname:
        return "university"
    if hostname.endswith(".org"):
        return "nonprofit"
    if hostname.endswith((".com", ".co", ".io", ".ai", ".biz", ".net")):
        return "commercial"
    return "unknown"


def classify_authority(hostname: str) -> dict[str, str]:
    hostname = str(hostname or "").lower().strip(".")
    compact = hostname.replace("-", "").replace(".", "")

    if hostname_matches(hostname, SOCIAL_PLATFORMS):
        category, rule, confidence = (
            "social_platform",
            "known_social_platform",
            "high",
        )
    elif hostname_matches(hostname, REFERENCE_SOURCES):
        category, rule, confidence = "reference", "known_reference", "high"
    elif hostname_matches(hostname, INTERGOVERNMENTAL_SOURCES):
        category, rule, confidence = (
            "intergovernmental",
            "known_intergovernmental",
            "high",
        )
    elif (
        hostname.endswith(".gov")
        or ".gov." in hostname
        or hostname.endswith(".mil")
    ):
        category, rule, confidence = "government", "government_suffix", "high"
    elif hostname_matches(hostname, POLICY_RESEARCH_SOURCES):
        category, rule, confidence = (
            "policy_research",
            "known_policy_research",
            "high",
        )
    elif hostname_matches(hostname, ACADEMIC_PUBLISHERS_AND_REPOSITORIES):
        category, rule, confidence = (
            "academic_research",
            "known_academic_publisher_or_repository",
            "high",
        )
    elif hostname_matches(hostname, NEWS_MEDIA_SOURCES):
        category, rule, confidence = "news_media", "known_news_media", "high"
    elif hostname_matches(hostname, ADVOCACY_SOURCES):
        category, rule, confidence = (
            "advocacy_community",
            "known_advocacy",
            "high",
        )
    elif hostname_matches(hostname, PROFESSIONAL_ASSOCIATIONS):
        category, rule, confidence = (
            "professional_association",
            "known_professional_association",
            "high",
        )
    elif hostname.endswith(".edu") or ".edu." in hostname or ".ac." in hostname:
        category, rule, confidence = (
            "academic_research",
            "academic_suffix",
            "medium",
        )
    elif any(token in compact for token in ADVOCACY_TOKENS):
        category, rule, confidence = (
            "advocacy_community",
            "advocacy_domain_token",
            "low",
        )
    elif any(token in compact for token in POLICY_TOKENS):
        category, rule, confidence = (
            "policy_research",
            "policy_domain_token",
            "low",
        )
    elif any(token in compact for token in PROFESSIONAL_TOKENS):
        category, rule, confidence = (
            "professional_association",
            "professional_domain_token",
            "low",
        )
    elif any(token in compact for token in HEALTHCARE_TOKENS):
        category, rule, confidence = (
            "healthcare_provider",
            "healthcare_domain_token",
            "low",
        )
    elif hostname.endswith((".com", ".co", ".io", ".ai", ".biz")):
        category, rule, confidence = (
            "commercial_service",
            "commercial_suffix_fallback",
            "low",
        )
    elif hostname.endswith(".org"):
        category, rule, confidence = (
            "nonprofit_general",
            "nonprofit_suffix_fallback",
            "low",
        )
    else:
        category, rule, confidence = "other", "unclassified_fallback", "low"

    return {
        "authority_category": category,
        "institutional_sector": institutional_sector(hostname, category),
        "classification_rule": rule,
        "classification_confidence": confidence,
    }


def add_authority_taxonomy(sources: pd.DataFrame) -> pd.DataFrame:
    classifications = pd.DataFrame(
        [
            classify_authority(hostname)
            for hostname in sources["source_hostname"].fillna("")
        ],
        index=sources.index,
    )
    return pd.concat([sources.copy(), classifications], axis=1)


def build_domain_crosswalk(classified: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "registrable_domain",
        "authority_category",
        "institutional_sector",
        "classification_rule",
        "classification_confidence",
    ]
    crosswalk = classified[columns].drop_duplicates()
    conflicting = (
        crosswalk.groupby("registrable_domain")["authority_category"]
        .transform("nunique")
        .gt(1)
    )
    crosswalk["domain_has_conflicting_labels"] = conflicting
    occurrences = (
        classified.groupby("registrable_domain")
        .size()
        .rename("source_occurrences")
        .reset_index()
    )
    return (
        crosswalk.merge(occurrences, on="registrable_domain", how="left")
        .sort_values(["source_occurrences", "registrable_domain"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_authority_composition(
    metrics: pd.DataFrame,
    classified: pd.DataFrame,
) -> pd.DataFrame:
    unique_sources = classified.drop_duplicates(["query_id", "normalized_url"])
    counts = (
        unique_sources.groupby(["query_id", "authority_category"])
        .size()
        .rename("category_source_count")
        .reset_index()
    )
    query_columns = [
        "query_id",
        "dimension",
        "condition",
        "group",
        "domain",
        "outcome",
        "n_unique_sources",
    ]
    queries = metrics[query_columns].copy()
    categories = pd.DataFrame({"authority_category": AUTHORITY_CATEGORIES})
    queries["_join"] = 1
    categories["_join"] = 1
    composition = queries.merge(categories, on="_join").drop(columns="_join")
    composition = composition.merge(
        counts,
        on=["query_id", "authority_category"],
        how="left",
    )
    composition["category_source_count"] = (
        composition["category_source_count"].fillna(0).astype(int)
    )
    composition["category_source_share"] = (
        composition["category_source_count"]
        / composition["n_unique_sources"].replace(0, np.nan)
    )
    return composition


def query_lookup(metrics: pd.DataFrame) -> tuple[dict, dict]:
    social: dict[tuple[str, str, str, str], pd.Series] = {}
    controls: dict[tuple[str, str], pd.Series] = {}
    for _, row in metrics.iterrows():
        if row["condition"] == "control":
            controls[(row["domain"], row["outcome"])] = row
        else:
            social[
                (
                    row["dimension"],
                    row["domain"],
                    row["outcome"],
                    row["condition"],
                )
            ] = row
    return social, controls


def paired_queries(metrics: pd.DataFrame):
    social, controls = query_lookup(metrics)
    units = metrics.loc[
        metrics["condition"].isin(["minority", "majority"]),
        ["dimension", "domain", "outcome"],
    ].drop_duplicates()
    dimension_rank = {value: index for index, value in enumerate(DIMENSION_ORDER)}
    units = units.assign(
        _dimension_rank=units["dimension"].map(dimension_rank).fillna(999)
    ).sort_values(["_dimension_rank", "domain", "outcome"])

    for row in units.itertuples(index=False):
        condition_rows = {
            condition: social.get(
                (row.dimension, row.domain, row.outcome, condition)
            )
            for condition in ("minority", "majority")
        }
        condition_rows["control"] = controls.get((row.domain, row.outcome))
        for comparison, condition_a, condition_b in COMPARISONS:
            a = condition_rows[condition_a]
            b = condition_rows[condition_b]
            if a is None or b is None:
                continue
            yield {
                "dimension": row.dimension,
                "domain": row.domain,
                "outcome": row.outcome,
                "comparison": comparison,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "group_a": a["group"],
                "group_b": b["group"],
                "query_id_a": a["query_id"],
                "query_id_b": b["query_id"],
            }


def build_authority_contrasts(
    metrics: pd.DataFrame,
    composition: pd.DataFrame,
) -> pd.DataFrame:
    indexed = composition.set_index(["query_id", "authority_category"])
    rows = []
    for pair in paired_queries(metrics):
        for category in AUTHORITY_CATEGORIES:
            a = indexed.loc[(pair["query_id_a"], category)]
            b = indexed.loc[(pair["query_id_b"], category)]
            rows.append(
                {
                    **pair,
                    "authority_category": category,
                    "n_unique_sources_a": int(a["n_unique_sources"]),
                    "n_unique_sources_b": int(b["n_unique_sources"]),
                    "category_count_a": int(a["category_source_count"]),
                    "category_count_b": int(b["category_source_count"]),
                    "category_count_delta": int(
                        a["category_source_count"] - b["category_source_count"]
                    ),
                    "category_share_a": a["category_source_share"],
                    "category_share_b": b["category_source_share"],
                    "category_share_delta": (
                        a["category_source_share"] - b["category_source_share"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def aggregate_authority_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "social_dimension": [
            "dimension",
            "comparison",
            "authority_category",
        ],
        "domain": ["domain", "comparison", "authority_category"],
        "outcome": [
            "domain",
            "outcome",
            "comparison",
            "authority_category",
        ],
    }
    outputs = []
    for level, columns in levels.items():
        grouped = (
            contrasts.groupby(columns, dropna=False)
            .agg(
                n_pairs=("query_id_a", "size"),
                n_share_pairs=("category_share_delta", "count"),
                mean_count_delta=("category_count_delta", "mean"),
                median_count_delta=("category_count_delta", "median"),
                mean_share_delta=("category_share_delta", "mean"),
                median_share_delta=("category_share_delta", "median"),
                positive_share_delta_n=(
                    "category_share_delta",
                    lambda values: int((values > 0).sum()),
                ),
                equal_share_delta_n=(
                    "category_share_delta",
                    lambda values: int((values == 0).sum()),
                ),
                negative_share_delta_n=(
                    "category_share_delta",
                    lambda values: int((values < 0).sum()),
                ),
            )
            .reset_index()
        )
        grouped.insert(0, "aggregation_level", level)
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True, sort=False)


def source_sets(
    classified: pd.DataFrame,
    query_id: str,
    column: str,
) -> set[str]:
    values = classified.loc[classified["query_id"] == query_id, column]
    return set(values.dropna().astype(str))


def turnover_pattern(a_only: int, b_only: int) -> str:
    if a_only == 0 and b_only == 0:
        return "identical"
    if b_only == 0:
        return "pure_expansion"
    if a_only == 0:
        return "pure_contraction"
    if a_only > b_only:
        return "substitution_plus_net_expansion"
    if a_only < b_only:
        return "substitution_plus_net_contraction"
    return "balanced_substitution"


def build_turnover(
    metrics: pd.DataFrame,
    classified: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    membership_rows = []
    url_metadata = (
        classified.sort_values("source_position")
        .drop_duplicates(["query_id", "normalized_url"])
        .set_index(["query_id", "normalized_url"])
    )

    for pair in paired_queries(metrics):
        for granularity, column in (
            ("canonical_url", "normalized_url"),
            ("registrable_domain", "registrable_domain"),
        ):
            set_a = source_sets(classified, pair["query_id_a"], column)
            set_b = source_sets(classified, pair["query_id_b"], column)
            shared = set_a & set_b
            a_only = set_a - set_b
            b_only = set_b - set_a
            union = set_a | set_b
            union_n = len(union)
            rows.append(
                {
                    **pair,
                    "granularity": granularity,
                    "set_size_a": len(set_a),
                    "set_size_b": len(set_b),
                    "shared_n": len(shared),
                    "a_only_n": len(a_only),
                    "b_only_n": len(b_only),
                    "union_n": union_n,
                    "shared_share_union": (
                        len(shared) / union_n if union_n else np.nan
                    ),
                    "a_only_share_union": (
                        len(a_only) / union_n if union_n else np.nan
                    ),
                    "b_only_share_union": (
                        len(b_only) / union_n if union_n else np.nan
                    ),
                    "replacement_mass": min(len(a_only), len(b_only)),
                    "net_a_minus_b": len(a_only) - len(b_only),
                    "net_expansion_normalized": (
                        (len(a_only) - len(b_only)) / union_n
                        if union_n
                        else np.nan
                    ),
                    "net_expansion_from_b_to_a": max(
                        len(a_only) - len(b_only), 0
                    ),
                    "net_contraction_from_b_to_a": max(
                        len(b_only) - len(a_only), 0
                    ),
                    "turnover_rate": (
                        (len(a_only) + len(b_only)) / union_n
                        if union_n
                        else np.nan
                    ),
                    "jaccard": len(shared) / union_n if union_n else np.nan,
                    "both_sets_nonempty": bool(set_a and set_b),
                    "pattern": turnover_pattern(len(a_only), len(b_only)),
                }
            )

            if granularity != "canonical_url":
                continue
            statuses = (
                [(value, "shared") for value in sorted(shared)]
                + [(value, "a_only") for value in sorted(a_only)]
                + [(value, "b_only") for value in sorted(b_only)]
            )
            for value, status in statuses:
                metadata_query = (
                    pair["query_id_b"] if status == "b_only" else pair["query_id_a"]
                )
                metadata = url_metadata.loc[(metadata_query, value)]
                membership_rows.append(
                    {
                        **pair,
                        "membership": status,
                        "normalized_url": value,
                        "registrable_domain": metadata["registrable_domain"],
                        "authority_category": metadata["authority_category"],
                        "institutional_sector": metadata["institutional_sector"],
                        "classification_rule": metadata["classification_rule"],
                        "classification_confidence": metadata[
                            "classification_confidence"
                        ],
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(membership_rows)


def aggregate_turnover(turnover: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "social_dimension": ["dimension", "comparison", "granularity"],
        "domain": ["domain", "comparison", "granularity"],
        "outcome": ["domain", "outcome", "comparison", "granularity"],
    }
    outputs = []
    for level, columns in levels.items():
        grouped = (
            turnover.groupby(columns, dropna=False)
            .agg(
                n_pairs=("query_id_a", "size"),
                both_sets_nonempty_n=("both_sets_nonempty", "sum"),
                mean_shared=("shared_n", "mean"),
                mean_a_only=("a_only_n", "mean"),
                mean_b_only=("b_only_n", "mean"),
                mean_replacement_mass=("replacement_mass", "mean"),
                mean_net_a_minus_b=("net_a_minus_b", "mean"),
                median_net_a_minus_b=("net_a_minus_b", "median"),
                mean_net_expansion_normalized=(
                    "net_expansion_normalized",
                    "mean",
                ),
                median_net_expansion_normalized=(
                    "net_expansion_normalized",
                    "median",
                ),
                mean_turnover_rate=("turnover_rate", "mean"),
                median_turnover_rate=("turnover_rate", "median"),
                pooled_shared=("shared_n", "sum"),
                pooled_a_only=("a_only_n", "sum"),
                pooled_b_only=("b_only_n", "sum"),
            )
            .reset_index()
        )
        pooled_union = (
            grouped["pooled_shared"]
            + grouped["pooled_a_only"]
            + grouped["pooled_b_only"]
        )
        grouped["pooled_shared_share"] = (
            grouped["pooled_shared"] / pooled_union.replace(0, np.nan)
        )
        grouped["pooled_a_only_share"] = (
            grouped["pooled_a_only"] / pooled_union.replace(0, np.nan)
        )
        grouped["pooled_b_only_share"] = (
            grouped["pooled_b_only"] / pooled_union.replace(0, np.nan)
        )
        grouped.insert(0, "aggregation_level", level)
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True, sort=False)


def aggregate_membership_authority(membership: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        membership.groupby(
            [
                "dimension",
                "domain",
                "outcome",
                "comparison",
                "membership",
                "authority_category",
            ],
            dropna=False,
        )
        .size()
        .rename("source_count")
        .reset_index()
    )
    totals = grouped.groupby(
        ["dimension", "domain", "outcome", "comparison", "membership"],
        dropna=False,
    )["source_count"].transform("sum")
    grouped["source_share_within_membership"] = grouped["source_count"] / totals
    return grouped


def seeded_rng(*parts: object) -> np.random.Generator:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    seed = (RANDOM_SEED + zlib.crc32(token)) % (2**32)
    return np.random.default_rng(seed)


def bootstrap_mean_ci(
    values: pd.Series,
    *seed_parts: object,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return np.nan, np.nan
    rng = seeded_rng(*seed_parts)
    indices = rng.integers(0, len(array), size=(n_resamples, len(array)))
    means = array[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def bootstrap_authority_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["dimension", "comparison", "authority_category"]
    for keys, group in contrasts.groupby(columns, sort=False):
        values = group["category_share_delta"].dropna()
        low, high = bootstrap_mean_ci(values, "authority", *keys)
        rows.append(
            {
                **dict(zip(columns, keys)),
                "n_outcomes": len(values),
                "mean_share_delta": values.mean(),
                "median_share_delta": values.median(),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_turnover_metrics(turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["dimension", "comparison", "granularity"]
    metrics = (
        "shared_share_union",
        "a_only_share_union",
        "b_only_share_union",
        "net_expansion_normalized",
        "turnover_rate",
    )
    for keys, group in turnover.groupby(columns, sort=False):
        row = {**dict(zip(columns, keys)), "n_outcomes": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            low, high = bootstrap_mean_ci(values, "turnover", metric, *keys)
            row[f"mean_{metric}"] = values.mean()
            row[f"median_{metric}"] = values.median()
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row["bootstrap_resamples"] = BOOTSTRAP_RESAMPLES
        rows.append(row)
    return pd.DataFrame(rows)


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2

    def kl_divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    return (kl_divergence(left) + kl_divergence(right)) / (2 * np.log(2))


def build_authority_distances(
    metrics: pd.DataFrame,
    composition: pd.DataFrame,
) -> pd.DataFrame:
    vectors = (
        composition.pivot(
            index="query_id",
            columns="authority_category",
            values="category_source_share",
        )
        .reindex(columns=AUTHORITY_CATEGORIES)
    )
    rows = []
    for pair in paired_queries(metrics):
        left = vectors.loc[pair["query_id_a"]].to_numpy(dtype=float)
        right = vectors.loc[pair["query_id_b"]].to_numpy(dtype=float)
        valid = not (np.isnan(left).any() or np.isnan(right).any())
        rows.append(
            {
                **pair,
                "composition_defined_both": valid,
                "authority_total_variation": (
                    0.5 * np.abs(left - right).sum() if valid else np.nan
                ),
                "authority_jensen_shannon": (
                    jensen_shannon_divergence(left, right)
                    if valid
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_spearman(
    group: pd.DataFrame,
    *seed_parts: object,
) -> tuple[float, float]:
    data = group.dropna(
        subset=["turnover_rate", "authority_total_variation"]
    ).copy()
    units = data["outcome_key"].unique()
    if len(units) < 4:
        return np.nan, np.nan
    unit_indices = [
        np.flatnonzero(data["outcome_key"].to_numpy() == unit)
        for unit in units
    ]
    x_ranks = data["turnover_rate"].rank().to_numpy(dtype=float)
    y_ranks = data["authority_total_variation"].rank().to_numpy(dtype=float)
    rng = seeded_rng("spearman", *seed_parts)
    estimates = []
    for _ in range(CORRELATION_BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, len(units), size=len(units))
        indices = np.concatenate([unit_indices[index] for index in sampled])
        x_sample = x_ranks[indices]
        y_sample = y_ranks[indices]
        if np.std(x_sample) == 0 or np.std(y_sample) == 0:
            continue
        estimate = np.corrcoef(x_sample, y_sample)[0, 1]
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return np.nan, np.nan
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def build_turnover_authority_link(
    turnover: pd.DataFrame,
    authority_distances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "dimension",
        "domain",
        "outcome",
        "comparison",
        "condition_a",
        "condition_b",
        "group_a",
        "group_b",
        "query_id_a",
        "query_id_b",
    ]
    linked = turnover.merge(
        authority_distances[
            keys
            + [
                "composition_defined_both",
                "authority_total_variation",
                "authority_jensen_shannon",
            ]
        ],
        on=keys,
        validate="many_to_one",
    )
    linked["outcome_key"] = linked["domain"] + " :: " + linked["outcome"]

    rows = []
    for (comparison, granularity), comparison_group in linked.groupby(
        ["comparison", "granularity"],
        sort=False,
    ):
        groups = [("All dimensions", comparison_group)]
        groups.extend(comparison_group.groupby("dimension", sort=False))
        for dimension, group in groups:
            valid = group.dropna(
                subset=["turnover_rate", "authority_total_variation"]
            )
            correlation = (
                spearmanr(
                    valid["turnover_rate"],
                    valid["authority_total_variation"],
                ).statistic
                if (
                    len(valid) >= 4
                    and valid["turnover_rate"].nunique() > 1
                    and valid["authority_total_variation"].nunique() > 1
                )
                else np.nan
            )
            low, high = bootstrap_spearman(
                valid,
                comparison,
                granularity,
                dimension,
            )
            rows.append(
                {
                    "dimension": dimension,
                    "comparison": comparison,
                    "granularity": granularity,
                    "n_pairs": len(valid),
                    "spearman_r": correlation,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_resamples": CORRELATION_BOOTSTRAP_RESAMPLES,
                }
            )
    return linked, pd.DataFrame(rows)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return adjusted
    raw = valid.to_numpy()
    corrected = np.maximum.accumulate(
        (len(raw) - np.arange(len(raw))) * raw
    )
    adjusted.loc[valid.index] = np.minimum(corrected, 1)
    return adjusted


def fit_authority_binary_models(
    classified: pd.DataFrame,
) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    unique_sources = classified.drop_duplicates(["query_id", "normalized_url"])
    controls = unique_sources.loc[unique_sources["condition"] == "control"]
    rows = []
    contrast_vectors = {
        "minority_majority": np.array([1.0, 0.0]),
        "minority_generic": np.array([1.0, -1.0]),
        "majority_generic": np.array([0.0, -1.0]),
    }

    for dimension in DIMENSION_ORDER:
        social = unique_sources.loc[
            (unique_sources["dimension"] == dimension)
            & unique_sources["condition"].isin(["minority", "majority"])
        ]
        model_base = pd.concat([social, controls], ignore_index=True)
        model_base["outcome_id"] = (
            model_base["domain"] + " :: " + model_base["outcome"]
        )
        model_base["minority_indicator"] = (
            model_base["condition"] == "minority"
        ).astype(int)
        model_base["generic_indicator"] = (
            model_base["condition"] == "control"
        ).astype(int)

        for category in AUTHORITY_CATEGORIES:
            data = model_base.copy()
            data["is_category"] = (
                data["authority_category"] == category
            ).astype(int)
            positives = int(data["is_category"].sum())
            negatives = int(len(data) - positives)
            if positives < 5 or negatives < 5:
                for comparison in contrast_vectors:
                    rows.append(
                        {
                            "dimension": dimension,
                            "authority_category": category,
                            "comparison": comparison,
                            "status": "insufficient_category_variation",
                            "n_sources": len(data),
                            "n_positive": positives,
                        }
                    )
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = smf.glm(
                        (
                            "is_category ~ minority_indicator"
                            " + generic_indicator + C(outcome_id)"
                        ),
                        data=data,
                        family=sm.families.Binomial(),
                    ).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": data["query_id"]},
                    )
                names = ["minority_indicator", "generic_indicator"]
                coefficients = result.params[names].to_numpy(dtype=float)
                covariance = result.cov_params().loc[names, names].to_numpy(
                    dtype=float
                )
                separated = bool(
                    np.abs(coefficients).max() > 8
                    or not np.isfinite(covariance).all()
                )
                for comparison, vector in contrast_vectors.items():
                    estimate = float(vector @ coefficients)
                    variance = float(vector @ covariance @ vector)
                    standard_error = np.sqrt(max(variance, 0))
                    z_value = (
                        estimate / standard_error
                        if standard_error > 0
                        else np.nan
                    )
                    p_value = (
                        2 * norm.sf(abs(z_value))
                        if np.isfinite(z_value) and not separated
                        else np.nan
                    )
                    rows.append(
                        {
                            "dimension": dimension,
                            "authority_category": category,
                            "comparison": comparison,
                            "status": (
                                "quasi_complete_separation"
                                if separated
                                else "fitted"
                            ),
                            "n_sources": len(data),
                            "n_positive": positives,
                            "log_odds_contrast": estimate,
                            "standard_error": standard_error,
                            "ci_low": estimate - 1.96 * standard_error,
                            "ci_high": estimate + 1.96 * standard_error,
                            "odds_ratio": np.exp(np.clip(estimate, -20, 20)),
                            "p_value": p_value,
                            "model": (
                                "binary GLM with outcome fixed effects; "
                                "query-clustered standard errors"
                            ),
                        }
                    )
            except Exception as exc:
                for comparison in contrast_vectors:
                    rows.append(
                        {
                            "dimension": dimension,
                            "authority_category": category,
                            "comparison": comparison,
                            "status": "failed",
                            "n_sources": len(data),
                            "n_positive": positives,
                            "error": str(exc),
                        }
                    )

    results = pd.DataFrame(rows)
    results["p_value_holm"] = (
        results.groupby(["dimension", "comparison"], group_keys=False)[
            "p_value"
        ]
        .apply(holm_adjust)
        .reindex(results.index)
    )
    return results


def fit_authority_hierarchical_models(
    classified: pd.DataFrame,
) -> pd.DataFrame:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    unique_sources = classified.drop_duplicates(["query_id", "normalized_url"])
    controls = unique_sources.loc[unique_sources["condition"] == "control"]
    contrast_vectors = {
        "minority_majority": np.array([1.0, 0.0]),
        "minority_generic": np.array([1.0, -1.0]),
        "majority_generic": np.array([0.0, -1.0]),
    }
    rows = []
    for dimension in DIMENSION_ORDER:
        social = unique_sources.loc[
            (unique_sources["dimension"] == dimension)
            & unique_sources["condition"].isin(["minority", "majority"])
        ]
        model_base = pd.concat([social, controls], ignore_index=True)
        model_base["outcome_id"] = (
            model_base["domain"] + " :: " + model_base["outcome"]
        )
        model_base["minority_indicator"] = (
            model_base["condition"] == "minority"
        ).astype(int)
        model_base["generic_indicator"] = (
            model_base["condition"] == "control"
        ).astype(int)

        for category in AUTHORITY_CATEGORIES:
            data = model_base.copy()
            data["is_category"] = (
                data["authority_category"] == category
            ).astype(int)
            positives = int(data["is_category"].sum())
            negatives = int(len(data) - positives)
            if positives < 5 or negatives < 5:
                for comparison in contrast_vectors:
                    rows.append(
                        {
                            "dimension": dimension,
                            "authority_category": category,
                            "comparison": comparison,
                            "status": "insufficient_category_variation",
                            "n_sources": len(data),
                            "n_positive": positives,
                        }
                    )
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = BinomialBayesMixedGLM.from_formula(
                        (
                            "is_category ~ minority_indicator"
                            " + generic_indicator"
                        ),
                        {"outcome": "0 + C(outcome_id)"},
                        data,
                    )
                    result = model.fit_vb()
                fixed_means = result.fe_mean[1:3]
                fixed_sds = result.fe_sd[1:3]
                separated = bool(
                    np.abs(fixed_means).max() > 8
                    or not np.isfinite(fixed_sds).all()
                )
                outcome_log_sd = float(result.vcp_mean[0])
                outcome_log_sd_se = float(result.vcp_sd[0])
                for comparison, vector in contrast_vectors.items():
                    estimate = float(vector @ fixed_means)
                    posterior_sd = float(
                        np.sqrt(np.sum((vector * fixed_sds) ** 2))
                    )
                    rows.append(
                        {
                            "dimension": dimension,
                            "authority_category": category,
                            "comparison": comparison,
                            "status": (
                                "quasi_complete_separation"
                                if separated
                                else "fitted"
                            ),
                            "n_sources": len(data),
                            "n_positive": positives,
                            "posterior_log_odds_contrast": estimate,
                            "posterior_sd": posterior_sd,
                            "credible_interval_low": (
                                estimate - 1.96 * posterior_sd
                            ),
                            "credible_interval_high": (
                                estimate + 1.96 * posterior_sd
                            ),
                            "posterior_odds_ratio": np.exp(
                                np.clip(estimate, -20, 20)
                            ),
                            "outcome_random_intercept_sd": np.exp(
                                outcome_log_sd
                            ),
                            "outcome_random_intercept_sd_low": np.exp(
                                outcome_log_sd - 1.96 * outcome_log_sd_se
                            ),
                            "outcome_random_intercept_sd_high": np.exp(
                                outcome_log_sd + 1.96 * outcome_log_sd_se
                            ),
                            "model": (
                                "variational-Bayes binomial mixed model; "
                                "random intercept by outcome"
                            ),
                        }
                    )
            except Exception as exc:
                for comparison in contrast_vectors:
                    rows.append(
                        {
                            "dimension": dimension,
                            "authority_category": category,
                            "comparison": comparison,
                            "status": "failed",
                            "n_sources": len(data),
                            "n_positive": positives,
                            "error": str(exc),
                        }
                    )
    return pd.DataFrame(rows)


def build_validation_sample(
    domain_crosswalk: pd.DataFrame,
    classified: pd.DataFrame,
    n_per_stratum: int = 6,
) -> pd.DataFrame:
    examples = (
        classified.sort_values("source_position")
        .drop_duplicates("registrable_domain")
        .loc[
            :,
            ["registrable_domain", "source_hostname", "normalized_url"],
        ]
        .rename(columns={"normalized_url": "example_url"})
    )
    frame = domain_crosswalk.merge(
        examples,
        on="registrable_domain",
        how="left",
        validate="one_to_one",
    )
    samples = []
    stratum_columns = ["authority_category", "classification_confidence"]
    for keys, group in frame.groupby(stratum_columns, sort=False):
        sample_n = min(n_per_stratum, len(group))
        rng = seeded_rng("validation", *keys)
        selected = group.iloc[
            rng.choice(len(group), size=sample_n, replace=False)
        ].copy()
        selected["stratum_population_n"] = len(group)
        selected["stratum_sample_n"] = sample_n
        selected["sampling_probability"] = sample_n / len(group)
        samples.append(selected)
    sample = pd.concat(samples, ignore_index=True)
    sample["human_authority_category"] = ""
    sample["human_institutional_sector"] = ""
    sample["reviewer"] = ""
    sample["annotation_notes"] = ""
    return sample.sort_values(
        ["authority_category", "classification_confidence", "registrable_domain"]
    )


def plot_authority_composition(
    composition: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    pooled = (
        composition.groupby(
            ["dimension", "condition", "group", "authority_category"],
            dropna=False,
        )["category_source_count"]
        .sum()
        .reset_index()
    )
    totals = pooled.groupby(
        ["dimension", "condition", "group"], dropna=False
    )["category_source_count"].transform("sum")
    pooled["share"] = pooled["category_source_count"] / totals.replace(0, np.nan)
    order = (
        pooled[["dimension", "condition", "group"]]
        .drop_duplicates()
        .assign(
            rank=lambda frame: frame["dimension"]
            .map({value: i for i, value in enumerate(DIMENSION_ORDER)})
            .fillna(-1),
            condition_rank=lambda frame: frame["condition"].map(
                {"control": 0, "majority": 1, "minority": 2}
            ),
        )
        .sort_values(["rank", "condition_rank"])
    )
    group_order = order["group"].tolist()
    pivot = (
        pooled.pivot_table(
            index="group",
            columns="authority_category",
            values="share",
            fill_value=0,
        )
        .reindex(group_order)
        .reindex(columns=AUTHORITY_CATEGORIES, fill_value=0)
    )
    colors = sns.color_palette("tab20", n_colors=len(AUTHORITY_CATEGORIES))
    fig, ax = plt.subplots(figsize=(15, 8))
    left = np.zeros(len(pivot))
    for category, color in zip(AUTHORITY_CATEGORIES, colors):
        values = pivot[category].to_numpy()
        ax.barh(pivot.index, values, left=left, label=category, color=color)
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of unique sources")
    ax.set_ylabel("")
    ax.set_title("Automatic source-authority composition by query condition")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "authority_composition_by_condition.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_authority_heatmap(
    contrasts: pd.DataFrame,
    output_dir: Path,
) -> None:
    data = contrasts.loc[
        contrasts["comparison"] == "minority_majority"
    ].copy()
    data["outcome_id"] = data["domain"] + " :: " + data["outcome"]
    max_abs = np.nanmax(np.abs(data["category_share_delta"]))
    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(24, 21))
    for index, (ax, dimension) in enumerate(zip(axes.flat, DIMENSION_ORDER)):
        subset = data.loc[data["dimension"] == dimension]
        heatmap = subset.pivot(
            index="outcome_id",
            columns="authority_category",
            values="category_share_delta",
        ).reindex(columns=AUTHORITY_CATEGORIES)
        sns.heatmap(
            heatmap,
            cmap="vlag",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            linewidths=0.2,
            cbar=index == len(DIMENSION_ORDER) - 1,
            cbar_kws={
                "label": "Share difference (minority − majority)"
            },
            ax=ax,
        )
        ax.set_xlabel("Automatic authority category")
        ax.set_ylabel("")
        ax.set_title(dimension, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle(
        "Outcome-level changes in source-authority composition",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(
        output_dir / "authority_share_difference_by_dimension.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_turnover(turnover: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "minority_majority": "Minority × majority",
        "minority_generic": "Minority × generic",
        "majority_generic": "Majority × generic",
    }
    granularities = ("canonical_url", "registrable_domain")
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    palette = dict(
        zip(DIMENSION_ORDER, sns.color_palette("colorblind", len(DIMENSION_ORDER)))
    )
    for row, granularity in enumerate(granularities):
        for column, (comparison, title) in enumerate(labels.items()):
            ax = axes[row, column]
            subset = turnover.loc[
                (turnover["granularity"] == granularity)
                & (turnover["comparison"] == comparison)
            ]
            sns.scatterplot(
                data=subset,
                x="net_expansion_normalized",
                y="turnover_rate",
                hue="dimension",
                hue_order=DIMENSION_ORDER,
                palette=palette,
                alpha=0.8,
                s=45,
                ax=ax,
                legend=row == 0 and column == 0,
            )
            ax.axvline(0, linestyle="--", color="#555555", linewidth=1)
            ax.set_xlim(-1.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("Net expansion E (A − B)")
            ax.set_ylabel("Turnover T = 1 − Jaccard")
            ax.set_title(
                f"{title}\n{granularity.replace('_', ' ').title()}"
            )
            if row == 0 and column == 0:
                ax.legend(
                    title="Social dimension",
                    fontsize=7,
                    title_fontsize=8,
                )
    fig.suptitle(
        "Normalized source expansion and turnover",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        output_dir / "normalized_source_expansion_turnover.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_set_components(turnover: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "minority_majority": "Minority × majority",
        "minority_generic": "Minority × generic",
        "majority_generic": "Majority × generic",
    }
    granularities = ("canonical_url", "registrable_domain")
    colors = {
        "shared": "#888888",
        "a_only": "#7B2CBF",
        "b_only": "#2A6FBB",
    }
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True)
    for row, granularity in enumerate(granularities):
        for column, (comparison, title) in enumerate(labels.items()):
            ax = axes[row, column]
            subset = turnover.loc[
                (turnover["granularity"] == granularity)
                & (turnover["comparison"] == comparison)
            ]
            pooled = (
                subset.groupby("dimension")[
                    ["shared_n", "a_only_n", "b_only_n"]
                ]
                .sum()
                .reindex(DIMENSION_ORDER)
            )
            totals = pooled.sum(axis=1).replace(0, np.nan)
            shares = pooled.div(totals, axis=0)
            y = np.arange(len(shares))
            left = np.zeros(len(shares))
            for column_name, legend_label in (
                ("shared_n", "Shared"),
                ("a_only_n", "A-only"),
                ("b_only_n", "B-only"),
            ):
                short_name = column_name.removesuffix("_n")
                values = shares[column_name].fillna(0).to_numpy()
                ax.barh(
                    y,
                    values,
                    left=left,
                    color=colors[short_name],
                    label=legend_label,
                )
                left += values
            ax.set_yticks(y, DIMENSION_ORDER, fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Pooled share of source union")
            ax.set_title(
                f"{title}\n{granularity.replace('_', ' ').title()}"
            )
            ax.grid(axis="y", visible=False)
            if row == 0 and column == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Shared and condition-exclusive source components",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        output_dir / "source_set_components.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_turnover_authority_link(
    linked: pd.DataFrame,
    output_dir: Path,
) -> None:
    labels = {
        "minority_majority": "Minority × majority",
        "minority_generic": "Minority × generic",
        "majority_generic": "Majority × generic",
    }
    granularities = ("canonical_url", "registrable_domain")
    palette = dict(
        zip(DIMENSION_ORDER, sns.color_palette("colorblind", len(DIMENSION_ORDER)))
    )
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False, sharey=True)
    for row, granularity in enumerate(granularities):
        for column, (comparison, title) in enumerate(labels.items()):
            ax = axes[row, column]
            subset = linked.loc[
                (linked["granularity"] == granularity)
                & (linked["comparison"] == comparison)
            ]
            sns.scatterplot(
                data=subset,
                x="turnover_rate",
                y="authority_total_variation",
                hue="dimension",
                hue_order=DIMENSION_ORDER,
                palette=palette,
                alpha=0.8,
                s=42,
                ax=ax,
                legend=row == 0 and column == 0,
            )
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("Source turnover T")
            ax.set_ylabel("Authority-composition distance")
            ax.set_title(
                f"{title}\n{granularity.replace('_', ' ').title()}"
            )
            if row == 0 and column == 0:
                ax.legend(
                    title="Social dimension",
                    fontsize=7,
                    title_fontsize=8,
                )
    fig.suptitle(
        "Does source turnover coincide with authority turnover?",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        output_dir / "source_turnover_vs_authority_change.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def write_readme(
    output_dir: Path,
    input_dir: Path,
    classified: pd.DataFrame,
    turnover: pd.DataFrame,
) -> None:
    text = f"""Automatic source authority and turnover pilot
=============================================

Input: {input_dir}
Source occurrences: {len(classified)}
Unique canonical URLs: {classified['normalized_url'].nunique()}
Unique registrable domains: {classified['registrable_domain'].nunique()}
Turnover rows: {len(turnover)}

Authority taxonomy
------------------
The authority taxonomy is fully automatic and deterministic. It uses known
domain families, public/academic suffixes, domain-name tokens, and fallback
suffix rules. No manual source annotations are used.

classification_rule records the rule used for every source.
classification_confidence describes rule specificity, not validated accuracy.
Low-confidence and fallback assignments must be treated as exploratory.

Composition uses unique canonical URLs within each query. For zero-source
queries, category shares are missing rather than zero.

Expansion and substitution
--------------------------
All three paired comparisons are included:
- minority_majority
- minority_generic
- majority_generic

For each pair and granularity:
- shared = A intersection B
- a_only = A minus B
- b_only = B minus A
- replacement_mass = min(a_only, b_only)
- net_a_minus_b = a_only minus b_only
- E = (a_only minus b_only) / union
- T = (a_only plus b_only) / union = 1 minus Jaccard

Uncertainty and inference
-------------------------
Mean authority-share contrasts and normalized turnover metrics receive
percentile confidence intervals from 10,000 paired outcome bootstraps.

Category-specific binary GLMs preserve each social dimension separately and
adjust for outcome fixed effects. Standard errors are clustered by query.
These models estimate composition conditional on a source being present; they
do not model whether an AIO or source list exists.
Rare-category models with quasi-complete separation are flagged and receive no
p-values. They must not be interpreted as evidence of very large effects.

A complementary variational-Bayes binomial mixed model uses a random intercept
by outcome. Its intervals are approximate posterior credible intervals under
the mean-field approximation and are reported as a sensitivity analysis.

authority_total_variation and normalized Jensen-Shannon divergence measure
how much the complete authority-composition vector changes. Their association
with source turnover is summarized with Spearman correlations and outcome-
clustered bootstrap intervals.

Taxonomy validation
-------------------
artifacts/annotation_inputs/authority_taxonomy_validation_sample.csv is a
deterministic stratified sample with empty human-label columns. Strata are
automatic category by heuristic confidence. Because rare strata are
oversampled, validation metrics should use sampling_probability or inverse-
probability weights.

The control query ("people") is reused across social dimensions. Comparisons
against control are substantively useful but are not independent replications
across dimensions.

Main files
----------
- source_authority_observations.csv
- automatic_authority_domain_crosswalk.csv
- authority_composition_by_query.csv
- authority_paired_contrasts.csv
- authority_contrasts_aggregated.csv
- authority_contrast_bootstrap.csv
- authority_binary_models.csv
- authority_hierarchical_models.csv
- artifacts/annotation_inputs/authority_taxonomy_validation_sample.csv
- source_turnover_by_pair.csv
- source_turnover_aggregated.csv
- source_turnover_bootstrap.csv
- source_set_membership.csv
- authority_by_set_membership.csv
- authority_distance_by_pair.csv
- source_turnover_authority_link.csv
- source_turnover_authority_association.csv
- analysis_summary.json
- authority_composition_by_condition.png
- authority_share_difference_by_dimension.png
- normalized_source_expansion_turnover.png
- source_set_components.png
- source_turnover_vs_authority_change.png
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def run_analysis(input_dir: Path, output_dir: Path) -> None:
    sources = pd.read_csv(input_dir / "source_observations.csv")
    metrics = pd.read_csv(input_dir / "condition_metrics.csv")
    classified = add_authority_taxonomy(sources)
    domain_crosswalk = build_domain_crosswalk(classified)
    composition = build_authority_composition(metrics, classified)
    authority_contrasts = build_authority_contrasts(metrics, composition)
    authority_aggregated = aggregate_authority_contrasts(authority_contrasts)
    authority_bootstrap = bootstrap_authority_contrasts(authority_contrasts)
    authority_models = fit_authority_binary_models(classified)
    authority_hierarchical = fit_authority_hierarchical_models(classified)
    validation_sample = build_validation_sample(domain_crosswalk, classified)
    turnover, membership = build_turnover(metrics, classified)
    turnover_aggregated = aggregate_turnover(turnover)
    turnover_bootstrap = bootstrap_turnover_metrics(turnover)
    membership_authority = aggregate_membership_authority(membership)
    authority_distances = build_authority_distances(metrics, composition)
    linked, association = build_turnover_authority_link(
        turnover,
        authority_distances,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    classified.to_csv(
        output_dir / "source_authority_observations.csv", index=False
    )
    domain_crosswalk.to_csv(
        output_dir / "automatic_authority_domain_crosswalk.csv", index=False
    )
    composition.to_csv(
        output_dir / "authority_composition_by_query.csv", index=False
    )
    authority_contrasts.to_csv(
        output_dir / "authority_paired_contrasts.csv", index=False
    )
    authority_aggregated.to_csv(
        output_dir / "authority_contrasts_aggregated.csv", index=False
    )
    authority_bootstrap.to_csv(
        output_dir / "authority_contrast_bootstrap.csv", index=False
    )
    authority_models.to_csv(
        output_dir / "authority_binary_models.csv", index=False
    )
    authority_hierarchical.to_csv(
        output_dir / "authority_hierarchical_models.csv", index=False
    )
    validation_sample.to_csv(
        output_dir / "authority_taxonomy_validation_sample.csv", index=False
    )
    DEFAULT_VALIDATION_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_sample.to_csv(DEFAULT_VALIDATION_SAMPLE_PATH, index=False)
    turnover.to_csv(output_dir / "source_turnover_by_pair.csv", index=False)
    turnover_aggregated.to_csv(
        output_dir / "source_turnover_aggregated.csv", index=False
    )
    turnover_bootstrap.to_csv(
        output_dir / "source_turnover_bootstrap.csv", index=False
    )
    membership.to_csv(output_dir / "source_set_membership.csv", index=False)
    membership_authority.to_csv(
        output_dir / "authority_by_set_membership.csv", index=False
    )
    authority_distances.to_csv(
        output_dir / "authority_distance_by_pair.csv", index=False
    )
    linked.to_csv(
        output_dir / "source_turnover_authority_link.csv", index=False
    )
    association.to_csv(
        output_dir / "source_turnover_authority_association.csv", index=False
    )

    plot_authority_composition(composition, output_dir)
    plot_authority_heatmap(authority_contrasts, output_dir)
    plot_turnover(turnover, output_dir)
    plot_set_components(turnover, output_dir)
    plot_turnover_authority_link(linked, output_dir)

    summary = {
        "input_dir": str(input_dir),
        "source_occurrences": len(classified),
        "unique_canonical_urls": int(classified["normalized_url"].nunique()),
        "unique_registrable_domains": int(
            classified["registrable_domain"].nunique()
        ),
        "authority_category_counts": {
            str(key): int(value)
            for key, value in classified["authority_category"]
            .value_counts()
            .items()
        },
        "classification_confidence_counts": {
            str(key): int(value)
            for key, value in classified["classification_confidence"]
            .value_counts()
            .items()
        },
        "turnover_pairs": int(
            turnover[
                ["dimension", "domain", "outcome", "comparison"]
            ].drop_duplicates().shape[0]
        ),
        "turnover_rows": len(turnover),
        "authority_models_fitted": int(
            (authority_models["status"] == "fitted").sum()
        ),
        "authority_models_not_fitted": int(
            (authority_models["status"] != "fitted").sum()
        ),
        "authority_hierarchical_models_fitted": int(
            (authority_hierarchical["status"] == "fitted").sum()
        ),
        "authority_hierarchical_models_not_fitted": int(
            (authority_hierarchical["status"] != "fitted").sum()
        ),
        "validation_sample_domains": len(validation_sample),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "correlation_bootstrap_resamples": CORRELATION_BOOTSTRAP_RESAMPLES,
        "comparisons": [comparison for comparison, _, _ in COMPARISONS],
        "granularities": ["canonical_url", "registrable_domain"],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, input_dir, classified, turnover)

    print(f"Sources classified: {len(classified)}")
    print(f"Turnover rows: {len(turnover)}")
    print(f"Results written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the automatic source-authority and paired source-turnover pilot."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing source_observations.csv and condition_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    run_analysis(args.input_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
