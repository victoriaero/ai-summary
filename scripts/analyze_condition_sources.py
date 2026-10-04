from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import tldextract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION_DIR = (
    PROJECT_ROOT / "annotations" / "v1_dallas" / "google_aio_collection"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "condition_source_analysis_dallas"

SECTION_HEADERS = (
    "QUERY",
    "LINKS",
    "AI OVERVIEW TEXT",
    "COLLECTION INFO",
    "METADATA — DO NOT EDIT",
)

SOURCE_CATEGORIES = (
    "government",
    "education",
    "nonprofit",
    "social_media",
    "reference",
    "commercial",
    "network",
    "other",
)

SOCIAL_MEDIA_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "reddit.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
)

REFERENCE_DOMAINS = (
    "britannica.com",
    "dictionary.com",
    "investopedia.com",
    "wikipedia.org",
)

TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}

COMPARISONS = (
    ("minority_majority", "minority", "majority"),
    ("minority_generic", "minority", "control"),
    ("majority_generic", "majority", "control"),
)

SOURCE_COLUMNS = (
    "dimension",
    "condition",
    "group",
    "domain",
    "outcome",
    "query_id",
    "source_position",
    "raw_url",
    "normalized_url",
    "source_hostname",
    "registrable_domain",
    "source_domain",
    "source_category",
)

DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def get_section(text: str, header: str) -> str:
    text = normalize_newlines(text)
    match = re.search(rf"(?m)^\s*{re.escape(header)}\s*$", text)
    if not match:
        return ""

    remainder = text[match.end() :]
    positions = []
    for next_header in SECTION_HEADERS:
        if next_header == header:
            continue
        next_match = re.search(
            rf"(?m)^\s*{re.escape(next_header)}\s*$",
            remainder,
        )
        if next_match:
            positions.append(next_match.start())
    if positions:
        remainder = remainder[: min(positions)]

    lines = remainder.splitlines()
    while lines and (not lines[0].strip() or re.fullmatch(r"=+", lines[0].strip())):
        lines.pop(0)
    while lines and (
        not lines[-1].strip() or re.fullmatch(r"=+", lines[-1].strip())
    ):
        lines.pop()
    return "\n".join(lines).strip()


def parse_key_values(section: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in section.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def parse_links(section: str) -> list[str]:
    urls: list[str] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        numbered = re.match(r"^\s*\d+\s*-\s*(.*)$", line)
        candidate = numbered.group(1).strip() if numbered else line
        if not candidate:
            continue

        markdown = re.search(r"\((https?://[^)]+)\)", candidate)
        if markdown:
            urls.append(markdown.group(1).strip())
            continue

        bracketed = re.match(r"^\s*\[\d+\]\s+(https?://\S+)\s*$", candidate)
        if bracketed:
            urls.append(bracketed.group(1).rstrip(".,;"))
            continue

        normal = re.search(r"https?://\S+", candidate)
        if normal:
            urls.append(normal.group(0).rstrip(".,;)"))
    return urls


def parse_yes_no(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    return None


def normalize_hostname(hostname: str | None) -> str:
    host = (hostname or "").lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def registrable_domain(hostname: str) -> str:
    extracted = DOMAIN_EXTRACTOR(hostname)
    return extracted.top_domain_under_public_suffix or hostname


def normalize_url(raw_url: str) -> str:
    value = raw_url.strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value

    host = normalize_hostname(parsed.hostname)
    if not host:
        return value

    port = parsed.port
    netloc = host
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"

    filtered_query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_PARAMETERS
    ]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower() or "https",
            netloc,
            path,
            urlencode(sorted(filtered_query)),
            "",
        )
    )


def hostname_matches(hostname: str, suffixes: Iterable[str]) -> bool:
    return any(hostname == suffix or hostname.endswith(f".{suffix}") for suffix in suffixes)


def classify_source(hostname: str) -> str:
    if hostname_matches(hostname, SOCIAL_MEDIA_DOMAINS):
        return "social_media"
    if hostname_matches(hostname, REFERENCE_DOMAINS):
        return "reference"
    if hostname.endswith(".gov") or ".gov." in hostname or hostname.endswith(".mil"):
        return "government"
    if hostname.endswith(".edu") or ".edu." in hostname or ".ac." in hostname:
        return "education"
    if hostname.endswith(".org"):
        return "nonprofit"
    if hostname.endswith((".com", ".co", ".io", ".ai", ".biz")):
        return "commercial"
    if hostname.endswith(".net"):
        return "network"
    return "other"


def parse_collection_file(filepath: Path, collection_dir: Path) -> tuple[dict, list[dict]]:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    aio_text = get_section(text, "AI OVERVIEW TEXT")
    collection = parse_key_values(get_section(text, "COLLECTION INFO"))
    metadata = parse_key_values(get_section(text, "METADATA — DO NOT EDIT"))
    raw_urls = parse_links(get_section(text, "LINKS"))

    sources: list[dict] = []
    for position, raw_url in enumerate(raw_urls, start=1):
        normalized_url = normalize_url(raw_url)
        hostname = normalize_hostname(urlsplit(normalized_url).hostname)
        domain = registrable_domain(hostname)
        sources.append(
            {
                "query_id": metadata.get("Query ID", ""),
                "source_position": position,
                "raw_url": raw_url,
                "normalized_url": normalized_url,
                "source_hostname": hostname,
                "registrable_domain": domain,
                "source_domain": domain,
                "source_category": classify_source(hostname),
            }
        )

    unique_urls = {source["normalized_url"] for source in sources}
    unique_hostnames = {
        source["source_hostname"] for source in sources if source["source_hostname"]
    }
    unique_domains = {
        source["registrable_domain"]
        for source in sources
        if source["registrable_domain"]
    }
    category_counts = Counter(source["source_category"] for source in sources)
    categories = set(category_counts)
    recorded = parse_yes_no(collection.get("AIO present", ""))

    row = {
        "file": str(filepath.relative_to(collection_dir)),
        "query_id": metadata.get("Query ID", ""),
        "dimension": metadata.get("Dimension", ""),
        "condition": metadata.get("Condition", ""),
        "group": metadata.get("Group", ""),
        "domain": metadata.get("Domain", ""),
        "outcome": metadata.get("Outcome", ""),
        "aio_present": bool(aio_text),
        "aio_present_recorded": recorded,
        "n_sources": len(sources),
        "n_unique_sources": len(unique_urls),
        "n_unique_hostnames": len(unique_hostnames),
        "n_unique_domains": len(unique_domains),
        "aio_chars": len(aio_text),
        "aio_words": len(aio_text.split()),
        "_url_set": unique_urls,
        "_domain_set": unique_domains,
        "_category_set": categories,
        "_category_counts": category_counts,
    }
    return row, sources


def load_collection(collection_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    source_rows: list[dict] = []
    for filepath in sorted(collection_dir.glob("*/*.txt")):
        row, sources = parse_collection_file(filepath, collection_dir)
        metric_rows.append(row)
        for source in sources:
            source_rows.append(
                {
                    "dimension": row["dimension"],
                    "condition": row["condition"],
                    "group": row["group"],
                    "domain": row["domain"],
                    "outcome": row["outcome"],
                    **source,
                }
            )

    if not metric_rows:
        raise ValueError(f"No collection .txt files found in {collection_dir}")
    metrics = pd.DataFrame(metric_rows)
    sources = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)

    duplicated_ids = metrics.loc[
        metrics["query_id"].ne("") & metrics["query_id"].duplicated(keep=False),
        "query_id",
    ]
    if not duplicated_ids.empty:
        ids = ", ".join(sorted(duplicated_ids.unique()))
        raise ValueError(f"Duplicate query IDs in collection: {ids}")
    return metrics, sources


def category_composition(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        counts = row["_category_counts"]
        for category in SOURCE_CATEGORIES:
            count = int(counts.get(category, 0))
            rows.append(
                {
                    "query_id": row["query_id"],
                    "dimension": row["dimension"],
                    "condition": row["condition"],
                    "group": row["group"],
                    "domain": row["domain"],
                    "outcome": row["outcome"],
                    "source_category": category,
                    "category_source_count": count,
                    "category_source_share": (
                        count / row["n_sources"] if row["n_sources"] else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def overlap_measures(left: set[str], right: set[str]) -> dict[str, int | float]:
    intersection = len(left & right)
    union = len(left | right)
    smaller = min(len(left), len(right))
    return {
        "set_size_a": len(left),
        "set_size_b": len(right),
        "intersection_size": intersection,
        "union_size": union,
        "smaller_set_size": smaller,
        "jaccard": intersection / union if union else np.nan,
        "overlap_coefficient": intersection / smaller if smaller else np.nan,
    }


def build_overlaps(metrics: pd.DataFrame) -> pd.DataFrame:
    social_dimensions = sorted(
        metrics.loc[metrics["condition"] != "control", "dimension"].unique()
    )
    controls = {
        (row["domain"], row["outcome"]): row
        for _, row in metrics.loc[metrics["condition"] == "control"].iterrows()
    }
    rows = []

    for dimension in social_dimensions:
        subset = metrics.loc[metrics["dimension"] == dimension]
        units = sorted(set(zip(subset["domain"], subset["outcome"])))
        for domain, outcome in units:
            unit = subset.loc[
                (subset["domain"] == domain) & (subset["outcome"] == outcome)
            ]
            condition_rows = {
                row["condition"]: row for _, row in unit.iterrows()
            }
            control = controls.get((domain, outcome))
            if control is not None:
                condition_rows["control"] = control

            for comparison, left_condition, right_condition in COMPARISONS:
                left = condition_rows.get(left_condition)
                right = condition_rows.get(right_condition)
                complete = left is not None and right is not None
                granularities = (
                    ("canonical_url", "_url_set"),
                    ("registrable_domain", "_domain_set"),
                    ("source_category", "_category_set"),
                )
                for granularity, set_column in granularities:
                    measures = (
                        overlap_measures(left[set_column], right[set_column])
                        if complete
                        else {
                            "set_size_a": 0,
                            "set_size_b": 0,
                            "intersection_size": 0,
                            "union_size": 0,
                            "smaller_set_size": 0,
                            "jaccard": np.nan,
                            "overlap_coefficient": np.nan,
                        }
                    )
                    rows.append(
                        {
                            "dimension": dimension,
                            "domain": domain,
                            "outcome": outcome,
                            "comparison": comparison,
                            "condition_a": left_condition,
                            "condition_b": right_condition,
                            "group_a": left["group"] if left is not None else "",
                            "group_b": right["group"] if right is not None else "",
                            "pair_complete": complete,
                            "n_source_occurrences_a": (
                                left["n_sources"] if left is not None else np.nan
                            ),
                            "n_source_occurrences_b": (
                                right["n_sources"] if right is not None else np.nan
                            ),
                            "granularity": granularity,
                            **measures,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "social_dimension": ["dimension", "condition", "group"],
        "domain": ["domain", "condition", "group"],
        "outcome": ["domain", "outcome", "condition", "group"],
    }
    outputs = []
    for level, columns in levels.items():
        grouped = (
            metrics.groupby(columns, dropna=False)
            .agg(
                n_queries=("query_id", "size"),
                aio_present_n=("aio_present", "sum"),
                aio_present_rate=("aio_present", "mean"),
                mean_sources=("n_sources", "mean"),
                median_sources=("n_sources", "median"),
                mean_unique_sources=("n_unique_sources", "mean"),
                mean_unique_hostnames=("n_unique_hostnames", "mean"),
                mean_unique_domains=("n_unique_domains", "mean"),
                mean_aio_chars=("aio_chars", "mean"),
                median_aio_chars=("aio_chars", "median"),
                mean_aio_words=("aio_words", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "aggregation_level", level)
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True, sort=False)


def aggregate_categories(composition: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "social_dimension": [
            "dimension",
            "condition",
            "group",
            "source_category",
        ],
        "domain": ["domain", "condition", "group", "source_category"],
        "outcome": [
            "domain",
            "outcome",
            "condition",
            "group",
            "source_category",
        ],
    }
    outputs = []
    for level, columns in levels.items():
        grouped = (
            composition.groupby(columns, dropna=False)
            .agg(
                n_queries=("query_id", "size"),
                total_category_sources=("category_source_count", "sum"),
                mean_category_sources=("category_source_count", "mean"),
                mean_category_share=("category_source_share", "mean"),
            )
            .reset_index()
        )
        totals = grouped.groupby(
            [column for column in columns if column != "source_category"],
            dropna=False,
        )["total_category_sources"].transform("sum")
        grouped["pooled_category_share"] = (
            grouped["total_category_sources"] / totals.replace(0, np.nan)
        )
        grouped.insert(0, "aggregation_level", level)
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True, sort=False)


def aggregate_overlaps(overlaps: pd.DataFrame) -> pd.DataFrame:
    levels = {
        "social_dimension": ["dimension", "comparison", "granularity"],
        "domain": ["domain", "comparison", "granularity"],
        "outcome": ["domain", "outcome", "comparison", "granularity"],
    }
    outputs = []
    for level, columns in levels.items():
        grouped = (
            overlaps.groupby(columns, dropna=False)
            .agg(
                n_pairs=("pair_complete", "size"),
                n_complete_pairs=("pair_complete", "sum"),
                mean_jaccard=("jaccard", "mean"),
                median_jaccard=("jaccard", "median"),
                sd_jaccard=("jaccard", "std"),
                mean_overlap_coefficient=("overlap_coefficient", "mean"),
                median_overlap_coefficient=("overlap_coefficient", "median"),
                sd_overlap_coefficient=("overlap_coefficient", "std"),
                intersection_total=("intersection_size", "sum"),
                union_total=("union_size", "sum"),
                smaller_set_total=("smaller_set_size", "sum"),
            )
            .reset_index()
        )
        grouped["pooled_jaccard"] = (
            grouped["intersection_total"]
            / grouped["union_total"].replace(0, np.nan)
        )
        grouped["pooled_overlap_coefficient"] = (
            grouped["intersection_total"]
            / grouped["smaller_set_total"].replace(0, np.nan)
        )
        grouped.insert(0, "aggregation_level", level)
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True, sort=False)


def write_readme(
    output_dir: Path,
    collection_dir: Path,
    metrics: pd.DataFrame,
    sources: pd.DataFrame,
    overlaps: pd.DataFrame,
) -> None:
    text = f"""Condition and source exploratory analysis
=========================================

Collection: {collection_dir}
Queries: {len(metrics)}
Source occurrences: {len(sources)}
Condition pairs: {len(overlaps) // 3}
Overlap rows (three granularities): {len(overlaps)}
Complete overlap rows: {int(overlaps['pair_complete'].sum())} / {len(overlaps)}

Main outputs:
- condition_metrics.csv
- source_observations.csv
- source_category_composition.csv
- source_overlap_per_query.csv
- condition_metrics_aggregated.csv
- source_category_composition_aggregated.csv
- source_overlap_aggregated.csv
- analysis_summary.json

Source overlap is reported at canonical-URL, registrable-domain, and category levels:
J(A, B) = |A intersection B| / |A union B|.
O(A, B) = |A intersection B| / min(|A|, |B|).

Canonical URLs remove fragments, common tracking parameters, default ports,
and superficial www/trailing-slash differences. Registrable domains use the
bundled Public Suffix List snapshot from tldextract. Category overlap compares
the set of source categories present, not their frequencies.

The overlap coefficient is undefined when either set is empty and is exported
as a missing value in that case.

Source categories are deterministic heuristics based on hostname suffixes.
They are intended for exploration and should be reviewed before confirmatory analysis.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def run_analysis(collection_dir: Path, output_dir: Path) -> None:
    metrics, sources = load_collection(collection_dir)
    composition = category_composition(metrics)
    overlaps = build_overlaps(metrics)
    metrics_aggregated = aggregate_metrics(metrics)
    categories_aggregated = aggregate_categories(composition)
    overlaps_aggregated = aggregate_overlaps(overlaps)

    output_dir.mkdir(parents=True, exist_ok=True)
    public_metrics = metrics.drop(
        columns=[
            "_url_set",
            "_domain_set",
            "_category_set",
            "_category_counts",
        ]
    )
    public_metrics.to_csv(output_dir / "condition_metrics.csv", index=False)
    sources.to_csv(output_dir / "source_observations.csv", index=False)
    composition.to_csv(output_dir / "source_category_composition.csv", index=False)
    overlaps.to_csv(output_dir / "source_overlap_per_query.csv", index=False)
    metrics_aggregated.to_csv(
        output_dir / "condition_metrics_aggregated.csv", index=False
    )
    categories_aggregated.to_csv(
        output_dir / "source_category_composition_aggregated.csv", index=False
    )
    overlaps_aggregated.to_csv(
        output_dir / "source_overlap_aggregated.csv", index=False
    )

    summary = {
        "collection_dir": str(collection_dir),
        "n_queries": len(metrics),
        "aio_present_n": int(metrics["aio_present"].sum()),
        "source_occurrences": len(sources),
        "unique_normalized_sources": int(sources["normalized_url"].nunique()),
        "unique_source_hostnames": int(sources["source_hostname"].nunique()),
        "unique_registrable_domains": int(
            sources["registrable_domain"].nunique()
        ),
        "condition_pairs": len(overlaps) // 3,
        "overlap_rows": len(overlaps),
        "complete_overlap_rows": int(overlaps["pair_complete"].sum()),
        "source_category_counts": {
            str(key): int(value)
            for key, value in sources["source_category"].value_counts().items()
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, collection_dir, metrics, sources, overlaps)

    print(f"Queries analyzed: {len(metrics)}")
    print(f"Source occurrences: {len(sources)}")
    print(f"Condition pairs: {len(overlaps) // 3}")
    print(f"Overlap rows: {len(overlaps)}")
    print(f"Results written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze AIO presence, source composition, and paired overlap."
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        default=DEFAULT_COLLECTION_DIR,
        help="Directory containing one subdirectory per condition/group.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which analysis files will be written.",
    )
    args = parser.parse_args()
    run_analysis(args.collection_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
