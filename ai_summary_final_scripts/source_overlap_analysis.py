from pathlib import Path
import os
from urllib.parse import (
    urlparse,
    urlunparse,
    parse_qsl,
    urlencode,
)
import itertools
import json
import math
import random
import re
import warnings
import zlib

import numpy as np
import pandas as pd

from scipy.stats import (
    friedmanchisquare,
    rankdata,
    wilcoxon,
)

from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    os.environ.get(
        "AIO_BASE_DIR",
        "/scratch/victoria.estanislau/ai-summary",
    )
)

COLLECTION_VERSION = os.environ.get(
    "AIO_COLLECTION_VERSION",
    "v1_dallas",
)

LOCATION_SLUG = os.environ.get(
    "AIO_LOCATION_SLUG",
    COLLECTION_VERSION.split("_", 1)[-1],
)

COLLECTION_DIR = (
    BASE_DIR
    / "annotations"
    / COLLECTION_VERSION
    / "google_aio_collection"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / f"source_overlap_analysis_{LOCATION_SLUG}"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ALPHA = 0.05

RANDOM_SEED = 42

N_PERMUTATIONS = 200_000

N_BOOTSTRAP = 20_000

EXACT_SIGNFLIP_MAX_N = 18


EXPECTED_GROUPS = {
    "black_people": {
        "group": "Black people",
        "dimension": "Race",
        "condition": "minority",
    },

    "white_people": {
        "group": "White people",
        "dimension": "Race",
        "condition": "majority",
    },

    "latino_people": {
        "group": "Latino people",
        "dimension": "Ethnicity",
        "condition": "minority",
    },

    "non_latino_people": {
        "group": "non-Latino people",
        "dimension": "Ethnicity",
        "condition": "majority",
    },

    "women": {
        "group": "Women",
        "dimension": "Gender",
        "condition": "minority",
    },

    "men": {
        "group": "Men",
        "dimension": "Gender",
        "condition": "majority",
    },

    "people_with_disabilities": {
        "group": "People with disabilities",
        "dimension": "Disability",
        "condition": "minority",
    },

    "people_without_disabilities": {
        "group": "People without disabilities",
        "dimension": "Disability",
        "condition": "majority",
    },

    "homosexual_people": {
        "group": "homosexual people",
        "dimension": "Sexual Orientation",
        "condition": "minority",
    },

    "heterosexual_people": {
        "group": "heterosexual people",
        "dimension": "Sexual Orientation",
        "condition": "majority",
    },

    "transgender_people": {
        "group": "transgender people",
        "dimension": "Gender Identity",
        "condition": "minority",
    },

    "cisgender_people": {
        "group": "cisgender people",
        "dimension": "Gender Identity",
        "condition": "majority",
    },

    "people": {
        "group": "people",
        "dimension": "Control",
        "condition": "control",
    },
}


EXPECTED_FILES_PER_GROUP = 21
EXPECTED_TOTAL = 273


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# HELPERS
# ============================================================

def stable_seed(label):

    return (
        RANDOM_SEED
        + zlib.crc32(
            str(label).encode(
                "utf-8"
            )
        )
    ) % (
        2**32 - 1
    )


def json_safe(value):

    if value is None:
        return None

    if isinstance(
        value,
        np.integer
    ):
        return int(value)

    if isinstance(
        value,
        np.floating
    ):

        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_
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
            str(k):
                json_safe(v)

            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        )
    ):

        return [
            json_safe(v)
            for v in value
        ]

    return value


def dataframe_records(df):

    if (
        df is None
        or len(df) == 0
    ):
        return []

    return json.loads(
        df.to_json(
            orient="records"
        )
    )


# ============================================================
# TEXT FILE PARSING
# ============================================================

def normalize_newlines(text):

    return (
        text
        .replace(
            "\r\n",
            "\n"
        )
        .replace(
            "\r",
            "\n"
        )
    )


def get_section(
    text,
    header,
    next_headers,
):

    text = normalize_newlines(
        text
    )

    match = re.search(
        rf"(?m)^\s*"
        rf"{re.escape(header)}"
        rf"\s*$",
        text
    )

    if not match:
        return None

    remainder = text[
        match.end():
    ]

    positions = []

    for next_header in next_headers:

        m = re.search(
            rf"(?m)^\s*"
            rf"{re.escape(next_header)}"
            rf"\s*$",
            remainder
        )

        if m:

            positions.append(
                m.start()
            )

    if positions:

        remainder = remainder[
            :min(
                positions
            )
        ]

    lines = (
        remainder
        .splitlines()
    )

    while (
        lines
        and (
            not lines[0].strip()
            or re.fullmatch(
                r"=+",
                lines[0].strip()
            )
        )
    ):
        lines.pop(0)

    while (
        lines
        and (
            not lines[-1].strip()
            or re.fullmatch(
                r"=+",
                lines[-1].strip()
            )
        )
    ):
        lines.pop()

    return "\n".join(
        lines
    ).strip()


def parse_key_values(section):

    result = {}

    if not section:
        return result

    for line in section.splitlines():

        if ":" not in line:
            continue

        key, value = (
            line.split(
                ":",
                1
            )
        )

        result[
            key.strip()
        ] = value.strip()

    return result


# ============================================================
# LINK PARSER
# ============================================================

def parse_links(section):
    """
    Accepts formats such as:

    1 - https://example.com

    [1] [example.com](https://example.com)

    [1] https://example.com

    https://example.com
    """

    urls = []

    malformed = []

    if not section:
        return (
            urls,
            malformed
        )

    for raw_line in (
        section.splitlines()
    ):

        line = raw_line.strip()

        if not line:
            continue


        # ----------------------------------------------------
        # 1 - URL
        # ----------------------------------------------------

        match = re.match(
            r"^\s*\d+\s*-\s*(.*)$",
            line
        )

        if match:

            rest = (
                match
                .group(1)
                .strip()
            )

            if not rest:
                continue

            markdown = re.search(
                r"\((https?://[^)]+)\)",
                rest
            )

            if markdown:

                urls.append(
                    markdown
                    .group(1)
                    .strip()
                )

                continue

            normal = re.search(
                r"https?://\S+",
                rest
            )

            if normal:

                urls.append(
                    normal
                    .group(0)
                    .rstrip(
                        ".,;"
                    )
                )

                continue

            malformed.append(
                line
            )

            continue


        # ----------------------------------------------------
        # [1] [text](URL)
        # ----------------------------------------------------

        match = re.match(
            r"^\s*\[\d+\]\s*"
            r"\[[^\]]+\]"
            r"\((https?://[^)]+)\)"
            r"\s*$",
            line
        )

        if match:

            urls.append(
                match
                .group(1)
                .strip()
            )

            continue


        # ----------------------------------------------------
        # [1] URL
        # ----------------------------------------------------

        match = re.match(
            r"^\s*\[\d+\]\s*"
            r"(https?://\S+)"
            r"\s*$",
            line
        )

        if match:

            urls.append(
                match
                .group(1)
                .rstrip(
                    ".,;"
                )
            )

            continue


        # ----------------------------------------------------
        # Plain URL
        # ----------------------------------------------------

        match = re.search(
            r"https?://\S+",
            line
        )

        if match:

            urls.append(
                match
                .group(0)
                .rstrip(
                    ".,;"
                )
            )

            continue


        malformed.append(
            line
        )


    return (
        urls,
        malformed
    )


# ============================================================
# URL NORMALIZATION
# ============================================================

TRACKING_PARAMETERS = {
    "gclid",
    "fbclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "_ga",
    "_gl",
}


def is_tracking_parameter(key):

    key = (
        key
        .strip()
        .lower()
    )

    return (
        key.startswith(
            "utm_"
        )
        or key in TRACKING_PARAMETERS
    )


def normalize_url(url):
    """
    Conservative URL normalization.

    Removes:
    - fragments
    - obvious tracking parameters
    - default ports
    - duplicate trailing slash

    Preserves substantive query parameters.
    """

    url = (
        url
        .strip()
        .rstrip(
            ".,;"
        )
    )

    try:

        parsed = urlparse(
            url
        )

    except Exception:
        return url


    scheme = (
        parsed.scheme.lower()
        if parsed.scheme
        else "https"
    )


    hostname = (
        parsed.hostname.lower()
        if parsed.hostname
        else ""
    )


    if hostname.startswith(
        "www."
    ):
        hostname = hostname[4:]


    port = parsed.port

    if (
        port
        and not (
            (
                scheme == "https"
                and port == 443
            )
            or (
                scheme == "http"
                and port == 80
            )
        )
    ):

        netloc = (
            f"{hostname}:{port}"
        )

    else:

        netloc = hostname


    path = (
        parsed.path
        or "/"
    )


    # Normalize repeated slash
    path = re.sub(
        r"/+",
        "/",
        path
    )


    # Treat /x and /x/ as equivalent
    if (
        len(path) > 1
        and path.endswith("/")
    ):
        path = path[:-1]


    query_items = []

    for key, value in parse_qsl(
        parsed.query,
        keep_blank_values=True
    ):

        if is_tracking_parameter(
            key
        ):
            continue

        query_items.append(
            (
                key,
                value
            )
        )


    query_items = sorted(
        query_items
    )


    query = urlencode(
        query_items,
        doseq=True
    )


    normalized = urlunparse(
        (
            scheme,
            netloc,
            path,
            "",
            query,
            "",
        )
    )


    return normalized


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

# Try to use registrable domains, e.g.
#   news.example.com -> example.com
#
# If tldextract is unavailable, fall back to hostname.

try:

    import tldextract

    _extractor = (
        tldextract.TLDExtract(
            suffix_list_urls=None
        )
    )

    HAVE_TLDEXTRACT = True

except ImportError:

    HAVE_TLDEXTRACT = False


def get_hostname(url):

    try:

        hostname = (
            urlparse(url)
            .hostname
        )

    except Exception:
        return None

    if not hostname:
        return None

    hostname = (
        hostname
        .lower()
        .strip(".")
    )

    if hostname.startswith(
        "www."
    ):
        hostname = hostname[4:]

    return hostname


def get_registrable_domain(url):

    hostname = get_hostname(
        url
    )

    if not hostname:
        return None


    if HAVE_TLDEXTRACT:

        result = _extractor(
            hostname
        )

        if (
            result.domain
            and result.suffix
        ):

            return (
                f"{result.domain}."
                f"{result.suffix}"
            )


    # Fallback:
    # hostname rather than potentially wrong
    # last-two-label approximation.
    return hostname


# ============================================================
# COLLECTION PARSER
# ============================================================

def parse_collection_file(
    filepath
):

    raw = filepath.read_text(
        encoding="utf-8",
        errors="replace"
    )


    query = get_section(
        raw,
        "QUERY",
        [
            "LINKS",
            "AI OVERVIEW TEXT",
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ]
    ) or ""


    links_section = get_section(
        raw,
        "LINKS",
        [
            "AI OVERVIEW TEXT",
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ]
    )


    metadata_section = get_section(
        raw,
        "METADATA — DO NOT EDIT",
        []
    )


    metadata = parse_key_values(
        metadata_section
    )


    urls_raw, malformed = (
        parse_links(
            links_section
        )
    )


    urls_normalized = [
        normalize_url(url)

        for url
        in urls_raw
    ]


    url_set = set(
        urls_normalized
    )


    domains = {
        d

        for d in (
            get_registrable_domain(
                url
            )

            for url
            in urls_normalized
        )

        if d
    }


    hosts = {
        h

        for h in (
            get_hostname(
                url
            )

            for url
            in urls_normalized
        )

        if h
    }


    return {

        "query":
            query.strip(),

        "metadata":
            metadata,

        "urls_raw":
            urls_raw,

        "urls_normalized":
            urls_normalized,

        "url_set":
            url_set,

        "domain_set":
            domains,

        "host_set":
            hosts,

        "malformed_links":
            malformed,
    }


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 80)
print("LOADING SOURCE SETS")
print("=" * 80)


rows = []


for folder_name, expected in (
    EXPECTED_GROUPS.items()
):

    folder = (
        COLLECTION_DIR
        / folder_name
    )

    if not folder.exists():

        raise RuntimeError(
            f"Missing folder: "
            f"{folder}"
        )


    files = sorted(
        folder.glob(
            "*.txt"
        )
    )


    if (
        len(files)
        != EXPECTED_FILES_PER_GROUP
    ):

        raise RuntimeError(
            f"{folder_name}: "
            f"expected "
            f"{EXPECTED_FILES_PER_GROUP} "
            f"files, found "
            f"{len(files)}"
        )


    for filepath in files:

        parsed = (
            parse_collection_file(
                filepath
            )
        )

        metadata = (
            parsed[
                "metadata"
            ]
        )


        group = metadata.get(
            "Group",
            expected[
                "group"
            ]
        )


        dimension = metadata.get(
            "Dimension",
            expected[
                "dimension"
            ]
        )


        condition = metadata.get(
            "Condition",
            expected[
                "condition"
            ]
        )


        domain = metadata.get(
            "Domain",
            ""
        )


        outcome = metadata.get(
            "Outcome",
            ""
        )


        outcome_id = (
            f"{domain} :: "
            f"{outcome}"
        )


        if parsed[
            "malformed_links"
        ]:

            print(
                "\nWARNING — "
                "unrecognized LINKS lines:"
            )

            print(
                filepath
            )

            for line in parsed[
                "malformed_links"
            ]:

                print(
                    f"  {line}"
                )


        rows.append({

            "file":
                str(
                    filepath.relative_to(
                        COLLECTION_DIR
                    )
                ),

            "folder":
                folder_name,

            "group":
                group,

            "dimension":
                dimension,

            "condition":
                condition,

            "domain":
                domain,

            "outcome":
                outcome,

            "outcome_id":
                outcome_id,

            "query":
                parsed[
                    "query"
                ],

            "n_links_raw":
                len(
                    parsed[
                        "urls_raw"
                    ]
                ),

            "n_urls_unique":
                len(
                    parsed[
                        "url_set"
                    ]
                ),

            "n_domains_unique":
                len(
                    parsed[
                        "domain_set"
                    ]
                ),

            "n_hosts_unique":
                len(
                    parsed[
                        "host_set"
                    ]
                ),

            "urls_raw":
                parsed[
                    "urls_raw"
                ],

            "urls_normalized":
                sorted(
                    parsed[
                        "url_set"
                    ]
                ),

            "domains":
                sorted(
                    parsed[
                        "domain_set"
                    ]
                ),

            "hosts":
                sorted(
                    parsed[
                        "host_set"
                    ]
                ),
        })


docs = pd.DataFrame(
    rows
)


if len(docs) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_TOTAL} files, "
        f"found {len(docs)}"
    )


# ============================================================
# DESIGN VALIDATION
# ============================================================

print(
    f"\nDocuments: "
    f"{len(docs)}"
)

print(
    f"Groups: "
    f"{docs['group'].nunique()}"
)

print(
    f"Outcomes: "
    f"{docs['outcome_id'].nunique()}"
)


outcome_counts = (
    docs
    .groupby(
        "outcome_id"
    )[
        "group"
    ]
    .nunique()
)


if not (
    outcome_counts
    == 13
).all():

    raise RuntimeError(
        "Not every outcome has "
        "all 13 conditions."
    )


group_counts = (
    docs
    .groupby(
        "group"
    )[
        "outcome_id"
    ]
    .nunique()
)


if not (
    group_counts
    == 21
).all():

    raise RuntimeError(
        "Not every group has "
        "all 21 outcomes."
    )


print(
    "✓ Experimental design validated."
)


# ============================================================
# JACCARD FUNCTIONS
# ============================================================

def set_overlap_metrics(
    explicit_set,
    generic_set,
):
    """
    If both sets are empty:
        similarity and distance = NaN

    If one is empty:
        similarity = 0
        distance = 1
    """

    explicit_set = set(
        explicit_set
    )

    generic_set = set(
        generic_set
    )


    intersection = (
        explicit_set
        &
        generic_set
    )

    union = (
        explicit_set
        |
        generic_set
    )


    if len(union) == 0:

        jaccard_similarity = (
            np.nan
        )

        jaccard_distance = (
            np.nan
        )

    else:

        jaccard_similarity = (
            len(intersection)
            /
            len(union)
        )

        jaccard_distance = (
            1.0
            -
            jaccard_similarity
        )


    explicit_coverage = (
        len(intersection)
        /
        len(explicit_set)

        if len(explicit_set)
        else np.nan
    )


    generic_coverage = (
        len(intersection)
        /
        len(generic_set)

        if len(generic_set)
        else np.nan
    )


    return {

        "intersection_n":
            len(
                intersection
            ),

        "union_n":
            len(
                union
            ),

        "explicit_n":
            len(
                explicit_set
            ),

        "generic_n":
            len(
                generic_set
            ),

        "jaccard_similarity":
            jaccard_similarity,

        "jaccard_distance":
            jaccard_distance,

        "explicit_coverage":
            explicit_coverage,

        "generic_coverage":
            generic_coverage,

        "shared":
            sorted(
                intersection
            ),

        "explicit_only":
            sorted(
                explicit_set
                -
                generic_set
            ),

        "generic_only":
            sorted(
                generic_set
                -
                explicit_set
            ),
    }


# ============================================================
# GENERIC BASELINE LOOKUP
# ============================================================

generic_docs = docs[
    docs[
        "condition"
    ]
    == "control"
].copy()


if len(generic_docs) != 21:

    raise RuntimeError(
        "Expected exactly 21 "
        "generic `people` queries."
    )


generic_by_outcome = {

    row[
        "outcome_id"
    ]:
        row

    for _, row
    in generic_docs.iterrows()
}


# ============================================================
# BUILD EXPLICIT-vs-GENERIC COMPARISONS
# ============================================================

comparison_rows = []


for _, row in docs.iterrows():

    if (
        row[
            "condition"
        ]
        == "control"
    ):
        continue


    generic = (
        generic_by_outcome[
            row[
                "outcome_id"
            ]
        ]
    )


    # --------------------------------------------------------
    # URL-level
    # --------------------------------------------------------

    url_metrics = (
        set_overlap_metrics(
            row[
                "urls_normalized"
            ],

            generic[
                "urls_normalized"
            ],
        )
    )


    # --------------------------------------------------------
    # Registrable-domain level
    # --------------------------------------------------------

    domain_metrics = (
        set_overlap_metrics(
            row[
                "domains"
            ],

            generic[
                "domains"
            ],
        )
    )


    # --------------------------------------------------------
    # Host-level, secondary diagnostic
    # --------------------------------------------------------

    host_metrics = (
        set_overlap_metrics(
            row[
                "hosts"
            ],

            generic[
                "hosts"
            ],
        )
    )


    comparison_rows.append({

        "dimension":
            row[
                "dimension"
            ],

        "condition":
            row[
                "condition"
            ],

        "group":
            row[
                "group"
            ],

        "domain":
            row[
                "domain"
            ],

        "outcome":
            row[
                "outcome"
            ],

        "outcome_id":
            row[
                "outcome_id"
            ],

        "file":
            row[
                "file"
            ],

        "generic_file":
            generic[
                "file"
            ],


        # ----------------------------------------------------
        # Source counts
        # ----------------------------------------------------

        "n_links_raw":
            row[
                "n_links_raw"
            ],

        "generic_n_links_raw":
            generic[
                "n_links_raw"
            ],

        "n_unique_urls":
            row[
                "n_urls_unique"
            ],

        "generic_n_unique_urls":
            generic[
                "n_urls_unique"
            ],

        "n_unique_domains":
            row[
                "n_domains_unique"
            ],

        "generic_n_unique_domains":
            generic[
                "n_domains_unique"
            ],


        # ----------------------------------------------------
        # URL overlap
        # ----------------------------------------------------

        "url_shared_n":
            url_metrics[
                "intersection_n"
            ],

        "url_union_n":
            url_metrics[
                "union_n"
            ],

        "url_jaccard_similarity":
            url_metrics[
                "jaccard_similarity"
            ],

        "url_jaccard_distance":
            url_metrics[
                "jaccard_distance"
            ],

        "url_explicit_coverage":
            url_metrics[
                "explicit_coverage"
            ],

        "url_generic_coverage":
            url_metrics[
                "generic_coverage"
            ],


        # ----------------------------------------------------
        # Domain overlap
        # ----------------------------------------------------

        "domain_shared_n":
            domain_metrics[
                "intersection_n"
            ],

        "domain_union_n":
            domain_metrics[
                "union_n"
            ],

        "domain_jaccard_similarity":
            domain_metrics[
                "jaccard_similarity"
            ],

        "domain_jaccard_distance":
            domain_metrics[
                "jaccard_distance"
            ],

        "domain_explicit_coverage":
            domain_metrics[
                "explicit_coverage"
            ],

        "domain_generic_coverage":
            domain_metrics[
                "generic_coverage"
            ],


        # ----------------------------------------------------
        # Host overlap
        # ----------------------------------------------------

        "host_jaccard_similarity":
            host_metrics[
                "jaccard_similarity"
            ],

        "host_jaccard_distance":
            host_metrics[
                "jaccard_distance"
            ],


        # ----------------------------------------------------
        # Exact sets — useful later
        # ----------------------------------------------------

        "shared_urls":
            url_metrics[
                "shared"
            ],

        "explicit_only_urls":
            url_metrics[
                "explicit_only"
            ],

        "generic_only_urls":
            url_metrics[
                "generic_only"
            ],

        "shared_domains":
            domain_metrics[
                "shared"
            ],

        "explicit_only_domains":
            domain_metrics[
                "explicit_only"
            ],

        "generic_only_domains":
            domain_metrics[
                "generic_only"
            ],
    })


comparisons = pd.DataFrame(
    comparison_rows
)


print(
    f"\nExplicit-vs-generic "
    f"comparisons: "
    f"{len(comparisons)}"
)


# ============================================================
# SAVE FLAT CSV WITHOUT LIST COLUMNS
# ============================================================

flat_columns = [
    col

    for col
    in comparisons.columns

    if col not in {
        "shared_urls",
        "explicit_only_urls",
        "generic_only_urls",
        "shared_domains",
        "explicit_only_domains",
        "generic_only_domains",
    }
]


comparisons[
    flat_columns
].to_csv(
    RESULTS_DIR
    / "source_overlap_vs_generic.csv",
    index=False,
)


# ============================================================
# DESCRIPTIVE SUMMARY BY GROUP
# ============================================================

group_summary = (
    comparisons
    .groupby(
        [
            "dimension",
            "condition",
            "group",
        ]
    )
    .agg(

        n=(
            "outcome_id",
            "size"
        ),

        mean_url_jaccard=(
            "url_jaccard_similarity",
            "mean"
        ),

        median_url_jaccard=(
            "url_jaccard_similarity",
            "median"
        ),

        mean_url_distance=(
            "url_jaccard_distance",
            "mean"
        ),

        mean_domain_jaccard=(
            "domain_jaccard_similarity",
            "mean"
        ),

        median_domain_jaccard=(
            "domain_jaccard_similarity",
            "median"
        ),

        mean_domain_distance=(
            "domain_jaccard_distance",
            "mean"
        ),

        mean_shared_urls=(
            "url_shared_n",
            "mean"
        ),

        mean_shared_domains=(
            "domain_shared_n",
            "mean"
        ),

        mean_unique_urls=(
            "n_unique_urls",
            "mean"
        ),

        mean_unique_domains=(
            "n_unique_domains",
            "mean"
        ),

        mean_generic_url_coverage=(
            "url_generic_coverage",
            "mean"
        ),

        mean_generic_domain_coverage=(
            "domain_generic_coverage",
            "mean"
        ),
    )
    .reset_index()
)


group_summary = (
    group_summary
    .sort_values(
        "mean_url_distance",
        ascending=False,
    )
)


group_summary.to_csv(
    RESULTS_DIR
    / "source_overlap_by_group.csv",
    index=False,
)


print()
print("=" * 80)
print("SOURCE OVERLAP BY GROUP")
print("=" * 80)

print(
    group_summary[
        [
            "dimension",
            "condition",
            "group",
            "mean_url_jaccard",
            "mean_url_distance",
            "mean_domain_jaccard",
            "mean_domain_distance",
            "mean_shared_urls",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# BY DOMAIN / OUTCOME
# ============================================================

domain_summary = (
    comparisons
    .groupby(
        "domain"
    )
    .agg(

        n=(
            "outcome_id",
            "size"
        ),

        mean_url_jaccard=(
            "url_jaccard_similarity",
            "mean"
        ),

        mean_url_distance=(
            "url_jaccard_distance",
            "mean"
        ),

        mean_domain_jaccard=(
            "domain_jaccard_similarity",
            "mean"
        ),

        mean_domain_distance=(
            "domain_jaccard_distance",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_url_distance",
        ascending=False,
    )
)


domain_summary.to_csv(
    RESULTS_DIR
    / "source_overlap_by_domain.csv",
    index=False,
)


outcome_summary = (
    comparisons
    .groupby(
        [
            "domain",
            "outcome",
        ]
    )
    .agg(

        n=(
            "group",
            "size"
        ),

        mean_url_jaccard=(
            "url_jaccard_similarity",
            "mean"
        ),

        mean_url_distance=(
            "url_jaccard_distance",
            "mean"
        ),

        mean_domain_jaccard=(
            "domain_jaccard_similarity",
            "mean"
        ),

        mean_domain_distance=(
            "domain_jaccard_distance",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_url_distance",
        ascending=False,
    )
)


outcome_summary.to_csv(
    RESULTS_DIR
    / "source_overlap_by_outcome.csv",
    index=False,
)


# ============================================================
# STATISTICAL HELPERS
# ============================================================

def paired_rank_biserial(
    differences
):

    d = np.asarray(
        differences,
        dtype=float
    )

    d = d[
        np.isfinite(d)
    ]

    d = d[
        d != 0
    ]

    if len(d) == 0:
        return 0.0


    ranks = rankdata(
        np.abs(d),
        method="average"
    )


    positive = (
        ranks[
            d > 0
        ].sum()
    )

    negative = (
        ranks[
            d < 0
        ].sum()
    )


    return float(
        (
            positive
            -
            negative
        )
        /
        (
            positive
            +
            negative
        )
    )


def paired_bootstrap_ci(
    differences,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
):

    d = np.asarray(
        differences,
        dtype=float
    )

    d = d[
        np.isfinite(d)
    ]


    if len(d) == 0:

        return (
            np.nan,
            np.nan
        )


    rng = (
        np.random.default_rng(
            seed
        )
    )


    n = len(d)

    values = np.empty(
        n_boot
    )


    for i in range(
        n_boot
    ):

        sample = rng.choice(
            d,
            size=n,
            replace=True
        )

        values[i] = (
            sample.mean()
        )


    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),

        float(
            np.percentile(
                values,
                97.5
            )
        ),
    )


def signflip_permutation_test(
    differences,
    n_resamples=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):

    d = np.asarray(
        differences,
        dtype=float
    )

    d = d[
        np.isfinite(d)
    ]

    d = d[
        d != 0
    ]


    n = len(d)


    if n == 0:

        return {
            "statistic":
                0.0,

            "p_value":
                1.0,

            "n_nonzero":
                0,

            "method":
                "all_zero",
        }


    observed = abs(
        d.mean()
    )


    # --------------------------------------------------------
    # EXACT
    # --------------------------------------------------------

    if (
        n
        <= EXACT_SIGNFLIP_MAX_N
    ):

        total = (
            2 ** n
        )

        extreme = 0

        powers = (
            1
            <<
            np.arange(
                n,
                dtype=np.uint64
            )
        )


        batch_size = 10_000


        for start in range(
            0,
            total,
            batch_size
        ):

            stop = min(
                start
                + batch_size,
                total
            )


            numbers = np.arange(
                start,
                stop,
                dtype=np.uint64
            )[:, None]


            bits = (
                (
                    numbers
                    & powers
                )
                > 0
            )


            signs = (
                bits.astype(
                    float
                )
                * 2
                - 1
            )


            permuted = np.abs(
                (
                    signs
                    * d[None, :]
                )
                .mean(
                    axis=1
                )
            )


            extreme += int(
                np.sum(
                    permuted
                    >= observed
                    - 1e-15
                )
            )


        p_value = (
            extreme
            /
            total
        )


        method = (
            f"exact_signflip_2^{n}"
        )


    # --------------------------------------------------------
    # MONTE CARLO
    # --------------------------------------------------------

    else:

        rng = (
            np.random.default_rng(
                seed
            )
        )


        extreme = 0

        completed = 0

        batch_size = 10_000


        while (
            completed
            < n_resamples
        ):

            batch = min(
                batch_size,
                n_resamples
                - completed
            )


            signs = rng.choice(
                [
                    -1.0,
                    1.0,
                ],
                size=(
                    batch,
                    n
                )
            )


            permuted = np.abs(
                (
                    signs
                    * d[None, :]
                )
                .mean(
                    axis=1
                )
            )


            extreme += int(
                np.sum(
                    permuted
                    >= observed
                    - 1e-15
                )
            )


            completed += batch


        p_value = (
            extreme + 1
        ) / (
            n_resamples + 1
        )


        method = (
            f"monte_carlo_signflip_"
            f"{n_resamples}"
        )


    return {

        "statistic":
            float(
                observed
            ),

        "p_value":
            float(
                p_value
            ),

        "n_nonzero":
            int(
                n
            ),

        "method":
            method,
    }


def safe_wilcoxon(
    differences
):

    d = np.asarray(
        differences,
        dtype=float
    )

    d = d[
        np.isfinite(d)
    ]

    d = np.round(
        d,
        12
    )


    if len(d) == 0:

        return (
            np.nan,
            np.nan
        )


    if np.all(
        d == 0
    ):

        return (
            0.0,
            1.0
        )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        try:

            result = wilcoxon(
                d,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

        except TypeError:

            result = wilcoxon(
                d,
                alternative="two-sided",
                zero_method="wilcox",
            )


    return (
        float(
            result.statistic
        ),

        float(
            result.pvalue
        )
    )


def complete_paired_test(
    minority,
    majority,
    label,
):

    minority = np.asarray(
        minority,
        dtype=float
    )

    majority = np.asarray(
        majority,
        dtype=float
    )


    valid = (
        np.isfinite(
            minority
        )
        &
        np.isfinite(
            majority
        )
    )


    minority = (
        minority[
            valid
        ]
    )

    majority = (
        majority[
            valid
        ]
    )


    difference = (
        minority
        -
        majority
    )


    permutation = (
        signflip_permutation_test(
            difference,
            seed=stable_seed(
                label
            )
        )
    )


    wilcoxon_W, wilcoxon_p = (
        safe_wilcoxon(
            difference
        )
    )


    ci_low, ci_high = (
        paired_bootstrap_ci(
            difference,
            seed=stable_seed(
                "bootstrap::"
                + label
            )
        )
    )


    return {

        "n":
            len(
                difference
            ),

        "minority_mean":
            float(
                np.mean(
                    minority
                )
            ),

        "majority_mean":
            float(
                np.mean(
                    majority
                )
            ),

        "mean_difference":
            float(
                np.mean(
                    difference
                )
            ),

        "median_difference":
            float(
                np.median(
                    difference
                )
            ),

        "bootstrap_ci_low":
            ci_low,

        "bootstrap_ci_high":
            ci_high,

        "rank_biserial":
            paired_rank_biserial(
                difference
            ),

        "permutation_statistic":
            permutation[
                "statistic"
            ],

        "p_permutation_raw":
            permutation[
                "p_value"
            ],

        "permutation_method":
            permutation[
                "method"
            ],

        "wilcoxon_W":
            wilcoxon_W,

        "p_wilcoxon_raw":
            wilcoxon_p,
    }


# ============================================================
# MINORITY vs MAJORITY BY DIMENSION
# ============================================================

def run_dimension_tests(
    metric,
    metric_name,
):

    results = []


    for dimension in sorted(
        comparisons[
            "dimension"
        ].unique()
    ):

        subset = comparisons[
            comparisons[
                "dimension"
            ]
            == dimension
        ]


        minority_groups = (
            subset.loc[
                subset[
                    "condition"
                ]
                == "minority",
                "group",
            ]
            .unique()
        )


        majority_groups = (
            subset.loc[
                subset[
                    "condition"
                ]
                == "majority",
                "group",
            ]
            .unique()
        )


        if (
            len(
                minority_groups
            ) != 1
            or
            len(
                majority_groups
            ) != 1
        ):

            raise RuntimeError(
                f"Invalid design: "
                f"{dimension}"
            )


        minority_group = (
            minority_groups[0]
        )

        majority_group = (
            majority_groups[0]
        )


        wide = (
            subset[
                [
                    "outcome_id",
                    "group",
                    metric,
                ]
            ]
            .pivot(
                index="outcome_id",
                columns="group",
                values=metric,
            )
        )


        wide = wide.dropna(
            subset=[
                minority_group,
                majority_group,
            ]
        )


        stats = complete_paired_test(

            wide[
                minority_group
            ].values,

            wide[
                majority_group
            ].values,

            label=(
                f"{metric_name}"
                f"::{dimension}"
            ),
        )


        stats.update({

            "metric":
                metric_name,

            "dimension":
                dimension,

            "minority_group":
                minority_group,

            "majority_group":
                majority_group,
        })


        results.append(
            stats
        )


    result_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Holm correction across six social dimensions
    # --------------------------------------------------------

    reject, p_adj, _, _ = (
        multipletests(
            result_df[
                "p_permutation_raw"
            ],
            alpha=ALPHA,
            method="holm",
        )
    )


    result_df[
        "p_permutation_holm"
    ] = p_adj


    result_df[
        "significant_permutation_holm"
    ] = reject


    reject_w, p_adj_w, _, _ = (
        multipletests(
            result_df[
                "p_wilcoxon_raw"
            ],
            alpha=ALPHA,
            method="holm",
        )
    )


    result_df[
        "p_wilcoxon_holm"
    ] = p_adj_w


    result_df[
        "significant_wilcoxon_holm"
    ] = reject_w


    return result_df


# ============================================================
# TEST 1 — URL SET DISTANCE
# ============================================================

url_dimension_tests = (
    run_dimension_tests(
        "url_jaccard_distance",
        "url_jaccard_distance",
    )
)


url_dimension_tests.to_csv(
    RESULTS_DIR
    / "minority_vs_majority_url_distance.csv",
    index=False,
)


# ============================================================
# TEST 2 — DOMAIN SET DISTANCE
# ============================================================

domain_dimension_tests = (
    run_dimension_tests(
        "domain_jaccard_distance",
        "domain_jaccard_distance",
    )
)


domain_dimension_tests.to_csv(
    RESULTS_DIR
    / "minority_vs_majority_domain_distance.csv",
    index=False,
)


# ============================================================
# GENERIC COVERAGE TESTS
#
# Secondary but useful:
#
# How much of the generic source set survives in
# each group-specific source set?
#
# Here LOWER generic coverage = more generic sources lost.
#
# We multiply by -1 conceptually only in interpretation;
# tests compare raw coverage directly.
# ============================================================

url_generic_coverage_tests = (
    run_dimension_tests(
        "url_generic_coverage",
        "url_generic_coverage",
    )
)


url_generic_coverage_tests.to_csv(
    RESULTS_DIR
    / "minority_vs_majority_url_generic_coverage.csv",
    index=False,
)


domain_generic_coverage_tests = (
    run_dimension_tests(
        "domain_generic_coverage",
        "domain_generic_coverage",
    )
)


domain_generic_coverage_tests.to_csv(
    RESULTS_DIR
    / "minority_vs_majority_domain_generic_coverage.csv",
    index=False,
)


# ============================================================
# PRINT PRIMARY TESTS
# ============================================================

print()
print("=" * 80)
print("URL-SET DISPLACEMENT FROM GENERIC")
print("=" * 80)

print(
    url_dimension_tests[
        [
            "dimension",
            "minority_group",
            "majority_group",
            "n",
            "minority_mean",
            "majority_mean",
            "mean_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "rank_biserial",
            "p_permutation_holm",
            "significant_permutation_holm",
        ]
    ]
    .round(5)
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("DOMAIN-SET DISPLACEMENT FROM GENERIC")
print("=" * 80)

print(
    domain_dimension_tests[
        [
            "dimension",
            "minority_group",
            "majority_group",
            "n",
            "minority_mean",
            "majority_mean",
            "mean_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "rank_biserial",
            "p_permutation_holm",
            "significant_permutation_holm",
        ]
    ]
    .round(5)
    .to_string(
        index=False
    )
)


# ============================================================
# AGGREGATED MINORITY vs MAJORITY
#
# Important:
# first average six minority / six majority conditions
# WITHIN EACH OUTCOME.
#
# Then use the 21 outcomes as paired units.
# ============================================================

def aggregate_minority_majority(
    metric,
    metric_name,
):

    temp = (
        comparisons
        .groupby(
            [
                "outcome_id",
                "condition",
            ]
        )[
            metric
        ]
        .mean()
        .unstack()
    )


    temp = temp.dropna(
        subset=[
            "minority",
            "majority",
        ]
    )


    result = complete_paired_test(

        temp[
            "minority"
        ].values,

        temp[
            "majority"
        ].values,

        label=(
            "aggregate::"
            + metric_name
        ),
    )


    result.update({

        "metric":
            metric_name,

        "unit":
            "outcome",

        "description":
            (
                "For each outcome, "
                "average across six minority "
                "and six majority conditions, "
                "then compare the 21 paired "
                "outcome-level values."
            ),
    })


    return result


aggregate_url = (
    aggregate_minority_majority(
        "url_jaccard_distance",
        "url_jaccard_distance",
    )
)


aggregate_domain = (
    aggregate_minority_majority(
        "domain_jaccard_distance",
        "domain_jaccard_distance",
    )
)


aggregate_url_coverage = (
    aggregate_minority_majority(
        "url_generic_coverage",
        "url_generic_coverage",
    )
)


aggregate_domain_coverage = (
    aggregate_minority_majority(
        "domain_generic_coverage",
        "domain_generic_coverage",
    )
)


aggregate_results = pd.DataFrame([
    aggregate_url,
    aggregate_domain,
    aggregate_url_coverage,
    aggregate_domain_coverage,
])


aggregate_results.to_csv(
    RESULTS_DIR
    / "aggregate_minority_vs_majority.csv",
    index=False,
)


print()
print("=" * 80)
print("AGGREGATED MINORITY vs MAJORITY")
print("=" * 80)

print(
    aggregate_results[
        [
            "metric",
            "n",
            "minority_mean",
            "majority_mean",
            "mean_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "rank_biserial",
            "p_permutation_raw",
        ]
    ]
    .round(5)
    .to_string(
        index=False
    )
)


# ============================================================
# GLOBAL GROUP EFFECT — FRIEDMAN
# ============================================================

def global_friedman(
    metric,
    name,
):

    wide = (
        comparisons
        .pivot(
            index="outcome_id",
            columns="group",
            values=metric,
        )
    )


    complete = wide.dropna()


    if (
        len(
            complete
        ) < 2
        or complete.shape[1] < 3
    ):

        return {

            "metric":
                name,

            "n_outcomes":
                len(
                    complete
                ),

            "n_groups":
                complete.shape[1],

            "friedman_chi2":
                None,

            "p_value":
                None,

            "kendall_W":
                None,
        }


    stat, p = (
        friedmanchisquare(
            *[
                complete[
                    column
                ].values

                for column
                in complete.columns
            ]
        )
    )


    n = (
        complete.shape[0]
    )

    k = (
        complete.shape[1]
    )


    kendall_W = (
        stat
        /
        (
            n
            *
            (
                k - 1
            )
        )
    )


    return {

        "metric":
            name,

        "n_outcomes":
            n,

        "n_groups":
            k,

        "friedman_chi2":
            float(
                stat
            ),

        "p_value":
            float(
                p
            ),

        "kendall_W":
            float(
                kendall_W
            ),

        "significant":
            bool(
                p < ALPHA
            ),
    }


global_tests = pd.DataFrame([

    global_friedman(
        "url_jaccard_distance",
        "url_jaccard_distance",
    ),

    global_friedman(
        "domain_jaccard_distance",
        "domain_jaccard_distance",
    ),
])


global_tests.to_csv(
    RESULTS_DIR
    / "global_friedman_tests.csv",
    index=False,
)


# ============================================================
# CASE COUNTS / ZERO-SOURCE DIAGNOSTICS
# ============================================================

zero_source_cases = docs[
    docs[
        "n_urls_unique"
    ]
    == 0
].copy()


zero_source_cases.to_csv(
    RESULTS_DIR
    / "zero_source_cases.csv",
    index=False,
)


both_empty_url_pairs = (
    comparisons[
        (
            comparisons[
                "n_unique_urls"
            ]
            == 0
        )
        &
        (
            comparisons[
                "generic_n_unique_urls"
            ]
            == 0
        )
    ]
)


one_empty_url_pairs = (
    comparisons[
        (
            (
                comparisons[
                    "n_unique_urls"
                ]
                == 0
            )
            ^
            (
                comparisons[
                    "generic_n_unique_urls"
                ]
                == 0
            )
        )
    ]
)


# ============================================================
# MOST / LEAST SIMILAR SOURCE SETS
# ============================================================

most_different_url = (
    comparisons
    .sort_values(
        "url_jaccard_distance",
        ascending=False,
    )
    [
        [
            "dimension",
            "condition",
            "group",
            "domain",
            "outcome",
            "url_jaccard_similarity",
            "url_jaccard_distance",
            "url_shared_n",
            "n_unique_urls",
            "generic_n_unique_urls",
        ]
    ]
    .head(
        30
    )
)


most_similar_url = (
    comparisons
    .sort_values(
        "url_jaccard_distance",
        ascending=True,
    )
    [
        [
            "dimension",
            "condition",
            "group",
            "domain",
            "outcome",
            "url_jaccard_similarity",
            "url_jaccard_distance",
            "url_shared_n",
            "n_unique_urls",
            "generic_n_unique_urls",
        ]
    ]
    .head(
        30
    )
)


most_different_domain = (
    comparisons
    .sort_values(
        "domain_jaccard_distance",
        ascending=False,
    )
    [
        [
            "dimension",
            "condition",
            "group",
            "domain",
            "outcome",
            "domain_jaccard_similarity",
            "domain_jaccard_distance",
            "domain_shared_n",
            "n_unique_domains",
            "generic_n_unique_domains",
        ]
    ]
    .head(
        30
    )
)


most_different_url.to_csv(
    RESULTS_DIR
    / "most_different_url_sets.csv",
    index=False,
)


most_similar_url.to_csv(
    RESULTS_DIR
    / "most_similar_url_sets.csv",
    index=False,
)


most_different_domain.to_csv(
    RESULTS_DIR
    / "most_different_domain_sets.csv",
    index=False,
)


# ============================================================
# MASTER JSON
# ============================================================

all_results = {

    "metadata": {

        "analysis":
            (
                "Google AI Overview "
                "source-set overlap analysis"
            ),

        "collection":
            COLLECTION_VERSION,

        "collection_directory":
            str(
                COLLECTION_DIR
            ),

        "results_directory":
            str(
                RESULTS_DIR
            ),

        "alpha":
            ALPHA,

        "random_seed":
            RANDOM_SEED,

        "permutation_resamples":
            N_PERMUTATIONS,

        "bootstrap_resamples":
            N_BOOTSTRAP,

        "domain_definition":
            (
                "Registrable domain when "
                "tldextract is available; "
                "hostname fallback otherwise."
            ),

        "url_normalization":
            (
                "Fragments and obvious "
                "tracking parameters removed; "
                "substantive query parameters "
                "preserved."
            ),

        "jaccard_empty_set_rule":
            (
                "If both source sets are empty, "
                "Jaccard is undefined and saved "
                "as null. If only one set is empty, "
                "similarity=0 and distance=1."
            ),

        "primary_statistical_unit":
            "matched outcome",

        "primary_inference":
            (
                "paired sign-flip permutation "
                "test with Holm correction across "
                "the six social dimensions"
            ),
    },


    "dataset_summary": {

        "n_documents":
            len(
                docs
            ),

        "n_explicit_vs_generic_pairs":
            len(
                comparisons
            ),

        "n_groups":
            docs[
                "group"
            ].nunique(),

        "n_outcomes":
            docs[
                "outcome_id"
            ].nunique(),

        "n_zero_source_documents":
            len(
                zero_source_cases
            ),

        "n_pairs_both_url_sets_empty":
            len(
                both_empty_url_pairs
            ),

        "n_pairs_one_url_set_empty":
            len(
                one_empty_url_pairs
            ),

        "tldextract_available":
            HAVE_TLDEXTRACT,
    },


    "descriptive": {

        "by_group":
            dataframe_records(
                group_summary
            ),

        "by_domain":
            dataframe_records(
                domain_summary
            ),

        "by_outcome":
            dataframe_records(
                outcome_summary
            ),
    },


    "minority_vs_majority": {

        "url_jaccard_distance":
            dataframe_records(
                url_dimension_tests
            ),

        "domain_jaccard_distance":
            dataframe_records(
                domain_dimension_tests
            ),

        "url_generic_coverage":
            dataframe_records(
                url_generic_coverage_tests
            ),

        "domain_generic_coverage":
            dataframe_records(
                domain_generic_coverage_tests
            ),

        "aggregate":
            dataframe_records(
                aggregate_results
            ),
    },


    "global_tests":
        dataframe_records(
            global_tests
        ),


    "extreme_cases": {

        "most_different_url_sets":
            dataframe_records(
                most_different_url
            ),

        "most_similar_url_sets":
            dataframe_records(
                most_similar_url
            ),

        "most_different_domain_sets":
            dataframe_records(
                most_different_domain
            ),
    },


    "zero_source_cases":
        dataframe_records(
            zero_source_cases
        ),


    "all_pairwise_observations":
        dataframe_records(
            comparisons
        ),
}


all_results = json_safe(
    all_results
)


with open(
    RESULTS_DIR
    / "all_results.json",
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
# HUMAN READABLE SUMMARY
# ============================================================

summary_lines = []

summary_lines.append(
    "GOOGLE AIO SOURCE OVERLAP ANALYSIS"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append(
    ""
)

summary_lines.append(
    "URL-SET DISTANCE:"
)

summary_lines.append(
    ""
)


for _, row in (
    url_dimension_tests
    .iterrows()
):

    summary_lines.append(

        (
            f"{row['dimension']}: "
            f"{row['minority_group']}="
            f"{row['minority_mean']:.4f}, "
            f"{row['majority_group']}="
            f"{row['majority_mean']:.4f}, "
            f"Delta="
            f"{row['mean_difference']:+.4f}, "
            f"95% CI "
            f"[{row['bootstrap_ci_low']:.4f}, "
            f"{row['bootstrap_ci_high']:.4f}], "
            f"Holm p="
            f"{row['p_permutation_holm']:.6g}"
        )
    )


summary_lines.append(
    ""
)

summary_lines.append(
    "DOMAIN-SET DISTANCE:"
)

summary_lines.append(
    ""
)


for _, row in (
    domain_dimension_tests
    .iterrows()
):

    summary_lines.append(

        (
            f"{row['dimension']}: "
            f"{row['minority_group']}="
            f"{row['minority_mean']:.4f}, "
            f"{row['majority_group']}="
            f"{row['majority_mean']:.4f}, "
            f"Delta="
            f"{row['mean_difference']:+.4f}, "
            f"95% CI "
            f"[{row['bootstrap_ci_low']:.4f}, "
            f"{row['bootstrap_ci_high']:.4f}], "
            f"Holm p="
            f"{row['p_permutation_holm']:.6g}"
        )
    )


summary_lines.append(
    ""
)

summary_lines.append(
    "AGGREGATED RESULTS:"
)

summary_lines.append(
    ""
)


for _, row in (
    aggregate_results
    .iterrows()
):

    summary_lines.append(

        (
            f"{row['metric']}: "
            f"minority="
            f"{row['minority_mean']:.4f}, "
            f"majority="
            f"{row['majority_mean']:.4f}, "
            f"Delta="
            f"{row['mean_difference']:+.4f}, "
            f"p="
            f"{row['p_permutation_raw']:.6g}"
        )
    )


summary_lines.append(
    ""
)

summary_lines.append(
    (
        "Interpretation for distance metrics: "
        "positive minority-majority difference "
        "means the minority condition uses a "
        "source set farther from the matched "
        "generic `people` query."
    )
)


summary_lines.append(
    ""
)

summary_lines.append(
    (
        "Interpretation for coverage metrics: "
        "positive minority-majority difference "
        "means the minority condition preserves "
        "a larger fraction of the generic source "
        "set."
    )
)


(
    RESULTS_DIR
    / "SUMMARY.txt"
).write_text(
    "\n".join(
        summary_lines
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# README
# ============================================================

README = f"""
Google AI Overview — Source Overlap Analysis
=============================================

Collection:
{COLLECTION_VERSION}

Primary questions
-----------------

1. How much does the exact URL set change when a social
   identity is explicitly mentioned relative to the matched
   generic "people" query?

2. How much does the publisher/domain set change?

3. Is the displacement systematically different for
   minority-marked and majority/comparison-marked conditions?

Primary measures
----------------

URL Jaccard similarity:

    |URL_group ∩ URL_people|
    -------------------------
    |URL_group ∪ URL_people|

URL displacement:

    1 - URL Jaccard similarity

Domain displacement:

    1 - Domain Jaccard similarity

Statistical design
------------------

The statistical unit is the matched domain/outcome.

Within each social dimension, the 21 minority observations
are paired with the same 21 majority/comparison observations.

Primary test:
paired sign-flip permutation test.

Secondary confirmation:
Wilcoxon signed-rank test.

Multiple comparisons:
Holm correction across the six social dimensions.

Files
-----

all_results.json
    Complete analysis output.

SUMMARY.txt
    Human-readable result summary.

source_overlap_vs_generic.csv
    All 252 explicit-group vs generic comparisons.

source_overlap_by_group.csv
    Descriptive results by social group.

source_overlap_by_domain.csv
    Descriptive results by domain.

source_overlap_by_outcome.csv
    Descriptive results by outcome.

minority_vs_majority_url_distance.csv
    Main URL-overlap statistical tests.

minority_vs_majority_domain_distance.csv
    Main domain-overlap statistical tests.

minority_vs_majority_url_generic_coverage.csv
    Fraction of generic URLs preserved.

minority_vs_majority_domain_generic_coverage.csv
    Fraction of generic domains preserved.

aggregate_minority_vs_majority.csv
    Aggregate 21-outcome minority-vs-majority tests.

global_friedman_tests.csv
    Global group effects.

zero_source_cases.csv
    Responses containing no sources.

most_different_url_sets.csv
    Largest source-set displacements.

most_similar_url_sets.csv
    Smallest source-set displacements.

most_different_domain_sets.csv
    Largest publisher/domain-set displacements.
"""


(
    RESULTS_DIR
    / "README.txt"
).write_text(
    README.strip()
    + "\n",
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nResults directory:"
)

print(
    RESULTS_DIR
)

print(
    "\nMain files:"
)

for filename in [
    "SUMMARY.txt",
    "all_results.json",
    "source_overlap_by_group.csv",
    "minority_vs_majority_url_distance.csv",
    "minority_vs_majority_domain_distance.csv",
    "aggregate_minority_vs_majority.csv",
]:

    print(
        f"  - {filename}"
    )

print()
print(
    "Done."
)