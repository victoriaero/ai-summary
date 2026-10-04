# ============================================================
# GOOGLE AI OVERVIEW
# INSTITUTIONAL SOURCE TYPE ANALYSIS — V2
#
# PURPOSE
# -------
# Determine what kinds of institutions populate the
# evidentiary pathways surfaced by Google AI Overviews.
#
# V2 FIXES
# --------
# - Larger classifier: Qwen2.5-14B-Instruct-AWQ
# - Proper chat template
# - JSON output
# - Robust parser
# - Automatic retry of malformed generations
# - Parse failures are NOT silently converted to Other/Unclear
# - Per-batch checkpointing
# - Sanity checks BEFORE statistical analysis
# - Separate classifier uncertainty from substantive "Other"
# - Known-category-only robustness analysis
#
# UNIT
# ----
# Unique normalized cited URL within each query.
#
# TAXONOMY
# --------
# 1. Government/Public Agency
# 2. Academic/Research
# 3. News/Media
# 4. NGO/Advocacy
# 5. Community/Lived Experience
# 6. Commercial/Professional
# 7. Other/Unclear
#
# ============================================================


from pathlib import Path
from urllib.parse import urlparse

import json
import os
import re
import warnings
import zlib

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata, wilcoxon

from statsmodels.stats.multitest import multipletests


# ============================================================
# 0. CONFIG
# ============================================================

BASE_DIR = Path(
    "/scratch/victoria.estanislau/ai-summary"
)

COLLECTION_VERSION = "v1_dallas"


AIO_COLLECTION_DIR = (
    BASE_DIR
    / "annotations"
    / COLLECTION_VERSION
    / "google_aio_collection"
)


SOURCE_CORPUS_DIR = (
    BASE_DIR
    / "annotations"
    / COLLECTION_VERSION
    / "source_corpus"
)


MANIFEST_FILE = (
    SOURCE_CORPUS_DIR
    / "manifest.csv"
)


URL_USAGE_FILE = (
    SOURCE_CORPUS_DIR
    / "url_usage.csv"
)


# IMPORTANT:
# Use a NEW directory.
# Do not reuse the broken V1 classifications.

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "source_type_analysis_dallas_v2"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CLASSIFICATION_FILE = (
    RESULTS_DIR
    / "source_type_catalog_v2.csv"
)


CHECKPOINT_FILE = (
    RESULTS_DIR
    / "source_type_classifier_checkpoint.csv"
)


# ============================================================
# 1. CLASSIFIER MODEL
# ============================================================

# Good default for a 32 GB RTX 5090.
#
# AWQ leaves substantially more memory headroom than BF16.
#
# Override from shell if desired:
#
# export SOURCE_TYPE_MODEL="Qwen/Qwen2.5-14B-Instruct"
#

MODEL_NAME = os.environ.get(
    "SOURCE_TYPE_MODEL",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
)


GPU_MEMORY_UTILIZATION = 0.90


# Prompts are small; we do not need giant context.
MAX_MODEL_LEN = 3072


# vLLM internally schedules sequences.
MAX_NUM_SEQS = 64


# Number of prompts supplied at a time.
CLASSIFIER_BATCH_SIZE = 128


MAX_OUTPUT_TOKENS = 160


TEMPERATURE = 0.0


# ============================================================
# 2. RESUME
# ============================================================

# This only reuses V2 classifications in RESULTS_DIR above.
#
# It does NOT touch the previous broken V1 directory.

REUSE_V2_CLASSIFICATIONS = True


# Save every batch so Ctrl+C / crash does not lose work.
SAVE_AFTER_EVERY_BATCH = True


# ============================================================
# 3. TAXONOMY
# ============================================================

SOURCE_TYPES = [

    "Government/Public Agency",

    "Academic/Research",

    "News/Media",

    "NGO/Advocacy",

    "Community/Lived Experience",

    "Commercial/Professional",

    "Other/Unclear",
]


KNOWN_SOURCE_TYPES = [

    category
    for category in SOURCE_TYPES
    if category != "Other/Unclear"
]


# ============================================================
# 4. MANUAL VALIDATION
# ============================================================

VALIDATION_PER_CATEGORY = 25


# Include all uncertain cases in the validation sheet.
INCLUDE_ALL_LOW_CONFIDENCE = True

INCLUDE_ALL_OTHER = True

INCLUDE_ALL_PARSE_FAILURES = True

INCLUDE_DOMAIN_CONFLICTS = True


# ============================================================
# 5. SANITY CHECKS
# ============================================================

STRICT_SANITY_CHECKS = True


# Parse failure should be virtually nonexistent after retry.
MAX_PARSE_FAILURE_RATE = 0.01


# 68% in V1 was clearly broken.
#
# This threshold is a safeguard, not a substantive assumption.
MAX_OTHER_SHARE = 0.45


# These should almost certainly occur in this corpus.
REQUIRE_NONZERO_CATEGORIES = {

    "News/Media",
    "NGO/Advocacy",
    "Commercial/Professional",
}


# ============================================================
# 6. STATISTICS
# ============================================================

ALPHA = 0.05

RANDOM_SEED = 42

N_PERMUTATIONS = 200_000

N_BOOTSTRAP = 20_000

EXACT_SIGNFLIP_MAX_N = 18


# ============================================================
# 7. CONDITION MAPPINGS
# ============================================================

FOCAL_CONDITIONS = {
    "minority",
    "focal",
}


COMPARISON_CONDITIONS = {
    "majority",
    "comparison",
}


CONTROL_CONDITIONS = {
    "control",
    "generic",
}


# ============================================================
# 8. HELPERS
# ============================================================

def stable_seed(label):

    return (
        RANDOM_SEED
        +
        zlib.crc32(
            str(label).encode(
                "utf-8"
            )
        )
    ) % (
        2**32 - 1
    )


def clean_text(text):

    text = str(
        text
        or ""
    )

    text = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def slugify_category(category):

    value = (
        category
        .lower()
        .replace(
            "/",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    value = re.sub(
        r"[^a-z0-9_]+",
        "",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


SOURCE_TYPE_SLUGS = {

    category:
        slugify_category(
            category
        )

    for category
    in SOURCE_TYPES
}


# ============================================================
# 9. AIO FILE PARSING
# ============================================================

def normalize_newlines(text):

    return (
        str(
            text
            or ""
        )
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
        text,
    )


    if not match:

        return ""


    remainder = text[
        match.end():
    ]


    positions = []


    for next_header in (
        next_headers
    ):

        m = re.search(
            rf"(?m)^\s*"
            rf"{re.escape(next_header)}"
            rf"\s*$",
            remainder,
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


    return remainder.strip()


def parse_key_values(
    section
):

    result = {}


    for line in (
        section
        or ""
    ).splitlines():

        if ":" not in line:

            continue


        key, value = line.split(
            ":",
            1,
        )


        result[
            key.strip()
        ] = value.strip()


    return result


# ============================================================
# 10. LOAD QUERY INDEX
# ============================================================

print()
print("=" * 80)
print("LOADING QUERY INDEX")
print("=" * 80)


query_rows = []


for filepath in sorted(
    AIO_COLLECTION_DIR.rglob(
        "*.txt"
    )
):

    raw = filepath.read_text(
        encoding="utf-8",
        errors="replace",
    )


    metadata = parse_key_values(

        get_section(
            raw,
            "METADATA — DO NOT EDIT",
            [],
        )
    )


    domain = metadata.get(
        "Domain",
        ""
    )


    outcome = metadata.get(
        "Outcome",
        ""
    )


    query_rows.append({

        "file":
            str(
                filepath.relative_to(
                    AIO_COLLECTION_DIR
                )
            ),

        "query_id":
            metadata.get(
                "Query ID",
                ""
            ),

        "group":
            metadata.get(
                "Group",
                ""
            ),

        "condition":
            metadata.get(
                "Condition",
                ""
            ),

        "dimension":
            metadata.get(
                "Dimension",
                ""
            ),

        "domain":
            domain,

        "outcome":
            outcome,

        "outcome_id":
            (
                f"{domain} :: "
                f"{outcome}"
            ),
    })


query_index = pd.DataFrame(
    query_rows
)


print(
    f"\nQueries: "
    f"{len(query_index):,}"
)


print(
    f"Outcomes: "
    f"{query_index['outcome_id'].nunique()}"
)


print(
    f"Groups: "
    f"{query_index['group'].nunique()}"
)


# ============================================================
# 11. LOAD SOURCE DATA
# ============================================================

if not MANIFEST_FILE.exists():

    raise RuntimeError(
        f"Missing manifest:\n"
        f"{MANIFEST_FILE}"
    )


if not URL_USAGE_FILE.exists():

    raise RuntimeError(
        f"Missing URL usage:\n"
        f"{URL_USAGE_FILE}"
    )


manifest = pd.read_csv(
    MANIFEST_FILE
)


usage = pd.read_csv(
    URL_USAGE_FILE
)


manifest[
    "source_id"
] = (
    manifest[
        "source_id"
    ]
    .astype(str)
)


usage[
    "source_id"
] = (
    usage[
        "source_id"
    ]
    .astype(str)
)


usage[
    "source_order"
] = pd.to_numeric(
    usage[
        "source_order"
    ],
    errors="coerce",
)


# ------------------------------------------------------------
# One source contributes at most once per answer.
# ------------------------------------------------------------

usage_unique = (
    usage
    .sort_values(
        [
            "file",
            "source_order",
        ]
    )
    .drop_duplicates(
        subset=[
            "file",
            "source_id",
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


print()
print(
    f"Source occurrences: "
    f"{len(usage):,}"
)


print(
    f"Unique query-source pairs: "
    f"{len(usage_unique):,}"
)


print(
    f"Unique sources: "
    f"{usage_unique['source_id'].nunique():,}"
)


# ============================================================
# 12. BUILD SOURCE CATALOG
# ============================================================

metadata_columns = [

    column

    for column in [

        "source_id",
        "registrable_domain",
        "final_url",
        "page_title",
        "site_name",
        "text_path",
        "extraction_status",

    ]

    if column
    in manifest.columns
]


manifest_small = (

    manifest[
        metadata_columns
    ]
    .drop_duplicates(
        subset=[
            "source_id"
        ],
        keep="last",
    )
)


source_catalog = (

    usage_unique[
        [
            "source_id",
            "normalized_url",
        ]
    ]
    .drop_duplicates(
        subset=[
            "source_id"
        ],
        keep="first",
    )
    .merge(
        manifest_small,
        on="source_id",
        how="left",
    )
)


# ============================================================
# 13. URL / HOST INFORMATION
# ============================================================

def get_hostname(url):

    try:

        return (
            urlparse(
                str(
                    url
                    or ""
                )
            )
            .hostname
            or ""
        ).lower()

    except Exception:

        return ""


source_catalog[
    "hostname"
] = (
    source_catalog[
        "normalized_url"
    ]
    .apply(
        get_hostname
    )
)


if (
    "registrable_domain"
    not in source_catalog.columns
):

    source_catalog[
        "registrable_domain"
    ] = (
        source_catalog[
            "hostname"
        ]
    )


source_catalog[
    "registrable_domain"
] = (

    source_catalog[
        "registrable_domain"
    ]
    .fillna(
        source_catalog[
            "hostname"
        ]
    )
    .astype(str)
)


# ============================================================
# 14. READ SHORT PAGE EXCERPTS
# ============================================================

MAX_EXCERPT_CHARS = 1200


def read_excerpt(row):

    text_path = row.get(
        "text_path"
    )


    if (
        not isinstance(
            text_path,
            str,
        )
        or
        not text_path
    ):

        return ""


    path = Path(
        text_path
    )


    if not path.exists():

        return ""


    try:

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )


        text = clean_text(
            text
        )


        return text[
            :MAX_EXCERPT_CHARS
        ]


    except Exception:

        return ""


print()
print(
    "Reading source excerpts..."
)


source_catalog[
    "source_excerpt"
] = source_catalog.apply(
    read_excerpt,
    axis=1,
)


# ============================================================
# 15. CLASSIFICATION COLUMNS
# ============================================================

for column in [

    "source_type",
    "classification_confidence",
    "classification_method",
    "classification_status",
    "classification_rationale",
    "classifier_raw_output",

]:

    source_catalog[
        column
    ] = None


# ============================================================
# 16. OPTIONAL V2 RESUME
# ============================================================

if (
    REUSE_V2_CLASSIFICATIONS
    and
    CHECKPOINT_FILE.exists()
):

    print()
    print(
        "Loading V2 classifier checkpoint..."
    )


    checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )


    checkpoint[
        "source_id"
    ] = (
        checkpoint[
            "source_id"
        ]
        .astype(str)
    )


    reusable = checkpoint[
        checkpoint[
            "classification_status"
        ]
        .isin(
            [
                "rule_valid",
                "llm_valid",
                "llm_retry_valid",
            ]
        )
    ]


    reusable = (
        reusable
        .drop_duplicates(
            subset=[
                "source_id"
            ],
            keep="last",
        )
        .set_index(
            "source_id"
        )
    )


    for index, row in (
        source_catalog.iterrows()
    ):

        source_id = (
            row[
                "source_id"
            ]
        )


        if (
            source_id
            not in reusable.index
        ):

            continue


        old = reusable.loc[
            source_id
        ]


        source_catalog.at[
            index,
            "source_type"
        ] = old[
            "source_type"
        ]


        source_catalog.at[
            index,
            "classification_confidence"
        ] = old[
            "classification_confidence"
        ]


        source_catalog.at[
            index,
            "classification_method"
        ] = "reused_v2"


        source_catalog.at[
            index,
            "classification_status"
        ] = old[
            "classification_status"
        ]


        source_catalog.at[
            index,
            "classification_rationale"
        ] = old.get(
            "classification_rationale",
            "",
        )


        source_catalog.at[
            index,
            "classifier_raw_output"
        ] = old.get(
            "classifier_raw_output",
            "",
        )


# ============================================================
# 17. HIGH-PRECISION RULES
# ============================================================

COMMUNITY_DOMAINS = {

    "reddit.com",
    "quora.com",
    "medium.com",
    "substack.com",
    "tumblr.com",
    "wordpress.com",
    "blogspot.com",
}


def high_precision_rule(
    row
):

    hostname = str(
        row.get(
            "hostname",
            ""
        )
        or ""
    ).lower()


    registrable = str(
        row.get(
            "registrable_domain",
            ""
        )
        or ""
    ).lower()


    # --------------------------------------------------------
    # US / obvious government.
    # --------------------------------------------------------

    if (
        hostname.endswith(
            ".gov"
        )
        or
        hostname.endswith(
            ".mil"
        )
        or
        ".gov."
        in hostname
        or
        ".mil."
        in hostname
    ):

        return {

            "category":
                "Government/Public Agency",

            "confidence":
                "high",

            "rationale":
                (
                    "Official government or "
                    "military domain."
                ),
        }


    # --------------------------------------------------------
    # Academic domains.
    # --------------------------------------------------------

    if (
        hostname.endswith(
            ".edu"
        )
        or
        ".edu."
        in hostname
        or
        ".ac."
        in hostname
    ):

        return {

            "category":
                "Academic/Research",

            "confidence":
                "high",

            "rationale":
                (
                    "Academic or university "
                    "domain."
                ),
        }


    # --------------------------------------------------------
    # Known user-generated/community platforms.
    # --------------------------------------------------------

    for domain in (
        COMMUNITY_DOMAINS
    ):

        if (
            registrable
            == domain
            or
            hostname.endswith(
                "."
                + domain
            )
        ):

            return {

                "category":
                    (
                        "Community/"
                        "Lived Experience"
                    ),

                "confidence":
                    "high",

                "rationale":
                    (
                        "Known community or "
                        "user-generated publishing "
                        "platform."
                    ),
            }


    return None


# ============================================================
# 18. APPLY RULES
# ============================================================

for index, row in (
    source_catalog.iterrows()
):

    if (
        source_catalog.at[
            index,
            "source_type"
        ]
        in SOURCE_TYPES
    ):

        continue


    result = high_precision_rule(
        row
    )


    if result is None:

        continue


    source_catalog.at[
        index,
        "source_type"
    ] = result[
        "category"
    ]


    source_catalog.at[
        index,
        "classification_confidence"
    ] = result[
        "confidence"
    ]


    source_catalog.at[
        index,
        "classification_method"
    ] = "rule"


    source_catalog.at[
        index,
        "classification_status"
    ] = "rule_valid"


    source_catalog.at[
        index,
        "classification_rationale"
    ] = result[
        "rationale"
    ]


# ============================================================
# 19. TAXONOMY INSTRUCTIONS
# ============================================================

TAXONOMY_TEXT = """
1. Government/Public Agency
Official government department, agency, court, public authority,
municipality, public health agency, military organization, or other
state institution.

2. Academic/Research
University, academic department, scholarly journal, scientific
publisher, scientific repository, research institute, or organization
whose primary function is research.

3. News/Media
Newspaper, news website, magazine, broadcaster, newsroom, wire service,
or other organization whose primary function is journalism or news
publishing.

4. NGO/Advocacy
Non-governmental nonprofit, charity, civil-rights organization,
advocacy organization, activist organization, mission-driven policy
organization, or nonprofit organization primarily engaged in public
advocacy.

5. Community/Lived Experience
Discussion forum, user-generated platform, support community,
personal blog, community publishing space, or source whose primary
function is individual/community experience or peer discussion.

6. Commercial/Professional
For-profit company, hospital or clinic that is not a government or
academic institution, law firm, financial firm, consultancy, employer,
trade organization, professional association, commercial health site,
or other professional/commercial service provider.

7. Other/Unclear
Use ONLY when the publisher genuinely cannot be identified from the
URL, hostname, metadata, and available text, or when the organization
does not reasonably fit any category above.
""".strip()


DECISION_RULES = """
IMPORTANT DECISION RULES

- Classify the PUBLISHER / HOSTING INSTITUTION, not the topic of the page.

- A government page discussing academic research is Government/Public Agency.

- A university page discussing advocacy is Academic/Research.

- A newspaper article about science is News/Media.

- A law firm page is Commercial/Professional.

- A non-governmental civil-rights organization is NGO/Advocacy.

- A hospital or clinic is Commercial/Professional unless it is clearly
  operated by a government or university.

- A scholarly journal or scientific publisher is Academic/Research.

- A mission-oriented nonprofit whose primary function is advocacy,
  civil rights, or public-interest campaigning is NGO/Advocacy.

- A research institute whose primary activity is research rather than
  advocacy is Academic/Research.

- Reddit, forums, community discussion sites, and personal blogs are
  Community/Lived Experience.

- Do not use Other/Unclear merely because you are slightly uncertain.
  If the institutional function is reasonably identifiable, choose the
  best-fitting substantive category.

- Use Other/Unclear only when there is genuinely insufficient evidence.
""".strip()


SYSTEM_PROMPT = """
You are a careful research annotator classifying web publishers for
an academic study.

Your task is to identify the institutional type of the organization
that publishes or hosts each cited web source.

Follow the supplied taxonomy exactly.

Do not classify political ideology, stance, sentiment, topic, or
quality.

Return ONLY a valid JSON object.
Do not use Markdown.
Do not add text before or after the JSON.
""".strip()


# ============================================================
# 20. BUILD USER PROMPT
# ============================================================

def source_field(
    row,
    field
):

    value = row.get(
        field,
        ""
    )


    if pd.isna(
        value
    ):

        return ""


    return str(
        value
    )


def build_user_prompt(
    row
):

    return f"""
SOURCE TYPE TAXONOMY

{TAXONOMY_TEXT}

{DECISION_RULES}

SOURCE TO CLASSIFY

URL:
{source_field(row, "normalized_url")}

Hostname:
{source_field(row, "hostname")}

Registrable domain:
{source_field(row, "registrable_domain")}

Site name:
{source_field(row, "site_name")}

Page title:
{source_field(row, "page_title")}

Beginning of extracted page, if available:
{source_field(row, "source_excerpt")}

Return exactly one JSON object with this schema:

{{
  "category": "ONE EXACT TAXONOMY CATEGORY",
  "confidence": "high|medium|low",
  "rationale": "one short sentence explaining the publisher classification"
}}
""".strip()


# ============================================================
# 21. JSON PARSER
# ============================================================

def extract_json_object(
    text
):

    raw = str(
        text
        or ""
    ).strip()


    # Remove common Markdown fences if model ignored instruction.

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )


    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )


    # Direct parse first.

    try:

        parsed = json.loads(
            raw
        )

        if isinstance(
            parsed,
            dict,
        ):

            return parsed

    except Exception:

        pass


    # Find first JSON-looking object.

    first = raw.find(
        "{"
    )

    last = raw.rfind(
        "}"
    )


    if (
        first >= 0
        and
        last > first
    ):

        candidate = raw[
            first:
            last + 1
        ]


        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except Exception:

            pass


    return None


def validate_classifier_result(
    parsed
):

    if not isinstance(
        parsed,
        dict,
    ):

        return (
            False,
            None,
        )


    category = str(
        parsed.get(
            "category",
            ""
        )
    ).strip()


    confidence = str(
        parsed.get(
            "confidence",
            ""
        )
    ).strip().lower()


    rationale = str(
        parsed.get(
            "rationale",
            ""
        )
    ).strip()


    # Exact category is preferred.

    matched_category = None


    for candidate in SOURCE_TYPES:

        if (
            category.casefold()
            ==
            candidate.casefold()
        ):

            matched_category = (
                candidate
            )

            break


    if (
        matched_category
        is None
    ):

        return (
            False,
            None,
        )


    if confidence not in {

        "high",
        "medium",
        "low",

    }:

        return (
            False,
            None,
        )


    if not rationale:

        rationale = (
            "No rationale provided."
        )


    return (

        True,

        {

            "category":
                matched_category,

            "confidence":
                confidence,

            "rationale":
                rationale,
        },
    )


# ============================================================
# 22. SAVE CHECKPOINT
# ============================================================

def save_checkpoint():

    temp = (
        CHECKPOINT_FILE.parent
        /
        (
            CHECKPOINT_FILE.name
            + ".tmp"
        )
    )


    source_catalog.to_csv(
        temp,
        index=False,
    )


    os.replace(
        temp,
        CHECKPOINT_FILE,
    )


# ============================================================
# 23. LOAD VLLM + TOKENIZER
# ============================================================

pending_mask = (
    source_catalog[
        "source_type"
    ]
    .isna()
)


pending_indices = (
    source_catalog[
        pending_mask
    ]
    .index
    .tolist()
)


print()
print("=" * 80)
print("SOURCE TYPE CLASSIFICATION V2")
print("=" * 80)


print(
    f"\nModel: "
    f"{MODEL_NAME}"
)


print(
    f"Already handled by "
    f"rules/resume: "
    f"{len(source_catalog) - len(pending_indices):,}"
)


print(
    f"Pending LLM classifications: "
    f"{len(pending_indices):,}"
)


if pending_indices:

    from transformers import (
        AutoTokenizer,
    )

    from vllm import (
        LLM,
        SamplingParams,
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(

            MODEL_NAME,

            trust_remote_code=True,
        )
    )


    print()
    print(
        "Loading vLLM model..."
    )


    llm = LLM(

        model=MODEL_NAME,

        trust_remote_code=True,

        dtype="auto",

        gpu_memory_utilization=
            GPU_MEMORY_UTILIZATION,

        max_model_len=
            MAX_MODEL_LEN,

        max_num_seqs=
            MAX_NUM_SEQS,
    )


    sampling_params = SamplingParams(

        temperature=
            TEMPERATURE,

        top_p=1.0,

        max_tokens=
            MAX_OUTPUT_TOKENS,
    )


    # ========================================================
    # CHAT TEMPLATE
    # ========================================================

    def render_chat_prompt(
        user_prompt
    ):

        messages = [

            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT,
            },

            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ]


        return (
            tokenizer
            .apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=True,
            )
        )


    # ========================================================
    # FIRST CLASSIFICATION PASS
    # ========================================================

    invalid_after_first_pass = []


    for start in range(
        0,
        len(pending_indices),
        CLASSIFIER_BATCH_SIZE,
    ):

        batch_indices = (
            pending_indices[
                start:
                start
                + CLASSIFIER_BATCH_SIZE
            ]
        )


        prompts = [

            render_chat_prompt(

                build_user_prompt(
                    source_catalog.loc[
                        index
                    ]
                )
            )

            for index
            in batch_indices
        ]


        outputs = llm.generate(

            prompts,

            sampling_params,

            use_tqdm=True,
        )


        for index, output in zip(
            batch_indices,
            outputs,
        ):

            raw_output = (

                output
                .outputs[0]
                .text
                .strip()
            )


            parsed = extract_json_object(
                raw_output
            )


            valid, result = (
                validate_classifier_result(
                    parsed
                )
            )


            source_catalog.at[
                index,
                "classifier_raw_output"
            ] = raw_output


            if valid:

                source_catalog.at[
                    index,
                    "source_type"
                ] = result[
                    "category"
                ]


                source_catalog.at[
                    index,
                    "classification_confidence"
                ] = result[
                    "confidence"
                ]


                source_catalog.at[
                    index,
                    "classification_method"
                ] = "llm"


                source_catalog.at[
                    index,
                    "classification_status"
                ] = "llm_valid"


                source_catalog.at[
                    index,
                    "classification_rationale"
                ] = result[
                    "rationale"
                ]


            else:

                source_catalog.at[
                    index,
                    "classification_method"
                ] = "llm"


                source_catalog.at[
                    index,
                    "classification_status"
                ] = "first_pass_invalid"


                invalid_after_first_pass.append(
                    index
                )


        if SAVE_AFTER_EVERY_BATCH:

            save_checkpoint()


        print(
            f"First pass: "
            f"{min(start + CLASSIFIER_BATCH_SIZE, len(pending_indices)):,}"
            f"/{len(pending_indices):,}"
        )


    # ========================================================
    # RETRY INVALID OUTPUTS
    # ========================================================

    print()
    print(
        f"Invalid after first pass: "
        f"{len(invalid_after_first_pass):,}"
    )


    def build_retry_prompt(
        row,
        previous_output,
    ):

        base = build_user_prompt(
            row
        )


        return f"""
{base}

Your previous answer could not be parsed because it did not follow
the required JSON schema.

PREVIOUS INVALID OUTPUT:
{previous_output}

Return ONLY the corrected JSON object.

No Markdown.
No explanation outside the JSON.
""".strip()


    still_invalid = []


    for start in range(
        0,
        len(invalid_after_first_pass),
        CLASSIFIER_BATCH_SIZE,
    ):

        batch_indices = (
            invalid_after_first_pass[
                start:
                start
                + CLASSIFIER_BATCH_SIZE
            ]
        )


        prompts = []


        for index in (
            batch_indices
        ):

            previous_output = str(

                source_catalog.at[
                    index,
                    "classifier_raw_output"
                ]

                or ""
            )


            prompts.append(

                render_chat_prompt(

                    build_retry_prompt(

                        source_catalog.loc[
                            index
                        ],

                        previous_output,
                    )
                )
            )


        outputs = llm.generate(

            prompts,

            sampling_params,

            use_tqdm=True,
        )


        for index, output in zip(
            batch_indices,
            outputs,
        ):

            retry_output = (

                output
                .outputs[0]
                .text
                .strip()
            )


            parsed = extract_json_object(
                retry_output
            )


            valid, result = (
                validate_classifier_result(
                    parsed
                )
            )


            previous_output = str(

                source_catalog.at[
                    index,
                    "classifier_raw_output"
                ]

                or ""
            )


            source_catalog.at[
                index,
                "classifier_raw_output"
            ] = (

                "FIRST PASS:\n"
                + previous_output
                + "\n\nRETRY:\n"
                + retry_output
            )


            if valid:

                source_catalog.at[
                    index,
                    "source_type"
                ] = result[
                    "category"
                ]


                source_catalog.at[
                    index,
                    "classification_confidence"
                ] = result[
                    "confidence"
                ]


                source_catalog.at[
                    index,
                    "classification_method"
                ] = "llm_retry"


                source_catalog.at[
                    index,
                    "classification_status"
                ] = "llm_retry_valid"


                source_catalog.at[
                    index,
                    "classification_rationale"
                ] = result[
                    "rationale"
                ]


            else:

                source_catalog.at[
                    index,
                    "classification_method"
                ] = "llm_retry"


                source_catalog.at[
                    index,
                    "classification_status"
                ] = "parse_failed"


                source_catalog.at[
                    index,
                    "classification_confidence"
                ] = "low"


                # IMPORTANT:
                # This label allows the table to be saved,
                # but classification_status keeps parse failure
                # separate from a genuine model choice of Other.

                source_catalog.at[
                    index,
                    "source_type"
                ] = "Other/Unclear"


                source_catalog.at[
                    index,
                    "classification_rationale"
                ] = (
                    "Classifier output remained "
                    "unparseable after retry."
                )


                still_invalid.append(
                    index
                )


        if SAVE_AFTER_EVERY_BATCH:

            save_checkpoint()


    print(
        f"Still invalid after retry: "
        f"{len(still_invalid):,}"
    )


# ============================================================
# 24. FINAL SAVE
# ============================================================

source_catalog.to_csv(
    CLASSIFICATION_FILE,
    index=False,
)


save_checkpoint()


# ============================================================
# 25. CLASSIFICATION SUMMARY
# ============================================================

print()
print("=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)


type_counts = (

    source_catalog[
        "source_type"
    ]
    .value_counts()
    .reindex(
        SOURCE_TYPES,
        fill_value=0,
    )
)


print()
print(
    type_counts.to_string()
)


print()
print(
    "Classification status:"
)


print(

    source_catalog[
        "classification_status"
    ]
    .fillna(
        "missing"
    )
    .value_counts()
    .to_string()
)


print()
print(
    "Classification methods:"
)


print(

    source_catalog[
        "classification_method"
    ]
    .fillna(
        "missing"
    )
    .value_counts()
    .to_string()
)


# ============================================================
# 26. DOMAIN CONSISTENCY AUDIT
# ============================================================

MULTITENANT_DOMAINS = {

    "medium.com",
    "substack.com",
    "wordpress.com",
    "blogspot.com",
}


domain_type_counts = (

    source_catalog[
        ~source_catalog[
            "registrable_domain"
        ]
        .isin(
            MULTITENANT_DOMAINS
        )
    ]
    .groupby(
        "registrable_domain"
    )[
        "source_type"
    ]
    .nunique()
)


conflicting_domains = (

    domain_type_counts[
        domain_type_counts > 1
    ]
    .index
)


domain_conflicts = (

    source_catalog[
        source_catalog[
            "registrable_domain"
        ]
        .isin(
            conflicting_domains
        )
    ]
    .sort_values(
        [
            "registrable_domain",
            "source_type",
        ]
    )
)


domain_conflicts.to_csv(

    RESULTS_DIR
    / "source_type_domain_conflicts.csv",

    index=False,
)


# ============================================================
# 27. MANUAL VALIDATION SAMPLE
# ============================================================

validation_parts = []


for category in SOURCE_TYPES:

    subset = source_catalog[
        source_catalog[
            "source_type"
        ]
        == category
    ]


    if len(
        subset
    ) == 0:

        continue


    n = min(
        VALIDATION_PER_CATEGORY,
        len(
            subset
        ),
    )


    validation_parts.append(

        subset.sample(

            n=n,

            random_state=stable_seed(
                "validation::"
                + category
            ),
        )
    )


validation = (

    pd.concat(
        validation_parts,
        ignore_index=True,
    )

    if validation_parts

    else pd.DataFrame()
)


if INCLUDE_ALL_LOW_CONFIDENCE:

    low = source_catalog[
        source_catalog[
            "classification_confidence"
        ]
        == "low"
    ]


    validation = pd.concat(
        [
            validation,
            low,
        ],
        ignore_index=True,
    )


if INCLUDE_ALL_OTHER:

    other = source_catalog[
        (
            source_catalog[
                "source_type"
            ]
            == "Other/Unclear"
        )
        &
        (
            source_catalog[
                "classification_status"
            ]
            != "parse_failed"
        )
    ]


    validation = pd.concat(
        [
            validation,
            other,
        ],
        ignore_index=True,
    )


if INCLUDE_ALL_PARSE_FAILURES:

    failed = source_catalog[
        source_catalog[
            "classification_status"
        ]
        == "parse_failed"
    ]


    validation = pd.concat(
        [
            validation,
            failed,
        ],
        ignore_index=True,
    )


if INCLUDE_DOMAIN_CONFLICTS:

    validation = pd.concat(
        [
            validation,
            domain_conflicts,
        ],
        ignore_index=True,
    )


validation = (

    validation
    .drop_duplicates(
        subset=[
            "source_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


validation[
    "human_label_1"
] = ""


validation[
    "human_label_2"
] = ""


validation[
    "adjudicated_label"
] = ""


validation[
    "validation_notes"
] = ""


validation.to_csv(

    RESULTS_DIR
    / "source_type_manual_validation_sample.csv",

    index=False,
)


# ============================================================
# 28. SANITY CHECKS
# ============================================================

n_sources = len(
    source_catalog
)


parse_failure_count = int(

    (
        source_catalog[
            "classification_status"
        ]
        == "parse_failed"
    )
    .sum()
)


parse_failure_rate = (

    parse_failure_count
    / n_sources

    if n_sources
    else np.nan
)


other_count = int(

    (
        source_catalog[
            "source_type"
        ]
        == "Other/Unclear"
    )
    .sum()
)


other_share = (

    other_count
    / n_sources

    if n_sources
    else np.nan
)


sanity_failures = []


if (
    np.isfinite(
        parse_failure_rate
    )
    and
    parse_failure_rate
    >
    MAX_PARSE_FAILURE_RATE
):

    sanity_failures.append(

        (
            "Parse failure rate "
            f"{parse_failure_rate:.2%} "
            f"> allowed "
            f"{MAX_PARSE_FAILURE_RATE:.2%}"
        )
    )


if (
    np.isfinite(
        other_share
    )
    and
    other_share
    >
    MAX_OTHER_SHARE
):

    sanity_failures.append(

        (
            "Other/Unclear share "
            f"{other_share:.2%} "
            f"> allowed "
            f"{MAX_OTHER_SHARE:.2%}"
        )
    )


for category in (
    REQUIRE_NONZERO_CATEGORIES
):

    count = int(
        type_counts.get(
            category,
            0,
        )
    )


    if count == 0:

        sanity_failures.append(

            (
                f"Category "
                f"{category!r} "
                f"has zero sources."
            )
        )


sanity_summary = {

    "n_unique_sources":
        n_sources,

    "parse_failure_count":
        parse_failure_count,

    "parse_failure_rate":
        parse_failure_rate,

    "other_count":
        other_count,

    "other_share":
        other_share,

    "category_counts":
        {
            key:
                int(value)

            for key, value
            in type_counts.items()
        },

    "sanity_failures":
        sanity_failures,

    "passed":
        (
            len(
                sanity_failures
            )
            == 0
        ),
}


with (
    RESULTS_DIR
    / "classification_sanity_summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        sanity_summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 80)
print("SANITY CHECK")
print("=" * 80)


print(
    f"\nParse failure rate: "
    f"{parse_failure_rate:.2%}"
)


print(
    f"Other/Unclear share: "
    f"{other_share:.2%}"
)


if sanity_failures:

    print()
    print(
        "SANITY CHECK FAILURES:"
    )


    for failure in (
        sanity_failures
    ):

        print(
            f"  - {failure}"
        )


    if STRICT_SANITY_CHECKS:

        raise RuntimeError(
            "\nClassification sanity checks failed.\n"
            "Statistics were NOT executed.\n"
            "Inspect source_type_catalog_v2.csv and "
            "source_type_manual_validation_sample.csv."
        )


else:

    print(
        "\nClassification sanity checks passed."
    )


# ============================================================
# 29. ATTACH SOURCE TYPES TO QUERIES
# ============================================================

typed_usage = (

    usage_unique[
        [
            "file",
            "source_id",
            "source_order",
            "normalized_url",
        ]
    ]
    .merge(

        source_catalog[
            [
                "source_id",
                "source_type",
                "classification_confidence",
                "classification_status",
                "registrable_domain",
            ]
        ],

        on="source_id",

        how="left",
    )
)


typed_usage.to_csv(

    RESULTS_DIR
    / "query_source_types.csv",

    index=False,
)


# ============================================================
# 30. QUERY-LEVEL COUNTS
# ============================================================

type_counts_query = (

    typed_usage
    .groupby(
        [
            "file",
            "source_type",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


for category in SOURCE_TYPES:

    if category not in (
        type_counts_query.columns
    ):

        type_counts_query[
            category
        ] = 0


type_counts_query = (

    type_counts_query[
        SOURCE_TYPES
    ]
    .reset_index()
)


rename_map = {

    category:
        (
            "count_"
            + SOURCE_TYPE_SLUGS[
                category
            ]
        )

    for category
    in SOURCE_TYPES
}


type_counts_query = (
    type_counts_query.rename(
        columns=rename_map
    )
)


query_composition = (

    query_index
    .merge(
        type_counts_query,
        on="file",
        how="left",
    )
)


count_columns = [

    "count_"
    + SOURCE_TYPE_SLUGS[
        category
    ]

    for category
    in SOURCE_TYPES
]


query_composition[
    count_columns
] = (

    query_composition[
        count_columns
    ]
    .fillna(0)
)


query_composition[
    "n_unique_cited_sources"
] = (

    query_composition[
        count_columns
    ]
    .sum(
        axis=1
    )
)


# ============================================================
# 31. SOURCE TYPE PROPORTIONS
# ============================================================

share_columns = []


for category in SOURCE_TYPES:

    slug = SOURCE_TYPE_SLUGS[
        category
    ]


    count_column = (
        "count_"
        + slug
    )


    share_column = (
        "share_"
        + slug
    )


    share_columns.append(
        share_column
    )


    query_composition[
        share_column
    ] = np.where(

        query_composition[
            "n_unique_cited_sources"
        ]
        > 0,

        query_composition[
            count_column
        ]
        /
        query_composition[
            "n_unique_cited_sources"
        ],

        np.nan,
    )


# ============================================================
# 32. KNOWN-CATEGORY-ONLY ROBUSTNESS
# ============================================================

other_count_column = (
    "count_"
    + SOURCE_TYPE_SLUGS[
        "Other/Unclear"
    ]
)


query_composition[
    "n_known_type_sources"
] = (

    query_composition[
        "n_unique_cited_sources"
    ]
    -
    query_composition[
        other_count_column
    ]
)


known_share_columns = []


for category in (
    KNOWN_SOURCE_TYPES
):

    slug = SOURCE_TYPE_SLUGS[
        category
    ]


    count_column = (
        "count_"
        + slug
    )


    known_share_column = (
        "known_share_"
        + slug
    )


    known_share_columns.append(
        known_share_column
    )


    query_composition[
        known_share_column
    ] = np.where(

        query_composition[
            "n_known_type_sources"
        ]
        > 0,

        query_composition[
            count_column
        ]
        /
        query_composition[
            "n_known_type_sources"
        ],

        np.nan,
    )


query_composition[
    "known_classification_share"
] = np.where(

    query_composition[
        "n_unique_cited_sources"
    ]
    > 0,

    query_composition[
        "n_known_type_sources"
    ]
    /
    query_composition[
        "n_unique_cited_sources"
    ],

    np.nan,
)


# ============================================================
# 33. CONDITION KIND
# ============================================================

def condition_kind(
    value
):

    value = str(
        value
        or ""
    ).strip().lower()


    if value in (
        FOCAL_CONDITIONS
    ):

        return "focal"


    if value in (
        COMPARISON_CONDITIONS
    ):

        return "comparison"


    if value in (
        CONTROL_CONDITIONS
    ):

        return "control"


    return value


query_composition[
    "condition_kind"
] = (

    query_composition[
        "condition"
    ]
    .apply(
        condition_kind
    )
)


# ============================================================
# 34. MATCH GENERIC CONDITION
# ============================================================

control = query_composition[
    query_composition[
        "condition_kind"
    ]
    == "control"
].copy()


generic_columns = (

    [
        "outcome_id",
        "file",
        "n_unique_cited_sources",
        "n_known_type_sources",
    ]

    +
    share_columns

    +
    known_share_columns
)


generic = control[
    generic_columns
].copy()


rename_generic = {

    "file":
        "generic_file",

    "n_unique_cited_sources":
        "generic_n_sources",

    "n_known_type_sources":
        "generic_n_known_sources",
}


for column in (
    share_columns
    +
    known_share_columns
):

    rename_generic[
        column
    ] = (
        "generic_"
        + column
    )


generic = generic.rename(
    columns=rename_generic
)


query_composition = (
    query_composition
    .merge(
        generic,
        on="outcome_id",
        how="left",
    )
)


# ============================================================
# 35. JENSEN-SHANNON DISTANCE
# ============================================================

def js_distance_from_columns(
    row,
    current_columns,
    generic_prefix,
):

    p = np.asarray(
        [
            row[
                column
            ]

            for column
            in current_columns
        ],
        dtype=float,
    )


    q = np.asarray(
        [
            row[
                generic_prefix
                + column
            ]

            for column
            in current_columns
        ],
        dtype=float,
    )


    if (
        not np.all(
            np.isfinite(
                p
            )
        )
        or
        not np.all(
            np.isfinite(
                q
            )
        )
    ):

        return np.nan


    if (
        p.sum() <= 0
        or
        q.sum() <= 0
    ):

        return np.nan


    return float(
        jensenshannon(
            p,
            q,
            base=2.0,
        )
    )


# ------------------------------------------------------------
# Primary: all seven categories.
# ------------------------------------------------------------

query_composition[
    "source_type_js_distance_to_generic"
] = (

    query_composition.apply(

        lambda row:
            js_distance_from_columns(
                row,
                share_columns,
                "generic_",
            ),

        axis=1,
    )
)


# ------------------------------------------------------------
# Robustness:
# remove Other/Unclear and renormalize known categories.
# ------------------------------------------------------------

query_composition[
    "source_type_js_known_only_to_generic"
] = (

    query_composition.apply(

        lambda row:
            js_distance_from_columns(
                row,
                known_share_columns,
                "generic_",
            ),

        axis=1,
    )
)


# ============================================================
# 36. EXPLICIT-vs-GENERIC TYPE DELTAS
# ============================================================

for share_column in (
    share_columns
):

    query_composition[
        "delta_generic_"
        + share_column
    ] = (

        query_composition[
            share_column
        ]
        -
        query_composition[
            "generic_"
            + share_column
        ]
    )


query_composition.to_csv(

    RESULTS_DIR
    / "query_source_type_composition.csv",

    index=False,
)


# ============================================================
# 37. GROUP DESCRIPTIVES
# ============================================================

aggregation = {

    "n_queries":
        (
            "file",
            "size",
        ),

    "mean_n_sources":
        (
            "n_unique_cited_sources",
            "mean",
        ),

    "mean_known_classification_share":
        (
            "known_classification_share",
            "mean",
        ),

    "mean_js_distance_to_generic":
        (
            "source_type_js_distance_to_generic",
            "mean",
        ),

    "mean_js_known_only_to_generic":
        (
            "source_type_js_known_only_to_generic",
            "mean",
        ),
}


for category in (
    SOURCE_TYPES
):

    slug = SOURCE_TYPE_SLUGS[
        category
    ]


    aggregation[
        "mean_share_"
        + slug
    ] = (

        "share_"
        + slug,

        "mean",
    )


group_summary = (

    query_composition
    .groupby(
        [
            "dimension",
            "condition_kind",
            "group",
        ],
        dropna=False,
    )
    .agg(
        **aggregation
    )
    .reset_index()
)


group_summary.to_csv(

    RESULTS_DIR
    / "source_type_composition_by_group.csv",

    index=False,
)


# ============================================================
# 38. STATISTICAL HELPERS
# ============================================================

def paired_rank_biserial(
    differences
):

    d = np.asarray(
        differences,
        dtype=float,
    )


    d = d[
        np.isfinite(
            d
        )
    ]


    d = d[
        d != 0
    ]


    if len(d) == 0:

        return 0.0


    ranks = rankdata(
        np.abs(
            d
        ),
        method="average",
    )


    positive = ranks[
        d > 0
    ].sum()


    negative = ranks[
        d < 0
    ].sum()


    return float(
        (
            positive
            - negative
        )
        /
        (
            positive
            + negative
        )
    )


def bootstrap_ci(
    differences,
    seed,
):

    d = np.asarray(
        differences,
        dtype=float,
    )


    d = d[
        np.isfinite(
            d
        )
    ]


    if len(d) == 0:

        return (
            np.nan,
            np.nan,
        )


    rng = np.random.default_rng(
        seed
    )


    indices = rng.integers(

        0,
        len(d),

        size=(
            N_BOOTSTRAP,
            len(d),
        ),
    )


    means = (
        d[
            indices
        ]
        .mean(
            axis=1
        )
    )


    return (

        float(
            np.percentile(
                means,
                2.5,
            )
        ),

        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def signflip_test(
    differences,
    seed,
):

    d = np.asarray(
        differences,
        dtype=float,
    )


    d = d[
        np.isfinite(
            d
        )
    ]


    d = d[
        d != 0
    ]


    n = len(
        d
    )


    if n == 0:

        return (
            1.0,
            "all_zero",
        )


    observed = abs(
        d.mean()
    )


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
                dtype=np.uint64,
            )
        )


        batch_size = 10_000


        for start in range(
            0,
            total,
            batch_size,
        ):

            stop = min(
                start
                + batch_size,
                total,
            )


            integers = np.arange(
                start,
                stop,
                dtype=np.uint64,
            )[:, None]


            signs = (

                (
                    (
                        integers
                        & powers
                    )
                    > 0
                )
                .astype(float)
                * 2.0
                - 1.0
            )


            values = np.abs(

                (
                    signs
                    * d
                )
                .mean(
                    axis=1
                )
            )


            extreme += int(

                np.sum(
                    values
                    >=
                    observed
                    - 1e-15
                )
            )


        return (

            float(
                extreme
                / total
            ),

            (
                f"exact_signflip_"
                f"2^{n}"
            ),
        )


    rng = np.random.default_rng(
        seed
    )


    extreme = 0

    completed = 0

    batch_size = 10_000


    while (
        completed
        <
        N_PERMUTATIONS
    ):

        batch = min(

            batch_size,

            N_PERMUTATIONS
            - completed,
        )


        signs = rng.choice(

            [
                -1.0,
                1.0,
            ],

            size=(
                batch,
                n,
            ),
        )


        values = np.abs(

            (
                signs
                * d
            )
            .mean(
                axis=1
            )
        )


        extreme += int(

            np.sum(
                values
                >=
                observed
                - 1e-15
            )
        )


        completed += batch


    p = (
        extreme + 1
    ) / (
        N_PERMUTATIONS + 1
    )


    return (

        float(
            p
        ),

        (
            "monte_carlo_signflip_"
            f"{N_PERMUTATIONS}"
        ),
    )


def safe_wilcoxon(
    differences
):

    d = np.asarray(
        differences,
        dtype=float,
    )


    d = d[
        np.isfinite(
            d
        )
    ]


    if (
        len(d) == 0
        or
        np.allclose(
            d,
            0
        )
    ):

        return (
            0.0,
            1.0,
        )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        result = wilcoxon(

            d,

            alternative="two-sided",

            zero_method="wilcox",

            method="auto",
        )


    return (

        float(
            result.statistic
        ),

        float(
            result.pvalue
        ),
    )


def summarize_pairs(
    focal,
    comparison,
    label,
):

    focal = np.asarray(
        focal,
        dtype=float,
    )


    comparison = np.asarray(
        comparison,
        dtype=float,
    )


    valid = (
        np.isfinite(
            focal
        )
        &
        np.isfinite(
            comparison
        )
    )


    focal = focal[
        valid
    ]


    comparison = comparison[
        valid
    ]


    if len(
        focal
    ) == 0:

        return None


    differences = (
        focal
        -
        comparison
    )


    p_perm, method = (
        signflip_test(

            differences,

            seed=stable_seed(
                label
            ),
        )
    )


    ci_low, ci_high = (
        bootstrap_ci(

            differences,

            seed=stable_seed(
                "bootstrap::"
                + label
            ),
        )
    )


    W, p_wilcoxon = (
        safe_wilcoxon(
            differences
        )
    )


    return {

        "n":
            len(
                differences
            ),

        "focal_mean":
            float(
                focal.mean()
            ),

        "comparison_mean":
            float(
                comparison.mean()
            ),

        "mean_difference":
            float(
                differences.mean()
            ),

        "median_difference":
            float(
                np.median(
                    differences
                )
            ),

        "bootstrap_ci_low":
            ci_low,

        "bootstrap_ci_high":
            ci_high,

        "rank_biserial":
            paired_rank_biserial(
                differences
            ),

        "p_permutation_raw":
            p_perm,

        "permutation_method":
            method,

        "wilcoxon_W":
            W,

        "p_wilcoxon_raw":
            p_wilcoxon,
    }


# ============================================================
# 39. GENERIC FUNCTION FOR DIMENSION TEST
# ============================================================

def run_dimension_tests(
    dataframe,
    metric,
    correction_name,
):

    explicit = dataframe[
        dataframe[
            "condition_kind"
        ]
        .isin(
            [
                "focal",
                "comparison",
            ]
        )
    ]


    rows = []


    for dimension in sorted(
        explicit[
            "dimension"
        ]
        .dropna()
        .unique()
    ):

        subset = explicit[
            explicit[
                "dimension"
            ]
            == dimension
        ]


        focal = (

            subset[
                subset[
                    "condition_kind"
                ]
                == "focal"
            ][
                [
                    "outcome_id",
                    "group",
                    metric,
                ]
            ]
            .rename(
                columns={

                    "group":
                        "focal_group",

                    metric:
                        "focal_value",
                }
            )
        )


        comparison = (

            subset[
                subset[
                    "condition_kind"
                ]
                == "comparison"
            ][
                [
                    "outcome_id",
                    "group",
                    metric,
                ]
            ]
            .rename(
                columns={

                    "group":
                        "comparison_group",

                    metric:
                        "comparison_value",
                }
            )
        )


        paired = focal.merge(

            comparison,

            on="outcome_id",

            how="inner",
        )


        stats = summarize_pairs(

            paired[
                "focal_value"
            ],

            paired[
                "comparison_value"
            ],

            label=(
                metric
                + "::"
                + dimension
            ),
        )


        if stats is None:

            continue


        stats.update({

            "dimension":
                dimension,

            "metric":
                metric,

            "focal_group":
                (
                    paired[
                        "focal_group"
                    ].iloc[0]
                ),

            "comparison_group":
                (
                    paired[
                        "comparison_group"
                    ].iloc[0]
                ),
        })


        rows.append(
            stats
        )


    result = pd.DataFrame(
        rows
    )


    if len(
        result
    ):

        reject, p_adj, _, _ = (
            multipletests(

                result[
                    "p_permutation_raw"
                ],

                alpha=ALPHA,

                method="holm",
            )
        )


        result[
            correction_name
        ] = p_adj


        result[
            "significant_holm"
        ] = reject


    return result


# ============================================================
# 40. PRIMARY:
# INSTITUTIONAL COMPOSITION DISTANCE TO GENERIC
# ============================================================

primary_tests = (
    run_dimension_tests(

        query_composition,

        (
            "source_type_"
            "js_distance_to_generic"
        ),

        "p_permutation_holm",
    )
)


primary_tests.to_csv(

    RESULTS_DIR
    / "primary_source_type_composition_displacement.csv",

    index=False,
)


# ============================================================
# 41. ROBUSTNESS:
# EXCLUDE OTHER/UNCLEAR
# ============================================================

known_only_tests = (
    run_dimension_tests(

        query_composition,

        (
            "source_type_"
            "js_known_only_to_generic"
        ),

        "p_permutation_holm",
    )
)


known_only_tests.to_csv(

    RESULTS_DIR
    / "robustness_known_types_only_js.csv",

    index=False,
)


# ============================================================
# 42. POST-HOC SOURCE-TYPE SHARE TESTS
# ============================================================

explicit = query_composition[
    query_composition[
        "condition_kind"
    ]
    .isin(
        [
            "focal",
            "comparison",
        ]
    )
]


posthoc_rows = []


for category in (
    SOURCE_TYPES
):

    slug = (
        SOURCE_TYPE_SLUGS[
            category
        ]
    )


    metric = (
        "share_"
        + slug
    )


    for dimension in sorted(
        explicit[
            "dimension"
        ]
        .dropna()
        .unique()
    ):

        subset = explicit[
            explicit[
                "dimension"
            ]
            == dimension
        ]


        focal = (

            subset[
                subset[
                    "condition_kind"
                ]
                == "focal"
            ][
                [
                    "outcome_id",
                    "group",
                    metric,
                ]
            ]
            .rename(
                columns={

                    "group":
                        "focal_group",

                    metric:
                        "focal_value",
                }
            )
        )


        comparison = (

            subset[
                subset[
                    "condition_kind"
                ]
                == "comparison"
            ][
                [
                    "outcome_id",
                    "group",
                    metric,
                ]
            ]
            .rename(
                columns={

                    "group":
                        "comparison_group",

                    metric:
                        "comparison_value",
                }
            )
        )


        paired = focal.merge(

            comparison,

            on="outcome_id",

            how="inner",
        )


        stats = summarize_pairs(

            paired[
                "focal_value"
            ],

            paired[
                "comparison_value"
            ],

            label=(
                "share::"
                + category
                + "::"
                + dimension
            ),
        )


        if stats is None:

            continue


        stats.update({

            "source_type":
                category,

            "dimension":
                dimension,

            "metric":
                metric,

            "focal_group":
                (
                    paired[
                        "focal_group"
                    ].iloc[0]
                ),

            "comparison_group":
                (
                    paired[
                        "comparison_group"
                    ].iloc[0]
                ),
        })


        posthoc_rows.append(
            stats
        )


posthoc = pd.DataFrame(
    posthoc_rows
)


if len(
    posthoc
):

    reject, p_adj, _, _ = (
        multipletests(

            posthoc[
                "p_permutation_raw"
            ],

            alpha=ALPHA,

            method="holm",
        )
    )


    posthoc[
        "p_permutation_holm_42"
    ] = p_adj


    posthoc[
        "significant_holm_42"
    ] = reject


posthoc.to_csv(

    RESULTS_DIR
    / "posthoc_source_type_share_differences.csv",

    index=False,
)


# ============================================================
# 43. EXPLICIT vs GENERIC DESCRIPTIVES
# ============================================================

delta_columns = [

    "delta_generic_"
    + share_column

    for share_column
    in share_columns
]


explicit_generic = (

    query_composition[
        query_composition[
            "condition_kind"
        ]
        != "control"
    ]
    .groupby(
        [
            "dimension",
            "condition_kind",
            "group",
        ]
    )[
        delta_columns
    ]
    .mean()
    .reset_index()
)


explicit_generic.to_csv(

    RESULTS_DIR
    / "explicit_vs_generic_source_type_deltas.csv",

    index=False,
)


# ============================================================
# 44. AGGREGATE COMPOSITION EFFECT
# ============================================================

pair_rows = []


for dimension in sorted(
    explicit[
        "dimension"
    ]
    .dropna()
    .unique()
):

    subset = explicit[
        explicit[
            "dimension"
        ]
        == dimension
    ]


    focal = (

        subset[
            subset[
                "condition_kind"
            ]
            == "focal"
        ][
            [
                "outcome_id",
                "source_type_js_distance_to_generic",
            ]
        ]
        .rename(
            columns={
                "source_type_js_distance_to_generic":
                    "focal_value"
            }
        )
    )


    comparison = (

        subset[
            subset[
                "condition_kind"
            ]
            == "comparison"
        ][
            [
                "outcome_id",
                "source_type_js_distance_to_generic",
            ]
        ]
        .rename(
            columns={
                "source_type_js_distance_to_generic":
                    "comparison_value"
            }
        )
    )


    paired = focal.merge(

        comparison,

        on="outcome_id",

        how="inner",
    )


    paired[
        "difference"
    ] = (

        paired[
            "focal_value"
        ]
        -
        paired[
            "comparison_value"
        ]
    )


    paired[
        "dimension"
    ] = dimension


    pair_rows.append(
        paired
    )


if pair_rows:

    all_pairs = pd.concat(
        pair_rows,
        ignore_index=True,
    )


    outcome_effects = (

        all_pairs
        .groupby(
            "outcome_id"
        )
        .agg(

            mean_difference=(
                "difference",
                "mean",
            ),

            n_dimensions=(
                "dimension",
                "nunique",
            ),
        )
        .reset_index()
    )


    outcome_effects = (
        outcome_effects[
            outcome_effects[
                "n_dimensions"
            ]
            >= 2
        ]
    )


    aggregate_stats = (
        summarize_pairs(

            outcome_effects[
                "mean_difference"
            ],

            np.zeros(
                len(
                    outcome_effects
                )
            ),

            label=(
                "aggregate_source_type_js"
            ),
        )
    )


    if aggregate_stats:

        aggregate_stats[
            "n_outcomes"
        ] = len(
            outcome_effects
        )


        aggregate_stats[
            "mean_dimensions_per_outcome"
        ] = float(

            outcome_effects[
                "n_dimensions"
            ]
            .mean()
        )


        aggregate_df = pd.DataFrame(
            [
                aggregate_stats
            ]
        )


    else:

        aggregate_df = (
            pd.DataFrame()
        )


else:

    aggregate_df = (
        pd.DataFrame()
    )


aggregate_df.to_csv(

    RESULTS_DIR
    / "aggregate_source_type_composition_effect.csv",

    index=False,
)


# ============================================================
# 45. MASTER JSON
# ============================================================

def records(
    dataframe
):

    if (
        dataframe is None
        or
        len(dataframe) == 0
    ):

        return []


    return json.loads(

        dataframe.to_json(
            orient="records"
        )
    )


master = {

    "metadata": {

        "analysis":
            (
                "Institutional composition "
                "of cited sources V2"
            ),

        "collection":
            COLLECTION_VERSION,

        "classifier_model":
            MODEL_NAME,

        "taxonomy":
            SOURCE_TYPES,

        "unit":
            (
                "unique normalized cited URL "
                "within each query"
            ),

        "primary_metric":
            (
                "Jensen-Shannon distance "
                "between source-type composition "
                "and matched generic condition"
            ),

        "robustness_metric":
            (
                "Jensen-Shannon distance "
                "after excluding Other/Unclear"
            ),
    },


    "classification": {

        "n_unique_sources":
            n_sources,

        "parse_failure_rate":
            parse_failure_rate,

        "other_share":
            other_share,

        "category_counts":
            {

                key:
                    int(
                        value
                    )

                for key, value
                in type_counts.items()
            },

        "classification_status_counts":
            (
                source_catalog[
                    "classification_status"
                ]
                .fillna(
                    "missing"
                )
                .value_counts()
                .to_dict()
            ),

        "manual_validation_rows":
            len(
                validation
            ),

        "domain_conflict_rows":
            len(
                domain_conflicts
            ),
    },


    "primary_tests":
        records(
            primary_tests
        ),


    "known_types_only_robustness":
        records(
            known_only_tests
        ),


    "posthoc_tests":
        records(
            posthoc
        ),


    "aggregate_test":
        records(
            aggregate_df
        ),
}


with (
    RESULTS_DIR
    / "all_results_v2.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(

        master,

        f,

        indent=2,

        ensure_ascii=False,
    )


# ============================================================
# 46. FINAL PRINT
# ============================================================

print()
print("=" * 80)
print("SOURCE TYPE ANALYSIS V2 COMPLETE")
print("=" * 80)


print()
print(
    "FINAL SOURCE TYPE COUNTS"
)


print(
    type_counts.to_string()
)


print()
print(
    "PRIMARY COMPOSITION TESTS"
)


if len(
    primary_tests
):

    print(

        primary_tests[
            [
                "dimension",
                "n",
                "focal_mean",
                "comparison_mean",
                "mean_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "rank_biserial",
                "p_permutation_holm",
                "significant_holm",
            ]
        ]
        .round(
            5
        )
        .to_string(
            index=False
        )
    )


print()
print(
    "KNOWN-TYPES-ONLY ROBUSTNESS"
)


if len(
    known_only_tests
):

    print(

        known_only_tests[
            [
                "dimension",
                "n",
                "focal_mean",
                "comparison_mean",
                "mean_difference",
                "p_permutation_holm",
                "significant_holm",
            ]
        ]
        .round(
            5
        )
        .to_string(
            index=False
        )
    )


print()
print(
    "SIGNIFICANT POST-HOC SOURCE-TYPE EFFECTS"
)


if len(
    posthoc
):

    significant = posthoc[
        posthoc[
            "significant_holm_42"
        ]
    ]


    if len(
        significant
    ):

        print(

            significant[
                [
                    "source_type",
                    "dimension",
                    "n",
                    "focal_mean",
                    "comparison_mean",
                    "mean_difference",
                    "rank_biserial",
                    "p_permutation_holm_42",
                ]
            ]
            .round(
                5
            )
            .to_string(
                index=False
            )
        )


    else:

        print(
            "None survive Holm correction."
        )


print()
print(
    "Output directory:"
)

print(
    RESULTS_DIR
)


print()
print(
    "Inspect these first:"
)

for filename in [

    "classification_sanity_summary.json",

    "source_type_catalog_v2.csv",

    "source_type_manual_validation_sample.csv",

    "source_type_composition_by_group.csv",

    "primary_source_type_composition_displacement.csv",

    "robustness_known_types_only_js.csv",

    "posthoc_source_type_share_differences.csv",

    "all_results_v2.json",

]:

    print(
        f"  {filename}"
    )


print()
print(
    "Done."
)