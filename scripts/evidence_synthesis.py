# ============================================================
# GOOGLE AI OVERVIEW
# EVIDENCE -> SYNTHESIS ANALYSIS V2
#
# Main improvements over V1:
#
# 1. MULTILINGUAL embeddings
#    English sources <-> Portuguese/English AIOs
#
# 2. TOKEN-BASED CHUNKING
#    No dependence on broken sentence punctuation/headings
#
# 3. SOURCE-BALANCED evidence alignment
#
# 4. SUBJECT-NORMALIZED semantic matching
#    Reduces trivial identity-word matching
#
# 5. COVERAGE SENSITIVITY
#    40%, 50%, 60%, 70%, 80%
#
# 6. Reference analysis:
#    >= 50% of cited sources recovered
#    AND >= 3 usable cited sources
#
# 7. Paired outcome-level inference
#
# 8. Holm correction across six dimensions
#
# 9. Block-aware pipeline correlations
#    respecting the 21 repeated outcomes
#
# 10. IMPORTANT TERMINOLOGY FIX:
#     We call 1-cosine a "semantic gap",
#     NOT "claim omission".
#
#     Formal claim survival / omission should be measured
#     later with claim extraction + entailment.
#
# ============================================================


from pathlib import Path
import json
import math
import re
import warnings
import zlib

import numpy as np
import pandas as pd
import torch

from scipy.stats import (
    friedmanchisquare,
    rankdata,
    spearmanr,
    wilcoxon,
)

from statsmodels.stats.multitest import (
    multipletests,
)

from sentence_transformers import (
    SentenceTransformer,
)


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


RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "evidence_synthesis_analysis_dallas_v2"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. OPTIONAL PREVIOUS ANALYSES
# ============================================================

SOURCE_OVERLAP_FILE = (
    BASE_DIR
    / "results"
    / "source_overlap_analysis_dallas"
    / "source_overlap_vs_generic.csv"
)


SEMANTIC_FILE = (
    BASE_DIR
    / "results"
    / "semantic_embedding_analysis_dallas"
    / "group_vs_generic_semantic_metrics.csv"
)


# ============================================================
# 2. EMBEDDING MODEL
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-mpnet-base-v2"
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BATCH_SIZE = 64


# ============================================================
# 3. CHUNKING
# ============================================================

# Source pages:
SOURCE_CHUNK_TOKENS = 160

SOURCE_CHUNK_OVERLAP = 40


# AIO:
#
# Shorter chunks are intentional.
# We want units substantially smaller than the whole answer.
AIO_CHUNK_TOKENS = 96

AIO_CHUNK_OVERLAP = 24


# Number of most query-relevant passages retained PER source.
TOP_PASSAGES_PER_SOURCE = 3


# Avoid selecting three almost identical overlapping windows.
MAX_SELECTED_TOKEN_OVERLAP = 0.50


# Avoid pathological enormous PDFs dominating memory.
#
# ~500 * 160 tokens = up to ~80k source tokens represented.
MAX_CHUNKS_PER_SOURCE = 500


MIN_SOURCE_WORDS = 80


# ============================================================
# 4. COVERAGE DESIGN
# ============================================================

# Reference analysis.
REFERENCE_COVERAGE_THRESHOLD = 0.50


# Sensitivity analyses.
COVERAGE_THRESHOLDS = [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]


# Require some evidentiary diversity even if percentage
# coverage is high on a query with few citations.
MIN_AVAILABLE_SOURCES = 3


# Exploratory aggregate result:
# require an outcome to contribute paired effects from
# at least this many social dimensions.
MIN_DIMENSIONS_FOR_AGGREGATE = 2


# ============================================================
# 5. DESCRIPTIVE SIMILARITY THRESHOLD
# ============================================================

# This is NOT the inferential endpoint.
#
# Used only for descriptive:
# "what fraction of evidence has a strong semantic match?"
DESCRIPTIVE_MATCH_THRESHOLD = 0.65


# ============================================================
# 6. STATISTICS
# ============================================================

ALPHA = 0.05

RANDOM_SEED = 42


N_PERMUTATIONS = 200_000

N_BOOTSTRAP = 20_000


# Exact sign flips up to this number of non-zero pairs.
EXACT_SIGNFLIP_MAX_N = 18


# Block-aware pipeline correlations
N_BLOCK_PERMUTATIONS = 20_000

N_BLOCK_BOOTSTRAP = 5_000


np.random.seed(
    RANDOM_SEED
)


torch.manual_seed(
    RANDOM_SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# 7. GENERAL HELPERS
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
        np.integer,
    ):

        return int(
            value
        )


    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(
            value
        ):
            return None

        return float(
            value
        )


    if isinstance(
        value,
        np.bool_,
    ):

        return bool(
            value
        )


    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )


    if isinstance(
        value,
        dict,
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
        ),
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
# 8. TXT PARSING
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
                lines[0].strip(),
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
                lines[-1].strip(),
            )
        )
    ):

        lines.pop()


    return "\n".join(
        lines
    ).strip()


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


        key, value = (
            line.split(
                ":",
                1,
            )
        )


        result[
            key.strip()
        ] = value.strip()


    return result


# ============================================================
# 9. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = normalize_newlines(
        text
    )


    # Markdown links -> anchor text.
    text = re.sub(
        r"\[([^\]]+)\]"
        r"\(https?://[^)]+\)",
        r"\1",
        text,
    )


    # Bare URLs.
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )


    # Citation markers.
    text = re.sub(
        r"\[\s*\d+"
        r"(?:\s*,\s*\d+)*"
        r"\s*\]",
        " ",
        text,
    )


    # Markdown headers.
    text = re.sub(
        r"(?m)^\s*#+\s*",
        "",
        text,
    )


    # Basic bullet syntax.
    text = re.sub(
        r"(?m)^\s*"
        r"[\*\-•]+\s*",
        "",
        text,
    )


    # Keep line breaks because they may contain structure,
    # but collapse pathological whitespace.
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


# ============================================================
# 10. SUBJECT NORMALIZATION
# ============================================================

def subject_normalize(
    text,
    group,
):
    """
    Replace only the EXACT experimental label with "people".

    This is intentionally conservative.

    We do NOT automatically replace arbitrary identity
    synonyms because doing that would become a substantive
    text transformation rather than a simple lexical control.
    """

    text = (
        text
        or ""
    )


    group = str(
        group
        or ""
    ).strip()


    if (
        not group
        or
        group.casefold()
        == "people"
    ):

        return text


    pattern = re.compile(
        re.escape(
            group
        ),
        flags=re.IGNORECASE,
    )


    return pattern.sub(
        "people",
        text,
    )


# ============================================================
# 11. LOAD AIO DOCUMENTS
# ============================================================

print()
print("=" * 80)
print("LOADING GOOGLE AIO DOCUMENTS")
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


    query = get_section(
        raw,
        "QUERY",
        [
            "LINKS",
            "AI OVERVIEW TEXT",
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ],
    )


    aio_text = get_section(
        raw,
        "AI OVERVIEW TEXT",
        [
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ],
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

        "query":
            clean_text(
                query
            ),

        "aio_text":
            clean_text(
                aio_text
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


queries = pd.DataFrame(
    query_rows
)


queries[
    "query_index"
] = np.arange(
    len(
        queries
    )
)


print(
    f"\nQueries loaded: "
    f"{len(queries):,}"
)


print(
    f"Groups: "
    f"{queries['group'].nunique()}"
)


print(
    f"Outcomes: "
    f"{queries['outcome_id'].nunique()}"
)


# ============================================================
# 12. LOAD SOURCE CORPUS
# ============================================================

if not MANIFEST_FILE.exists():

    raise RuntimeError(
        f"Missing:\n{MANIFEST_FILE}"
    )


if not URL_USAGE_FILE.exists():

    raise RuntimeError(
        f"Missing:\n{URL_USAGE_FILE}"
    )


manifest = pd.read_csv(
    MANIFEST_FILE
)


usage = pd.read_csv(
    URL_USAGE_FILE
)


# Guarantee source IDs are strings.
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


print()
print(
    f"Manifest records: "
    f"{len(manifest):,}"
)


print(
    f"Source occurrences: "
    f"{len(usage):,}"
)


# ============================================================
# 13. UNIQUE SOURCE OCCURRENCE PER QUERY
# ============================================================

usage[
    "source_order"
] = pd.to_numeric(
    usage[
        "source_order"
    ],
    errors="coerce",
)


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


# ============================================================
# 14. LOAD USABLE SOURCE TEXTS
# ============================================================

usable_source_texts = {}

source_text_metadata = []


for _, row in (
    manifest.iterrows()
):

    source_id = str(
        row[
            "source_id"
        ]
    )


    extraction_status = str(
        row.get(
            "extraction_status",
            "",
        )
    )


    if (
        extraction_status
        != "success"
    ):

        continue


    text_path = row.get(
        "text_path"
    )


    if (
        not isinstance(
            text_path,
            str,
        )
        or not text_path
    ):

        continue


    path = Path(
        text_path
    )


    if not path.exists():

        continue


    text = clean_text(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )


    word_count = len(
        text.split()
    )


    if (
        word_count
        < MIN_SOURCE_WORDS
    ):

        continue


    usable_source_texts[
        source_id
    ] = text


    source_text_metadata.append({

        "source_id":
            source_id,

        "word_count":
            word_count,

        "normalized_url":
            row.get(
                "normalized_url",
                "",
            ),

        "registrable_domain":
            row.get(
                "registrable_domain",
                "",
            ),
    })


print(
    f"\nUsable source texts: "
    f"{len(usable_source_texts):,}"
)


# ============================================================
# 15. QUERY-LEVEL SOURCE COVERAGE
# ============================================================

coverage_rows = []


for _, query_row in (
    queries.iterrows()
):

    file = (
        query_row[
            "file"
        ]
    )


    cited = (
        usage_unique[
            usage_unique[
                "file"
            ]
            == file
        ]
    )


    cited_ids = (
        cited[
            "source_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    available_ids = [

        source_id

        for source_id
        in cited_ids

        if source_id
        in usable_source_texts
    ]


    n_cited = len(
        cited_ids
    )


    n_available = len(
        available_ids
    )


    fetch_coverage = (

        n_available
        / n_cited

        if n_cited > 0
        else np.nan
    )


    coverage_rows.append({

        "file":
            file,

        "n_cited_sources":
            n_cited,

        "n_available_sources":
            n_available,

        "fetch_coverage":
            fetch_coverage,
    })


coverage_df = pd.DataFrame(
    coverage_rows
)


queries = queries.merge(
    coverage_df,
    on="file",
    how="left",
)


# ============================================================
# 16. COVERAGE SUMMARY
# ============================================================

coverage_by_group = (
    queries
    .groupby(
        [
            "dimension",
            "condition",
            "group",
        ],
        dropna=False,
    )
    .agg(

        n_queries=(
            "file",
            "size",
        ),

        mean_fetch_coverage=(
            "fetch_coverage",
            "mean",
        ),

        median_fetch_coverage=(
            "fetch_coverage",
            "median",
        ),

        mean_cited_sources=(
            "n_cited_sources",
            "mean",
        ),

        mean_available_sources=(
            "n_available_sources",
            "mean",
        ),
    )
    .reset_index()
)


for threshold in (
    COVERAGE_THRESHOLDS
):

    column = (
        f"n_eligible_"
        f"{int(threshold * 100)}"
    )


    eligibility = (
        (
            queries[
                "fetch_coverage"
            ]
            >= threshold
        )
        &
        (
            queries[
                "n_available_sources"
            ]
            >= MIN_AVAILABLE_SOURCES
        )
    )


    temp = (
        queries[
            [
                "dimension",
                "condition",
                "group",
            ]
        ]
        .copy()
    )


    temp[
        "eligible"
    ] = eligibility


    counts = (
        temp
        .groupby(
            [
                "dimension",
                "condition",
                "group",
            ]
        )[
            "eligible"
        ]
        .sum()
        .reset_index(
            name=column
        )
    )


    coverage_by_group = (
        coverage_by_group
        .merge(
            counts,
            on=[
                "dimension",
                "condition",
                "group",
            ],
            how="left",
        )
    )


coverage_by_group.to_csv(
    RESULTS_DIR
    / "source_coverage_by_group_v2.csv",
    index=False,
)


# ============================================================
# 17. LOAD MULTILINGUAL MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING MULTILINGUAL EMBEDDING MODEL")
print("=" * 80)


print(
    f"\nModel: {MODEL_NAME}"
)


print(
    f"Device: {DEVICE}"
)


model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
)


tokenizer = (
    model.tokenizer
)


MODEL_MAX_TOKENS = int(
    model.max_seq_length
)


print(
    f"Model max sequence length: "
    f"{MODEL_MAX_TOKENS}"
)


SOURCE_CHUNK_TOKENS = min(
    SOURCE_CHUNK_TOKENS,
    MODEL_MAX_TOKENS - 16,
)


AIO_CHUNK_TOKENS = min(
    AIO_CHUNK_TOKENS,
    MODEL_MAX_TOKENS - 16,
)


# ============================================================
# 18. TOKEN CHUNKING
# ============================================================

def token_chunks(
    text,
    max_tokens,
    overlap_tokens,
):
    """
    Robust sliding token windows.

    This deliberately does NOT depend on sentence boundaries.

    Useful when AIO headings/bullets are concatenated or
    punctuation is malformed.
    """

    text = clean_text(
        text
    )


    if not text:

        return []


    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )


    if not ids:

        return []


    if (
        overlap_tokens
        >= max_tokens
    ):

        raise ValueError(
            "overlap_tokens must be "
            "smaller than max_tokens"
        )


    step = (
        max_tokens
        - overlap_tokens
    )


    chunks = []


    chunk_index = 0


    for start in range(
        0,
        len(ids),
        step,
    ):

        end = min(
            start
            + max_tokens,
            len(ids),
        )


        chunk_ids = ids[
            start:end
        ]


        text_chunk = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
        ).strip()


        if (
            len(
                text_chunk
            )
            >= 30
        ):

            chunks.append({

                "chunk_index":
                    chunk_index,

                "token_start":
                    start,

                "token_end":
                    end,

                "token_count":
                    (
                        end
                        - start
                    ),

                "text":
                    text_chunk,
            })


            chunk_index += 1


        if (
            end
            >= len(ids)
        ):

            break


    return chunks


# ============================================================
# 19. BUILD SOURCE CHUNKS
# ============================================================

print()
print("=" * 80)
print("BUILDING SOURCE CHUNKS")
print("=" * 80)


source_chunk_rows = []


truncated_sources = []


for source_id, text in (
    usable_source_texts.items()
):

    chunks = token_chunks(

        text,

        max_tokens=
            SOURCE_CHUNK_TOKENS,

        overlap_tokens=
            SOURCE_CHUNK_OVERLAP,
    )


    was_truncated = (
        len(chunks)
        >
        MAX_CHUNKS_PER_SOURCE
    )


    if was_truncated:

        truncated_sources.append(
            source_id
        )


        chunks = chunks[
            :MAX_CHUNKS_PER_SOURCE
        ]


    for chunk in chunks:

        source_chunk_rows.append({

            "source_id":
                source_id,

            "chunk_index":
                chunk[
                    "chunk_index"
                ],

            "token_start":
                chunk[
                    "token_start"
                ],

            "token_end":
                chunk[
                    "token_end"
                ],

            "token_count":
                chunk[
                    "token_count"
                ],

            "chunk_text":
                chunk[
                    "text"
                ],

            "source_truncated":
                was_truncated,
        })


source_chunks = pd.DataFrame(
    source_chunk_rows
)


source_chunks[
    "global_chunk_id"
] = np.arange(
    len(
        source_chunks
    )
)


print(
    f"\nSource chunks: "
    f"{len(source_chunks):,}"
)


print(
    f"Sources truncated at "
    f"{MAX_CHUNKS_PER_SOURCE} chunks: "
    f"{len(truncated_sources):,}"
)


# ============================================================
# 20. EMBED SOURCE CHUNKS
# ============================================================

print(
    "\nEmbedding source chunks..."
)


source_chunk_embeddings = model.encode(

    source_chunks[
        "chunk_text"
    ].tolist(),

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True,
)


source_chunk_embeddings = np.asarray(
    source_chunk_embeddings,
    dtype=np.float32,
)


source_to_chunk_ids = {}


for source_id, group_df in (
    source_chunks.groupby(
        "source_id"
    )
):

    source_to_chunk_ids[
        str(source_id)
    ] = (
        group_df[
            "global_chunk_id"
        ]
        .astype(int)
        .tolist()
    )


# ============================================================
# 21. QUERY EMBEDDINGS
# ============================================================

print(
    "\nEmbedding queries..."
)


query_embeddings = model.encode(

    queries[
        "query"
    ]
    .fillna("")
    .tolist(),

    batch_size=BATCH_SIZE,

    show_progress_bar=False,

    convert_to_numpy=True,

    normalize_embeddings=True,
)


query_embeddings = np.asarray(
    query_embeddings,
    dtype=np.float32,
)


# ============================================================
# 22. OVERLAP BETWEEN TOKEN WINDOWS
# ============================================================

def token_overlap_fraction(
    a_start,
    a_end,
    b_start,
    b_end,
):

    overlap = max(
        0,
        min(
            a_end,
            b_end,
        )
        -
        max(
            a_start,
            b_start,
        ),
    )


    if overlap <= 0:

        return 0.0


    shorter = min(
        a_end - a_start,
        b_end - b_start,
    )


    if shorter <= 0:

        return 0.0


    return (
        overlap
        / shorter
    )


# ============================================================
# 23. SELECT DIVERSE TOP-K PASSAGES
# ============================================================

def select_diverse_top_chunks(
    candidate_chunk_ids,
    similarities,
    k,
):

    order = (
        np.argsort(
            similarities
        )[::-1]
    )


    selected = []


    for position in order:

        chunk_id = int(
            candidate_chunk_ids[
                int(
                    position
                )
            ]
        )


        row = (
            source_chunks.loc[
                source_chunks[
                    "global_chunk_id"
                ]
                == chunk_id
            ]
            .iloc[0]
        )


        too_overlapping = False


        for selected_id in (
            selected
        ):

            selected_row = (
                source_chunks.loc[
                    source_chunks[
                        "global_chunk_id"
                    ]
                    == selected_id
                ]
                .iloc[0]
            )


            overlap = (
                token_overlap_fraction(

                    int(
                        row[
                            "token_start"
                        ]
                    ),

                    int(
                        row[
                            "token_end"
                        ]
                    ),

                    int(
                        selected_row[
                            "token_start"
                        ]
                    ),

                    int(
                        selected_row[
                            "token_end"
                        ]
                    ),
                )
            )


            if (
                overlap
                >
                MAX_SELECTED_TOKEN_OVERLAP
            ):

                too_overlapping = True

                break


        if not too_overlapping:

            selected.append(
                chunk_id
            )


        if (
            len(selected)
            >= k
        ):

            break


    # --------------------------------------------------------
    # If diversity filtering yielded fewer than k,
    # fill from highest-ranking remaining windows.
    # --------------------------------------------------------

    if (
        len(selected)
        < k
    ):

        for position in order:

            chunk_id = int(
                candidate_chunk_ids[
                    int(
                        position
                    )
                ]
            )


            if (
                chunk_id
                in selected
            ):

                continue


            selected.append(
                chunk_id
            )


            if (
                len(selected)
                >= k
            ):

                break


    return selected


# ============================================================
# 24. SELECT QUERY-RELEVANT EVIDENCE PER SOURCE
# ============================================================

print()
print("=" * 80)
print("SELECTING QUERY-RELEVANT EVIDENCE")
print("=" * 80)


selected_rows = []


manifest_lookup = (
    manifest
    .drop_duplicates(
        subset=[
            "source_id"
        ]
    )
    .set_index(
        "source_id"
    )
)


for _, query_row in (
    queries.iterrows()
):

    file = (
        query_row[
            "file"
        ]
    )


    query_index = int(
        query_row[
            "query_index"
        ]
    )


    query_embedding = (
        query_embeddings[
            query_index
        ]
    )


    citations = (
        usage_unique[
            usage_unique[
                "file"
            ]
            == file
        ]
        .sort_values(
            "source_order"
        )
    )


    for _, citation in (
        citations.iterrows()
    ):

        source_id = str(
            citation[
                "source_id"
            ]
        )


        if (
            source_id
            not in source_to_chunk_ids
        ):

            continue


        candidate_ids = (
            source_to_chunk_ids[
                source_id
            ]
        )


        candidate_vectors = (
            source_chunk_embeddings[
                candidate_ids
            ]
        )


        similarities = (
            candidate_vectors
            @ query_embedding
        )


        selected_ids = (
            select_diverse_top_chunks(

                candidate_ids,

                similarities,

                TOP_PASSAGES_PER_SOURCE,
            )
        )


        # Map global chunk ID -> similarity.
        similarity_lookup = {

            int(chunk_id):
                float(
                    similarities[i]
                )

            for i, chunk_id
            in enumerate(
                candidate_ids
            )
        }


        for source_rank, chunk_id in enumerate(
            selected_ids,
            start=1,
        ):

            chunk_row = (
                source_chunks.loc[
                    source_chunks[
                        "global_chunk_id"
                    ]
                    == chunk_id
                ]
                .iloc[0]
            )


            source_url = ""

            source_domain = ""


            try:

                source_meta = (
                    manifest_lookup.loc[
                        source_id
                    ]
                )


                source_url = (
                    source_meta.get(
                        "normalized_url",
                        ""
                    )
                )


                source_domain = (
                    source_meta.get(
                        "registrable_domain",
                        ""
                    )
                )


            except Exception:

                pass


            selected_rows.append({

                "file":
                    file,

                "query_id":
                    query_row[
                        "query_id"
                    ],

                "query":
                    query_row[
                        "query"
                    ],

                "group":
                    query_row[
                        "group"
                    ],

                "condition":
                    query_row[
                        "condition"
                    ],

                "dimension":
                    query_row[
                        "dimension"
                    ],

                "domain":
                    query_row[
                        "domain"
                    ],

                "outcome":
                    query_row[
                        "outcome"
                    ],

                "outcome_id":
                    query_row[
                        "outcome_id"
                    ],

                "source_id":
                    source_id,

                "source_order":
                    citation[
                        "source_order"
                    ],

                "source_rank":
                    source_rank,

                "source_url":
                    source_url,

                "source_domain":
                    source_domain,

                "source_chunk_id":
                    chunk_id,

                "token_start":
                    chunk_row[
                        "token_start"
                    ],

                "token_end":
                    chunk_row[
                        "token_end"
                    ],

                "passage_text":
                    chunk_row[
                        "chunk_text"
                    ],

                "query_relevance":
                    similarity_lookup[
                        int(
                            chunk_id
                        )
                    ],
            })


selected = pd.DataFrame(
    selected_rows
)


print(
    f"\nSelected evidence passages: "
    f"{len(selected):,}"
)


# ============================================================
# 25. BUILD AIO TOKEN CHUNKS
# ============================================================

print()
print("=" * 80)
print("BUILDING AIO CHUNKS")
print("=" * 80)


aio_chunk_rows = []


for _, query_row in (
    queries.iterrows()
):

    chunks = token_chunks(

        query_row[
            "aio_text"
        ],

        max_tokens=
            AIO_CHUNK_TOKENS,

        overlap_tokens=
            AIO_CHUNK_OVERLAP,
    )


    for chunk in chunks:

        aio_chunk_rows.append({

            "file":
                query_row[
                    "file"
                ],

            "query_id":
                query_row[
                    "query_id"
                ],

            "group":
                query_row[
                    "group"
                ],

            "condition":
                query_row[
                    "condition"
                ],

            "dimension":
                query_row[
                    "dimension"
                ],

            "domain":
                query_row[
                    "domain"
                ],

            "outcome":
                query_row[
                    "outcome"
                ],

            "outcome_id":
                query_row[
                    "outcome_id"
                ],

            "aio_chunk_index":
                chunk[
                    "chunk_index"
                ],

            "token_start":
                chunk[
                    "token_start"
                ],

            "token_end":
                chunk[
                    "token_end"
                ],

            "chunk_text":
                chunk[
                    "text"
                ],
        })


aio_chunks = pd.DataFrame(
    aio_chunk_rows
)


aio_chunks[
    "global_aio_chunk_id"
] = np.arange(
    len(
        aio_chunks
    )
)


print(
    f"\nAIO chunks: "
    f"{len(aio_chunks):,}"
)


# ============================================================
# 26. SUBJECT-NORMALIZED MATCHING TEXT
# ============================================================

selected[
    "matching_text"
] = selected.apply(

    lambda row:
        subject_normalize(

            row[
                "passage_text"
            ],

            row[
                "group"
            ],
        ),

    axis=1,
)


aio_chunks[
    "matching_text"
] = aio_chunks.apply(

    lambda row:
        subject_normalize(

            row[
                "chunk_text"
            ],

            row[
                "group"
            ],
        ),

    axis=1,
)


# ============================================================
# 27. EMBED MATCHING UNITS
# ============================================================

print()
print("=" * 80)
print("EMBEDDING EVIDENCE + AIO CHUNKS")
print("=" * 80)


selected_embeddings = model.encode(

    selected[
        "matching_text"
    ].tolist(),

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True,
)


selected_embeddings = np.asarray(
    selected_embeddings,
    dtype=np.float32,
)


selected[
    "selected_embedding_index"
] = np.arange(
    len(
        selected
    )
)


aio_embeddings = model.encode(

    aio_chunks[
        "matching_text"
    ].tolist(),

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True,
)


aio_embeddings = np.asarray(
    aio_embeddings,
    dtype=np.float32,
)


aio_chunks[
    "aio_embedding_index"
] = np.arange(
    len(
        aio_chunks
    )
)


# ============================================================
# 28. EVIDENCE -> AIO MATCHING
# ============================================================

print()
print("=" * 80)
print("MATCHING EVIDENCE TO AIO")
print("=" * 80)


query_metric_rows = []

source_metric_rows = []

passage_match_rows = []

aio_support_rows = []


for query_counter, (
    _,
    query_row,
) in enumerate(
    queries.iterrows(),
    start=1,
):

    file = (
        query_row[
            "file"
        ]
    )


    evidence = (
        selected[
            selected[
                "file"
            ]
            == file
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    answer = (
        aio_chunks[
            aio_chunks[
                "file"
            ]
            == file
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    base = {

        "file":
            file,

        "query_id":
            query_row[
                "query_id"
            ],

        "query":
            query_row[
                "query"
            ],

        "group":
            query_row[
                "group"
            ],

        "condition":
            query_row[
                "condition"
            ],

        "dimension":
            query_row[
                "dimension"
            ],

        "domain":
            query_row[
                "domain"
            ],

        "outcome":
            query_row[
                "outcome"
            ],

        "outcome_id":
            query_row[
                "outcome_id"
            ],

        "n_cited_sources":
            query_row[
                "n_cited_sources"
            ],

        "n_available_sources":
            query_row[
                "n_available_sources"
            ],

        "fetch_coverage":
            query_row[
                "fetch_coverage"
            ],

        "n_selected_passages":
            len(
                evidence
            ),

        "n_aio_chunks":
            len(
                answer
            ),
    }


    if (
        len(evidence) == 0
        or len(answer) == 0
    ):

        base.update({

            "evidence_alignment_source_balanced":
                np.nan,

            "evidence_semantic_gap_source_balanced":
                np.nan,

            "evidence_alignment_passage_mean":
                np.nan,

            "evidence_semantic_gap_passage_mean":
                np.nan,

            "answer_support_alignment":
                np.nan,

            "answer_support_semantic_gap":
                np.nan,

            "evidence_match_rate_065":
                np.nan,

            "answer_support_rate_065":
                np.nan,

            "source_match_rate_065":
                np.nan,

            "n_contributing_sources_065":
                np.nan,

            "dominant_source_share_065":
                np.nan,
        })


        query_metric_rows.append(
            base
        )


        continue


    evidence_indices = (
        evidence[
            "selected_embedding_index"
        ]
        .astype(int)
        .to_numpy()
    )


    answer_indices = (
        answer[
            "aio_embedding_index"
        ]
        .astype(int)
        .to_numpy()
    )


    evidence_vectors = (
        selected_embeddings[
            evidence_indices
        ]
    )


    answer_vectors = (
        aio_embeddings[
            answer_indices
        ]
    )


    similarity_matrix = (
        evidence_vectors
        @ answer_vectors.T
    )


    similarity_matrix = np.clip(
        similarity_matrix,
        -1.0,
        1.0,
    )


    # --------------------------------------------------------
    # EVIDENCE -> AIO
    # --------------------------------------------------------

    evidence_best_similarity = (
        similarity_matrix.max(
            axis=1
        )
    )


    evidence_best_answer_position = (
        similarity_matrix.argmax(
            axis=1
        )
    )


    evidence[
        "best_aio_similarity"
    ] = (
        evidence_best_similarity
    )


    # --------------------------------------------------------
    # AIO -> EVIDENCE
    # --------------------------------------------------------

    answer_best_similarity = (
        similarity_matrix.max(
            axis=0
        )
    )


    answer_best_evidence_position = (
        similarity_matrix.argmax(
            axis=0
        )
    )


    # ========================================================
    # SOURCE-BALANCED EVIDENCE ALIGNMENT
    # ========================================================

    source_alignment_values = []


    for source_id, source_group in (
        evidence.groupby(
            "source_id"
        )
    ):

        source_alignment = float(
            source_group[
                "best_aio_similarity"
            ].mean()
        )


        source_alignment_values.append(
            source_alignment
        )


        source_metric_rows.append({

            "file":
                file,

            "group":
                query_row[
                    "group"
                ],

            "condition":
                query_row[
                    "condition"
                ],

            "dimension":
                query_row[
                    "dimension"
                ],

            "domain":
                query_row[
                    "domain"
                ],

            "outcome":
                query_row[
                    "outcome"
                ],

            "outcome_id":
                query_row[
                    "outcome_id"
                ],

            "source_id":
                source_id,

            "source_url":
                source_group[
                    "source_url"
                ].iloc[0],

            "source_domain":
                source_group[
                    "source_domain"
                ].iloc[0],

            "n_selected_passages":
                len(
                    source_group
                ),

            "mean_query_relevance":
                float(
                    source_group[
                        "query_relevance"
                    ].mean()
                ),

            "evidence_alignment":
                source_alignment,

            "evidence_semantic_gap":
                (
                    1.0
                    - source_alignment
                ),

            "max_passage_alignment":
                float(
                    source_group[
                        "best_aio_similarity"
                    ].max()
                ),
        })


    evidence_alignment_source_balanced = float(
        np.mean(
            source_alignment_values
        )
    )


    evidence_alignment_passage_mean = float(
        np.mean(
            evidence_best_similarity
        )
    )


    answer_support_alignment = float(
        np.mean(
            answer_best_similarity
        )
    )


    # ========================================================
    # THRESHOLD DESCRIPTIVES
    # ========================================================

    evidence_match_rate = float(
        np.mean(
            evidence_best_similarity
            >= DESCRIPTIVE_MATCH_THRESHOLD
        )
    )


    answer_support_rate = float(
        np.mean(
            answer_best_similarity
            >= DESCRIPTIVE_MATCH_THRESHOLD
        )
    )


    source_maximums = (
        evidence
        .groupby(
            "source_id"
        )[
            "best_aio_similarity"
        ]
        .max()
    )


    source_match_rate = float(
        np.mean(
            source_maximums
            >= DESCRIPTIVE_MATCH_THRESHOLD
        )
    )


    # --------------------------------------------------------
    # Which source is the nearest evidence for each AIO chunk?
    # --------------------------------------------------------

    contributing_source_ids = []


    for answer_position, evidence_position in enumerate(
        answer_best_evidence_position
    ):

        best_similarity = float(
            answer_best_similarity[
                answer_position
            ]
        )


        evidence_row = (
            evidence.iloc[
                int(
                    evidence_position
                )
            ]
        )


        source_id = (
            evidence_row[
                "source_id"
            ]
        )


        if (
            best_similarity
            >= DESCRIPTIVE_MATCH_THRESHOLD
        ):

            contributing_source_ids.append(
                source_id
            )


        aio_support_rows.append({

            "file":
                file,

            "group":
                query_row[
                    "group"
                ],

            "dimension":
                query_row[
                    "dimension"
                ],

            "domain":
                query_row[
                    "domain"
                ],

            "outcome":
                query_row[
                    "outcome"
                ],

            "outcome_id":
                query_row[
                    "outcome_id"
                ],

            "aio_chunk_index":
                answer.iloc[
                    answer_position
                ][
                    "aio_chunk_index"
                ],

            "aio_chunk_text":
                answer.iloc[
                    answer_position
                ][
                    "chunk_text"
                ],

            "best_evidence_similarity":
                best_similarity,

            "best_source_id":
                source_id,

            "best_source_url":
                evidence_row[
                    "source_url"
                ],

            "best_evidence_passage":
                evidence_row[
                    "passage_text"
                ],
        })


    if contributing_source_ids:

        counts = pd.Series(
            contributing_source_ids
        ).value_counts()


        n_contributing_sources = int(
            len(
                counts
            )
        )


        dominant_source_share = float(
            counts.max()
            / counts.sum()
        )


    else:

        n_contributing_sources = 0

        dominant_source_share = np.nan


    # ========================================================
    # PASSAGE-LEVEL AUDIT OUTPUT
    # ========================================================

    for evidence_position, (
        _,
        evidence_row,
    ) in enumerate(
        evidence.iterrows()
    ):

        best_answer_position = int(
            evidence_best_answer_position[
                evidence_position
            ]
        )


        similarity = float(
            evidence_best_similarity[
                evidence_position
            ]
        )


        relevance = float(
            evidence_row[
                "query_relevance"
            ]
        )


        # Useful only for qualitative case prioritization.
        audit_priority_score = (
            max(
                relevance,
                0.0,
            )
            *
            (
                1.0
                - similarity
            )
        )


        passage_match_rows.append({

            "file":
                file,

            "query":
                query_row[
                    "query"
                ],

            "group":
                query_row[
                    "group"
                ],

            "condition":
                query_row[
                    "condition"
                ],

            "dimension":
                query_row[
                    "dimension"
                ],

            "domain":
                query_row[
                    "domain"
                ],

            "outcome":
                query_row[
                    "outcome"
                ],

            "outcome_id":
                query_row[
                    "outcome_id"
                ],

            "source_id":
                evidence_row[
                    "source_id"
                ],

            "source_url":
                evidence_row[
                    "source_url"
                ],

            "source_domain":
                evidence_row[
                    "source_domain"
                ],

            "source_order":
                evidence_row[
                    "source_order"
                ],

            "source_rank":
                evidence_row[
                    "source_rank"
                ],

            "query_relevance":
                relevance,

            "evidence_passage":
                evidence_row[
                    "passage_text"
                ],

            "best_aio_similarity":
                similarity,

            "best_aio_chunk":
                answer.iloc[
                    best_answer_position
                ][
                    "chunk_text"
                ],

            "semantic_gap":
                (
                    1.0
                    - similarity
                ),

            "audit_priority_score":
                audit_priority_score,
        })


    # ========================================================
    # QUERY METRICS
    # ========================================================

    base.update({

        # ----------------------------------------------------
        # PRIMARY SEMANTIC EVIDENCE METRIC
        # ----------------------------------------------------

        "evidence_alignment_source_balanced":
            evidence_alignment_source_balanced,

        "evidence_semantic_gap_source_balanced":
            (
                1.0
                - evidence_alignment_source_balanced
            ),


        # ----------------------------------------------------
        # PASSAGE-WEIGHTED SENSITIVITY
        # ----------------------------------------------------

        "evidence_alignment_passage_mean":
            evidence_alignment_passage_mean,

        "evidence_semantic_gap_passage_mean":
            (
                1.0
                - evidence_alignment_passage_mean
            ),


        # ----------------------------------------------------
        # AIO -> EVIDENCE
        # ----------------------------------------------------

        "answer_support_alignment":
            answer_support_alignment,

        "answer_support_semantic_gap":
            (
                1.0
                - answer_support_alignment
            ),


        # ----------------------------------------------------
        # DESCRIPTIVE ONLY
        # ----------------------------------------------------

        "evidence_match_rate_065":
            evidence_match_rate,

        "answer_support_rate_065":
            answer_support_rate,

        "source_match_rate_065":
            source_match_rate,

        "n_contributing_sources_065":
            n_contributing_sources,

        "dominant_source_share_065":
            dominant_source_share,
    })


    query_metric_rows.append(
        base
    )


    if (
        query_counter % 25
        == 0
    ):

        print(
            f"Processed "
            f"{query_counter}/"
            f"{len(queries)} queries"
        )


# ============================================================
# 29. RESULTS DATAFRAMES
# ============================================================

query_metrics = pd.DataFrame(
    query_metric_rows
)


source_metrics = pd.DataFrame(
    source_metric_rows
)


passage_matches = pd.DataFrame(
    passage_match_rows
)


aio_support = pd.DataFrame(
    aio_support_rows
)


# ============================================================
# 30. SAVE RAW ANALYSIS TABLES
# ============================================================

query_metrics.to_csv(
    RESULTS_DIR
    / "query_evidence_alignment_metrics_v2.csv",
    index=False,
)


source_metrics.to_csv(
    RESULTS_DIR
    / "source_evidence_alignment_metrics_v2.csv",
    index=False,
)


passage_matches.to_csv(
    RESULTS_DIR
    / "evidence_passage_matches_v2.csv",
    index=False,
)


aio_support.to_csv(
    RESULTS_DIR
    / "aio_chunk_support_v2.csv",
    index=False,
)


selected.to_csv(
    RESULTS_DIR
    / "selected_query_relevant_evidence_v2.csv",
    index=False,
)


aio_chunks.to_csv(
    RESULTS_DIR
    / "aio_chunks_v2.csv",
    index=False,
)


# ============================================================
# 31. QUALITATIVE AUDIT CASES
# ============================================================

weakly_aligned = (
    passage_matches
    .sort_values(
        [
            "audit_priority_score",
            "query_relevance",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        300
    )
)


strongly_aligned = (
    passage_matches
    .sort_values(
        [
            "best_aio_similarity",
            "query_relevance",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        300
    )
)


weakly_aligned.to_csv(
    RESULTS_DIR
    / "weakly_aligned_relevant_evidence.csv",
    index=False,
)


strongly_aligned.to_csv(
    RESULTS_DIR
    / "strongly_aligned_relevant_evidence.csv",
    index=False,
)


# ============================================================
# 32. GROUP DESCRIPTIVES
# ============================================================

group_summary = (
    query_metrics
    .groupby(
        [
            "dimension",
            "condition",
            "group",
        ],
        dropna=False,
    )
    .agg(

        n_queries=(
            "file",
            "size",
        ),

        mean_fetch_coverage=(
            "fetch_coverage",
            "mean",
        ),

        mean_available_sources=(
            "n_available_sources",
            "mean",
        ),

        mean_evidence_alignment=(
            "evidence_alignment_source_balanced",
            "mean",
        ),

        mean_evidence_semantic_gap=(
            "evidence_semantic_gap_source_balanced",
            "mean",
        ),

        mean_answer_support_alignment=(
            "answer_support_alignment",
            "mean",
        ),

        mean_answer_support_gap=(
            "answer_support_semantic_gap",
            "mean",
        ),

        mean_evidence_match_rate=(
            "evidence_match_rate_065",
            "mean",
        ),

        mean_source_match_rate=(
            "source_match_rate_065",
            "mean",
        ),

        mean_contributing_sources=(
            "n_contributing_sources_065",
            "mean",
        ),
    )
    .reset_index()
)


group_summary.to_csv(
    RESULTS_DIR
    / "evidence_alignment_by_group_v2.csv",
    index=False,
)


# ============================================================
# 33. STATISTICAL HELPERS
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


def bootstrap_mean_ci(
    differences,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
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


    values = np.empty(
        n_boot,
        dtype=float,
    )


    for i in range(
        n_boot
    ):

        sample = rng.choice(
            d,
            size=len(d),
            replace=True,
        )


        values[
            i
        ] = sample.mean()


    return (

        float(
            np.percentile(
                values,
                2.5,
            )
        ),

        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


def signflip_test(
    differences,
    seed=RANDOM_SEED,
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

        return {

            "statistic":
                0.0,

            "p_value":
                1.0,

            "method":
                "all_zero",

            "n_nonzero":
                0,
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


            bits = (
                (
                    integers
                    & powers
                )
                > 0
            )


            signs = (
                bits.astype(
                    float
                )
                * 2.0
                - 1.0
            )


            stats = np.abs(
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
                    stats
                    >= observed
                    - 1e-15
                )
            )


        p_value = (
            extreme
            / total
        )


        method = (
            f"exact_signflip_2^{n}"
        )


    # --------------------------------------------------------
    # MONTE CARLO
    # --------------------------------------------------------

    else:

        rng = np.random.default_rng(
            seed
        )


        extreme = 0

        completed = 0

        batch_size = 10_000


        while (
            completed
            < N_PERMUTATIONS
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


            stats = np.abs(
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
                    stats
                    >= observed
                    - 1e-15
                )
            )


            completed += (
                batch
            )


        p_value = (
            extreme + 1
        ) / (
            N_PERMUTATIONS + 1
        )


        method = (
            "monte_carlo_signflip_"
            f"{N_PERMUTATIONS}"
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

        "method":
            method,

        "n_nonzero":
            n,
    }


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


    d = np.round(
        d,
        12,
    )


    if len(d) == 0:

        return (
            np.nan,
            np.nan,
        )


    if np.all(
        d == 0
    ):

        return (
            0.0,
            1.0,
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
        ),
    )


def summarize_differences(
    differences,
    label,
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

        return {}


    permutation = signflip_test(

        d,

        seed=stable_seed(
            label
        ),
    )


    W, wilcoxon_p = (
        safe_wilcoxon(
            d
        )
    )


    ci_low, ci_high = (
        bootstrap_mean_ci(

            d,

            seed=stable_seed(
                "bootstrap::"
                + label
            ),
        )
    )


    return {

        "n":
            len(
                d
            ),

        "mean_difference":
            float(
                np.mean(
                    d
                )
            ),

        "median_difference":
            float(
                np.median(
                    d
                )
            ),

        "bootstrap_ci_low":
            ci_low,

        "bootstrap_ci_high":
            ci_high,

        "rank_biserial":
            paired_rank_biserial(
                d
            ),

        "p_permutation_raw":
            permutation[
                "p_value"
            ],

        "permutation_method":
            permutation[
                "method"
            ],

        "wilcoxon_W":
            W,

        "p_wilcoxon_raw":
            wilcoxon_p,
    }


# ============================================================
# 34. ELIGIBILITY
# ============================================================

def eligible_data(
    dataframe,
    threshold,
    metric,
):

    return dataframe[
        (
            dataframe[
                "fetch_coverage"
            ]
            >= threshold
        )
        &
        (
            dataframe[
                "n_available_sources"
            ]
            >= MIN_AVAILABLE_SOURCES
        )
        &
        (
            dataframe[
                metric
            ]
            .notna()
        )
    ].copy()


# ============================================================
# 35. DIMENSION-LEVEL PAIRED TESTS
# ============================================================

def dimension_tests(
    dataframe,
    metric,
    threshold,
):

    eligible = (
        eligible_data(
            dataframe,
            threshold,
            metric,
        )
    )


    explicit = eligible[
        eligible[
            "condition"
        ]
        .isin(
            [
                "minority",
                "majority",
            ]
        )
    ]


    rows = []


    for dimension in sorted(
        explicit[
            "dimension"
        ].unique()
    ):

        subset = explicit[
            explicit[
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

            continue


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


        if (
            minority_group
            not in wide.columns
            or
            majority_group
            not in wide.columns
        ):

            continue


        wide = wide.dropna(
            subset=[
                minority_group,
                majority_group,
            ]
        )


        if len(wide) == 0:

            continue


        minority_values = (
            wide[
                minority_group
            ]
            .to_numpy(
                dtype=float
            )
        )


        majority_values = (
            wide[
                majority_group
            ]
            .to_numpy(
                dtype=float
            )
        )


        differences = (
            minority_values
            - majority_values
        )


        stats = summarize_differences(

            differences,

            label=(
                f"{metric}::"
                f"{threshold}::"
                f"{dimension}"
            ),
        )


        stats.update({

            "coverage_threshold":
                threshold,

            "metric":
                metric,

            "dimension":
                dimension,

            "minority_group":
                minority_group,

            "majority_group":
                majority_group,

            "minority_mean":
                float(
                    np.mean(
                        minority_values
                    )
                ),

            "majority_mean":
                float(
                    np.mean(
                        majority_values
                    )
                ),
        })


        rows.append(
            stats
        )


    result = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Correct across six planned dimension tests
    # separately at each coverage threshold.
    # --------------------------------------------------------

    if len(result):

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
            "p_permutation_holm"
        ] = p_adj


        result[
            "significant_permutation_holm"
        ] = reject


        reject_w, p_adj_w, _, _ = (
            multipletests(

                result[
                    "p_wilcoxon_raw"
                ],

                alpha=ALPHA,

                method="holm",
            )
        )


        result[
            "p_wilcoxon_holm"
        ] = p_adj_w


        result[
            "significant_wilcoxon_holm"
        ] = reject_w


    return result


# ============================================================
# 36. COVERAGE SENSITIVITY — PRIMARY METRIC
# ============================================================

PRIMARY_METRIC = (
    "evidence_semantic_gap_source_balanced"
)


sensitivity_tables = []


for threshold in (
    COVERAGE_THRESHOLDS
):

    table = dimension_tests(

        query_metrics,

        PRIMARY_METRIC,

        threshold,
    )


    if len(table):

        sensitivity_tables.append(
            table
        )


if sensitivity_tables:

    sensitivity_dimension_tests = pd.concat(
        sensitivity_tables,
        ignore_index=True,
    )

else:

    sensitivity_dimension_tests = pd.DataFrame()


sensitivity_dimension_tests.to_csv(
    RESULTS_DIR
    / "sensitivity_dimension_tests_v2.csv",
    index=False,
)


# ============================================================
# 37. REFERENCE 50% ANALYSIS
# ============================================================

reference_dimension_tests = (
    sensitivity_dimension_tests[
        np.isclose(
            sensitivity_dimension_tests[
                "coverage_threshold"
            ],
            REFERENCE_COVERAGE_THRESHOLD,
        )
    ]
    .copy()
    if len(
        sensitivity_dimension_tests
    )
    else pd.DataFrame()
)


reference_dimension_tests.to_csv(
    RESULTS_DIR
    / "reference_50pct_dimension_tests_v2.csv",
    index=False,
)


# ============================================================
# 38. SECONDARY — ANSWER SUPPORT
# ============================================================

answer_support_tables = []


for threshold in (
    COVERAGE_THRESHOLDS
):

    table = dimension_tests(

        query_metrics,

        "answer_support_semantic_gap",

        threshold,
    )


    if len(table):

        answer_support_tables.append(
            table
        )


answer_support_sensitivity = (

    pd.concat(
        answer_support_tables,
        ignore_index=True,
    )

    if answer_support_tables

    else pd.DataFrame()
)


answer_support_sensitivity.to_csv(
    RESULTS_DIR
    / "answer_support_sensitivity_v2.csv",
    index=False,
)


# ============================================================
# 39. BUILD PAIRED DIFFERENCES FOR AGGREGATE ANALYSIS
# ============================================================

def pair_difference_table(
    dataframe,
    metric,
    threshold,
):

    eligible = eligible_data(
        dataframe,
        threshold,
        metric,
    )


    explicit = eligible[
        eligible[
            "condition"
        ]
        .isin(
            [
                "minority",
                "majority",
            ]
        )
    ]


    rows = []


    for dimension in sorted(
        explicit[
            "dimension"
        ].unique()
    ):

        subset = explicit[
            explicit[
                "dimension"
            ]
            == dimension
        ]


        minority = subset[
            subset[
                "condition"
            ]
            == "minority"
        ][
            [
                "outcome_id",
                metric,
            ]
        ].rename(
            columns={
                metric:
                    "minority_value"
            }
        )


        majority = subset[
            subset[
                "condition"
            ]
            == "majority"
        ][
            [
                "outcome_id",
                metric,
            ]
        ].rename(
            columns={
                metric:
                    "majority_value"
            }
        )


        merged = minority.merge(
            majority,
            on="outcome_id",
            how="inner",
        )


        for _, row in (
            merged.iterrows()
        ):

            rows.append({

                "coverage_threshold":
                    threshold,

                "dimension":
                    dimension,

                "outcome_id":
                    row[
                        "outcome_id"
                    ],

                "minority_value":
                    row[
                        "minority_value"
                    ],

                "majority_value":
                    row[
                        "majority_value"
                    ],

                "difference":
                    (
                        row[
                            "minority_value"
                        ]
                        -
                        row[
                            "majority_value"
                        ]
                    ),
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# 40. AGGREGATE OUTCOME-BLOCKED EFFECT
# ============================================================

aggregate_rows = []


for threshold in (
    COVERAGE_THRESHOLDS
):

    pair_diffs = pair_difference_table(

        query_metrics,

        PRIMARY_METRIC,

        threshold,
    )


    if pair_diffs.empty:

        continue


    outcome_summary = (
        pair_diffs
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


    outcome_summary = (
        outcome_summary[
            outcome_summary[
                "n_dimensions"
            ]
            >= MIN_DIMENSIONS_FOR_AGGREGATE
        ]
    )


    if len(
        outcome_summary
    ) == 0:

        continue


    stats = summarize_differences(

        outcome_summary[
            "mean_difference"
        ].to_numpy(),

        label=(
            f"aggregate::"
            f"{threshold}"
        ),
    )


    stats.update({

        "coverage_threshold":
            threshold,

        "n_outcomes":
            len(
                outcome_summary
            ),

        "mean_dimensions_per_outcome":
            float(
                outcome_summary[
                    "n_dimensions"
                ].mean()
            ),

        "minimum_dimensions_required":
            MIN_DIMENSIONS_FOR_AGGREGATE,

        "metric":
            PRIMARY_METRIC,
    })


    aggregate_rows.append(
        stats
    )


aggregate_sensitivity = pd.DataFrame(
    aggregate_rows
)


aggregate_sensitivity.to_csv(
    RESULTS_DIR
    / "aggregate_sensitivity_v2.csv",
    index=False,
)


# ============================================================
# 41. COVERAGE BIAS TEST
#
# Does source recovery itself differ minority vs majority?
# ============================================================

coverage_rows_stats = []


for dimension in sorted(
    query_metrics[
        "dimension"
    ].dropna().unique()
):

    if (
        dimension
        == "Control"
    ):

        continue


    subset = query_metrics[
        query_metrics[
            "dimension"
        ]
        == dimension
    ]


    minority = subset[
        subset[
            "condition"
        ]
        == "minority"
    ][
        [
            "outcome_id",
            "fetch_coverage",
        ]
    ].rename(
        columns={
            "fetch_coverage":
                "minority"
        }
    )


    majority = subset[
        subset[
            "condition"
        ]
        == "majority"
    ][
        [
            "outcome_id",
            "fetch_coverage",
        ]
    ].rename(
        columns={
            "fetch_coverage":
                "majority"
        }
    )


    paired = minority.merge(
        majority,
        on="outcome_id",
        how="inner",
    ).dropna()


    if len(
        paired
    ) == 0:

        continue


    differences = (
        paired[
            "minority"
        ]
        -
        paired[
            "majority"
        ]
    )


    stats = summarize_differences(

        differences,

        label=(
            f"coverage::{dimension}"
        ),
    )


    stats.update({

        "dimension":
            dimension,

        "minority_mean":
            float(
                paired[
                    "minority"
                ].mean()
            ),

        "majority_mean":
            float(
                paired[
                    "majority"
                ].mean()
            ),
    })


    coverage_rows_stats.append(
        stats
    )


coverage_bias_tests = pd.DataFrame(
    coverage_rows_stats
)


if len(
    coverage_bias_tests
):

    reject, p_adj, _, _ = (
        multipletests(

            coverage_bias_tests[
                "p_permutation_raw"
            ],

            alpha=ALPHA,

            method="holm",
        )
    )


    coverage_bias_tests[
        "p_permutation_holm"
    ] = p_adj


    coverage_bias_tests[
        "significant_holm"
    ] = reject


coverage_bias_tests.to_csv(
    RESULTS_DIR
    / "coverage_bias_tests_v2.csv",
    index=False,
)


# ============================================================
# 42. METRIC vs COVERAGE DIAGNOSTICS
# ============================================================

coverage_diagnostic_rows = []


diagnostic_metrics = [

    "evidence_alignment_source_balanced",

    "evidence_semantic_gap_source_balanced",

    "answer_support_alignment",

    "answer_support_semantic_gap",
]


for metric in diagnostic_metrics:

    temp = (
        query_metrics[
            [
                "fetch_coverage",
                "n_available_sources",
                metric,
            ]
        ]
        .dropna()
    )


    if len(
        temp
    ) < 10:

        continue


    rho_cov, p_cov = spearmanr(

        temp[
            "fetch_coverage"
        ],

        temp[
            metric
        ],
    )


    rho_n, p_n = spearmanr(

        temp[
            "n_available_sources"
        ],

        temp[
            metric
        ],
    )


    coverage_diagnostic_rows.append({

        "metric":
            metric,

        "predictor":
            "fetch_coverage",

        "n":
            len(
                temp
            ),

        "spearman_rho":
            float(
                rho_cov
            ),

        "p_raw":
            float(
                p_cov
            ),
    })


    coverage_diagnostic_rows.append({

        "metric":
            metric,

        "predictor":
            "n_available_sources",

        "n":
            len(
                temp
            ),

        "spearman_rho":
            float(
                rho_n
            ),

        "p_raw":
            float(
                p_n
            ),
    })


coverage_diagnostics = pd.DataFrame(
    coverage_diagnostic_rows
)


if len(
    coverage_diagnostics
):

    reject, p_adj, _, _ = (
        multipletests(

            coverage_diagnostics[
                "p_raw"
            ],

            alpha=ALPHA,

            method="holm",
        )
    )


    coverage_diagnostics[
        "p_holm"
    ] = p_adj


    coverage_diagnostics[
        "significant_holm"
    ] = reject


coverage_diagnostics.to_csv(
    RESULTS_DIR
    / "metric_vs_coverage_diagnostics_v2.csv",
    index=False,
)


# ============================================================
# 43. GLOBAL FRIEDMAN AT REFERENCE THRESHOLD
# ============================================================

reference_eligible = eligible_data(

    query_metrics,

    REFERENCE_COVERAGE_THRESHOLD,

    PRIMARY_METRIC,
)


wide = reference_eligible.pivot(

    index="outcome_id",

    columns="group",

    values=PRIMARY_METRIC,
)


complete = wide.dropna()


global_test = {}


if (
    len(
        complete
    ) >= 2
    and
    complete.shape[1]
    >= 3
):

    stat, p_value = (
        friedmanchisquare(

            *[

                complete[
                    column
                ].to_numpy()

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


    global_test = {

        "coverage_threshold":
            REFERENCE_COVERAGE_THRESHOLD,

        "n_complete_outcomes":
            n,

        "n_groups":
            k,

        "friedman_chi2":
            float(
                stat
            ),

        "p_value":
            float(
                p_value
            ),

        "kendall_W":
            float(
                stat
                /
                (
                    n
                    * (
                        k - 1
                    )
                )
            ),
    }


pd.DataFrame(
    [
        global_test
    ]
    if global_test
    else []
).to_csv(
    RESULTS_DIR
    / "global_group_test_v2.csv",
    index=False,
)


# ============================================================
# 44. MERGE PREVIOUS PIPELINE METRICS
# ============================================================

mechanism_map = (
    query_metrics[
        query_metrics[
            "condition"
        ]
        != "control"
    ][
        [
            "file",
            "group",
            "condition",
            "dimension",
            "domain",
            "outcome",
            "outcome_id",
            "fetch_coverage",
            "n_available_sources",
            "evidence_alignment_source_balanced",
            "evidence_semantic_gap_source_balanced",
            "answer_support_alignment",
            "answer_support_semantic_gap",
        ]
    ]
    .copy()
)


# ------------------------------------------------------------
# SOURCE OVERLAP
# ------------------------------------------------------------

if SOURCE_OVERLAP_FILE.exists():

    overlap = pd.read_csv(
        SOURCE_OVERLAP_FILE
    )


    desired = [

        "group",
        "outcome_id",
        "url_jaccard_distance",
        "domain_jaccard_distance",
        "url_generic_coverage",
        "domain_generic_coverage",
    ]


    desired = [

        column

        for column
        in desired

        if column
        in overlap.columns
    ]


    if (
        "group"
        in desired
        and
        "outcome_id"
        in desired
    ):

        mechanism_map = (
            mechanism_map.merge(

                overlap[
                    desired
                ],

                on=[
                    "group",
                    "outcome_id",
                ],

                how="left",
            )
        )


# ------------------------------------------------------------
# PREVIOUS AIO SEMANTIC DISPLACEMENT
# ------------------------------------------------------------

if SEMANTIC_FILE.exists():

    semantic = pd.read_csv(
        SEMANTIC_FILE
    )


    desired = [

        "group",
        "outcome_id",
        "dense_distance_subject_normalized",
        "generic_coverage_distance",
        "group_novelty_distance",
        "symmetric_sentence_distance",
    ]


    desired = [

        column

        for column
        in desired

        if column
        in semantic.columns
    ]


    if (
        "group"
        in desired
        and
        "outcome_id"
        in desired
    ):

        mechanism_map = (
            mechanism_map.merge(

                semantic[
                    desired
                ],

                on=[
                    "group",
                    "outcome_id",
                ],

                how="left",
            )
        )


mechanism_map.to_csv(
    RESULTS_DIR
    / "mechanism_map_v2.csv",
    index=False,
)


# ============================================================
# 45. BLOCK-AWARE RANK CORRELATION
#
# Each outcome is a repeated-measures block.
#
# Method:
# 1. Rank X and Y WITHIN each outcome.
# 2. Center the ranks within outcome.
# 3. Compute Pearson correlation between centered ranks.
#
# Permutation:
# shuffle Y WITHIN each outcome.
#
# Bootstrap:
# resample entire outcomes, not individual rows.
# ============================================================

def centered_within_block_ranks(
    dataframe,
    x,
    y,
    block,
):

    pieces = []


    for _, group_df in (
        dataframe.groupby(
            block
        )
    ):

        group_df = (
            group_df[
                [
                    x,
                    y,
                ]
            ]
            .dropna()
            .copy()
        )


        if len(
            group_df
        ) < 3:

            continue


        x_rank = rankdata(
            group_df[
                x
            ],
            method="average",
        )


        y_rank = rankdata(
            group_df[
                y
            ],
            method="average",
        )


        x_centered = (
            x_rank
            - np.mean(
                x_rank
            )
        )


        y_centered = (
            y_rank
            - np.mean(
                y_rank
            )
        )


        pieces.append(
            (
                x_centered,
                y_centered,
            )
        )


    if not pieces:

        return (
            np.array([]),
            np.array([]),
        )


    return (

        np.concatenate(
            [
                x
                for x, _
                in pieces
            ]
        ),

        np.concatenate(
            [
                y
                for _, y
                in pieces
            ]
        ),
    )


def correlation_statistic(
    x,
    y,
):

    if (
        len(x) < 3
        or len(y) < 3
    ):

        return np.nan


    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):

        return np.nan


    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


def blocked_rank_correlation(
    dataframe,
    x,
    y,
    block="outcome_id",
    seed=RANDOM_SEED,
):

    data = (
        dataframe[
            [
                block,
                x,
                y,
            ]
        ]
        .dropna()
        .copy()
    )


    # Keep blocks with >= 3 observations.
    block_sizes = (
        data.groupby(
            block
        )
        .size()
    )


    valid_blocks = (
        block_sizes[
            block_sizes
            >= 3
        ]
        .index
        .tolist()
    )


    data = data[
        data[
            block
        ]
        .isin(
            valid_blocks
        )
    ]


    if (
        len(valid_blocks)
        < 3
    ):

        return {}


    cx, cy = (
        centered_within_block_ranks(

            data,

            x,

            y,

            block,
        )
    )


    observed = correlation_statistic(
        cx,
        cy,
    )


    if not np.isfinite(
        observed
    ):

        return {}


    rng = np.random.default_rng(
        seed
    )


    # --------------------------------------------------------
    # BLOCKED PERMUTATION
    # --------------------------------------------------------

    groups = {

        block_value:
            group_df.copy()

        for block_value, group_df
        in data.groupby(
            block
        )
    }


    extreme = 0


    for _ in range(
        N_BLOCK_PERMUTATIONS
    ):

        permuted_parts = []


        for block_value in (
            valid_blocks
        ):

            group_df = (
                groups[
                    block_value
                ]
                .copy()
            )


            values = (
                group_df[
                    y
                ]
                .to_numpy(
                    copy=True
                )
            )


            rng.shuffle(
                values
            )


            group_df[
                y
            ] = values


            permuted_parts.append(
                group_df
            )


        permuted = pd.concat(
            permuted_parts,
            ignore_index=True,
        )


        px, py = (
            centered_within_block_ranks(

                permuted,

                x,

                y,

                block,
            )
        )


        statistic = correlation_statistic(
            px,
            py,
        )


        if (
            np.isfinite(
                statistic
            )
            and
            abs(
                statistic
            )
            >= abs(
                observed
            )
            - 1e-15
        ):

            extreme += 1


    p_value = (
        extreme + 1
    ) / (
        N_BLOCK_PERMUTATIONS + 1
    )


    # --------------------------------------------------------
    # CLUSTER BOOTSTRAP BY OUTCOME
    # --------------------------------------------------------

    bootstrap_values = []


    for _ in range(
        N_BLOCK_BOOTSTRAP
    ):

        sampled_blocks = rng.choice(

            valid_blocks,

            size=len(
                valid_blocks
            ),

            replace=True,
        )


        boot_parts = []


        for sample_index, (
            block_value
        ) in enumerate(
            sampled_blocks
        ):

            group_df = (
                groups[
                    block_value
                ]
                .copy()
            )


            # New artificial block ID is essential
            # when a block is sampled more than once.
            group_df[
                "__boot_block"
            ] = (
                f"{sample_index}"
            )


            boot_parts.append(
                group_df
            )


        boot = pd.concat(
            boot_parts,
            ignore_index=True,
        )


        bx, by = (
            centered_within_block_ranks(

                boot,

                x,

                y,

                "__boot_block",
            )
        )


        statistic = correlation_statistic(
            bx,
            by,
        )


        if np.isfinite(
            statistic
        ):

            bootstrap_values.append(
                statistic
            )


    if bootstrap_values:

        ci_low = float(
            np.percentile(
                bootstrap_values,
                2.5,
            )
        )


        ci_high = float(
            np.percentile(
                bootstrap_values,
                97.5,
            )
        )


    else:

        ci_low = np.nan

        ci_high = np.nan


    return {

        "metric_x":
            x,

        "metric_y":
            y,

        "block":
            block,

        "n_observations":
            len(
                data
            ),

        "n_blocks":
            len(
                valid_blocks
            ),

        "blocked_rank_correlation":
            observed,

        "cluster_bootstrap_ci_low":
            ci_low,

        "cluster_bootstrap_ci_high":
            ci_high,

        "p_blocked_permutation_raw":
            p_value,

        "n_permutations":
            N_BLOCK_PERMUTATIONS,

        "n_bootstrap":
            N_BLOCK_BOOTSTRAP,
    }


# ============================================================
# 46. PIPELINE CORRELATIONS
# ============================================================

pipeline_pairs = [

    (
        "url_jaccard_distance",
        "dense_distance_subject_normalized",
    ),

    (
        "domain_jaccard_distance",
        "dense_distance_subject_normalized",
    ),

    (
        "url_jaccard_distance",
        "evidence_semantic_gap_source_balanced",
    ),

    (
        "domain_jaccard_distance",
        "evidence_semantic_gap_source_balanced",
    ),

    (
        "evidence_semantic_gap_source_balanced",
        "dense_distance_subject_normalized",
    ),

    (
        "answer_support_semantic_gap",
        "dense_distance_subject_normalized",
    ),
]


pipeline_rows = []


for x, y in (
    pipeline_pairs
):

    if (
        x not in mechanism_map.columns
        or
        y not in mechanism_map.columns
    ):

        continue


    # --------------------------------------------------------
    # If evidence metrics are involved, enforce reference
    # collection quality.
    # --------------------------------------------------------

    if (
        "evidence_"
        in x
        or
        "evidence_"
        in y
        or
        "answer_support"
        in x
        or
        "answer_support"
        in y
    ):

        temp = mechanism_map[
            (
                mechanism_map[
                    "fetch_coverage"
                ]
                >=
                REFERENCE_COVERAGE_THRESHOLD
            )
            &
            (
                mechanism_map[
                    "n_available_sources"
                ]
                >=
                MIN_AVAILABLE_SOURCES
            )
        ].copy()


    else:

        temp = (
            mechanism_map.copy()
        )


    result = blocked_rank_correlation(

        temp,

        x,

        y,

        block="outcome_id",

        seed=stable_seed(
            f"blocked::{x}::{y}"
        ),
    )


    if result:

        pipeline_rows.append(
            result
        )


pipeline_correlations = pd.DataFrame(
    pipeline_rows
)


if len(
    pipeline_correlations
):

    reject, p_adj, _, _ = (
        multipletests(

            pipeline_correlations[
                "p_blocked_permutation_raw"
            ],

            alpha=ALPHA,

            method="holm",
        )
    )


    pipeline_correlations[
        "p_blocked_permutation_holm"
    ] = p_adj


    pipeline_correlations[
        "significant_holm"
    ] = reject


pipeline_correlations.to_csv(
    RESULTS_DIR
    / "pipeline_blocked_correlations_v2.csv",
    index=False,
)


# ============================================================
# 47. SIMPLE RAW SPEARMAN FOR REFERENCE ONLY
#
# Not the inferential result.
# ============================================================

raw_correlation_rows = []


for x, y in (
    pipeline_pairs
):

    if (
        x not in mechanism_map.columns
        or
        y not in mechanism_map.columns
    ):

        continue


    temp = (
        mechanism_map[
            [
                x,
                y,
            ]
        ]
        .dropna()
    )


    if len(
        temp
    ) < 10:

        continue


    rho, p_value = spearmanr(
        temp[
            x
        ],
        temp[
            y
        ],
    )


    raw_correlation_rows.append({

        "metric_x":
            x,

        "metric_y":
            y,

        "n":
            len(
                temp
            ),

        "spearman_rho_naive":
            float(
                rho
            ),

        "p_naive":
            float(
                p_value
            ),
    })


pd.DataFrame(
    raw_correlation_rows
).to_csv(
    RESULTS_DIR
    / "pipeline_naive_correlations_for_comparison.csv",
    index=False,
)


# ============================================================
# 48. COVERAGE SENSITIVITY STABILITY TABLE
#
# Makes it easy to see whether the effect flips as the
# threshold increases.
# ============================================================

if len(
    sensitivity_dimension_tests
):

    stability_table = (
        sensitivity_dimension_tests[
            [
                "coverage_threshold",
                "dimension",
                "n",
                "minority_mean",
                "majority_mean",
                "mean_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "p_permutation_raw",
                "p_permutation_holm",
            ]
        ]
        .copy()
    )


else:

    stability_table = (
        pd.DataFrame()
    )


stability_table.to_csv(
    RESULTS_DIR
    / "coverage_threshold_stability_v2.csv",
    index=False,
)


# ============================================================
# 49. MASTER JSON
# ============================================================

reference_eligible_count = int(
    (
        (
            query_metrics[
                "fetch_coverage"
            ]
            >=
            REFERENCE_COVERAGE_THRESHOLD
        )
        &
        (
            query_metrics[
                "n_available_sources"
            ]
            >=
            MIN_AVAILABLE_SOURCES
        )
    ).sum()
)


all_results = {

    "metadata": {

        "analysis":
            (
                "Evidence-to-synthesis "
                "semantic alignment V2"
            ),

        "collection":
            COLLECTION_VERSION,

        "embedding_model":
            MODEL_NAME,

        "embedding_type":
            "multilingual",

        "device":
            DEVICE,

        "source_chunk_tokens":
            SOURCE_CHUNK_TOKENS,

        "source_chunk_overlap":
            SOURCE_CHUNK_OVERLAP,

        "aio_chunk_tokens":
            AIO_CHUNK_TOKENS,

        "aio_chunk_overlap":
            AIO_CHUNK_OVERLAP,

        "top_passages_per_source":
            TOP_PASSAGES_PER_SOURCE,

        "reference_coverage_threshold":
            REFERENCE_COVERAGE_THRESHOLD,

        "coverage_sensitivity_thresholds":
            COVERAGE_THRESHOLDS,

        "minimum_available_sources":
            MIN_AVAILABLE_SOURCES,

        "primary_metric":
            PRIMARY_METRIC,

        "primary_metric_definition":
            (
                "1 minus source-balanced mean maximum "
                "cosine similarity between each source's "
                "query-relevant passages and AIO chunks."
            ),

        "terminology_caveat":
            (
                "Semantic gap is not equivalent to factual "
                "or claim omission. Formal claim survival "
                "requires claim extraction and entailment."
            ),

        "coverage_threshold_note":
            (
                "The 50% reference threshold is accompanied "
                "by 40-80% sensitivity analyses. Threshold "
                "choice concerns source availability, not "
                "observed evidence-alignment effect sizes."
            ),
    },


    "collection_quality": {

        "n_queries":
            len(
                query_metrics
            ),

        "mean_fetch_coverage":
            float(
                query_metrics[
                    "fetch_coverage"
                ].mean()
            ),

        "median_fetch_coverage":
            float(
                query_metrics[
                    "fetch_coverage"
                ].median()
            ),

        "reference_eligible_queries":
            reference_eligible_count,

        "n_usable_source_texts":
            len(
                usable_source_texts
            ),

        "n_source_chunks":
            len(
                source_chunks
            ),

        "n_truncated_sources":
            len(
                truncated_sources
            ),

        "coverage_by_group":
            dataframe_records(
                coverage_by_group
            ),

        "coverage_bias_tests":
            dataframe_records(
                coverage_bias_tests
            ),

        "metric_vs_coverage":
            dataframe_records(
                coverage_diagnostics
            ),
    },


    "descriptive": {

        "by_group":
            dataframe_records(
                group_summary
            ),
    },


    "reference_analysis_50pct": {

        "dimension_tests":
            dataframe_records(
                reference_dimension_tests
            ),

        "global_group_test":
            json_safe(
                global_test
            ),
    },


    "coverage_sensitivity": {

        "dimension_tests":
            dataframe_records(
                sensitivity_dimension_tests
            ),

        "aggregate_outcome_blocked":
            dataframe_records(
                aggregate_sensitivity
            ),
    },


    "secondary_answer_support": {

        "dimension_tests":
            dataframe_records(
                answer_support_sensitivity
            ),
    },


    "pipeline": {

        "blocked_correlations":
            dataframe_records(
                pipeline_correlations
            ),

        "naive_correlations_for_comparison":
            dataframe_records(
                pd.DataFrame(
                    raw_correlation_rows
                )
            ),
    },


    "audit_examples": {

        "weakly_aligned_relevant_evidence":
            dataframe_records(
                weakly_aligned.head(
                    100
                )
            ),

        "strongly_aligned_relevant_evidence":
            dataframe_records(
                strongly_aligned.head(
                    100
                )
            ),
    },
}


with open(
    RESULTS_DIR
    / "all_results_v2.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(

        json_safe(
            all_results
        ),

        f,

        indent=2,

        ensure_ascii=False,

        allow_nan=False,
    )


# ============================================================
# 50. README
# ============================================================

README = f"""
Google AI Overview — Evidence -> Synthesis V2
==============================================

MAIN PURPOSE
------------

Estimate semantic alignment between query-relevant information
in cited sources and the final Google AI Overview.

This is NOT a formal claim-survival analysis.

The central metric should be described as an
"evidence-AIO semantic gap" or
"semantic evidence alignment".

Do NOT call 1-cosine a literal omission probability.


WHY V2
------

V1 had two important problems:

1. English source passages could be compared against
   Portuguese AIO text using an English-centric model.

2. Broken headings / formatting could cause almost an entire
   AIO to be treated as one "sentence".

V2 uses:

- {MODEL_NAME}
- token-based source chunks
- token-based AIO chunks
- source-balanced aggregation
- exact experimental-label normalization
- coverage sensitivity analyses


SOURCE CHUNKS
-------------

Maximum:
    {SOURCE_CHUNK_TOKENS} tokens

Overlap:
    {SOURCE_CHUNK_OVERLAP} tokens

Selected per cited source:
    {TOP_PASSAGES_PER_SOURCE}


AIO CHUNKS
----------

Maximum:
    {AIO_CHUNK_TOKENS} tokens

Overlap:
    {AIO_CHUNK_OVERLAP} tokens


PRIMARY / REFERENCE SAMPLE
--------------------------

Reference threshold:
    >= {REFERENCE_COVERAGE_THRESHOLD:.0%}
    of cited sources successfully extracted

AND:
    >= {MIN_AVAILABLE_SOURCES}
    usable cited sources.


SENSITIVITY
-----------

Coverage thresholds:

    {", ".join(str(int(x * 100)) + "%" for x in COVERAGE_THRESHOLDS)}


PRIMARY METRIC
--------------

evidence_semantic_gap_source_balanced

For each source:

1. Identify top query-relevant passages.
2. Match each passage to its most similar AIO chunk.
3. Average passage similarities within source.
4. Average sources equally.
5. Gap = 1 - alignment.


INFERENCE
---------

Within each social dimension:

minority/focal condition
vs
majority/comparison condition

matched on the same 21 outcomes.

Tests:

- paired sign-flip permutation
- bootstrap CI
- Wilcoxon confirmation
- Holm correction across six dimensions


PIPELINE CORRELATIONS
---------------------

Repeated observations are blocked by outcome.

The inferential correlation:

1. ranks variables within outcome;
2. centers ranks within outcome;
3. measures within-outcome rank association;
4. permutes Y only within outcome;
5. cluster-bootstraps entire outcomes.


IMPORTANT FILES
---------------

query_evidence_alignment_metrics_v2.csv

source_evidence_alignment_metrics_v2.csv

selected_query_relevant_evidence_v2.csv

aio_chunks_v2.csv

evidence_passage_matches_v2.csv

aio_chunk_support_v2.csv

evidence_alignment_by_group_v2.csv

reference_50pct_dimension_tests_v2.csv

sensitivity_dimension_tests_v2.csv

coverage_threshold_stability_v2.csv

aggregate_sensitivity_v2.csv

answer_support_sensitivity_v2.csv

coverage_bias_tests_v2.csv

metric_vs_coverage_diagnostics_v2.csv

mechanism_map_v2.csv

pipeline_blocked_correlations_v2.csv

weakly_aligned_relevant_evidence.csv

strongly_aligned_relevant_evidence.csv

all_results_v2.json
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
# 51. PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("EVIDENCE -> SYNTHESIS V2 COMPLETE")
print("=" * 80)


print(
    f"\nMean fetch coverage: "
    f"{query_metrics['fetch_coverage'].mean():.3f}"
)


print(
    f"Median fetch coverage: "
    f"{query_metrics['fetch_coverage'].median():.3f}"
)


print(
    f"\nReference eligible queries "
    f"(>= {REFERENCE_COVERAGE_THRESHOLD:.0%} "
    f"and >= {MIN_AVAILABLE_SOURCES} sources): "
    f"{reference_eligible_count}"
)


print()
print(
    "REFERENCE 50% DIMENSION TESTS"
)


if len(
    reference_dimension_tests
):

    print(

        reference_dimension_tests[
            [
                "dimension",
                "n",
                "minority_group",
                "majority_group",
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
        .round(
            5
        )
        .to_string(
            index=False
        )
    )


else:

    print(
        "No valid paired tests "
        "at reference threshold."
    )


print()
print(
    "AGGREGATE COVERAGE SENSITIVITY"
)


if len(
    aggregate_sensitivity
):

    print(

        aggregate_sensitivity[
            [
                "coverage_threshold",
                "n_outcomes",
                "mean_dimensions_per_outcome",
                "mean_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "p_permutation_raw",
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
    "BLOCK-AWARE PIPELINE CORRELATIONS"
)


if len(
    pipeline_correlations
):

    print(

        pipeline_correlations[
            [
                "metric_x",
                "metric_y",
                "n_observations",
                "n_blocks",
                "blocked_rank_correlation",
                "cluster_bootstrap_ci_low",
                "cluster_bootstrap_ci_high",
                "p_blocked_permutation_holm",
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
    "Results directory:"
)

print(
    RESULTS_DIR
)


print()
print(
    "Main JSON:"
)

print(
    RESULTS_DIR
    / "all_results_v2.json"
)


print()
print(
    "Done."
)