# ============================================================
# GOOGLE AI OVERVIEW
# SEMANTIC DISPLACEMENT FROM GENERIC BASELINE
#
# Dallas version
#
# Outputs:
# /scratch/victoria.estanislau/ai-summary/results/
#     semantic_embedding_analysis_dallas/
#
# Main ideas:
#
# 1. Compare each explicit social-group AIO against the
#    matched generic "people" AIO.
#
# 2. Primary metric:
#       cosine distance from generic baseline
#
# 3. Robustness:
#       - raw text
#       - subject-normalized text
#       - TF-IDF lexical distance
#
# 4. Sentence-level:
#       - generic coverage
#       - group novelty
#       - symmetric sentence-set distance
#
# 5. Statistics:
#       - paired sign-flip permutation test
#       - Wilcoxon signed-rank
#       - paired bootstrap CI
#       - Holm correction
#       - Friedman global test
#       - Kendall's W
#
# ============================================================


from pathlib import Path
import os
from collections import defaultdict
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
    spearmanr,
    wilcoxon,
)

from statsmodels.stats.multitest import multipletests

from sklearn.feature_extraction.text import TfidfVectorizer

import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# 0. CONFIG
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
    / f"semantic_embedding_analysis_{LOCATION_SLUG}"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# EMBEDDING MODEL
# ------------------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/all-mpnet-base-v2"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

ALPHA = 0.05

RANDOM_SEED = 42

N_PERMUTATIONS = 200_000
N_BOOTSTRAP = 20_000

# Exact sign-flip test becomes 2^n.
# We use exact when <= 18 non-zero differences.
EXACT_SIGNFLIP_MAX_N = 18


# ------------------------------------------------------------
# SENTENCE SIMILARITY
# ------------------------------------------------------------

# Not used for significance testing.
# Continuous similarity is the main measure.
# Saved only as descriptive information if desired.
SIMILARITY_THRESHOLD = 0.75


# ------------------------------------------------------------
# EXPECTED DESIGN
# ------------------------------------------------------------

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
# 1. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# 2. GENERIC HELPERS
# ============================================================

def stable_seed(label):
    """
    Deterministic seed independent of Python's randomized hash().
    """

    return (
        RANDOM_SEED
        + zlib.crc32(
            str(label).encode("utf-8")
        )
    ) % (2**32 - 1)


def json_safe(value):

    if value is None:
        return None

    if isinstance(
        value,
        (np.integer,)
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,)
    ):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        (np.bool_,)
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
        (list, tuple)
    ):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def dataframe_records(df):

    if df is None or len(df) == 0:
        return []

    return json.loads(
        df.to_json(
            orient="records"
        )
    )


def l2_normalize(v):

    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm


def cosine_distance(v1, v2):

    similarity = float(
        np.dot(v1, v2)
    )

    similarity = np.clip(
        similarity,
        -1.0,
        1.0
    )

    return 1.0 - similarity


# ============================================================
# 3. TXT PARSING
# ============================================================

def normalize_newlines(text):

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
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
        rf"(?m)^\s*{re.escape(header)}\s*$",
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
            rf"(?m)^\s*{re.escape(next_header)}\s*$",
            remainder
        )

        if m:
            positions.append(
                m.start()
            )

    if positions:
        remainder = remainder[
            :min(positions)
        ]

    lines = remainder.splitlines()

    while lines and (
        not lines[0].strip()
        or re.fullmatch(
            r"=+",
            lines[0].strip()
        )
    ):
        lines.pop(0)

    while lines and (
        not lines[-1].strip()
        or re.fullmatch(
            r"=+",
            lines[-1].strip()
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

        key, value = line.split(
            ":",
            1
        )

        result[
            key.strip()
        ] = value.strip()

    return result


def parse_collection_file(filepath):

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

    aio = get_section(
        raw,
        "AI OVERVIEW TEXT",
        [
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ]
    ) or ""

    metadata_section = get_section(
        raw,
        "METADATA — DO NOT EDIT",
        []
    )

    metadata = parse_key_values(
        metadata_section
    )

    return {
        "query": query.strip(),
        "aio_text": aio.strip(),
        "metadata": metadata,
    }


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_semantic_text(text):
    """
    Removes markup/citation artifacts without changing
    substantive wording.
    """

    text = normalize_newlines(
        text
    )

    # Markdown links:
    # [Williams Institute](https://...)
    # ->
    # Williams Institute
    text = re.sub(
        r"\[([^\]]+)\]\(https?://[^)]+\)",
        r"\1",
        text
    )

    # Bare URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Numeric citations:
    # [1], [1, 2], [1, 2, 3]
    text = re.sub(
        r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]",
        " ",
        text
    )

    # Markdown headers
    text = re.sub(
        r"(?m)^\s*#+\s*",
        "",
        text
    )

    # Long separator lines
    text = re.sub(
        r"(?m)^\s*[-_=]{5,}\s*$",
        "\n",
        text
    )

    # Bullet markup
    text = re.sub(
        r"(?m)^\s*[\*\-•]\s+",
        "",
        text
    )

    # Collapse spaces, preserve newlines
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def subject_normalize(
    text,
    group
):
    """
    Sensitivity analysis.

    Replaces the exact experimental group label with
    the generic term 'people'.

    Examples:
        Black people -> people
        White people -> people
        Women -> people
        transgender people -> people

    Generic 'people' responses are unchanged.

    This tests whether the semantic-distance result is
    driven trivially by the explicit group string itself.
    """

    if not text:
        return text

    if group.casefold() == "people":
        return text

    pattern = re.compile(
        re.escape(group),
        flags=re.IGNORECASE
    )

    return pattern.sub(
        "people",
        text
    )


# ============================================================
# 5. SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Lightweight dependency-free splitter.

    Treats bullet/heading lines as semantic units and
    also splits normal sentences at punctuation.
    """

    if not text:
        return []

    text = clean_semantic_text(
        text
    )

    units = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        pieces = re.split(
            r"(?<=[.!?])\s+",
            line
        )

        for piece in pieces:

            piece = piece.strip()

            if len(piece) >= 15:
                units.append(
                    piece
                )

    return units


# ============================================================
# 6. LOAD COLLECTION
# ============================================================

print()
print("=" * 80)
print("LOADING COLLECTION")
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
            f"Missing folder: {folder}"
        )

    files = sorted(
        folder.glob("*.txt")
    )

    if len(files) != EXPECTED_FILES_PER_GROUP:

        raise RuntimeError(
            f"{folder_name}: "
            f"expected {EXPECTED_FILES_PER_GROUP} files, "
            f"found {len(files)}"
        )

    for filepath in files:

        parsed = parse_collection_file(
            filepath
        )

        m = parsed["metadata"]

        group = m.get(
            "Group",
            expected["group"]
        )

        dimension = m.get(
            "Dimension",
            expected["dimension"]
        )

        condition = m.get(
            "Condition",
            expected["condition"]
        )

        domain = m.get(
            "Domain",
            ""
        )

        outcome = m.get(
            "Outcome",
            ""
        )

        query_id = m.get(
            "Query ID",
            ""
        )

        aio_raw = parsed[
            "aio_text"
        ]

        aio_clean = clean_semantic_text(
            aio_raw
        )

        aio_normalized = (
            subject_normalize(
                aio_clean,
                group
            )
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

            "query_id":
                query_id,

            "dimension":
                dimension,

            "condition":
                condition,

            "group":
                group,

            "domain":
                domain,

            "outcome":
                outcome,

            "outcome_id":
                f"{domain} :: {outcome}",

            "query":
                parsed["query"],

            "aio_text":
                aio_raw,

            "semantic_text_raw":
                aio_clean,

            "semantic_text_subject_normalized":
                aio_normalized,

            "aio_present":
                bool(
                    aio_clean.strip()
                ),
        })


docs = pd.DataFrame(
    rows
)

docs = docs.sort_values(
    [
        "outcome_id",
        "group",
    ]
).reset_index(
    drop=True
)

docs["doc_id"] = np.arange(
    len(docs)
)


if len(docs) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} documents, "
        f"found {len(docs)}"
    )


# ============================================================
# 7. DESIGN VALIDATION
# ============================================================

print(
    f"\nDocuments: {len(docs)}"
)

print(
    f"Groups: {docs['group'].nunique()}"
)

print(
    f"Outcomes: {docs['outcome_id'].nunique()}"
)


# Each outcome should have all 13 conditions.
design_counts = (
    docs
    .groupby("outcome_id")
    ["group"]
    .nunique()
)

bad_design = design_counts[
    design_counts != 13
]

if len(bad_design):

    raise RuntimeError(
        "Some outcomes do not contain all 13 groups:\n"
        + bad_design.to_string()
    )


# Every group should have all 21 outcomes.
group_counts = (
    docs
    .groupby("group")
    ["outcome_id"]
    .nunique()
)

bad_groups = group_counts[
    group_counts != 21
]

if len(bad_groups):

    raise RuntimeError(
        "Some groups do not contain all 21 outcomes:\n"
        + bad_groups.to_string()
    )


# Empty AIO check.
empty_aio = docs[
    ~docs["aio_present"]
]

if len(empty_aio):

    print(
        "\nWARNING: empty AIOs detected."
    )

    print(
        empty_aio[
            [
                "group",
                "domain",
                "outcome",
                "file",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nThese observations will be excluded "
        "pairwise rather than converted to zero."
    )


print(
    "\n✓ Matched experimental design validated."
)


# ============================================================
# 8. LENGTH / VERBOSITY FEATURES
# ============================================================

for column, prefix in [
    (
        "semantic_text_raw",
        "raw"
    ),
    (
        "semantic_text_subject_normalized",
        "normalized"
    ),
]:

    docs[
        f"{prefix}_chars"
    ] = docs[column].str.len()

    docs[
        f"{prefix}_words"
    ] = docs[column].apply(
        lambda x: len(
            x.split()
        )
    )

    docs[
        f"{prefix}_sentences"
    ] = docs[column].apply(
        lambda x: len(
            split_sentences(x)
        )
    )


# ============================================================
# 9. LOAD EMBEDDING MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING EMBEDDING MODEL")
print("=" * 80)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Device: {DEVICE}"
)

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

tokenizer = model.tokenizer

model_max_length = int(
    model.max_seq_length
)

# Leave room for special tokens.
CHUNK_MAX_TOKENS = min(
    320,
    max(
        64,
        model_max_length - 16
    )
)

print(
    f"Model max sequence length: "
    f"{model_max_length}"
)

print(
    f"Chunk token target: "
    f"{CHUNK_MAX_TOKENS}"
)


# ============================================================
# 10. LONG-DOCUMENT CHUNKING
# ============================================================

def token_count(text):

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


def split_long_unit(
    text,
    max_tokens
):

    ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if len(ids) <= max_tokens:

        return [
            (
                text,
                max(
                    len(ids),
                    1
                )
            )
        ]

    parts = []

    for start in range(
        0,
        len(ids),
        max_tokens
    ):

        piece_ids = ids[
            start:
            start + max_tokens
        ]

        piece = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True
        )

        parts.append(
            (
                piece,
                len(piece_ids)
            )
        )

    return parts


def document_to_chunks(
    text,
    max_tokens
):

    sentences = split_sentences(
        text
    )

    if not sentences:

        return []

    atomic_units = []

    for sentence in sentences:

        atomic_units.extend(
            split_long_unit(
                sentence,
                max_tokens
            )
        )

    chunks = []

    current_text = []
    current_tokens = 0

    for unit_text, unit_tokens in atomic_units:

        if (
            current_text
            and
            current_tokens + unit_tokens
            > max_tokens
        ):

            chunks.append(
                (
                    " ".join(
                        current_text
                    ),
                    current_tokens
                )
            )

            current_text = []
            current_tokens = 0

        current_text.append(
            unit_text
        )

        current_tokens += (
            unit_tokens
        )

    if current_text:

        chunks.append(
            (
                " ".join(
                    current_text
                ),
                current_tokens
            )
        )

    return chunks


# ============================================================
# 11. CHUNKED DOCUMENT EMBEDDINGS
# ============================================================

def embed_documents_chunked(
    texts,
    label
):

    all_chunks = []

    all_weights = []

    document_chunk_indices = []

    cursor = 0

    for text in texts:

        chunks = document_to_chunks(
            text,
            CHUNK_MAX_TOKENS
        )

        indices = []

        for chunk_text, weight in chunks:

            all_chunks.append(
                chunk_text
            )

            all_weights.append(
                weight
            )

            indices.append(
                cursor
            )

            cursor += 1

        document_chunk_indices.append(
            indices
        )

    print(
        f"\n{label}: "
        f"{len(texts)} documents -> "
        f"{len(all_chunks)} chunks"
    )

    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    chunk_embeddings = np.asarray(
        chunk_embeddings,
        dtype=np.float32
    )

    doc_embeddings = []

    for indices in document_chunk_indices:

        if not indices:

            doc_embeddings.append(
                np.full(
                    chunk_embeddings.shape[1],
                    np.nan,
                    dtype=np.float32
                )
            )

            continue

        weights = np.asarray(
            [
                all_weights[i]
                for i in indices
            ],
            dtype=float
        )

        vecs = chunk_embeddings[
            indices
        ]

        pooled = np.average(
            vecs,
            axis=0,
            weights=weights
        )

        pooled = l2_normalize(
            pooled
        )

        doc_embeddings.append(
            pooled.astype(
                np.float32
            )
        )

    return np.vstack(
        doc_embeddings
    )


# ------------------------------------------------------------
# Raw presented text
# ------------------------------------------------------------

dense_raw = embed_documents_chunked(
    docs[
        "semantic_text_raw"
    ].tolist(),
    "RAW"
)


# ------------------------------------------------------------
# Subject-normalized sensitivity
# ------------------------------------------------------------

dense_normalized = (
    embed_documents_chunked(
        docs[
            "semantic_text_subject_normalized"
        ].tolist(),
        "SUBJECT NORMALIZED"
    )
)


# ============================================================
# 12. SAVE DOCUMENT EMBEDDINGS
# ============================================================

np.savez_compressed(
    RESULTS_DIR
    / "document_embeddings.npz",

    doc_id=docs[
        "doc_id"
    ].to_numpy(),

    dense_raw=
        dense_raw,

    dense_subject_normalized=
        dense_normalized,
)


# ============================================================
# 13. TF-IDF ROBUSTNESS BASELINE
# ============================================================

print()
print("=" * 80)
print("TF-IDF ROBUSTNESS BASELINE")
print("=" * 80)


tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
)

tfidf_matrix = tfidf.fit_transform(
    docs[
        "semantic_text_subject_normalized"
    ]
)


print(
    f"TF-IDF vocabulary size: "
    f"{len(tfidf.vocabulary_)}"
)


# ============================================================
# 14. SENTENCE EMBEDDINGS
# ============================================================

print()
print("=" * 80)
print("SENTENCE-LEVEL EMBEDDINGS")
print("=" * 80)


sentence_lists = []

all_sentences = []

sentence_doc_slices = []

cursor = 0

for text in docs[
    "semantic_text_subject_normalized"
]:

    sentences = split_sentences(
        text
    )

    sentence_lists.append(
        sentences
    )

    start = cursor

    all_sentences.extend(
        sentences
    )

    cursor += len(
        sentences
    )

    sentence_doc_slices.append(
        (
            start,
            cursor
        )
    )


print(
    f"Total semantic units: "
    f"{len(all_sentences)}"
)


sentence_embeddings = model.encode(
    all_sentences,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

sentence_embeddings = np.asarray(
    sentence_embeddings,
    dtype=np.float32
)


def get_sentence_embeddings(
    doc_id
):

    start, end = (
        sentence_doc_slices[
            int(doc_id)
        ]
    )

    return (
        sentence_embeddings[
            start:end
        ]
    )


# ============================================================
# 15. MATCH EACH EXPLICIT GROUP TO GENERIC
# ============================================================

generic_docs = docs[
    docs["condition"] == "control"
].copy()

if len(generic_docs) != 21:

    raise RuntimeError(
        "Expected exactly 21 generic 'people' documents."
    )


generic_by_outcome = {
    row["outcome_id"]:
        int(row["doc_id"])

    for _, row
    in generic_docs.iterrows()
}


pair_rows = []


for _, row in docs.iterrows():

    if row[
        "condition"
    ] == "control":
        continue

    explicit_id = int(
        row["doc_id"]
    )

    generic_id = (
        generic_by_outcome[
            row["outcome_id"]
        ]
    )

    generic_row = docs.loc[
        docs["doc_id"]
        == generic_id
    ].iloc[0]


    # --------------------------------------------------------
    # Dense raw distance
    # --------------------------------------------------------

    raw_distance = (
        cosine_distance(
            dense_raw[
                explicit_id
            ],
            dense_raw[
                generic_id
            ]
        )
    )


    # --------------------------------------------------------
    # Dense subject-normalized distance
    # --------------------------------------------------------

    normalized_distance = (
        cosine_distance(
            dense_normalized[
                explicit_id
            ],
            dense_normalized[
                generic_id
            ]
        )
    )


    # --------------------------------------------------------
    # TF-IDF distance
    # --------------------------------------------------------

    tfidf_similarity = float(
        tfidf_matrix[
            explicit_id
        ].multiply(
            tfidf_matrix[
                generic_id
            ]
        ).sum()
    )

    tfidf_similarity = np.clip(
        tfidf_similarity,
        0.0,
        1.0
    )

    tfidf_distance = (
        1.0
        - tfidf_similarity
    )


    # --------------------------------------------------------
    # Sentence-set comparison
    # --------------------------------------------------------

    explicit_sent = (
        get_sentence_embeddings(
            explicit_id
        )
    )

    generic_sent = (
        get_sentence_embeddings(
            generic_id
        )
    )


    if (
        len(explicit_sent)
        and len(generic_sent)
    ):

        sim_matrix = (
            generic_sent
            @ explicit_sent.T
        )

        sim_matrix = np.clip(
            sim_matrix,
            -1.0,
            1.0
        )


        # For every generic sentence:
        # nearest sentence in explicit-group response.
        generic_best = (
            sim_matrix.max(
                axis=1
            )
        )


        # For every explicit-group sentence:
        # nearest sentence in generic response.
        group_best = (
            sim_matrix.max(
                axis=0
            )
        )


        generic_coverage_similarity = float(
            generic_best.mean()
        )

        generic_coverage_distance = (
            1.0
            - generic_coverage_similarity
        )


        group_to_generic_similarity = float(
            group_best.mean()
        )

        group_novelty_distance = (
            1.0
            - group_to_generic_similarity
        )


        symmetric_sentence_distance = (
            1.0
            -
            (
                generic_coverage_similarity
                +
                group_to_generic_similarity
            )
            / 2.0
        )


        generic_coverage_rate_075 = float(
            np.mean(
                generic_best
                >= SIMILARITY_THRESHOLD
            )
        )

        group_matched_rate_075 = float(
            np.mean(
                group_best
                >= SIMILARITY_THRESHOLD
            )
        )

    else:

        generic_coverage_similarity = np.nan
        generic_coverage_distance = np.nan

        group_to_generic_similarity = np.nan
        group_novelty_distance = np.nan

        symmetric_sentence_distance = np.nan

        generic_coverage_rate_075 = np.nan
        group_matched_rate_075 = np.nan


    # --------------------------------------------------------
    # Length diagnostics
    # --------------------------------------------------------

    explicit_words = (
        row[
            "normalized_words"
        ]
    )

    generic_words = (
        generic_row[
            "normalized_words"
        ]
    )

    abs_word_difference = abs(
        explicit_words
        - generic_words
    )

    signed_word_difference = (
        explicit_words
        - generic_words
    )

    if generic_words > 0:

        word_length_ratio = (
            explicit_words
            / generic_words
        )

        abs_log_word_ratio = abs(
            math.log(
                max(
                    word_length_ratio,
                    1e-12
                )
            )
        )

    else:

        word_length_ratio = np.nan
        abs_log_word_ratio = np.nan


    pair_rows.append({

        "outcome_id":
            row["outcome_id"],

        "domain":
            row["domain"],

        "outcome":
            row["outcome"],

        "dimension":
            row["dimension"],

        "condition":
            row["condition"],

        "group":
            row["group"],

        "explicit_doc_id":
            explicit_id,

        "generic_doc_id":
            generic_id,


        # Full-answer distances
        "dense_distance_raw":
            raw_distance,

        "dense_distance_subject_normalized":
            normalized_distance,

        "tfidf_distance_subject_normalized":
            tfidf_distance,


        # Sentence-level
        "generic_coverage_similarity":
            generic_coverage_similarity,

        "generic_coverage_distance":
            generic_coverage_distance,

        "group_to_generic_similarity":
            group_to_generic_similarity,

        "group_novelty_distance":
            group_novelty_distance,

        "symmetric_sentence_distance":
            symmetric_sentence_distance,

        "generic_coverage_rate_075":
            generic_coverage_rate_075,

        "group_matched_rate_075":
            group_matched_rate_075,


        # Verbosity
        "explicit_words":
            explicit_words,

        "generic_words":
            generic_words,

        "signed_word_difference":
            signed_word_difference,

        "abs_word_difference":
            abs_word_difference,

        "word_length_ratio":
            word_length_ratio,

        "abs_log_word_ratio":
            abs_log_word_ratio,
    })


pairs = pd.DataFrame(
    pair_rows
)


pairs.to_csv(
    RESULTS_DIR
    / "group_vs_generic_semantic_metrics.csv",
    index=False,
)


# ============================================================
# 16. PRIMARY STATISTICAL FUNCTIONS
# ============================================================

def paired_rank_biserial(
    differences
):
    """
    Paired rank-biserial correlation.

    Positive:
        first condition tends to be larger.

    Negative:
        second tends to be larger.
    """

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

    pos = ranks[
        d > 0
    ].sum()

    neg = ranks[
        d < 0
    ].sum()

    return float(
        (pos - neg)
        / (pos + neg)
    )


def paired_bootstrap_ci(
    differences,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
):
    """
    Resamples MATCHED OUTCOMES, not individual responses.
    """

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

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        n_boot
    )

    n = len(d)

    for i in range(
        n_boot
    ):

        sampled = rng.choice(
            d,
            size=n,
            replace=True
        )

        boot[i] = (
            sampled.mean()
        )

    return (
        float(
            np.percentile(
                boot,
                2.5
            )
        ),

        float(
            np.percentile(
                boot,
                97.5
            )
        )
    )


def signflip_permutation_test(
    differences,
    n_resamples=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):
    """
    Two-sided paired sign-flip permutation test.

    H0:
        distribution of paired differences is symmetric
        around zero.

    Statistic:
        absolute mean paired difference.

    Exact enumeration when small enough.
    Monte Carlo otherwise.
    """

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
    # Exact
    # --------------------------------------------------------

    if n <= EXACT_SIGNFLIP_MAX_N:

        total = (
            2 ** n
        )

        extreme = 0

        batch_size = 10_000

        powers = (
            1
            << np.arange(
                n,
                dtype=np.uint64
            )
        )

        for start in range(
            0,
            total,
            batch_size
        ):

            stop = min(
                start + batch_size,
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
                    np.float64
                )
                * 2.0
                - 1.0
            )

            perm_stats = np.abs(
                (
                    signs
                    * d[None, :]
                ).mean(
                    axis=1
                )
            )

            extreme += int(
                np.sum(
                    perm_stats
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
    # Monte Carlo
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
            < n_resamples
        ):

            batch = min(
                batch_size,
                n_resamples
                - completed
            )

            signs = rng.choice(
                [-1.0, 1.0],
                size=(
                    batch,
                    n
                )
            )

            perm_stats = np.abs(
                (
                    signs
                    * d[None, :]
                ).mean(
                    axis=1
                )
            )

            extreme += int(
                np.sum(
                    perm_stats
                    >= observed
                    - 1e-15
                )
            )

            completed += batch


        # +1 correction for Monte Carlo tests.
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
            float(observed),

        "p_value":
            float(p_value),

        "n_nonzero":
            int(n),

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

    # Avoid microscopic floating-point differences.
    d = np.round(
        d,
        decimals=12
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

            # Compatibility with older scipy.
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
    x,
    y,
    label
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    difference = (
        x - y
    )

    permutation = (
        signflip_permutation_test(
            difference,
            seed=stable_seed(
                label
            )
        )
    )

    w_stat, w_p = (
        safe_wilcoxon(
            difference
        )
    )

    ci_low, ci_high = (
        paired_bootstrap_ci(
            difference,
            seed=stable_seed(
                f"bootstrap::{label}"
            )
        )
    )

    return {
        "n":
            len(difference),

        "mean_1":
            float(
                np.mean(x)
            ),

        "mean_2":
            float(
                np.mean(y)
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
            w_stat,

        "p_wilcoxon_raw":
            w_p,
    }


# ============================================================
# 17. WITHIN-DIMENSION MINORITY vs MAJORITY
# ============================================================

def dimension_tests(
    metric_column,
    metric_label,
):

    results = []

    dimensions = sorted(
        pairs[
            "dimension"
        ].unique()
    )

    for dimension in dimensions:

        subset = pairs[
            pairs[
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
            len(minority_groups) != 1
            or
            len(majority_groups) != 1
        ):
            raise RuntimeError(
                f"Invalid design for {dimension}"
            )

        minority = (
            minority_groups[0]
        )

        majority = (
            majority_groups[0]
        )

        wide = (
            subset[
                [
                    "outcome_id",
                    "group",
                    metric_column,
                ]
            ]
            .pivot(
                index="outcome_id",
                columns="group",
                values=metric_column,
            )
        )

        wide = wide.dropna(
            subset=[
                minority,
                majority,
            ]
        )

        stats = complete_paired_test(
            wide[
                minority
            ].values,

            wide[
                majority
            ].values,

            label=(
                f"{metric_label}"
                f"::{dimension}"
            )
        )

        stats.update({
            "metric":
                metric_label,

            "dimension":
                dimension,

            "minority_group":
                minority,

            "majority_group":
                majority,
        })

        results.append(
            stats
        )


    result_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Holm correction:
    # six planned dimension-level tests form one family.
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


    # Also correct secondary Wilcoxon confirmation.
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
# 18. PRIMARY ANALYSIS
#
# Subject-normalized semantic distance is the strongest
# version because it reduces trivial identity-string effects.
# ============================================================

print()
print("=" * 80)
print("PRIMARY: SUBJECT-NORMALIZED SEMANTIC DISTANCE")
print("=" * 80)


primary_dimension = dimension_tests(
    "dense_distance_subject_normalized",
    "dense_subject_normalized_distance"
)

print(
    primary_dimension[
        [
            "dimension",
            "minority_group",
            "majority_group",
            "n",
            "mean_1",
            "mean_2",
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

primary_dimension.to_csv(
    RESULTS_DIR
    / "primary_dimension_tests_dense_subject_normalized.csv",
    index=False,
)


# ============================================================
# 19. RAW-TEXT ROBUSTNESS
# ============================================================

raw_dimension = dimension_tests(
    "dense_distance_raw",
    "dense_raw_distance"
)

raw_dimension.to_csv(
    RESULTS_DIR
    / "robustness_dimension_tests_dense_raw.csv",
    index=False,
)


# ============================================================
# 20. TF-IDF ROBUSTNESS
# ============================================================

tfidf_dimension = dimension_tests(
    "tfidf_distance_subject_normalized",
    "tfidf_subject_normalized_distance"
)

tfidf_dimension.to_csv(
    RESULTS_DIR
    / "robustness_dimension_tests_tfidf.csv",
    index=False,
)


# ============================================================
# 21. SENTENCE-LEVEL ANALYSES
# ============================================================

coverage_dimension = dimension_tests(
    "generic_coverage_distance",
    "generic_coverage_distance"
)

coverage_dimension.to_csv(
    RESULTS_DIR
    / "secondary_dimension_tests_generic_coverage.csv",
    index=False,
)


novelty_dimension = dimension_tests(
    "group_novelty_distance",
    "group_novelty_distance"
)

novelty_dimension.to_csv(
    RESULTS_DIR
    / "secondary_dimension_tests_group_novelty.csv",
    index=False,
)


sentence_symmetric_dimension = (
    dimension_tests(
        "symmetric_sentence_distance",
        "symmetric_sentence_distance"
    )
)

sentence_symmetric_dimension.to_csv(
    RESULTS_DIR
    / "secondary_dimension_tests_sentence_set_distance.csv",
    index=False,
)


# ============================================================
# 22. GLOBAL MINORITY vs MAJORITY
#
# To avoid pseudo-replication:
#
# For each of the 21 outcomes:
#     mean across 6 minority conditions
#     mean across 6 majority conditions
#
# Then compare those 21 matched values.
# ============================================================

def aggregate_condition_test(
    metric_column,
    metric_label,
):

    temp = (
        pairs
        .groupby(
            [
                "outcome_id",
                "condition",
            ]
        )[
            metric_column
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
            f"aggregate::{metric_label}"
        )
    )

    result.update({
        "metric":
            metric_label,

        "unit":
            "outcome",

        "description":
            (
                "For each outcome, average "
                "across six minority conditions "
                "and six majority conditions, "
                "then compare the 21 matched "
                "outcome-level means."
            )
    })

    return result


aggregate_primary = (
    aggregate_condition_test(
        "dense_distance_subject_normalized",
        "dense_subject_normalized_distance"
    )
)


aggregate_raw = (
    aggregate_condition_test(
        "dense_distance_raw",
        "dense_raw_distance"
    )
)


aggregate_tfidf = (
    aggregate_condition_test(
        "tfidf_distance_subject_normalized",
        "tfidf_subject_normalized_distance"
    )
)


aggregate_coverage = (
    aggregate_condition_test(
        "generic_coverage_distance",
        "generic_coverage_distance"
    )
)


aggregate_novelty = (
    aggregate_condition_test(
        "group_novelty_distance",
        "group_novelty_distance"
    )
)


aggregate_df = pd.DataFrame([
    aggregate_primary,
    aggregate_raw,
    aggregate_tfidf,
    aggregate_coverage,
    aggregate_novelty,
])


aggregate_df.to_csv(
    RESULTS_DIR
    / "aggregate_minority_vs_majority.csv",
    index=False,
)


# ============================================================
# 23. GLOBAL GROUP EFFECT: FRIEDMAN
#
# Compare 12 explicit groups on their semantic distance
# from the same generic baseline.
# ============================================================

def friedman_group_test(
    metric_column,
    metric_label,
):

    wide = pairs.pivot(
        index="outcome_id",
        columns="group",
        values=metric_column,
    )

    complete = wide.dropna()

    if (
        len(complete) < 2
        or complete.shape[1] < 3
    ):

        return {
            "metric":
                metric_label,

            "n_outcomes":
                len(complete),

            "n_groups":
                complete.shape[1],

            "friedman_chi2":
                None,

            "p_value":
                None,

            "kendall_W":
                None,
        }


    stat, p = friedmanchisquare(
        *[
            complete[col].values
            for col in complete.columns
        ]
    )


    n = complete.shape[0]
    k = complete.shape[1]

    kendall_w = (
        stat
        /
        (
            n
            *
            (k - 1)
        )
    )


    return {
        "metric":
            metric_label,

        "n_outcomes":
            n,

        "n_groups":
            k,

        "friedman_chi2":
            float(stat),

        "p_value":
            float(p),

        "kendall_W":
            float(kendall_w),

        "significant":
            bool(
                p < ALPHA
            ),
    }


global_tests = pd.DataFrame([
    friedman_group_test(
        "dense_distance_subject_normalized",
        "dense_subject_normalized_distance"
    ),

    friedman_group_test(
        "dense_distance_raw",
        "dense_raw_distance"
    ),

    friedman_group_test(
        "tfidf_distance_subject_normalized",
        "tfidf_subject_normalized_distance"
    ),

    friedman_group_test(
        "generic_coverage_distance",
        "generic_coverage_distance"
    ),

    friedman_group_test(
        "group_novelty_distance",
        "group_novelty_distance"
    ),
])


global_tests.to_csv(
    RESULTS_DIR
    / "global_friedman_tests.csv",
    index=False,
)


# ============================================================
# 24. POST-HOC EXPLICIT-GROUP COMPARISONS
#
# Exploratory.
# Only for the PRIMARY metric.
#
# 12 groups => 66 comparisons.
# Holm correction across all 66.
# ============================================================

def posthoc_group_tests(
    metric_column
):

    wide = pairs.pivot(
        index="outcome_id",
        columns="group",
        values=metric_column,
    )

    groups = list(
        wide.columns
    )

    results = []

    for g1, g2 in itertools.combinations(
        groups,
        2
    ):

        temp = wide[
            [
                g1,
                g2,
            ]
        ].dropna()

        stats = complete_paired_test(
            temp[
                g1
            ].values,

            temp[
                g2
            ].values,

            label=(
                f"posthoc::{g1}::{g2}"
            )
        )

        stats.update({
            "group_1":
                g1,

            "group_2":
                g2,
        })

        results.append(
            stats
        )


    result_df = pd.DataFrame(
        results
    )


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
        "significant_holm"
    ] = reject


    result_df = (
        result_df
        .sort_values(
            [
                "significant_holm",
                "p_permutation_holm",
            ],
            ascending=[
                False,
                True,
            ]
        )
    )

    return result_df


posthoc = posthoc_group_tests(
    "dense_distance_subject_normalized"
)

posthoc.to_csv(
    RESULTS_DIR
    / "exploratory_posthoc_group_tests.csv",
    index=False,
)


# ============================================================
# 25. TRIANGLE ANALYSIS
#
# For each dimension/outcome:
#
# Minority <-> Generic
# Majority <-> Generic
# Minority <-> Majority
#
# Positive generic_proximity_margin:
#     minority is CLOSER to generic than majority.
#
# Negative:
#     majority is closer.
# ============================================================

triangle_rows = []


for dimension in sorted(
    pairs[
        "dimension"
    ].unique()
):

    subset = pairs[
        pairs[
            "dimension"
        ]
        == dimension
    ]

    minority_group = (
        subset.loc[
            subset[
                "condition"
            ] == "minority",
            "group",
        ]
        .unique()[0]
    )

    majority_group = (
        subset.loc[
            subset[
                "condition"
            ] == "majority",
            "group",
        ]
        .unique()[0]
    )


    for outcome_id in sorted(
        subset[
            "outcome_id"
        ].unique()
    ):

        min_row = subset[
            (
                subset[
                    "outcome_id"
                ]
                == outcome_id
            )
            &
            (
                subset[
                    "group"
                ]
                == minority_group
            )
        ].iloc[0]

        maj_row = subset[
            (
                subset[
                    "outcome_id"
                ]
                == outcome_id
            )
            &
            (
                subset[
                    "group"
                ]
                == majority_group
            )
        ].iloc[0]


        min_id = int(
            min_row[
                "explicit_doc_id"
            ]
        )

        maj_id = int(
            maj_row[
                "explicit_doc_id"
            ]
        )


        d_min_generic = float(
            min_row[
                "dense_distance_subject_normalized"
            ]
        )

        d_maj_generic = float(
            maj_row[
                "dense_distance_subject_normalized"
            ]
        )


        d_min_majority = (
            cosine_distance(
                dense_normalized[
                    min_id
                ],
                dense_normalized[
                    maj_id
                ]
            )
        )


        # Positive:
        # majority farther from generic,
        # therefore minority closer to generic.
        generic_proximity_margin = (
            d_maj_generic
            -
            d_min_generic
        )


        original = docs[
            docs[
                "outcome_id"
            ]
            == outcome_id
        ].iloc[0]


        triangle_rows.append({

            "dimension":
                dimension,

            "minority_group":
                minority_group,

            "majority_group":
                majority_group,

            "outcome_id":
                outcome_id,

            "domain":
                original["domain"],

            "outcome":
                original["outcome"],

            "distance_minority_generic":
                d_min_generic,

            "distance_majority_generic":
                d_maj_generic,

            "distance_minority_majority":
                d_min_majority,

            "generic_proximity_margin":
                generic_proximity_margin,

            "minority_closer_to_generic":
                bool(
                    generic_proximity_margin
                    > 0
                ),
        })


triangles = pd.DataFrame(
    triangle_rows
)


triangles.to_csv(
    RESULTS_DIR
    / "semantic_triangle_metrics.csv",
    index=False,
)


# ============================================================
# 26. TRIANGLE SUMMARY
# ============================================================

triangle_summary_rows = []


for dimension, g in (
    triangles.groupby(
        "dimension"
    )
):

    margins = g[
        "generic_proximity_margin"
    ].values


    permutation = (
        signflip_permutation_test(
            margins,

            seed=stable_seed(
                f"triangle::{dimension}"
            )
        )
    )


    ci_low, ci_high = (
        paired_bootstrap_ci(
            margins,

            seed=stable_seed(
                f"triangle_bootstrap::{dimension}"
            )
        )
    )


    triangle_summary_rows.append({

        "dimension":
            dimension,

        "minority_group":
            g[
                "minority_group"
            ].iloc[0],

        "majority_group":
            g[
                "majority_group"
            ].iloc[0],

        "n_outcomes":
            len(g),

        "mean_generic_proximity_margin":
            margins.mean(),

        "median_generic_proximity_margin":
            np.median(
                margins
            ),

        "bootstrap_ci_low":
            ci_low,

        "bootstrap_ci_high":
            ci_high,

        "proportion_minority_closer_to_generic":
            np.mean(
                margins > 0
            ),

        "proportion_majority_closer_to_generic":
            np.mean(
                margins < 0
            ),

        "p_permutation_raw":
            permutation[
                "p_value"
            ],
    })


triangle_summary = pd.DataFrame(
    triangle_summary_rows
)


reject, p_adj, _, _ = (
    multipletests(
        triangle_summary[
            "p_permutation_raw"
        ],
        alpha=ALPHA,
        method="holm",
    )
)

triangle_summary[
    "p_permutation_holm"
] = p_adj

triangle_summary[
    "significant_holm"
] = reject


triangle_summary.to_csv(
    RESULTS_DIR
    / "semantic_triangle_summary.csv",
    index=False,
)


# ============================================================
# 27. LENGTH-CONFOUND DIAGNOSTICS
#
# These do NOT "control away" length automatically.
# They tell us whether semantic distance is strongly associated
# with answer-length differences.
# ============================================================

diagnostic_metrics = [
    "dense_distance_raw",
    "dense_distance_subject_normalized",
    "tfidf_distance_subject_normalized",
    "generic_coverage_distance",
    "group_novelty_distance",
    "symmetric_sentence_distance",
]


length_variables = [
    "abs_word_difference",
    "abs_log_word_ratio",
]


diagnostics = []


for metric in diagnostic_metrics:

    for length_var in length_variables:

        temp = pairs[
            [
                metric,
                length_var,
            ]
        ].dropna()

        rho, p = spearmanr(
            temp[
                metric
            ],
            temp[
                length_var
            ]
        )

        diagnostics.append({

            "metric":
                metric,

            "length_variable":
                length_var,

            "n":
                len(temp),

            "spearman_rho":
                float(rho),

            "p_raw":
                float(p),
        })


diagnostics = pd.DataFrame(
    diagnostics
)


reject, p_adj, _, _ = (
    multipletests(
        diagnostics[
            "p_raw"
        ],
        alpha=ALPHA,
        method="holm",
    )
)

diagnostics[
    "p_holm"
] = p_adj

diagnostics[
    "significant_holm"
] = reject


diagnostics.to_csv(
    RESULTS_DIR
    / "length_confound_diagnostics.csv",
    index=False,
)


# ============================================================
# 28. AGREEMENT BETWEEN EMBEDDING / LEXICAL METRICS
# ============================================================

agreement_rows = []


metric_pairs = [

    (
        "dense_distance_raw",
        "dense_distance_subject_normalized"
    ),

    (
        "dense_distance_subject_normalized",
        "tfidf_distance_subject_normalized"
    ),

    (
        "dense_distance_subject_normalized",
        "symmetric_sentence_distance"
    ),

    (
        "dense_distance_subject_normalized",
        "generic_coverage_distance"
    ),
]


for m1, m2 in metric_pairs:

    temp = pairs[
        [
            m1,
            m2,
        ]
    ].dropna()

    rho, p = spearmanr(
        temp[m1],
        temp[m2]
    )

    agreement_rows.append({

        "metric_1":
            m1,

        "metric_2":
            m2,

        "n":
            len(temp),

        "spearman_rho":
            float(rho),

        "p_value":
            float(p),
    })


metric_agreement = pd.DataFrame(
    agreement_rows
)

metric_agreement.to_csv(
    RESULTS_DIR
    / "metric_agreement.csv",
    index=False,
)


# ============================================================
# 29. DESCRIPTIVE TABLE BY GROUP
# ============================================================

group_summary = (
    pairs
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

        mean_dense_distance=(
            "dense_distance_subject_normalized",
            "mean"
        ),

        median_dense_distance=(
            "dense_distance_subject_normalized",
            "median"
        ),

        sd_dense_distance=(
            "dense_distance_subject_normalized",
            "std"
        ),

        mean_dense_raw_distance=(
            "dense_distance_raw",
            "mean"
        ),

        mean_tfidf_distance=(
            "tfidf_distance_subject_normalized",
            "mean"
        ),

        mean_generic_coverage_similarity=(
            "generic_coverage_similarity",
            "mean"
        ),

        mean_group_novelty_distance=(
            "group_novelty_distance",
            "mean"
        ),

        mean_sentence_set_distance=(
            "symmetric_sentence_distance",
            "mean"
        ),

        mean_word_length_ratio=(
            "word_length_ratio",
            "mean"
        ),
    )
    .reset_index()
)


group_summary = (
    group_summary
    .sort_values(
        "mean_dense_distance",
        ascending=False,
    )
)


group_summary.to_csv(
    RESULTS_DIR
    / "semantic_metrics_by_group.csv",
    index=False,
)


print()
print("=" * 80)
print("MEAN SEMANTIC DISTANCE FROM GENERIC")
print("=" * 80)

print(
    group_summary[
        [
            "dimension",
            "condition",
            "group",
            "mean_dense_distance",
            "mean_generic_coverage_similarity",
            "mean_group_novelty_distance",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 30. DESCRIPTIVE TABLE BY DOMAIN
# ============================================================

domain_summary = (
    pairs
    .groupby(
        "domain"
    )
    .agg(

        n=(
            "outcome_id",
            "size"
        ),

        mean_dense_distance=(
            "dense_distance_subject_normalized",
            "mean"
        ),

        median_dense_distance=(
            "dense_distance_subject_normalized",
            "median"
        ),

        mean_generic_coverage_similarity=(
            "generic_coverage_similarity",
            "mean"
        ),

        mean_group_novelty_distance=(
            "group_novelty_distance",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_dense_distance",
        ascending=False,
    )
)


domain_summary.to_csv(
    RESULTS_DIR
    / "semantic_metrics_by_domain.csv",
    index=False,
)


# ============================================================
# 31. DESCRIPTIVE TABLE BY OUTCOME
# ============================================================

outcome_summary = (
    pairs
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

        mean_dense_distance=(
            "dense_distance_subject_normalized",
            "mean"
        ),

        median_dense_distance=(
            "dense_distance_subject_normalized",
            "median"
        ),

        mean_generic_coverage_similarity=(
            "generic_coverage_similarity",
            "mean"
        ),

        mean_group_novelty_distance=(
            "group_novelty_distance",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "mean_dense_distance",
        ascending=False,
    )
)


outcome_summary.to_csv(
    RESULTS_DIR
    / "semantic_metrics_by_outcome.csv",
    index=False,
)


# ============================================================
# 32. SAVE DOCUMENT-LEVEL TABLE
#
# Do not place full AIO text in CSV by default because it makes
# the file awkward. It remains in the original collection.
# ============================================================

document_export_columns = [
    "doc_id",
    "file",
    "query_id",
    "dimension",
    "condition",
    "group",
    "domain",
    "outcome",
    "outcome_id",
    "query",
    "aio_present",
    "raw_chars",
    "raw_words",
    "raw_sentences",
    "normalized_chars",
    "normalized_words",
    "normalized_sentences",
]


docs[
    document_export_columns
].to_csv(
    RESULTS_DIR
    / "document_metadata.csv",
    index=False,
)


# ============================================================
# 33. BUILD MASTER JSON
# ============================================================

all_results = {

    "metadata": {

        "analysis":
            (
                "Google AI Overview semantic "
                "displacement from generic baseline"
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

        "embedding_model":
            MODEL_NAME,

        "device":
            DEVICE,

        "model_max_sequence_length":
            model_max_length,

        "chunk_max_tokens":
            CHUNK_MAX_TOKENS,

        "random_seed":
            RANDOM_SEED,

        "alpha":
            ALPHA,

        "permutation_resamples":
            N_PERMUTATIONS,

        "bootstrap_resamples":
            N_BOOTSTRAP,

        "primary_metric":
            (
                "dense_distance_subject_normalized"
            ),

        "primary_test":
            (
                "paired sign-flip permutation test; "
                "Holm correction across six "
                "social dimensions"
            ),

        "secondary_confirmatory_test":
            (
                "Wilcoxon signed-rank test"
            ),

        "notes": [
            (
                "Statistical unit is matched outcome, "
                "not individual AIO response."
            ),

            (
                "Full answers are embedded via "
                "token-aware chunking and weighted "
                "embedding pooling."
            ),

            (
                "Subject-normalized analysis replaces "
                "the exact experimental group label "
                "with 'people' before embedding."
            ),

            (
                "TF-IDF is treated as a lexical "
                "robustness baseline, not independent "
                "confirmatory evidence."
            ),

            (
                "Sentence-level coverage and novelty "
                "are secondary analyses."
            ),
        ],
    },


    "dataset": {

        "n_documents":
            len(docs),

        "n_explicit_group_documents":
            len(pairs),

        "n_groups_total":
            docs[
                "group"
            ].nunique(),

        "n_explicit_groups":
            pairs[
                "group"
            ].nunique(),

        "n_outcomes":
            docs[
                "outcome_id"
            ].nunique(),

        "n_dimensions":
            pairs[
                "dimension"
            ].nunique(),

        "n_empty_aio":
            int(
                (
                    ~docs[
                        "aio_present"
                    ]
                ).sum()
            ),
    },


    "primary_analysis": {

        "dimension_tests":
            dataframe_records(
                primary_dimension
            ),

        "aggregate_minority_vs_majority":
            json_safe(
                aggregate_primary
            ),

        "global_friedman":
            json_safe(
                friedman_group_test(
                    "dense_distance_subject_normalized",
                    "dense_subject_normalized_distance"
                )
            ),

        "posthoc_group_tests":
            dataframe_records(
                posthoc
            ),

        "triangle_summary":
            dataframe_records(
                triangle_summary
            ),
    },


    "robustness": {

        "raw_dense_dimension_tests":
            dataframe_records(
                raw_dimension
            ),

        "tfidf_dimension_tests":
            dataframe_records(
                tfidf_dimension
            ),

        "aggregate_raw_dense":
            json_safe(
                aggregate_raw
            ),

        "aggregate_tfidf":
            json_safe(
                aggregate_tfidf
            ),

        "metric_agreement":
            dataframe_records(
                metric_agreement
            ),
    },


    "sentence_level": {

        "generic_coverage_dimension_tests":
            dataframe_records(
                coverage_dimension
            ),

        "group_novelty_dimension_tests":
            dataframe_records(
                novelty_dimension
            ),

        "symmetric_sentence_distance_tests":
            dataframe_records(
                sentence_symmetric_dimension
            ),

        "aggregate_generic_coverage":
            json_safe(
                aggregate_coverage
            ),

        "aggregate_group_novelty":
            json_safe(
                aggregate_novelty
            ),
    },


    "diagnostics": {

        "length_confound":
            dataframe_records(
                diagnostics
            ),
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


    "triangle_metrics":
        dataframe_records(
            triangles
        ),


    "group_vs_generic_observations":
        dataframe_records(
            pairs
        ),


    "document_metadata":
        dataframe_records(
            docs[
                document_export_columns
            ]
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
# 34. HUMAN-READABLE SUMMARY
# ============================================================

summary_lines = []

summary_lines.append(
    "GOOGLE AI OVERVIEW SEMANTIC ANALYSIS"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append(
    ""
)

summary_lines.append(
    "PRIMARY METRIC:"
)

summary_lines.append(
    "Subject-normalized dense cosine distance "
    "from matched generic 'people' response."
)

summary_lines.append(
    ""
)

summary_lines.append(
    "PRIMARY DIMENSION TESTS:"
)

summary_lines.append(
    ""
)


for _, row in (
    primary_dimension.iterrows()
):

    summary_lines.append(
        (
            f"{row['dimension']}: "
            f"{row['minority_group']}="
            f"{row['mean_1']:.4f}, "
            f"{row['majority_group']}="
            f"{row['mean_2']:.4f}, "
            f"Delta={row['mean_difference']:+.4f}, "
            f"95% bootstrap CI "
            f"[{row['bootstrap_ci_low']:.4f}, "
            f"{row['bootstrap_ci_high']:.4f}], "
            f"Holm permutation p="
            f"{row['p_permutation_holm']:.6g}"
        )
    )


summary_lines.append(
    ""
)

summary_lines.append(
    "AGGREGATE MINORITY vs MAJORITY:"
)

summary_lines.append(
    (
        f"Minority mean distance="
        f"{aggregate_primary['mean_1']:.4f}"
    )
)

summary_lines.append(
    (
        f"Majority mean distance="
        f"{aggregate_primary['mean_2']:.4f}"
    )
)

summary_lines.append(
    (
        f"Mean paired difference="
        f"{aggregate_primary['mean_difference']:+.4f}"
    )
)

summary_lines.append(
    (
        f"Permutation p="
        f"{aggregate_primary['p_permutation_raw']:.6g}"
    )
)


summary_lines.append(
    ""
)

summary_lines.append(
    "TRIANGLE / GENERIC PROXIMITY:"
)

summary_lines.append(
    ""
)


for _, row in (
    triangle_summary.iterrows()
):

    summary_lines.append(
        (
            f"{row['dimension']}: "
            f"generic proximity margin="
            f"{row['mean_generic_proximity_margin']:+.4f}; "
            f"minority closer in "
            f"{row['proportion_minority_closer_to_generic']:.1%} "
            f"of outcomes; "
            f"Holm p="
            f"{row['p_permutation_holm']:.6g}"
        )
    )


summary_lines.append(
    ""
)

summary_lines.append(
    "IMPORTANT:"
)

summary_lines.append(
    (
        "Semantic distance measures how much the "
        "information presented changes. It does not, "
        "by itself, establish unfairness, harm, "
        "incorrectness, or discrimination."
    )
)


summary_file = (
    RESULTS_DIR
    / "SUMMARY.txt"
)

summary_file.write_text(
    "\n".join(
        summary_lines
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# 35. README
# ============================================================

readme = f"""
Google AI Overview Semantic Analysis
====================================

Collection:
{COLLECTION_VERSION}

Embedding model:
{MODEL_NAME}

Primary metric:
dense_distance_subject_normalized

Primary inferential test:
paired sign-flip permutation test

Multiple testing:
Holm correction across the six planned social-dimension contrasts

Statistical unit:
matched domain/outcome

Output files
------------

all_results.json
    Master JSON containing essentially all derived metrics
    and statistical results.

SUMMARY.txt
    Human-readable high-level output.

document_metadata.csv
    Metadata and verbosity measures for all 273 documents.

group_vs_generic_semantic_metrics.csv
    Main 252 explicit-group vs generic matched comparisons.

semantic_metrics_by_group.csv
    Descriptive semantic metrics by group.

semantic_metrics_by_domain.csv
    Descriptive semantic metrics by domain.

semantic_metrics_by_outcome.csv
    Descriptive semantic metrics by outcome.

primary_dimension_tests_dense_subject_normalized.csv
    Main minority-vs-majority tests.

robustness_dimension_tests_dense_raw.csv
    Same tests using unnormalized presented text.

robustness_dimension_tests_tfidf.csv
    Lexical TF-IDF robustness check.

secondary_dimension_tests_generic_coverage.csv
    Sentence-level loss of generic content.

secondary_dimension_tests_group_novelty.csv
    Sentence-level content added by group marking.

secondary_dimension_tests_sentence_set_distance.csv
    Symmetric sentence-set distance.

aggregate_minority_vs_majority.csv
    Outcome-level aggregate comparisons.

global_friedman_tests.csv
    Global group effects.

exploratory_posthoc_group_tests.csv
    All 66 explicit-group comparisons, Holm corrected.

semantic_triangle_metrics.csv
    Minority/generic/majority geometry for every outcome.

semantic_triangle_summary.csv
    Triangle results summarized by dimension.

length_confound_diagnostics.csv
    Association of semantic distance with response-length change.

metric_agreement.csv
    Agreement between dense, lexical, and sentence metrics.

document_embeddings.npz
    Raw and subject-normalized document embeddings.
"""

(
    RESULTS_DIR
    / "README.txt"
).write_text(
    readme.strip()
    + "\n",
    encoding="utf-8",
)


# ============================================================
# 36. FINAL PRINT
# ============================================================

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)

print(
    "\nMain files:"
)

print(
    "  all_results.json"
)

print(
    "  SUMMARY.txt"
)

print(
    "  primary_dimension_tests_dense_subject_normalized.csv"
)

print(
    "  semantic_triangle_summary.csv"
)

print(
    "  group_vs_generic_semantic_metrics.csv"
)

print(
    "\nPrimary dimension results:\n"
)

print(
    primary_dimension[
        [
            "dimension",
            "minority_group",
            "majority_group",
            "mean_1",
            "mean_2",
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
print(
    "Done."
)