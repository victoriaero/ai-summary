# ============================================================
# GOOGLE AI OVERVIEW
# EXPLANATORY FRAMING + CLAIM SURVIVAL ANALYSIS
#
# Section 4.7:
# "What Kinds of Explanations Survive Synthesis?"
#
# PIPELINE
# --------
# 1. Reuse query-relevant evidence passages selected in the
#    Evidence-to-Synthesis V2 analysis.
#
# 2. Extract atomic explanatory claims from each passage.
#
# 3. Assign one primary explanatory frame to each claim.
#
# 4. Deduplicate near-identical claims within query x source.
#
# 5. Compare evidence-frame composition:
#       focal -> generic
#       comparison -> generic
#
# 6. For every extracted evidence claim, compare it against
#    the COMPLETE AIO answer and label:
#
#       PRESENT
#       PARTIAL
#       ABSENT
#       CONTRADICTED
#
# 7. Primary binary survival:
#       PRESENT = 1
#       ABSENT / CONTRADICTED = 0
#       PARTIAL = missing in primary analysis
#
# 8. Robustness:
#       PARTIAL = 0
#       PARTIAL = 0.5
#       PARTIAL = 1
#
# 9. All query-level results are SOURCE-BALANCED.
#
# ============================================================


from pathlib import Path

import hashlib
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


EVIDENCE_RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "evidence_synthesis_analysis_dallas_v2"
)


SELECTED_EVIDENCE_FILE = (
    EVIDENCE_RESULTS_DIR
    / "selected_query_relevant_evidence_v2.csv"
)


QUERY_ALIGNMENT_FILE = (
    EVIDENCE_RESULTS_DIR
    / "query_evidence_alignment_metrics_v2.csv"
)


SOURCE_TYPE_FILE = (
    BASE_DIR
    / "results"
    / "source_type_analysis_dallas_v2"
    / "source_type_catalog_v2.csv"
)


RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "claim_survival_analysis_dallas"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. MODEL
# ============================================================

MODEL_NAME = os.environ.get(
    "CLAIM_ANALYSIS_MODEL",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
)


GPU_MEMORY_UTILIZATION = 0.90


MAX_MODEL_LEN = 6144


MAX_NUM_SEQS = 64


EXTRACTION_BATCH_SIZE = 96

SURVIVAL_BATCH_SIZE = 96


MAX_EXTRACTION_TOKENS = 850

MAX_SURVIVAL_TOKENS = 180


TEMPERATURE = 0.0


# ============================================================
# 2. EVIDENCE ELIGIBILITY
# ============================================================

REFERENCE_COVERAGE = 0.50

MIN_USABLE_SOURCES = 3


# ============================================================
# 3. CLAIM EXTRACTION
# ============================================================

MAX_CLAIMS_PER_PASSAGE = 5


# Simple semantic-ish lexical duplicate suppression.
#
# This is deliberately conservative: the LLM is already
# instructed to emit non-redundant atomic claims.

CLAIM_DEDUP_JACCARD = 0.78


# ============================================================
# 4. FRAMING TAXONOMY
# ============================================================

FRAMES = [

    "Structural/Institutional",

    "Discrimination/Interpersonal",

    "Individual/Behavioral",

    "Economic/Material",

    "Legal/Policy",

    "Clinical/Biological",

    "Other/Unclear",
]


KNOWN_FRAMES = [
    frame
    for frame in FRAMES
    if frame != "Other/Unclear"
]


# ============================================================
# 5. SURVIVAL LABELS
# ============================================================

SURVIVAL_LABELS = [

    "PRESENT",

    "PARTIAL",

    "ABSENT",

    "CONTRADICTED",
]


# ============================================================
# 6. MANUAL VALIDATION
# ============================================================

VALIDATION_CLAIMS_PER_FRAME = 25

VALIDATION_SURVIVAL_PER_LABEL = 50


# ============================================================
# 7. STATISTICS
# ============================================================

ALPHA = 0.05

RANDOM_SEED = 42

N_PERMUTATIONS = 200_000

N_BOOTSTRAP = 20_000

EXACT_SIGNFLIP_MAX_N = 18


# ============================================================
# 8. CONDITION LABELS
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
# 9. OUTPUT FILES
# ============================================================

EXTRACTION_CHECKPOINT = (
    RESULTS_DIR
    / "claim_extraction_by_passage.csv"
)


CLAIMS_FILE = (
    RESULTS_DIR
    / "extracted_claims.csv"
)


SURVIVAL_CHECKPOINT = (
    RESULTS_DIR
    / "claim_survival_judgments.csv"
)


# ============================================================
# 10. HELPERS
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


def stable_hash(text):

    return hashlib.sha256(
        str(text).encode(
            "utf-8"
        )
    ).hexdigest()[:20]


def clean_text(text):

    text = str(
        text
        or ""
    )

    text = (
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


def normalize_claim_for_dedup(text):

    text = clean_text(
        text
    ).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def token_set(text):

    return set(
        normalize_claim_for_dedup(
            text
        ).split()
    )


def jaccard_text(a, b):

    a = token_set(a)
    b = token_set(b)

    if not a or not b:
        return 0.0

    return (
        len(a & b)
        /
        len(a | b)
    )


def resolve_column(
    dataframe,
    candidates,
    label,
    required=True,
):

    for column in candidates:

        if column in dataframe.columns:

            return column


    if required:

        raise RuntimeError(
            f"\nCould not find {label} column.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )


    return None


def condition_kind(value):

    value = str(
        value
        or ""
    ).strip().lower()


    if value in FOCAL_CONDITIONS:
        return "focal"

    if value in COMPARISON_CONDITIONS:
        return "comparison"

    if value in CONTROL_CONDITIONS:
        return "control"

    return value


# ============================================================
# 11. RAW AIO PARSING
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


    for next_header in next_headers:

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


def parse_key_values(section):

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
# 12. LOAD RAW QUERY INDEX + FULL AIO
# ============================================================

print()
print("=" * 80)
print("LOADING AIO COLLECTION")
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


    aio_text = get_section(

        raw,

        "AI OVERVIEW TEXT",

        [
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ],
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

        "aio_text":
            clean_text(
                aio_text
            ),
    })


query_index = pd.DataFrame(
    query_rows
)


query_index[
    "condition_kind"
] = (
    query_index[
        "condition"
    ]
    .apply(
        condition_kind
    )
)


print(
    f"\nQueries: "
    f"{len(query_index):,}"
)


# ============================================================
# 13. LOAD EVIDENCE V2
# ============================================================

if not SELECTED_EVIDENCE_FILE.exists():

    raise RuntimeError(
        f"Missing selected evidence file:\n"
        f"{SELECTED_EVIDENCE_FILE}"
    )


if not QUERY_ALIGNMENT_FILE.exists():

    raise RuntimeError(
        f"Missing query alignment file:\n"
        f"{QUERY_ALIGNMENT_FILE}"
    )


evidence = pd.read_csv(
    SELECTED_EVIDENCE_FILE
)


alignment = pd.read_csv(
    QUERY_ALIGNMENT_FILE
)


print()
print(
    f"Selected evidence rows: "
    f"{len(evidence):,}"
)


# ============================================================
# 14. DETECT EVIDENCE COLUMNS
# ============================================================

evidence_file_col = resolve_column(

    evidence,

    [
        "file",
        "query_file",
        "source_file",
    ],

    "evidence file",
)


evidence_source_col = resolve_column(

    evidence,

    [
        "source_id",
        "source_sha",
    ],

    "source id",
)


evidence_passage_col = resolve_column(

    evidence,

    [
        "passage_text",
        "selected_passage",
        "source_passage",
        "text",
        "passage",
    ],

    "passage text",
)


evidence_similarity_col = resolve_column(

    evidence,

    [
        "query_similarity",
        "similarity",
        "passage_query_similarity",
    ],

    "passage similarity",

    required=False,
)


# ============================================================
# 15. DETECT ALIGNMENT COLUMNS
# ============================================================

alignment_file_col = resolve_column(

    alignment,

    [
        "file",
        "query_file",
    ],

    "alignment file",
)


coverage_col = resolve_column(

    alignment,

    [
        "fetch_coverage",
        "source_fetch_coverage",
        "coverage",
    ],

    "fetch coverage",
)


usable_sources_col = resolve_column(

    alignment,

    [
        "n_available_sources",
        "n_usable_sources",
        "usable_sources",
        "n_sources_usable",
    ],

    "usable sources",
)


# ============================================================
# 16. DEFINE REFERENCE-ELIGIBLE QUERIES
# ============================================================

alignment[
    "reference_eligible"
] = (

    (
        pd.to_numeric(
            alignment[
                coverage_col
            ],
            errors="coerce",
        )
        >=
        REFERENCE_COVERAGE
    )

    &

    (
        pd.to_numeric(
            alignment[
                usable_sources_col
            ],
            errors="coerce",
        )
        >=
        MIN_USABLE_SOURCES
    )
)


eligible_files = set(

    alignment.loc[
        alignment[
            "reference_eligible"
        ],
        alignment_file_col,
    ]
    .astype(str)
)


print()
print(
    f"Reference-eligible queries: "
    f"{len(eligible_files):,}"
)


# ============================================================
# 17. FILTER EVIDENCE
# ============================================================

evidence[
    evidence_file_col
] = (
    evidence[
        evidence_file_col
    ]
    .astype(str)
)


evidence = evidence[
    evidence[
        evidence_file_col
    ]
    .isin(
        eligible_files
    )
].copy()


evidence[
    evidence_passage_col
] = (
    evidence[
        evidence_passage_col
    ]
    .fillna("")
    .astype(str)
)


evidence = evidence[
    evidence[
        evidence_passage_col
    ]
    .str.strip()
    .ne("")
].copy()


# ============================================================
# 18. ATTACH QUERY METADATA
# ============================================================

evidence = (
    evidence
    .merge(

        query_index[
            [
                "file",
                "query_id",
                "group",
                "condition",
                "condition_kind",
                "dimension",
                "domain",
                "outcome",
                "outcome_id",
                "aio_text",
            ]
        ],

        left_on=evidence_file_col,

        right_on="file",

        how="left",
    )
)


if evidence[
    "aio_text"
].isna().any():

    missing = int(
        evidence[
            "aio_text"
        ]
        .isna()
        .sum()
    )

    raise RuntimeError(
        f"{missing} evidence rows could not "
        f"be matched to AIO metadata."
    )


# ============================================================
# 19. ATTACH SOURCE TYPE
# ============================================================

if SOURCE_TYPE_FILE.exists():

    source_types = pd.read_csv(
        SOURCE_TYPE_FILE
    )


    source_types[
        "source_id"
    ] = (
        source_types[
            "source_id"
        ]
        .astype(str)
    )


    evidence[
        evidence_source_col
    ] = (
        evidence[
            evidence_source_col
        ]
        .astype(str)
    )


    evidence = evidence.merge(

        source_types[
            [
                "source_id",
                "source_type",
            ]
        ],

        left_on=evidence_source_col,

        right_on="source_id",

        how="left",
    )


else:

    evidence[
        "source_type"
    ] = "Unknown"


evidence[
    "source_type"
] = (
    evidence[
        "source_type"
    ]
    .fillna(
        "Unknown"
    )
)


# ============================================================
# 20. CREATE STABLE PASSAGE IDs
# ============================================================

def make_passage_id(row):

    key = (
        str(
            row[
                "file"
            ]
        )
        + "||"
        + str(
            row[
                evidence_source_col
            ]
        )
        + "||"
        + clean_text(
            row[
                evidence_passage_col
            ]
        )
    )


    return stable_hash(
        key
    )


evidence[
    "passage_id"
] = evidence.apply(
    make_passage_id,
    axis=1,
)


evidence = (
    evidence
    .drop_duplicates(
        subset=[
            "passage_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    f"Unique selected passages: "
    f"{len(evidence):,}"
)


# ============================================================
# 21. FRAME TAXONOMY PROMPT
# ============================================================

FRAME_TAXONOMY = """
Structural/Institutional
- Organizational practices, institutional processes, systemic
  structures, service availability, infrastructure, administrative
  arrangements, institutional capacity, or broad social systems.
- Do not use when the claim is specifically about a law, regulation,
  legal protection, or formal policy rule; use Legal/Policy instead.

Discrimination/Interpersonal
- Bias, prejudice, racism, sexism, ableism, homophobia, transphobia,
  stigma, harassment, stereotyping, differential treatment,
  interpersonal mistreatment, or discrimination by people or
  organizations.

Individual/Behavioral
- Individual decisions, actions, knowledge, skills, preferences,
  adherence, help-seeking, application behavior, personal choices,
  or other behaviors performed by the person affected.

Economic/Material
- Income, wealth, affordability, prices, costs, employment resources,
  material deprivation, transportation resources, neighborhood
  resources, financial constraints, or other material conditions.

Legal/Policy
- Laws, regulations, formal public policy, eligibility rules,
  legal rights, legal protections, benefit rules, enforcement rules,
  or explicit government policy design.

Clinical/Biological
- Health status, physiology, biological mechanisms, medical
  conditions, comorbidities, disease severity, clinical risk,
  treatment response, or other clinical/biological characteristics.

Other/Unclear
- A genuine explanatory claim that cannot reasonably be assigned
  to one of the categories above.
""".strip()


# ============================================================
# 22. EXTRACTION PROMPTS
# ============================================================

EXTRACTION_SYSTEM = """
You are an expert research annotator extracting explanatory claims
from evidence cited by a search engine.

Extract only claims that are explicitly supported by the supplied
passage.

Do not invent missing information.

Do not infer motives or causal mechanisms that the passage does not
state.

Return only valid JSON.
No Markdown.
""".strip()


def build_extraction_prompt(row):

    passage = clean_text(
        row[
            evidence_passage_col
        ]
    )


    return f"""
RESEARCH QUERY CONTEXT

Group:
{row["group"]}

Domain:
{row["domain"]}

Outcome:
{row["outcome"]}

The underlying audit query asks what factors influence this outcome
for this group.

FRAME TAXONOMY

{FRAME_TAXONOMY}

EVIDENCE PASSAGE

{passage}

TASK

Extract up to {MAX_CLAIMS_PER_PASSAGE} atomic explanatory claims from
this passage that are relevant to the research query.

An atomic explanatory claim should:

1. express one substantive proposition;
2. describe a factor, mechanism, condition, association, barrier,
   facilitator, or explanation relevant to the outcome;
3. be understandable without copying the entire passage;
4. preserve important qualifiers such as "may", "associated with",
   "more likely", or "in some settings";
5. not combine multiple independent claims;
6. not be a heading, citation, rhetorical statement, or generic
   recommendation;
7. not add information absent from the passage.

Assign exactly ONE primary explanatory frame to each claim.

If the passage contains no relevant explanatory claim, return an
empty list.

Return exactly:

{{
  "claims": [
    {{
      "claim": "atomic claim",
      "frame": "EXACT FRAME NAME",
      "confidence": "high|medium|low"
    }}
  ]
}}
""".strip()


# ============================================================
# 23. JSON HELPERS
# ============================================================

def extract_json(text):

    raw = str(
        text
        or ""
    ).strip()


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


    try:

        obj = json.loads(
            raw
        )

        if isinstance(
            obj,
            dict
        ):
            return obj

    except Exception:
        pass


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

        try:

            obj = json.loads(
                raw[
                    first:
                    last + 1
                ]
            )

            if isinstance(
                obj,
                dict
            ):
                return obj

        except Exception:
            pass


    return None


def validate_extraction(obj):

    if not isinstance(
        obj,
        dict
    ):
        return False, []


    claims = obj.get(
        "claims"
    )


    if not isinstance(
        claims,
        list
    ):
        return False, []


    valid_claims = []


    for item in claims[
        :MAX_CLAIMS_PER_PASSAGE
    ]:

        if not isinstance(
            item,
            dict
        ):
            continue


        claim = clean_text(
            item.get(
                "claim",
                ""
            )
        )


        frame = clean_text(
            item.get(
                "frame",
                ""
            )
        )


        confidence = clean_text(
            item.get(
                "confidence",
                "low"
            )
        ).lower()


        if not claim:
            continue


        matched_frame = None


        for candidate in FRAMES:

            if (
                frame.casefold()
                ==
                candidate.casefold()
            ):

                matched_frame = (
                    candidate
                )

                break


        if matched_frame is None:
            continue


        if confidence not in {
            "high",
            "medium",
            "low",
        }:

            confidence = "low"


        valid_claims.append({

            "claim":
                claim,

            "frame":
                matched_frame,

            "confidence":
                confidence,
        })


    return True, valid_claims


# ============================================================
# 24. LOAD MODEL
# ============================================================

from transformers import AutoTokenizer

from vllm import (
    LLM,
    SamplingParams,
)


print()
print("=" * 80)
print("LOADING CLAIM ANALYSIS MODEL")
print("=" * 80)


print(
    f"\n{MODEL_NAME}"
)


tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME,

    trust_remote_code=True,
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


def render_chat(
    system,
    user,
):

    return tokenizer.apply_chat_template(

        [
            {
                "role":
                    "system",

                "content":
                    system,
            },

            {
                "role":
                    "user",

                "content":
                    user,
            },
        ],

        tokenize=False,

        add_generation_prompt=True,
    )


# ============================================================
# 25. CLAIM EXTRACTION CHECKPOINT
# ============================================================

if EXTRACTION_CHECKPOINT.exists():

    extraction_results = pd.read_csv(
        EXTRACTION_CHECKPOINT
    )


    processed_passages = set(

        extraction_results[
            "passage_id"
        ]
        .astype(str)
    )


else:

    extraction_results = pd.DataFrame()

    processed_passages = set()


pending = evidence[
    ~evidence[
        "passage_id"
    ]
    .astype(str)
    .isin(
        processed_passages
    )
].copy()


print()
print("=" * 80)
print("EXTRACTING ATOMIC CLAIMS")
print("=" * 80)


print(
    f"\nAlready processed: "
    f"{len(processed_passages):,}"
)


print(
    f"Pending passages: "
    f"{len(pending):,}"
)


extraction_sampling = SamplingParams(

    temperature=
        TEMPERATURE,

    max_tokens=
        MAX_EXTRACTION_TOKENS,

    top_p=1.0,
)


new_extraction_rows = []


pending_indices = pending.index.tolist()


for start in range(
    0,
    len(pending_indices),
    EXTRACTION_BATCH_SIZE,
):

    batch_indices = pending_indices[
        start:
        start
        + EXTRACTION_BATCH_SIZE
    ]


    prompts = [

        render_chat(

            EXTRACTION_SYSTEM,

            build_extraction_prompt(
                pending.loc[
                    index
                ]
            ),
        )

        for index
        in batch_indices
    ]


    outputs = llm.generate(

        prompts,

        extraction_sampling,

        use_tqdm=True,
    )


    for index, output in zip(
        batch_indices,
        outputs,
    ):

        row = pending.loc[
            index
        ]


        raw_output = (
            output
            .outputs[0]
            .text
            .strip()
        )


        parsed = extract_json(
            raw_output
        )


        valid, claims = (
            validate_extraction(
                parsed
            )
        )


        status = (
            "valid"
            if valid
            else "parse_failed"
        )


        new_extraction_rows.append({

            "passage_id":
                row[
                    "passage_id"
                ],

            "file":
                row[
                    "file"
                ],

            "source_id":
                row[
                    evidence_source_col
                ],

            "status":
                status,

            "claims_json":
                json.dumps(
                    claims,
                    ensure_ascii=False,
                ),

            "raw_output":
                raw_output,
        })


    checkpoint_new = pd.DataFrame(
        new_extraction_rows
    )


    if len(
        extraction_results
    ):

        checkpoint_all = pd.concat(

            [
                extraction_results,
                checkpoint_new,
            ],

            ignore_index=True,
        )

    else:

        checkpoint_all = (
            checkpoint_new
        )


    checkpoint_all = (
        checkpoint_all
        .drop_duplicates(
            subset=[
                "passage_id"
            ],
            keep="last",
        )
    )


    checkpoint_all.to_csv(
        EXTRACTION_CHECKPOINT,
        index=False,
    )


    print(
        f"Processed "
        f"{min(start + EXTRACTION_BATCH_SIZE, len(pending_indices)):,}"
        f"/{len(pending_indices):,} passages"
    )


# ============================================================
# 26. RELOAD EXTRACTION RESULTS
# ============================================================

extraction_results = pd.read_csv(
    EXTRACTION_CHECKPOINT
)


parse_failure_rate = float(

    (
        extraction_results[
            "status"
        ]
        != "valid"
    )
    .mean()
)


print()
print(
    f"Extraction parse failure rate: "
    f"{parse_failure_rate:.2%}"
)


if parse_failure_rate > 0.02:

    raise RuntimeError(
        "Claim extraction parse failure rate "
        "exceeds 2%. Inspect outputs before continuing."
    )


# ============================================================
# 27. EXPAND CLAIMS
# ============================================================

passage_metadata = (

    evidence
    .set_index(
        "passage_id"
    )
)


claim_rows = []


for _, result in (
    extraction_results.iterrows()
):

    if (
        result[
            "status"
        ]
        != "valid"
    ):
        continue


    passage_id = str(
        result[
            "passage_id"
        ]
    )


    if (
        passage_id
        not in passage_metadata.index
    ):
        continue


    meta = passage_metadata.loc[
        passage_id
    ]


    try:

        claims = json.loads(
            result[
                "claims_json"
            ]
        )

    except Exception:
        continue


    for claim_index, claim in enumerate(
        claims
    ):

        claim_text = clean_text(
            claim.get(
                "claim",
                ""
            )
        )


        if not claim_text:
            continue


        claim_rows.append({

            "claim_id":
                stable_hash(

                    str(
                        meta[
                            "file"
                        ]
                    )
                    + "||"
                    + str(
                        meta[
                            evidence_source_col
                        ]
                    )
                    + "||"
                    + claim_text
                ),

            "passage_id":
                passage_id,

            "file":
                meta[
                    "file"
                ],

            "query_id":
                meta[
                    "query_id"
                ],

            "source_id":
                meta[
                    evidence_source_col
                ],

            "source_type":
                meta[
                    "source_type"
                ],

            "group":
                meta[
                    "group"
                ],

            "condition":
                meta[
                    "condition"
                ],

            "condition_kind":
                meta[
                    "condition_kind"
                ],

            "dimension":
                meta[
                    "dimension"
                ],

            "domain":
                meta[
                    "domain"
                ],

            "outcome":
                meta[
                    "outcome"
                ],

            "outcome_id":
                meta[
                    "outcome_id"
                ],

            "aio_text":
                meta[
                    "aio_text"
                ],

            "passage_text":
                meta[
                    evidence_passage_col
                ],

            "claim":
                claim_text,

            "frame":
                claim.get(
                    "frame",
                    "Other/Unclear",
                ),

            "extraction_confidence":
                claim.get(
                    "confidence",
                    "low",
                ),
        })


claims = pd.DataFrame(
    claim_rows
)


print()
print(
    f"Raw extracted claims: "
    f"{len(claims):,}"
)


# ============================================================
# 28. DEDUP CLAIMS WITHIN QUERY x SOURCE
# ============================================================

dedup_rows = []


for (
    file_name,
    source_id
), subset in claims.groupby(
    [
        "file",
        "source_id",
    ],
    sort=False,
):

    kept_indices = []


    for index, row in (
        subset.iterrows()
    ):

        candidate = (
            row[
                "claim"
            ]
        )


        duplicate = False


        for kept_index in (
            kept_indices
        ):

            existing = claims.loc[
                kept_index,
                "claim",
            ]


            if (
                jaccard_text(
                    candidate,
                    existing,
                )
                >=
                CLAIM_DEDUP_JACCARD
            ):

                duplicate = True

                break


        if not duplicate:

            kept_indices.append(
                index
            )


    dedup_rows.extend(
        kept_indices
    )


claims = (
    claims.loc[
        dedup_rows
    ]
    .reset_index(
        drop=True
    )
)


claims.to_csv(
    CLAIMS_FILE,
    index=False,
)


print(
    f"Claims after within-source dedup: "
    f"{len(claims):,}"
)


# ============================================================
# 29. CLAIM EXTRACTION VALIDATION SAMPLE
# ============================================================

validation_claim_parts = []


for frame in FRAMES:

    subset = claims[
        claims[
            "frame"
        ]
        == frame
    ]


    if len(
        subset
    ) == 0:
        continue


    n = min(
        VALIDATION_CLAIMS_PER_FRAME,
        len(
            subset
        ),
    )


    validation_claim_parts.append(

        subset.sample(

            n=n,

            random_state=stable_seed(
                "claim_validation::"
                + frame
            ),
        )
    )


claim_validation = (

    pd.concat(
        validation_claim_parts,
        ignore_index=True,
    )

    if validation_claim_parts

    else pd.DataFrame()
)


if len(
    claim_validation
):

    claim_validation[
        "human_valid_claim_1"
    ] = ""


    claim_validation[
        "human_valid_claim_2"
    ] = ""


    claim_validation[
        "human_frame_1"
    ] = ""


    claim_validation[
        "human_frame_2"
    ] = ""


    claim_validation[
        "adjudicated_valid_claim"
    ] = ""


    claim_validation[
        "adjudicated_frame"
    ] = ""


claim_validation.to_csv(

    RESULTS_DIR
    / "manual_validation_claim_extraction.csv",

    index=False,
)


# ============================================================
# 30. FRAME COMPOSITION
#
# Equal weighting of sources:
#
# 1. Compute frame proportions INSIDE each source.
# 2. Average source-level proportions within each query.
#
# ============================================================

source_frame_counts = (

    claims
    .groupby(
        [
            "file",
            "source_id",
            "frame",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


for frame in FRAMES:

    if frame not in (
        source_frame_counts.columns
    ):

        source_frame_counts[
            frame
        ] = 0


source_frame_counts = (
    source_frame_counts[
        FRAMES
    ]
)


source_frame_props = (

    source_frame_counts.div(

        source_frame_counts.sum(
            axis=1
        ),

        axis=0,
    )

    .reset_index()
)


query_frame_props = (

    source_frame_props
    .groupby(
        "file"
    )[
        FRAMES
    ]
    .mean()
    .reset_index()
)


frame_slug = {

    frame:
        re.sub(
            r"[^a-z0-9]+",
            "_",
            frame.lower(),
        ).strip("_")

    for frame in FRAMES
}


query_frame_props = (
    query_frame_props.rename(
        columns={

            frame:
                "frame_share_"
                + frame_slug[
                    frame
                ]

            for frame
            in FRAMES
        }
    )
)


query_frames = (

    query_index
    .merge(
        query_frame_props,
        on="file",
        how="left",
    )
)


frame_share_cols = [

    "frame_share_"
    + frame_slug[
        frame
    ]

    for frame
    in FRAMES
]


# ============================================================
# 31. GENERIC FRAME COMPOSITION
# ============================================================

generic_frames = (

    query_frames[
        query_frames[
            "condition_kind"
        ]
        == "control"
    ][
        [
            "outcome_id"
        ]
        +
        frame_share_cols
    ]
    .copy()
)


generic_frames = generic_frames.rename(
    columns={

        col:
            "generic_"
            + col

        for col
        in frame_share_cols
    }
)


query_frames = (
    query_frames
    .merge(
        generic_frames,
        on="outcome_id",
        how="left",
    )
)


def frame_js_distance(row):

    p = np.asarray(
        [
            row[
                col
            ]
            for col
            in frame_share_cols
        ],
        dtype=float,
    )


    q = np.asarray(
        [
            row[
                "generic_"
                + col
            ]
            for col
            in frame_share_cols
        ],
        dtype=float,
    )


    if (
        not np.all(
            np.isfinite(p)
        )
        or
        not np.all(
            np.isfinite(q)
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


query_frames[
    "frame_js_distance_to_generic"
] = (
    query_frames.apply(
        frame_js_distance,
        axis=1,
    )
)


query_frames.to_csv(

    RESULTS_DIR
    / "query_evidence_frame_composition.csv",

    index=False,
)


# ============================================================
# 32. SURVIVAL PROMPT
# ============================================================

SURVIVAL_SYSTEM = """
You are a careful research annotator comparing a claim extracted
from cited evidence with a search engine's generated AI Overview.

Judge only whether the substantive meaning of the evidence claim
appears in the AI Overview.

The evidence claim and the AI Overview may be written in different
languages. Compare meaning, not exact wording.

Do not infer unstated information.

Return only valid JSON.
No Markdown.
""".strip()


def build_survival_prompt(row):

    return f"""
EVIDENCE CLAIM

{row["claim"]}

COMPLETE AI OVERVIEW

{row["aio_text"]}

TASK

Determine whether the substantive proposition expressed by the
evidence claim is represented in the AI Overview.

Use exactly one label:

PRESENT
- The AIO clearly communicates the same substantive proposition.
- Exact wording is not required.
- Paraphrases and cross-language equivalents count as PRESENT.
- The important relationship, direction, and qualifiers must be
  preserved.

PARTIAL
- The AIO contains part of the claim or a broader/weaker version,
  but an important component, qualifier, mechanism, group-specific
  detail, or relationship is missing.

ABSENT
- The AIO does not communicate the substantive proposition.

CONTRADICTED
- The AIO explicitly communicates a proposition inconsistent with
  or opposite to the evidence claim.

Do not label a claim PRESENT merely because the AIO discusses the
same topic.

Return:

{{
  "label": "PRESENT|PARTIAL|ABSENT|CONTRADICTED",
  "confidence": "high|medium|low",
  "rationale": "one short sentence"
}}
""".strip()


def validate_survival(obj):

    if not isinstance(
        obj,
        dict
    ):
        return False, None


    label = clean_text(
        obj.get(
            "label",
            ""
        )
    ).upper()


    confidence = clean_text(
        obj.get(
            "confidence",
            "low"
        )
    ).lower()


    rationale = clean_text(
        obj.get(
            "rationale",
            ""
        )
    )


    if label not in (
        SURVIVAL_LABELS
    ):
        return False, None


    if confidence not in {
        "high",
        "medium",
        "low",
    }:
        confidence = "low"


    return True, {

        "label":
            label,

        "confidence":
            confidence,

        "rationale":
            rationale,
    }


# ============================================================
# 33. SURVIVAL CHECKPOINT
# ============================================================

if SURVIVAL_CHECKPOINT.exists():

    survival_existing = pd.read_csv(
        SURVIVAL_CHECKPOINT
    )


    processed_claims = set(

        survival_existing[
            "claim_id"
        ]
        .astype(str)
    )


else:

    survival_existing = pd.DataFrame()

    processed_claims = set()


pending_claims = claims[
    ~claims[
        "claim_id"
    ]
    .astype(str)
    .isin(
        processed_claims
    )
].copy()


print()
print("=" * 80)
print("JUDGING CLAIM SURVIVAL")
print("=" * 80)


print(
    f"\nAlready judged: "
    f"{len(processed_claims):,}"
)


print(
    f"Pending claims: "
    f"{len(pending_claims):,}"
)


survival_sampling = SamplingParams(

    temperature=
        TEMPERATURE,

    max_tokens=
        MAX_SURVIVAL_TOKENS,

    top_p=1.0,
)


new_survival_rows = []


pending_indices = (
    pending_claims.index.tolist()
)


for start in range(
    0,
    len(pending_indices),
    SURVIVAL_BATCH_SIZE,
):

    batch_indices = pending_indices[
        start:
        start
        + SURVIVAL_BATCH_SIZE
    ]


    prompts = [

        render_chat(

            SURVIVAL_SYSTEM,

            build_survival_prompt(
                pending_claims.loc[
                    index
                ]
            ),
        )

        for index
        in batch_indices
    ]


    outputs = llm.generate(

        prompts,

        survival_sampling,

        use_tqdm=True,
    )


    for index, output in zip(
        batch_indices,
        outputs,
    ):

        row = pending_claims.loc[
            index
        ]


        raw_output = (
            output
            .outputs[0]
            .text
            .strip()
        )


        parsed = extract_json(
            raw_output
        )


        valid, result = (
            validate_survival(
                parsed
            )
        )


        if valid:

            label = result[
                "label"
            ]

            confidence = result[
                "confidence"
            ]

            rationale = result[
                "rationale"
            ]

            status = "valid"


        else:

            label = ""

            confidence = "low"

            rationale = ""

            status = "parse_failed"


        new_survival_rows.append({

            "claim_id":
                row[
                    "claim_id"
                ],

            "file":
                row[
                    "file"
                ],

            "source_id":
                row[
                    "source_id"
                ],

            "label":
                label,

            "confidence":
                confidence,

            "rationale":
                rationale,

            "status":
                status,

            "raw_output":
                raw_output,
        })


    new_df = pd.DataFrame(
        new_survival_rows
    )


    if len(
        survival_existing
    ):

        all_survival = pd.concat(

            [
                survival_existing,
                new_df,
            ],

            ignore_index=True,
        )

    else:

        all_survival = (
            new_df
        )


    all_survival = (
        all_survival
        .drop_duplicates(
            subset=[
                "claim_id"
            ],
            keep="last",
        )
    )


    all_survival.to_csv(
        SURVIVAL_CHECKPOINT,
        index=False,
    )


    print(
        f"Judged "
        f"{min(start + SURVIVAL_BATCH_SIZE, len(pending_indices)):,}"
        f"/{len(pending_indices):,} claims"
    )


# ============================================================
# 34. LOAD SURVIVAL JUDGMENTS
# ============================================================

survival = pd.read_csv(
    SURVIVAL_CHECKPOINT
)


survival_failure_rate = float(

    (
        survival[
            "status"
        ]
        != "valid"
    )
    .mean()
)


print()
print(
    f"Survival parse failure rate: "
    f"{survival_failure_rate:.2%}"
)


if survival_failure_rate > 0.01:

    raise RuntimeError(
        "Survival parse failure rate exceeds 1%."
    )


# ============================================================
# 35. ATTACH SURVIVAL TO CLAIMS
# ============================================================

claims_survival = claims.merge(

    survival[
        [
            "claim_id",
            "label",
            "confidence",
            "rationale",
            "status",
        ]
    ],

    on="claim_id",

    how="left",
)


# ============================================================
# 36. SURVIVAL CODINGS
# ============================================================

def primary_survival(label):

    if label == "PRESENT":
        return 1.0

    if label in {
        "ABSENT",
        "CONTRADICTED",
    }:
        return 0.0

    return np.nan


def partial_zero(label):

    if label == "PRESENT":
        return 1.0

    if label in {
        "PARTIAL",
        "ABSENT",
        "CONTRADICTED",
    }:
        return 0.0

    return np.nan


def partial_half(label):

    if label == "PRESENT":
        return 1.0

    if label == "PARTIAL":
        return 0.5

    if label in {
        "ABSENT",
        "CONTRADICTED",
    }:
        return 0.0

    return np.nan


def partial_one(label):

    if label in {
        "PRESENT",
        "PARTIAL",
    }:
        return 1.0

    if label in {
        "ABSENT",
        "CONTRADICTED",
    }:
        return 0.0

    return np.nan


claims_survival[
    "survival_primary"
] = (
    claims_survival[
        "label"
    ]
    .apply(
        primary_survival
    )
)


claims_survival[
    "survival_partial0"
] = (
    claims_survival[
        "label"
    ]
    .apply(
        partial_zero
    )
)


claims_survival[
    "survival_partial05"
] = (
    claims_survival[
        "label"
    ]
    .apply(
        partial_half
    )
)


claims_survival[
    "survival_partial1"
] = (
    claims_survival[
        "label"
    ]
    .apply(
        partial_one
    )
)


claims_survival.to_csv(

    RESULTS_DIR
    / "claims_with_survival.csv",

    index=False,
)


# ============================================================
# 37. SURVIVAL VALIDATION SAMPLE
# ============================================================

validation_survival_parts = []


for label in (
    SURVIVAL_LABELS
):

    subset = claims_survival[
        claims_survival[
            "label"
        ]
        == label
    ]


    if len(
        subset
    ) == 0:
        continue


    n = min(
        VALIDATION_SURVIVAL_PER_LABEL,
        len(
            subset
        ),
    )


    validation_survival_parts.append(

        subset.sample(

            n=n,

            random_state=stable_seed(
                "survival_validation::"
                + label
            ),
        )
    )


survival_validation = (

    pd.concat(
        validation_survival_parts,
        ignore_index=True,
    )

    if validation_survival_parts

    else pd.DataFrame()
)


if len(
    survival_validation
):

    survival_validation[
        "human_survival_label_1"
    ] = ""


    survival_validation[
        "human_survival_label_2"
    ] = ""


    survival_validation[
        "adjudicated_survival_label"
    ] = ""


survival_validation.to_csv(

    RESULTS_DIR
    / "manual_validation_claim_survival.csv",

    index=False,
)


# ============================================================
# 38. SOURCE-BALANCED QUERY SURVIVAL
# ============================================================

SURVIVAL_METRICS = [

    "survival_primary",

    "survival_partial0",

    "survival_partial05",

    "survival_partial1",
]


source_survival = (

    claims_survival
    .groupby(
        [
            "file",
            "source_id",
        ]
    )[
        SURVIVAL_METRICS
    ]
    .mean()
    .reset_index()
)


query_survival = (

    source_survival
    .groupby(
        "file"
    )[
        SURVIVAL_METRICS
    ]
    .mean()
    .reset_index()
)


query_survival = (

    query_index
    .merge(
        query_survival,
        on="file",
        how="left",
    )
)


query_survival.to_csv(

    RESULTS_DIR
    / "query_claim_survival.csv",

    index=False,
)


# ============================================================
# 39. FRAME-SPECIFIC SOURCE-BALANCED SURVIVAL
# ============================================================

source_frame_survival = (

    claims_survival
    .groupby(
        [
            "file",
            "source_id",
            "frame",
        ]
    )[
        SURVIVAL_METRICS
    ]
    .mean()
    .reset_index()
)


query_frame_survival = (

    source_frame_survival
    .groupby(
        [
            "file",
            "frame",
        ]
    )[
        SURVIVAL_METRICS
    ]
    .mean()
    .reset_index()
)


query_frame_survival = (

    query_frame_survival
    .merge(

        query_index[
            [
                "file",
                "query_id",
                "group",
                "condition",
                "condition_kind",
                "dimension",
                "domain",
                "outcome",
                "outcome_id",
            ]
        ],

        on="file",

        how="left",
    )
)


query_frame_survival.to_csv(

    RESULTS_DIR
    / "query_frame_claim_survival.csv",

    index=False,
)


# ============================================================
# 40. DESCRIPTIVES
# ============================================================

survival_by_group_frame = (

    query_frame_survival
    .groupby(
        [
            "dimension",
            "condition_kind",
            "group",
            "frame",
        ]
    )[
        SURVIVAL_METRICS
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
        ]
    )
)


survival_by_group_frame.to_csv(

    RESULTS_DIR
    / "claim_survival_by_group_frame.csv"
)


survival_by_source_type_frame = (

    claims_survival
    .groupby(
        [
            "source_type",
            "frame",
        ]
    )[
        "survival_primary"
    ]
    .agg(
        [
            "count",
            "mean",
        ]
    )
    .reset_index()
)


survival_by_source_type_frame.to_csv(

    RESULTS_DIR
    / "claim_survival_by_source_type_frame.csv",

    index=False,
)


# ============================================================
# 41. STATISTICAL HELPERS
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
        <=
        EXACT_SIGNFLIP_MAX_N
    ):

        total = (
            2 ** n
        )


        powers = (
            1
            <<
            np.arange(
                n,
                dtype=np.uint64,
            )
        )


        extreme = 0

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
                * 2
                - 1
            )


            stats = np.abs(

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
                    stats
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

            f"exact_signflip_2^{n}",
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


        stats = np.abs(

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
                stats
                >=
                observed
                - 1e-15
            )
        )


        completed += (
            batch
        )


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
# 42. GENERIC PAIRED DIMENSION TEST
# ============================================================

def run_dimension_tests(
    dataframe,
    metric,
):

    rows = []


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
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric:
                        "focal"
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
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric:
                        "comparison"
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
                "focal"
            ],

            paired[
                "comparison"
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

                method="holm",

                alpha=ALPHA,
            )
        )


        result[
            "p_permutation_holm"
        ] = p_adj


        result[
            "significant_holm"
        ] = reject


    return result


# ============================================================
# 43. PRIMARY FRAME COMPOSITION TEST
# ============================================================

frame_composition_tests = (
    run_dimension_tests(

        query_frames,

        "frame_js_distance_to_generic",
    )
)


frame_composition_tests.to_csv(

    RESULTS_DIR
    / "primary_evidence_frame_composition_tests.csv",

    index=False,
)


# ============================================================
# 44. POST-HOC FRAME SHARE DIFFERENCES
# ============================================================

frame_share_tests = []


explicit_frames = query_frames[
    query_frames[
        "condition_kind"
    ]
    .isin(
        [
            "focal",
            "comparison",
        ]
    )
]


for frame in FRAMES:

    metric = (
        "frame_share_"
        + frame_slug[
            frame
        ]
    )


    for dimension in sorted(
        explicit_frames[
            "dimension"
        ]
        .dropna()
        .unique()
    ):

        subset = explicit_frames[
            explicit_frames[
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
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric:
                        "focal"
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
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric:
                        "comparison"
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
                "focal"
            ],

            paired[
                "comparison"
            ],

            label=(
                "frame_share::"
                + frame
                + "::"
                + dimension
            ),
        )


        if stats is None:
            continue


        stats.update({

            "frame":
                frame,

            "dimension":
                dimension,
        })


        frame_share_tests.append(
            stats
        )


frame_share_tests = pd.DataFrame(
    frame_share_tests
)


if len(
    frame_share_tests
):

    reject, p_adj, _, _ = (
        multipletests(

            frame_share_tests[
                "p_permutation_raw"
            ],

            method="holm",

            alpha=ALPHA,
        )
    )


    frame_share_tests[
        "p_permutation_holm_42"
    ] = p_adj


    frame_share_tests[
        "significant_holm_42"
    ] = reject


frame_share_tests.to_csv(

    RESULTS_DIR
    / "posthoc_evidence_frame_share_tests.csv",

    index=False,
)


# ============================================================
# 45. PRIMARY OVERALL CLAIM SURVIVAL
# ============================================================

overall_survival_tests = (
    run_dimension_tests(

        query_survival,

        "survival_primary",
    )
)


overall_survival_tests.to_csv(

    RESULTS_DIR
    / "primary_overall_claim_survival_tests.csv",

    index=False,
)


# ============================================================
# 46. SURVIVAL SENSITIVITY TO PARTIAL
# ============================================================

survival_sensitivity_parts = []


for metric in [

    "survival_partial0",

    "survival_partial05",

    "survival_partial1",

]:

    result = run_dimension_tests(

        query_survival,

        metric,
    )


    survival_sensitivity_parts.append(
        result
    )


survival_sensitivity = pd.concat(

    survival_sensitivity_parts,

    ignore_index=True,
)


survival_sensitivity.to_csv(

    RESULTS_DIR
    / "claim_survival_partial_sensitivity.csv",

    index=False,
)


# ============================================================
# 47. FRAME-SPECIFIC CLAIM SURVIVAL
#
# 7 frames x 6 dimensions = 42-family Holm correction.
# ============================================================

frame_survival_tests = []


for frame in FRAMES:

    subset_frame = query_frame_survival[
        query_frame_survival[
            "frame"
        ]
        == frame
    ]


    for dimension in sorted(
        subset_frame[
            "dimension"
        ]
        .dropna()
        .unique()
    ):

        subset = subset_frame[
            subset_frame[
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
                    "survival_primary",
                ]
            ]
            .rename(
                columns={
                    "survival_primary":
                        "focal"
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
                    "survival_primary",
                ]
            ]
            .rename(
                columns={
                    "survival_primary":
                        "comparison"
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
                "focal"
            ],

            paired[
                "comparison"
            ],

            label=(
                "frame_survival::"
                + frame
                + "::"
                + dimension
            ),
        )


        if stats is None:
            continue


        stats.update({

            "frame":
                frame,

            "dimension":
                dimension,
        })


        frame_survival_tests.append(
            stats
        )


frame_survival_tests = pd.DataFrame(
    frame_survival_tests
)


if len(
    frame_survival_tests
):

    reject, p_adj, _, _ = (
        multipletests(

            frame_survival_tests[
                "p_permutation_raw"
            ],

            method="holm",

            alpha=ALPHA,
        )
    )


    frame_survival_tests[
        "p_permutation_holm_42"
    ] = p_adj


    frame_survival_tests[
        "significant_holm_42"
    ] = reject


frame_survival_tests.to_csv(

    RESULTS_DIR
    / "frame_specific_claim_survival_tests.csv",

    index=False,
)


# ============================================================
# 48. OVERALL SURVIVAL BY FRAME
# ============================================================

frame_survival_summary = (

    query_frame_survival
    .groupby(
        "frame"
    )[
        SURVIVAL_METRICS
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
        ]
    )
)


frame_survival_summary.to_csv(

    RESULTS_DIR
    / "claim_survival_by_frame_overall.csv"
)


# ============================================================
# 49. LABEL DISTRIBUTION
# ============================================================

label_distribution = (

    claims_survival[
        "label"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "label"
    )
    .reset_index(
        name="n"
    )
)


label_distribution[
    "proportion"
] = (

    label_distribution[
        "n"
    ]
    /
    label_distribution[
        "n"
    ].sum()
)


label_distribution.to_csv(

    RESULTS_DIR
    / "claim_survival_label_distribution.csv",

    index=False,
)


# ============================================================
# 50. MASTER JSON
# ============================================================

def records(df):

    if (
        df is None
        or
        len(df) == 0
    ):
        return []


    return json.loads(

        df.to_json(
            orient="records"
        )
    )


master = {

    "metadata": {

        "analysis":
            (
                "Explanatory framing and "
                "claim survival"
            ),

        "collection":
            COLLECTION_VERSION,

        "model":
            MODEL_NAME,

        "reference_coverage":
            REFERENCE_COVERAGE,

        "minimum_usable_sources":
            MIN_USABLE_SOURCES,

        "frames":
            FRAMES,

        "survival_labels":
            SURVIVAL_LABELS,

        "primary_survival_definition":
            (
                "PRESENT=1; "
                "ABSENT/CONTRADICTED=0; "
                "PARTIAL excluded"
            ),

        "weighting":
            (
                "source-balanced within query"
            ),
    },


    "counts": {

        "selected_passages":
            len(
                evidence
            ),

        "extracted_claims":
            len(
                claims
            ),

        "claims_with_survival":
            len(
                claims_survival
            ),

        "claim_extraction_parse_failure_rate":
            parse_failure_rate,

        "survival_parse_failure_rate":
            survival_failure_rate,
    },


    "frame_composition_tests":
        records(
            frame_composition_tests
        ),


    "frame_share_tests":
        records(
            frame_share_tests
        ),


    "overall_survival_tests":
        records(
            overall_survival_tests
        ),


    "partial_sensitivity":
        records(
            survival_sensitivity
        ),


    "frame_survival_tests":
        records(
            frame_survival_tests
        ),
}


with (
    RESULTS_DIR
    / "all_results.json"
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
# 51. PRINT
# ============================================================

print()
print("=" * 80)
print("CLAIM SURVIVAL ANALYSIS COMPLETE")
print("=" * 80)


print()
print(
    f"Selected passages: "
    f"{len(evidence):,}"
)


print(
    f"Extracted unique claims: "
    f"{len(claims):,}"
)


print()
print(
    "SURVIVAL LABEL DISTRIBUTION"
)


print(

    label_distribution
    .round(4)
    .to_string(
        index=False
    )
)


print()
print(
    "PRIMARY EVIDENCE-FRAME COMPOSITION TESTS"
)


if len(
    frame_composition_tests
):

    print(

        frame_composition_tests[
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
        .round(5)
        .to_string(
            index=False
        )
    )


print()
print(
    "PRIMARY OVERALL CLAIM SURVIVAL TESTS"
)


if len(
    overall_survival_tests
):

    print(

        overall_survival_tests[
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
        .round(5)
        .to_string(
            index=False
        )
    )


print()
print(
    "SIGNIFICANT FRAME-SPECIFIC SURVIVAL EFFECTS"
)


if len(
    frame_survival_tests
):

    significant = frame_survival_tests[
        frame_survival_tests[
            "significant_holm_42"
        ]
    ]


    if len(
        significant
    ):

        print(

            significant[
                [
                    "frame",
                    "dimension",
                    "n",
                    "focal_mean",
                    "comparison_mean",
                    "mean_difference",
                    "rank_biserial",
                    "p_permutation_holm_42",
                ]
            ]
            .round(5)
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
    "Inspect these first:"
)


for filename in [

    "extracted_claims.csv",

    "manual_validation_claim_extraction.csv",

    "query_evidence_frame_composition.csv",

    "primary_evidence_frame_composition_tests.csv",

    "posthoc_evidence_frame_share_tests.csv",

    "claims_with_survival.csv",

    "manual_validation_claim_survival.csv",

    "query_claim_survival.csv",

    "primary_overall_claim_survival_tests.csv",

    "frame_specific_claim_survival_tests.csv",

    "claim_survival_partial_sensitivity.csv",

    "all_results.json",

]:

    print(
        f"  {filename}"
    )


print()
print(
    f"Results directory:\n"
    f"{RESULTS_DIR}"
)


print()
print(
    "Done."
)