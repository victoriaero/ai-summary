from pathlib import Path
from collections import Counter
import argparse
import pandas as pd
import re


# ============================================================
# CONFIG
# ============================================================

parser = argparse.ArgumentParser(description="Check completeness and consistency of a Google AIO collection.")
parser.add_argument(
    "--collection-dir",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "annotations/v1_dallas/google_aio_collection",
    help="Collection folder to validate (default: Dallas).",
)
parser.add_argument(
    "--expected-files-per-group",
    type=int,
    help="Expected TXT files per group. By default this is inferred from query_manifest.csv when available.",
)
args = parser.parse_args()
ROOT = args.collection_dir.resolve()
if not ROOT.is_dir():
    parser.error(f"Collection folder does not exist: {ROOT}")

EXPECTED_GROUPS = {
    "black_people": "Black people",
    "white_people": "White people",

    "latino_people": "Latino people",
    "non_latino_people": "non-Latino people",

    "women": "Women",
    "men": "Men",

    "people_with_disabilities": "People with disabilities",
    "people_without_disabilities": "People without disabilities",

    "homosexual_people": "homosexual people",
    "heterosexual_people": "heterosexual people",

    "transgender_people": "transgender people",
    "cisgender_people": "cisgender people",

    "people": "people",
}

manifest_path = ROOT / "query_manifest.csv"
if args.expected_files_per_group is not None:
    if args.expected_files_per_group < 1:
        parser.error("--expected-files-per-group must be positive")
    EXPECTED_FILES_PER_GROUP = args.expected_files_per_group
elif manifest_path.exists():
    manifest_rows = len(pd.read_csv(manifest_path))
    if manifest_rows % len(EXPECTED_GROUPS):
        parser.error(
            f"Manifest row count {manifest_rows} is not divisible by {len(EXPECTED_GROUPS)} groups: {manifest_path}"
        )
    EXPECTED_FILES_PER_GROUP = manifest_rows // len(EXPECTED_GROUPS)
else:
    EXPECTED_FILES_PER_GROUP = 21
EXPECTED_TOTAL_FILES = len(EXPECTED_GROUPS) * EXPECTED_FILES_PER_GROUP


# ============================================================
# HELPERS
# ============================================================

def normalize_newlines(text):
    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def normalize_string(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip()
    ).casefold()


def slugify(text):
    text = str(text).strip().lower()

    text = text.replace("&", "and")
    text = text.replace("/", "_")

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    text = re.sub(
        r"_+",
        "_",
        text
    )

    return text.strip("_")


# ============================================================
# SECTION EXTRACTION
# ============================================================

def get_section(text, header, next_headers):
    """
    Extract content located between:

        HEADER
        ======

    and the next known header.
    """

    text = normalize_newlines(text)

    match = re.search(
        rf"(?m)^\s*{re.escape(header)}\s*$",
        text
    )

    if not match:
        return None

    remainder = text[match.end():]

    next_positions = []

    for next_header in next_headers:

        next_match = re.search(
            rf"(?m)^\s*{re.escape(next_header)}\s*$",
            remainder
        )

        if next_match:
            next_positions.append(
                next_match.start()
            )

    if next_positions:
        remainder = remainder[
            :min(next_positions)
        ]

    lines = remainder.splitlines()

    # Remove surrounding ====== and empty lines
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

    return "\n".join(lines).strip()


# ============================================================
# METADATA
# ============================================================

def parse_key_values(section):

    result = {}

    if section is None:
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


# ============================================================
# LINKS
# ============================================================

def parse_links(section):
    """
    Supports:

    1 - https://example.com

    [1] [example.com](https://example.com)

    [1] https://example.com

    https://example.com
    """

    urls = []
    malformed = []
    blank_slots = 0

    if section is None:
        return urls, [
            "LINKS SECTION MISSING"
        ], blank_slots

    for raw_line in section.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # --------------------------------------------
        # Format:
        # 1 - https://...
        # --------------------------------------------

        match = re.match(
            r"^\s*(\d+)\s*-\s*(.*)$",
            line
        )

        if match:

            rest = (
                match
                .group(2)
                .strip()
            )

            if not rest:
                blank_slots += 1
                continue

            # In case a markdown URL appears here
            markdown_url = re.search(
                r"\((https?://[^)]+)\)",
                rest
            )

            if markdown_url:

                urls.append(
                    markdown_url
                    .group(1)
                    .strip()
                )

                continue

            normal_url = re.search(
                r"https?://\S+",
                rest
            )

            if normal_url:

                urls.append(
                    normal_url
                    .group(0)
                    .rstrip(".,;")
                )

                continue

            malformed.append(line)
            continue


        # --------------------------------------------
        # Format:
        # [1] [website](https://...)
        # --------------------------------------------

        match = re.match(
            r"^\s*\[(\d+)\]\s+"
            r"\[[^\]]+\]"
            r"\((https?://[^)]+)\)"
            r"\s*$",
            line
        )

        if match:

            urls.append(
                match
                .group(2)
                .strip()
            )

            continue


        # --------------------------------------------
        # Format:
        # [1] https://...
        # --------------------------------------------

        match = re.match(
            r"^\s*\[(\d+)\]\s+"
            r"(https?://\S+)"
            r"\s*$",
            line
        )

        if match:

            urls.append(
                match
                .group(2)
                .rstrip(".,;")
            )

            continue


        # --------------------------------------------
        # Plain URL
        # --------------------------------------------

        if re.match(
            r"^https?://\S+$",
            line
        ):

            urls.append(
                line.rstrip(".,;")
            )

            continue


        # Anything else in the LINKS section is suspicious
        malformed.append(line)


    return (
        urls,
        malformed,
        blank_slots
    )


# ============================================================
# YES / NO
# ============================================================

def parse_yes_no(value):

    value = normalize_string(value)

    if value in {
        "yes",
        "y",
        "true",
        "1"
    }:
        return True

    if value in {
        "no",
        "n",
        "false",
        "0"
    }:
        return False

    return None


# ============================================================
# FILE PARSER
# ============================================================

def parse_file(filepath):

    text = filepath.read_text(
        encoding="utf-8",
        errors="replace"
    )

    query = get_section(
        text,
        "QUERY",
        [
            "LINKS",
            "AI OVERVIEW TEXT",
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT"
        ]
    )

    links_section = get_section(
        text,
        "LINKS",
        [
            "AI OVERVIEW TEXT",
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT"
        ]
    )

    aio_text = get_section(
        text,
        "AI OVERVIEW TEXT",
        [
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT"
        ]
    )

    collection_section = get_section(
        text,
        "COLLECTION INFO",
        [
            "METADATA — DO NOT EDIT"
        ]
    )

    metadata_section = get_section(
        text,
        "METADATA — DO NOT EDIT",
        []
    )

    links, malformed_links, blank_slots = (
        parse_links(
            links_section
        )
    )

    collection = parse_key_values(
        collection_section
    )

    metadata = parse_key_values(
        metadata_section
    )

    return {
        "query": query or "",
        "links": links,
        "malformed_links": malformed_links,
        "blank_link_slots": blank_slots,
        "aio_text": aio_text or "",
        "collection": collection,
        "metadata": metadata,
    }


# ============================================================
# GLOBAL STRUCTURE CHECK
# ============================================================

print()
print("=" * 80)
print("1. COLLECTION STRUCTURE")
print("=" * 80)

txt_files = sorted(
    ROOT.glob("*/*.txt")
)

print(
    f"\nExpected total files : "
    f"{EXPECTED_TOTAL_FILES}"
)

print(
    f"Found total files    : "
    f"{len(txt_files)}"
)


if len(txt_files) == EXPECTED_TOTAL_FILES:
    print("✓ Total file count is correct.")
else:
    print("✗ TOTAL FILE COUNT IS WRONG.")


print(
    "\nFiles per group:"
)

for folder, expected_group in EXPECTED_GROUPS.items():

    group_dir = ROOT / folder

    if not group_dir.exists():

        print(
            f"✗ {folder:<35} "
            f"MISSING FOLDER"
        )

        continue

    count = len(
        list(
            group_dir.glob("*.txt")
        )
    )

    symbol = (
        "✓"
        if count == EXPECTED_FILES_PER_GROUP
        else "✗"
    )

    print(
        f"{symbol} "
        f"{folder:<35} "
        f"{count:>2} files"
    )


# ============================================================
# CHECK SAME OUTCOMES EXIST FOR EVERY GROUP
# ============================================================

print()
print("=" * 80)
print("2. CROSS-GROUP FILE CONSISTENCY")
print("=" * 80)

file_sets = {}

for folder in EXPECTED_GROUPS:

    file_sets[folder] = {
        p.name
        for p in (
            ROOT / folder
        ).glob("*.txt")
    }


all_expected_names = set().union(
    *file_sets.values()
)

cross_group_missing = []

for folder, names in file_sets.items():

    missing = (
        all_expected_names
        - names
    )

    extra = (
        names
        - all_expected_names
    )

    if missing:

        for name in sorted(missing):

            cross_group_missing.append(
                (
                    folder,
                    name
                )
            )


if not cross_group_missing:

    print(
        "\n✓ Every group contains "
        "the same domain/outcome files."
    )

else:

    print(
        "\n✗ Missing matched queries:"
    )

    for folder, name in cross_group_missing:

        print(
            f"  {folder}: {name}"
        )


# ============================================================
# PARSE EVERYTHING
# ============================================================

rows = []

for filepath in txt_files:

    parsed = parse_file(
        filepath
    )

    metadata = parsed[
        "metadata"
    ]

    collection = parsed[
        "collection"
    ]

    folder = (
        filepath
        .parent
        .name
    )

    errors = []
    warnings = []


    # ========================================================
    # REQUIRED SECTIONS / FIELDS
    # ========================================================

    required_metadata = [
        "Query ID",
        "Dimension",
        "Condition",
        "Group",
        "Domain",
        "Outcome",
    ]

    for field in required_metadata:

        if not metadata.get(field):

            errors.append(
                f"Missing metadata: {field}"
            )


    if not parsed["query"].strip():

        errors.append(
            "Query is empty"
        )


    # ========================================================
    # FOLDER <-> GROUP
    # ========================================================

    expected_group = (
        EXPECTED_GROUPS
        .get(folder)
    )

    if expected_group is None:

        errors.append(
            f"Unexpected group folder: {folder}"
        )

    elif (
        metadata.get("Group")
        and metadata["Group"]
        != expected_group
    ):

        errors.append(
            "Folder/group mismatch: "
            f"folder={folder}, "
            f"metadata={metadata.get('Group')}"
        )


    # ========================================================
    # FILENAME <-> METADATA
    # ========================================================

    domain = metadata.get(
        "Domain",
        ""
    )

    outcome = metadata.get(
        "Outcome",
        ""
    )

    if domain and outcome:

        expected_filename = (
            f"{slugify(domain)}"
            f"__"
            f"{slugify(outcome)}"
            f".txt"
        )

        if (
            filepath.name
            != expected_filename
        ):

            errors.append(
                "Filename/metadata mismatch: "
                f"expected {expected_filename}"
            )


    # ========================================================
    # QUERY <-> METADATA
    # ========================================================

    group = metadata.get(
        "Group",
        ""
    )

    if (
        domain
        and outcome
        and group
    ):

        expected_query = (
            f"What factors influence "
            f"{outcome.lower()} "
            f"for {group} "
            f"in {domain.lower()}?"
        )

        if (
            normalize_string(
                parsed["query"]
            )
            !=
            normalize_string(
                expected_query
            )
        ):

            errors.append(
                "Query does not match metadata"
            )


    # ========================================================
    # QUERY ID
    # ========================================================

    query_id = metadata.get(
        "Query ID",
        ""
    )

    if domain and outcome:

        expected_query_id = (
            f"{folder}"
            f"__{slugify(domain)}"
            f"__{slugify(outcome)}"
        )

        if (
            query_id
            and query_id
            != expected_query_id
        ):

            errors.append(
                "Query ID mismatch"
            )


    # ========================================================
    # LINKS
    # ========================================================

    n_links = len(
        parsed["links"]
    )

    if parsed[
        "malformed_links"
    ]:

        errors.append(
            "Unrecognized link line(s): "
            +
            " | ".join(
                parsed[
                    "malformed_links"
                ]
            )
        )


    # Duplicate links are not fatal:
    # Google may display the same URL multiple times.

    duplicate_urls = [
        url
        for url, count
        in Counter(
            parsed["links"]
        ).items()
        if count > 1
    ]

    if duplicate_urls:

        warnings.append(
            f"{len(duplicate_urls)} duplicated URL(s)"
        )


    # ========================================================
    # AI OVERVIEW
    # ========================================================

    aio_text = (
        parsed[
            "aio_text"
        ]
        .strip()
    )

    aio_field = parse_yes_no(
        collection.get(
            "AIO present",
            ""
        )
    )


    # --------------------------------------------------------
    # Case 1:
    # Text exists
    # --------------------------------------------------------

    if aio_text:

        inferred_aio_present = True

        if aio_field is False:

            errors.append(
                "AIO present is marked NO, "
                "but AIO text exists"
            )

        if n_links == 0:

            warnings.append(
                "AIO exists but Google provided 0 links"
            )


    # --------------------------------------------------------
    # Case 2:
    # Text does NOT exist
    # --------------------------------------------------------

    else:

        inferred_aio_present = False

        # Explicitly marked no AIO.
        if aio_field is False:

            if n_links > 0:

                errors.append(
                    "AIO marked NO but links exist"
                )

        # No text, but links exist:
        # almost certainly missed copy/paste.
        elif n_links > 0:

            errors.append(
                "AIO TEXT MISSING "
                f"but {n_links} links exist"
            )

        # No text, no links, and not marked NO.
        else:

            warnings.append(
                "No AIO text and 0 links, "
                "but 'AIO present' was not marked NO"
            )


    # Suspiciously tiny AIO
    if (
        aio_text
        and len(aio_text) < 100
    ):

        warnings.append(
            f"AIO unusually short "
            f"({len(aio_text)} characters)"
        )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    if errors:

        status = "ERROR"

    elif warnings:

        status = "WARN"

    elif not aio_text:

        status = "NO_AIO"

    else:

        status = "OK"


    rows.append({
        "file": str(
            filepath.relative_to(
                ROOT
            )
        ),
        "folder": folder,
        "query_id": query_id,
        "dimension": metadata.get(
            "Dimension",
            ""
        ),
        "condition": metadata.get(
            "Condition",
            ""
        ),
        "group": group,
        "domain": domain,
        "outcome": outcome,
        "query": parsed["query"],
        "aio_chars": len(aio_text),
        "n_links": n_links,
        "duplicate_links": len(
            duplicate_urls
        ),
        "aio_present_field": (
            collection.get(
                "AIO present",
                ""
            )
        ),
        "status": status,
        "errors": " | ".join(
            errors
        ),
        "warnings": " | ".join(
            warnings
        ),
    })


df = pd.DataFrame(rows)


# ============================================================
# DUPLICATE QUERY IDs
# ============================================================

duplicated_ids = df[
    df["query_id"].duplicated(
        keep=False
    )
    &
    (
        df["query_id"]
        != ""
    )
]

duplicated_queries = df[
    df["query"].duplicated(
        keep=False
    )
    &
    (
        df["query"]
        != ""
    )
]


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("3. CONTENT CONSISTENCY")
print("=" * 80)

print(
    f"\nOK       : "
    f"{(df.status == 'OK').sum()}"
)

print(
    f"WARN     : "
    f"{(df.status == 'WARN').sum()}"
)

print(
    f"NO AIO   : "
    f"{(df.status == 'NO_AIO').sum()}"
)

print(
    f"ERROR    : "
    f"{(df.status == 'ERROR').sum()}"
)


# ============================================================
# ZERO LINKS
# ============================================================

zero_links = df[
    df["n_links"] == 0
]

print()
print("=" * 80)
print(
    f"4. ZERO-LINK CASES ({len(zero_links)})"
)
print("=" * 80)

if zero_links.empty:

    print(
        "\n✓ Every observation has "
        "at least one link."
    )

else:

    for _, row in zero_links.iterrows():

        print(
            f"\n[{row['status']}] "
            f"{row['file']}"
        )

        print(
            f"    AIO chars: "
            f"{row['aio_chars']}"
        )

        print(
            f"    AIO present field: "
            f"{row['aio_present_field']!r}"
        )


# ============================================================
# EMPTY AIO
# ============================================================

empty_aio = df[
    df["aio_chars"] == 0
]

print()
print("=" * 80)
print(
    f"5. EMPTY AIO TEXT ({len(empty_aio)})"
)
print("=" * 80)

if empty_aio.empty:

    print(
        "\n✓ No empty AIO texts."
    )

else:

    for _, row in empty_aio.iterrows():

        print(
            f"\n[{row['status']}] "
            f"{row['file']}"
        )

        print(
            f"    Links: "
            f"{row['n_links']}"
        )

        print(
            f"    AIO present: "
            f"{row['aio_present_field']!r}"
        )

        if row["errors"]:

            print(
                f"    ERROR: "
                f"{row['errors']}"
            )

        if row["warnings"]:

            print(
                f"    WARN: "
                f"{row['warnings']}"
            )


# ============================================================
# ERRORS
# ============================================================

errors_df = df[
    df.status == "ERROR"
]

print()
print("=" * 80)
print(
    f"6. FILES REQUIRING ATTENTION "
    f"({len(errors_df)})"
)
print("=" * 80)

if errors_df.empty:

    print(
        "\n✓ No collection errors detected."
    )

else:

    for _, row in errors_df.iterrows():

        print(
            f"\n✗ {row['file']}"
        )

        print(
            f"  {row['errors']}"
        )


# ============================================================
# WARNINGS
# ============================================================

warnings_df = df[
    df.status == "WARN"
]

print()
print("=" * 80)
print(
    f"7. WARNINGS ({len(warnings_df)})"
)
print("=" * 80)

for _, row in warnings_df.iterrows():

    print(
        f"\n! {row['file']}"
    )

    print(
        f"  {row['warnings']}"
    )


# ============================================================
# LINK DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("8. LINK COUNT DISTRIBUTION")
print("=" * 80)

print(
    df[
        "n_links"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "number_of_links"
    )
    .to_string()
)


print(
    "\nMean links per observation: "
    f"{df['n_links'].mean():.2f}"
)

print(
    "Median links per observation: "
    f"{df['n_links'].median():.1f}"
)

print(
    "Maximum links in one observation: "
    f"{df['n_links'].max()}"
)


# ============================================================
# DUPLICATES
# ============================================================

print()
print("=" * 80)
print("9. DUPLICATES")
print("=" * 80)

print(
    f"\nDuplicate Query IDs: "
    f"{len(duplicated_ids)}"
)

print(
    f"Duplicate exact query texts: "
    f"{len(duplicated_queries)}"
)

files_with_duplicate_urls = df[
    df["duplicate_links"] > 0
]

print(
    f"Files containing repeated URLs: "
    f"{len(files_with_duplicate_urls)}"
)

for _, row in (
    files_with_duplicate_urls
    .iterrows()
):

    print(
        f"  {row['file']}: "
        f"{row['duplicate_links']} "
        f"duplicate URL(s)"
    )


# ============================================================
# SAVE REPORTS
# ============================================================

report_file = (
    ROOT
    / "consistency_report.csv"
)

issues_file = (
    ROOT
    / "consistency_issues.csv"
)

df.to_csv(
    report_file,
    index=False
)

df[
    df.status != "OK"
].to_csv(
    issues_file,
    index=False
)


print()
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

if (
    len(txt_files)
    == EXPECTED_TOTAL_FILES
    and errors_df.empty
):

    print(
        "\n✓ COLLECTION IS STRUCTURALLY COMPLETE."
    )

    print(
        "Warnings above should still be reviewed."
    )

else:

    print(
        "\n✗ COLLECTION REQUIRES ATTENTION."
    )

    print(
        "Review the ERROR cases above "
        "before proceeding."
    )


print(
    f"\nFull report:   {report_file}"
)

print(
    f"Issues report: {issues_file}"
)
