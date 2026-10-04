from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "artifacts" / "annotation_results" / "selected_top3_outcomes.csv"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
VERSIONS = ("v1", "v2", "v3")
COLLECTION_DIRNAME = "google_aio_collection"

QUERY_TEMPLATE = "What factors influence {outcome} for {group} in {domain}?"
NUMBER_OF_LINK_SLOTS = 15

GROUPS = [
    {"dimension": "Race", "group": "Black people", "condition": "minority", "folder": "black_people"},
    {"dimension": "Race", "group": "White people", "condition": "majority", "folder": "white_people"},
    {"dimension": "Ethnicity", "group": "Latino people", "condition": "minority", "folder": "latino_people"},
    {"dimension": "Ethnicity", "group": "non-Latino people", "condition": "majority", "folder": "non_latino_people"},
    {"dimension": "Gender", "group": "Women", "condition": "minority", "folder": "women"},
    {"dimension": "Gender", "group": "Men", "condition": "majority", "folder": "men"},
    {"dimension": "Disability", "group": "People with disabilities", "condition": "minority", "folder": "people_with_disabilities"},
    {"dimension": "Disability", "group": "People without disabilities", "condition": "majority", "folder": "people_without_disabilities"},
    {"dimension": "Sexual Orientation", "group": "homosexual people", "condition": "minority", "folder": "homosexual_people"},
    {"dimension": "Sexual Orientation", "group": "heterosexual people", "condition": "majority", "folder": "heterosexual_people"},
    {"dimension": "Gender Identity", "group": "transgender people", "condition": "minority", "folder": "transgender_people"},
    {"dimension": "Gender Identity", "group": "cisgender people", "condition": "majority", "folder": "cisgender_people"},
    {"dimension": "Control", "group": "people", "condition": "control", "folder": "people"},
]


def slugify(text: object) -> str:
    value = str(text).strip().lower().replace("&", "and").replace("/", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def make_query(domain: object, outcome: object, group: str) -> str:
    return QUERY_TEMPLATE.format(
        outcome=str(outcome).strip().lower(),
        group=group,
        domain=str(domain).strip().lower(),
    )


def load_outcomes(input_csv: Path = INPUT_CSV) -> pd.DataFrame:
    dataframe = pd.read_csv(input_csv)
    required_columns = {"Domain", "Outcome"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    if "rank" in dataframe.columns:
        dataframe = dataframe[dataframe["rank"] <= 3].copy()

    return dataframe.drop_duplicates(subset=["Domain", "Outcome"]).reset_index(drop=True)


def build_collection_file(
    query: str,
    query_id: str,
    group_info: dict[str, str],
    domain: str,
    outcome: str,
) -> str:
    links_block = "\n".join(
        f"{number} - " for number in range(1, NUMBER_OF_LINK_SLOTS + 1)
    )
    return f"""============================================================
QUERY
============================================================

{query}


============================================================
LINKS
============================================================

{links_block}


============================================================
AI OVERVIEW TEXT
============================================================




============================================================
COLLECTION INFO
============================================================

AIO present:

Date/time:

VPN location:

Public IP:

Language: English

Notes:



============================================================
METADATA — DO NOT EDIT
============================================================

Query ID: {query_id}
Dimension: {group_info["dimension"]}
Condition: {group_info["condition"]}
Group: {group_info["group"]}
Domain: {domain}
Outcome: {outcome}

"""


def generate_collection(
    dataframe: pd.DataFrame, collection_dir: Path
) -> tuple[int, int]:
    collection_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    created = 0
    skipped = 0
    query_number = 1

    for group_info in GROUPS:
        for _, row in dataframe.iterrows():
            domain = str(row["Domain"]).strip()
            outcome = str(row["Outcome"]).strip()
            query = make_query(domain, outcome, group_info["group"])
            filename = f"{slugify(domain)}__{slugify(outcome)}.txt"
            relative_file = Path(group_info["folder"]) / filename
            filepath = collection_dir / relative_file
            query_id = (
                f"{group_info['folder']}__{slugify(domain)}__{slugify(outcome)}"
            )

            filepath.parent.mkdir(parents=True, exist_ok=True)
            if filepath.exists():
                skipped += 1
            else:
                filepath.write_text(
                    build_collection_file(
                        query, query_id, group_info, domain, outcome
                    ),
                    encoding="utf-8",
                )
                created += 1

            manifest.append(
                {
                    "query_number": query_number,
                    "query_id": query_id,
                    "dimension": group_info["dimension"],
                    "condition": group_info["condition"],
                    "group": group_info["group"],
                    "domain": domain,
                    "outcome": outcome,
                    "query": query,
                    "file": str(Path(COLLECTION_DIRNAME) / relative_file),
                }
            )
            query_number += 1

    pd.DataFrame(manifest).to_csv(
        collection_dir / "query_manifest.csv", index=False
    )
    return created, skipped


def main() -> None:
    dataframe = load_outcomes()
    total_queries = len(dataframe) * len(GROUPS)

    print("=" * 70)
    print("GOOGLE AI OVERVIEW MANUAL COLLECTION GENERATOR")
    print("=" * 70)
    print(f"Outcomes: {len(dataframe)}")
    print(f"Conditions: {len(GROUPS)}")
    print(f"Queries per version: {total_queries}")

    for version in VERSIONS:
        collection_dir = ANNOTATIONS_DIR / version / COLLECTION_DIRNAME
        created, skipped = generate_collection(dataframe, collection_dir)
        print(
            f"{version}: {created} created, {skipped} preserved — "
            f"{collection_dir}"
        )


if __name__ == "__main__":
    main()
