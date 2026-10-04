#!/usr/bin/env python3
"""Run the complete final three-location Google AIO audit pipeline.

Default project layout:
  /scratch/victoria.estanislau/ai-summary/
    annotations/
      v1_dallas/google_aio_collection/
      v2_ny/google_aio_collection/
      v3_la/google_aio_collection/
    artifacts/annotation_inputs/      # optional query-selection annotation files
    results/

Primary paper analyses produced per location:
  1) collection consistency checks
  2) citation/link count analysis
  3) source-set overlap/displacement analysis
  4) semantic answer displacement analysis
  5) cited-source download/extraction (resumable)
  6) evidence-to-synthesis semantic alignment analysis

Then:
  7) three-location compact summary
  8) secondary cross-location pooled analysis (outcomes remain the inferential unit)

Source-type and claim-survival analyses are intentionally NOT included because the
current paper version does not rely on those unvalidated model-derived taxonomies.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = Path("/scratch/victoria.estanislau/ai-summary")

DEFAULT_LOCATIONS = {
    "dallas": {
        "version": "v1_dallas",
        "label": "Dallas, Texas",
    },
    "ny": {
        "version": "v2_ny",
        "label": "New York City, New York",
    },
    "la": {
        "version": "v3_la",
        "label": "Los Angeles, California",
    },
}

PIPELINE = [
    ("link_count", "link_count_analysis.py"),
    ("source_overlap", "source_overlap_analysis.py"),
    ("semantic_displacement", "semantic_displacement_analysis.py"),
    ("source_download", "collect_cited_sources.py"),
    ("evidence_alignment", "evidence_synthesis_analysis.py"),
]


def tee_run(cmd: list[str], *, env: dict[str, str], log_file: Path) -> None:
    """Run command, stream output to terminal, and save exact log."""
    print("\n" + "=" * 100)
    print("RUNNING:", " ".join(cmd))
    print("LOG:", log_file)
    print("=" * 100 + "\n")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        rc = proc.wait()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def check_consistency_report(collection_dir: Path) -> None:
    report = collection_dir / "consistency_report.csv"
    if not report.exists():
        raise RuntimeError(f"Consistency report was not created: {report}")

    with report.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    error_rows = [r for r in rows if str(r.get("status", "")).strip().upper() == "ERROR"]
    if error_rows:
        preview = "\n".join(
            f"  - {r.get('file', '<unknown>')}: {r.get('errors', '')}"
            for r in error_rows[:10]
        )
        raise RuntimeError(
            f"Collection has {len(error_rows)} ERROR rows and should not be analyzed.\n{preview}"
        )


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the final Dallas + NYC + LA Google AI Overview audit pipeline."
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    p.add_argument(
        "--locations",
        nargs="+",
        choices=["dallas", "ny", "la"],
        default=["dallas", "ny", "la"],
        help="Locations to run. Default: all three.",
    )
    p.add_argument("--dallas-version", default="v1_dallas")
    p.add_argument("--ny-version", default="v2_ny")
    p.add_argument("--la-version", default="v3_la")
    p.add_argument(
        "--skip-source-download",
        action="store_true",
        help="Skip URL downloading/extraction. Only use if each location already has a source_corpus.",
    )
    p.add_argument(
        "--skip-evidence",
        action="store_true",
        help="Skip evidence-to-synthesis analysis.",
    )
    p.add_argument(
        "--skip-query-agreement",
        action="store_true",
        help="Do not run the optional Gwet query-selection agreement step.",
    )
    p.add_argument(
        "--skip-pooled-secondary",
        action="store_true",
        help="Do not produce the secondary cross-location pooled analysis.",
    )
    p.add_argument(
        "--only",
        choices=[
            "consistency",
            "link_count",
            "source_overlap",
            "semantic_displacement",
            "source_download",
            "evidence_alignment",
        ],
        help="Run only one per-location stage (plus consistency unless the stage itself is consistency).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Default: current interpreter.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_dir.resolve()
    # IMPORTANT: do not resolve a virtualenv Python symlink.
    # venv/bin/python often points to /usr/bin/python3.x; Path.resolve() would
    # silently discard the virtualenv and run the system interpreter.
    if os.path.sep in args.python:
        python_path = Path(args.python).expanduser()
        if not python_path.is_absolute():
            python_path = Path.cwd() / python_path
        python = str(python_path.absolute())  # preserves the venv symlink
    else:
        python = shutil.which(args.python) or args.python

    versions = {
        "dallas": args.dallas_version,
        "ny": args.ny_version,
        "la": args.la_version,
    }
    locations = {
        slug: {
            "version": versions[slug],
            "label": DEFAULT_LOCATIONS[slug]["label"],
        }
        for slug in args.locations
    }

    # Project-level dependency/data checks.
    require_path(base / "annotations", "annotations directory")
    if shutil.which("curl") is None and not args.skip_source_download and args.only in (None, "source_download"):
        raise RuntimeError("curl is required by collect_cited_sources.py but was not found in PATH.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = base / "results" / "run_logs" / run_stamp
    log_root.mkdir(parents=True, exist_ok=True)

    print("\nFINAL THREE-LOCATION AIO AUDIT")
    print("=" * 100)
    print(f"Base directory : {base}")
    print(f"Python         : {python}")
    print("Locations      : " + ", ".join(f"{k}={v['version']}" for k, v in locations.items()))
    print(f"Logs           : {log_root}")
    print("=" * 100)

    # Early interpreter sanity check.
    probe = subprocess.run(
        [python, "-c", "import sys, pandas; print(sys.executable); print(pandas.__version__)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "The selected Python interpreter cannot import pandas.\n"
            f"Interpreter: {python}\n"
            f"Probe output:\n{probe.stdout}"
        )
    probe_lines = [x.strip() for x in probe.stdout.splitlines() if x.strip()]
    if probe_lines:
        print(f"Interpreter check: {probe_lines[0]}")
    if len(probe_lines) > 1:
        print(f"pandas          : {probe_lines[1]}")

    # Optional query-design agreement. It is location-independent.
    if not args.skip_query_agreement and args.only is None:
        ann_input = base / "artifacts" / "annotation_inputs"
        a1 = ann_input / "outcomes_annotator1.csv"
        a2 = ann_input / "outcomes_annotator2.csv"
        if a1.exists() and a2.exists():
            env = os.environ.copy()
            env["AIO_BASE_DIR"] = str(base)
            tee_run(
                [python, str(SCRIPT_DIR / "query_selection_agreement.py")],
                env=env,
                log_file=log_root / "query_selection_agreement.log",
            )
        else:
            print(
                "\n[query agreement] Annotation input CSVs were not found; skipping.\n"
                f"Expected:\n  {a1}\n  {a2}\n"
            )

    for slug, info in locations.items():
        version = info["version"]
        label = info["label"]
        collection_dir = base / "annotations" / version / "google_aio_collection"
        require_path(collection_dir, f"{label} AIO collection")

        print("\n" + "#" * 100)
        print(f"LOCATION: {label}  [{version}]")
        print("#" * 100)

        env = os.environ.copy()
        env.update({
            "AIO_BASE_DIR": str(base),
            "AIO_COLLECTION_VERSION": version,
            "AIO_LOCATION_SLUG": slug,
            "AIO_LOCATION_LABEL": label,
        })

        # Consistency is always first for any stage that uses the collection.
        if args.only in (None, "consistency", "link_count", "source_overlap", "semantic_displacement", "source_download", "evidence_alignment"):
            tee_run(
                [
                    python,
                    str(SCRIPT_DIR / "consistency.py"),
                    "--collection-dir",
                    str(collection_dir),
                ],
                env=env,
                log_file=log_root / slug / "00_consistency.log",
            )
            check_consistency_report(collection_dir)

        if args.only == "consistency":
            continue

        for stage, filename in PIPELINE:
            if args.only is not None and stage != args.only:
                continue
            if stage == "source_download" and args.skip_source_download:
                print(f"\n[{slug}] Skipping source download by request.")
                continue
            if stage == "evidence_alignment" and args.skip_evidence:
                print(f"\n[{slug}] Skipping evidence alignment by request.")
                continue
            if stage == "evidence_alignment":
                corpus = base / "annotations" / version / "source_corpus"
                require_path(corpus / "manifest.csv", f"{label} source corpus manifest")
                require_path(corpus / "url_usage.csv", f"{label} source corpus URL usage table")

            step_num = {
                "link_count": "01",
                "source_overlap": "02",
                "semantic_displacement": "03",
                "source_download": "04",
                "evidence_alignment": "05",
            }[stage]

            tee_run(
                [python, str(SCRIPT_DIR / filename)],
                env=env,
                log_file=log_root / slug / f"{step_num}_{stage}.log",
            )

    # Summaries require all three locations and a full run.
    full_three = set(locations) == {"dallas", "ny", "la"}
    if args.only is None and full_three:
        if not args.skip_evidence:
            tee_run(
                [python, str(SCRIPT_DIR / "summarize_three_locations.py"), "--base-dir", str(base)],
                env=os.environ.copy(),
                log_file=log_root / "90_three_location_summary.log",
            )

            if not args.skip_pooled_secondary:
                tee_run(
                    [python, str(SCRIPT_DIR / "pooled_three_location_secondary.py"), "--base-dir", str(base)],
                    env=os.environ.copy(),
                    log_file=log_root / "91_pooled_three_location_secondary.log",
                )
        else:
            print("\nSkipping final three-location summary because evidence analysis was skipped.")

    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print(f"Logs: {log_root}")
    print(f"Results root: {base / 'results'}")
    if full_three and args.only is None and not args.skip_evidence:
        print(f"Three-location summary: {base / 'results' / 'three_location_summary'}")
        if not args.skip_pooled_secondary:
            print(f"Secondary pooled analysis: {base / 'results' / 'three_location_pooled_secondary'}")


if __name__ == "__main__":
    main()
