"""Finalize validation links, diagnostics JSON, and proximity forest labels."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from complete_epistemic_validation_entries import complete


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ["Race", "Ethnicity", "Gender", "Disability", "Sexual Orientation", "Gender Identity"]


def valid_json(value):
    if isinstance(value, dict):
        return {key: valid_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [valid_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fix_json_file(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(valid_json(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def proximity_forest(data: pd.DataFrame, analysis: str, label: str, path: Path):
    part = data.loc[data.analysis.eq(analysis)].set_index("dimension").reindex(DIMENSIONS)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, dimension in enumerate(DIMENSIONS):
        row = part.loc[dimension]
        if not np.isfinite(row.estimate):
            continue
        y = len(DIMENSIONS) - 1 - i
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            ax.errorbar(row.estimate, y, xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                        fmt="o", capsize=3, color="#2878B5", markersize=6)
        else:
            ax.scatter(row.estimate, y, color="#2878B5", s=35)
        ax.text(row.estimate, y + .17, f"n={int(row.n_outcomes)}", ha="center", fontsize=8)
    ax.axvline(0, linestyle="--", linewidth=1, color="#444444")
    ax.set_yticks(range(len(DIMENSIONS)-1, -1, -1), DIMENSIONS)
    ax.set_xlabel("Mean [|People − Majority| − |People − Minority|]")
    ax.set_title(f"Mean People proximity · {label}", loc="left", fontweight="bold")
    fig.text(.5, .01, "Positive: People closer to Minority. Negative: closer to Majority. Intervals bootstrap outcomes; veridicality uses covered triplets only.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(.02, .035, .99, .98))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def finalize(source: Path, output: Path):
    complete(source, output)
    diagnostics = output / "models" / "occurrence_model_diagnostics.json"
    if diagnostics.exists():
        fix_json_file(diagnostics)
    manifest = output / "models" / "analysis_manifest.json"
    if manifest.exists():
        fix_json_file(manifest)
    summary = pd.read_csv(output / "tables" / "people_proximity_summary.csv")
    main_dir = output / "figures" / "main"
    proximity_forest(summary, "contextual_hedging", "contextual hedging", main_dir / "people_proximity_contextual_forest.png")
    proximity_forest(summary, "veridicality_score", "strict veridicality score", main_dir / "people_proximity_veridicality_forest.png")
    validation_table = output / "validation" / "measurement_validation.csv"
    if validation_table.exists():
        shutil.copyfile(validation_table, output / "tables" / "measurement_validation.csv")
    print(f"Finalized validation resource IDs and proximity figures under {output}")


def main():
    parser = argparse.ArgumentParser(description="Finalize epistemic commitment v2 outputs.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "results" / "epistemic_commitment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "epistemic_commitment_v2")
    args = parser.parse_args()
    finalize(args.source_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
