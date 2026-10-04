from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align lexical and contextual hedge metrics by response for sensitivity analyses."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "epistemic_commitment",
    )
    args = parser.parse_args()
    root = args.results_dir.resolve()
    lexical_path = root / "lexical_bioscope_overlap" / "hedging_response_metrics.csv"
    contextual_path = root / "contextual_hedging" / "contextual_hedging_response_metrics.csv"
    missing = [path for path in (lexical_path, contextual_path) if not path.is_file()]
    if missing:
        absent = "\n".join(f"  - {path}" for path in missing)
        parser.error(
            "inputs not found:\n"
            f"{absent}\n"
            "Generate the missing tables first. In particular, if the contextual "
            "file is absent, run from the repository root:\n"
            "  python scripts/analyze_contextual_hedging_v2.py\n"
            "Then rerun:\n"
            "  python scripts/compare_hedging_sensitivity.py"
        )

    lexical = pd.read_csv(lexical_path)
    contextual = pd.read_csv(contextual_path)
    cols = [
        "response_id", "query_id", "outcome", "dimension", "condition",
        "group", "group_type", "location", "replica_id",
    ]
    keep_lex = cols + [
        "n_sentences", "n_hedged_sentences", "hedged_sentence_rate",
        "n_hedge_occurrences", "hedge_per_100_tokens",
    ]
    keep_ctx = [
        "response_id", "n_contextual_hedges", "n_hedged_sentences",
        "contextual_hedges_per_100_tokens", "contextual_hedged_sentence_rate",
    ]
    data = lexical[keep_lex].merge(
        contextual[keep_ctx],
        on="response_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_lexical", "_contextual"),
    )
    output_path = root / "hedging_sensitivity_comparison.csv"
    data.to_csv(output_path, index=False)
    print(f"Aligned hedge sensitivity table written to {output_path}")


if __name__ == "__main__":
    main()
