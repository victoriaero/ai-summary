"""Descriptive figures for the experimental AIO claim extraction pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ORDER = ["people", "minority", "majority"]
PALETTE = {"people": "#667085", "minority": "#B85920", "majority": "#2676A5"}


def _display(metrics: pd.DataFrame) -> pd.DataFrame:
    """Repeat People only for faceted visualization, not response statistics."""
    dimensions = sorted(metrics.loc[metrics.group_type.ne("people"), "demographic_dimension"].dropna().unique())
    people = metrics.loc[metrics.group_type.eq("people")]
    return pd.concat([metrics.loc[metrics.group_type.ne("people")]] +
                     [people.assign(demographic_dimension=d) for d in dimensions], ignore_index=True)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _note(fig, text: str = "Dallas pilot · descriptive only · People is the same shared control response across dimensions"):
    fig.text(.5, .005, text, ha="center", fontsize=8, color="#555")


def plot_all(output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output / "response_metrics.csv")
    sentences = pd.read_csv(output / "sentences.csv")
    claims = pd.read_csv(output / "extracted_claims.csv") if (output / "extracted_claims.csv").stat().st_size > 1 else pd.DataFrame()
    valid = pd.read_csv(output / "valid_claims.csv") if (output / "valid_claims.csv").stat().st_size > 1 else pd.DataFrame()
    matching = pd.read_csv(output / "claim_matching_candidates.csv") if (output / "claim_matching_candidates.csv").stat().st_size > 1 else pd.DataFrame()
    display = _display(metrics)
    dims = sorted(display.demographic_dimension.dropna().unique())
    sns.set_theme(style="whitegrid", context="notebook")

    for field, title, filename in (
        ("proportion_sentences_with_claim", "Sentence rate with verifiable content", "factual_claim_sentence_rate.png"),
        ("n_extracted_claims", "Extracted claims per response", "claims_per_response.png"),
        ("mean_claims_per_claim_sentence", "Claims per claim-bearing sentence", "claims_per_claim_sentence.png"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
        for ax, dimension in zip(axes.flat, dims):
            sub = display.loc[display.demographic_dimension.eq(dimension)]
            observed = sub.loc[sub[field].notna()]
            counts = observed.groupby("group_type").size()
            # A boxplot is undefined for the mostly missing conditional metric.
            # Keep its observed points visible without inventing zero values.
            if all(counts.get(kind, 0) >= 2 for kind in ORDER):
                sns.boxplot(data=observed, x="group_type", y=field, order=ORDER, hue="group_type",
                            hue_order=ORDER, palette=PALETTE, dodge=False, showfliers=False, ax=ax, legend=False)
            if len(observed):
                sns.stripplot(data=observed, x="group_type", y=field, order=ORDER, color="#232323",
                              alpha=.65, jitter=.17, size=4, ax=ax)
            else:
                ax.set_xticks(range(len(ORDER)), ORDER)
                ax.text(.5, .5, "No defined values", transform=ax.transAxes,
                        ha="center", va="center", color="#667085")
            ax.set_title(dimension)
            ax.set_xlabel("")
            ax.set_ylabel(field.replace("_", " ") if ax.get_subplotspec().is_first_col() else "")
            for ix, kind in enumerate(ORDER):
                ax.text(ix, 1.01, f"n={counts.get(kind, 0)}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8)
        fig.suptitle(title, fontsize=15)
        _note(fig)
        fig.tight_layout(rect=(0, .025, 1, .96))
        _save(fig, figures / filename)

    # Funnel counts sentence units; claims themselves are reported separately.
    if len(sentences):
        # The full-stage decisions are reconstructed from checkpoint files, not the sampled sheet.
        from claim_extraction_qwen14b_inference import checkpoint_records
        checkpoints = {stage: checkpoint_records(output / path) for stage, path in (
            ("selection", "selection_outputs.jsonl"), ("ambiguity", "disambiguation_outputs.jsonl"),
            ("decomposition", "logs/decomposition_outputs.jsonl"))}
        def by_source(stage):
            return {r["source_record_id"]: r["parsed"] for r in checkpoints[stage].values()}
        select, ambiguity, decomposition = (by_source(name) for name in ("selection", "ambiguity", "decomposition"))
        funnel_rows = []
        for response in metrics.itertuples():
            sids = sentences.loc[sentences.response_id.eq(response.response_id), "sentence_id"].tolist()
            selected = [sid for sid in sids if (select.get(sid) or {}).get("selection_status") == "HAS_VERIFIABLE_CLAIM"]
            resolved = [sid for sid in selected if (ambiguity.get(sid) or {}).get("ambiguity_status") in
                        {"NO_AMBIGUITY", "RESOLVABLE_AMBIGUITY"}]
            extracted = [sid for sid in resolved if (decomposition.get(sid) or {}).get("claims")]
            entailed = [sid for sid in extracted if len(claims) and ((claims.sentence_id == sid) & claims.entailment.eq("ENTAILED")).any()]
            qualified = [sid for sid in entailed if len(valid) and valid.sentence_id.eq(sid).any()]
            for stage, count in zip(("All", "Selected", "Disambiguated", "Extracted", "Entailed", "Qualified"),
                                    (len(sids), len(selected), len(resolved), len(extracted), len(entailed), len(qualified))):
                funnel_rows.append({"group_type": response.group_type, "stage": stage, "n_sentences": count})
        funnel = pd.DataFrame(funnel_rows).groupby(["group_type", "stage"], as_index=False).n_sentences.sum()
        save_path = output / "funnel_sentence_counts.csv"
        funnel.to_csv(save_path, index=False)
        order = ["All", "Selected", "Disambiguated", "Extracted", "Entailed", "Qualified"]
        fig, ax = plt.subplots(figsize=(11, 5))
        for kind in ORDER:
            sub = funnel.loc[funnel.group_type.eq(kind)].set_index("stage").reindex(order)
            ax.plot(order, sub.n_sentences, marker="o", label=kind.title(), color=PALETTE[kind])
            for x, y in zip(order, sub.n_sentences):
                baseline = sub.loc["All", "n_sentences"]
                ax.annotate(f"{int(y)}\n({y / baseline:.0%})" if baseline else "0", (x, y),
                            textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        ax.set_ylabel("Sentence-like units, not number of claims")
        ax.set_title("Claim extraction funnel by condition")
        ax.legend()
        fig.tight_layout()
        _save(fig, figures / "claim_extraction_funnel.png")

    if len(sentences):
        from claim_extraction_qwen14b_inference import checkpoint_records
        decisions = {r["source_record_id"]: r["parsed"] for r in checkpoint_records(
            output / "disambiguation_outputs.jsonl").values()}
        ambiguity = sentences[["sentence_id", "group_type", "demographic_dimension"]].copy()
        ambiguity["status"] = ambiguity.sentence_id.map(lambda i: (decisions.get(i) or {}).get("ambiguity_status", "NOT_RUN"))
        ambiguity = _display(ambiguity.loc[ambiguity.status.ne("NOT_RUN")]).groupby(
            ["group_type", "demographic_dimension"], as_index=False).agg(
                n=("sentence_id", "size"), n_unresolvable=("status", lambda s: int(s.eq("UNRESOLVABLE_AMBIGUITY").sum())))
        ambiguity["rate"] = ambiguity.n_unresolvable / ambiguity.n
        if len(ambiguity):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=ambiguity, x="demographic_dimension", y="rate", hue="group_type", hue_order=ORDER, palette=PALETTE, ax=ax)
            ax.set_title("Unresolvable ambiguity among selected sentences")
            ax.set_ylabel("Rate among selected sentences")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            _save(fig, figures / "ambiguity_rate.png")

    if len(valid):
        matched_ids = set()
        if len(matching):
            equiv = matching.loc[matching.match_status.eq("EQUIVALENT")]
            matched_ids.update(equiv.claim_a_id)
            matched_ids.update(equiv.claim_b_id)
        coverage = valid.copy()
        coverage["has_counterpart"] = coverage.claim_id.isin(matched_ids)
        summary = coverage.groupby("group_type", as_index=False).agg(n=("claim_id", "size"), rate=("has_counterpart", "mean"))
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=summary, x="group_type", y="rate", order=ORDER, hue="group_type", palette=PALETTE, legend=False, ax=ax)
        for i, row in enumerate(summary.set_index("group_type").reindex(ORDER).itertuples()):
            if np.isfinite(row.rate):
                ax.text(i, row.rate, f"n={int(row.n)}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Claims with a provisional counterpart")
        ax.set_ylabel("Share of accepted claims")
        fig.tight_layout()
        _save(fig, figures / "matched_claim_coverage.png")

        matrix = np.eye(3)
        for i, left in enumerate(ORDER):
            for j, right in enumerate(ORDER):
                if i >= j:
                    continue
                sub = matching.loc[matching.match_status.eq("EQUIVALENT") &
                                   (((matching.group_type_a == left) & (matching.group_type_b == right)) |
                                    ((matching.group_type_a == right) & (matching.group_type_b == left)))] if len(matching) else matching
                eligible = set(coverage.loc[coverage.group_type.eq(left), "claim_id"])
                both = set(sub.claim_a_id).union(sub.claim_b_id) & eligible if len(sub) else set()
                other = set(coverage.loc[coverage.group_type.eq(right), "claim_id"])
                both_other = set(sub.claim_a_id).union(sub.claim_b_id) & other if len(sub) else set()
                left_rate = len(both) / len(eligible) if eligible else np.nan
                right_rate = len(both_other) / len(other) if other else np.nan
                matrix[i, j] = matrix[j, i] = np.nanmean([left_rate, right_rate]) if np.isfinite([left_rate, right_rate]).any() else np.nan
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", xticklabels=[x.title() for x in ORDER],
                    yticklabels=[x.title() for x in ORDER], ax=ax)
        ax.set_title("Provisional claim overlap matrix\n(mean of both directional coverage rates)")
        fig.tight_layout()
        _save(fig, figures / "claim_overlap_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "claim_extraction_pilot" / "microsoft-phi-4")
    args = parser.parse_args()
    plot_all(args.output_dir)


if __name__ == "__main__":
    main()
