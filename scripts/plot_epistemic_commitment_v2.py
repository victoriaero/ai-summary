from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DIMENSIONS = ["Race", "Ethnicity", "Gender", "Disability", "Sexual Orientation", "Gender Identity"]
GROUPS = ["people", "minority", "majority"]
LABELS = {"people": "People control", "minority": "Minority", "majority": "Majority"}
COLORS = {"People control": "#686868", "Minority": "#2878B5", "Majority": "#E07A2D"}


def expand_people(data: pd.DataFrame) -> pd.DataFrame:
    people = data.loc[data.group_type.eq("people")].copy()
    explicit = data.loc[data.group_type.isin(["minority", "majority"])].copy()
    copies = []
    for dimension in DIMENSIONS:
        p = people.copy(); p["dimension"] = dimension; copies.append(p)
    out = pd.concat([explicit, *copies], ignore_index=True)
    out["group_label"] = out.group_type.map(LABELS)
    return out


def save_group_plot(data: pd.DataFrame, value: str, title: str, ylabel: str, path: Path) -> None:
    data = data.loc[data[value].notna()].copy()
    fig, axes = plt.subplots(len(DIMENSIONS), 1, figsize=(12, 18), sharex=True, sharey=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        part = data.loc[data.dimension.eq(dimension)]
        sns.boxplot(data=part, x="group_label", y=value, order=list(COLORS), color="white", showfliers=False, ax=ax)
        sns.stripplot(data=part, x="group_label", y=value, order=list(COLORS), hue="group_label", hue_order=list(COLORS),
                      palette=COLORS, jitter=.16, size=4, alpha=.72, legend=False, ax=ax)
        ax.set_title(dimension, loc="left", fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel(ylabel if dimension == "Gender" else "")
    fig.suptitle(title, y=.995, fontsize=16, fontweight="bold")
    fig.text(.5, .005, "Points are response-level observations. People controls are repeated descriptively across dimensions; they are not independent replicas.", ha="center", fontsize=9)
    fig.tight_layout(rect=(.02, .025, .99, .97)); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create descriptive PNG figures from epistemic-commitment v2 outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/epistemic_commitment"))
    args = parser.parse_args()
    base = args.results_dir.resolve(); output = base / "figures_v2"; output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=.8)
    contextual = pd.read_csv(base / "contextual_hedging" / "contextual_hedging_response_metrics.csv")
    save_group_plot(expand_people(contextual), "contextual_hedged_sentence_rate",
                    "Contextual hedging in Google AI Overviews", "Hedged sentence rate", output / "contextual_hedging_sentence_rate.png")
    if (base / "lexical_bioscope_overlap" / "hedging_response_metrics.csv").exists():
        lexical = pd.read_csv(base / "lexical_bioscope_overlap" / "hedging_response_metrics.csv")
        l = expand_people(lexical); c = expand_people(contextual)
        joined = l[["response_id", "dimension", "group_type", "hedge_per_100_tokens"]].merge(
            c[["response_id", "dimension", "group_type", "contextual_hedges_per_100_tokens"]],
            on=["response_id", "dimension", "group_type"], validate="one_to_one")
        tidy = joined.melt(id_vars=["response_id", "dimension", "group_type"], var_name="measure", value_name="value")
        tidy["measure"] = tidy.measure.map({"hedge_per_100_tokens": "Lexical BioScope overlap", "contextual_hedges_per_100_tokens": "Contextual detector"})
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=tidy, x="measure", y="value", hue="measure", palette={"Lexical BioScope overlap": "#999999", "Contextual detector": "#2878B5"}, showfliers=False, ax=ax)
        sns.stripplot(data=tidy, x="measure", y="value", color="#222222", alpha=.28, size=2.5, jitter=.18, ax=ax)
        ax.set_title("Lexical baseline and contextual hedge rate", loc="left", fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("Occurrences per 100 tokens")
        fig.tight_layout(); fig.savefig(output / "lexical_vs_contextual_hedging.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    ver_path = base / "veridicality_v2" / "veridicality_response_metrics_v2.csv"
    if ver_path.exists():
        ver = pd.read_csv(ver_path)
        save_group_plot(expand_people(ver), "mean_veridicality", "Strictly matched MegaVeridicality scores", "Mean score per covered response", output / "veridicality_mean_by_condition.png")
        coverage = expand_people(ver).groupby(["dimension", "group_type"], as_index=False).predicate_coverage.mean()
        matrix = coverage.pivot(index="dimension", columns="group_type", values="predicate_coverage").reindex(index=DIMENSIONS, columns=GROUPS)
        fig, ax = plt.subplots(figsize=(9, 6)); sns.heatmap(matrix * 100, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100, cbar_kws={"label": "Responses covered (%)"}, ax=ax)
        ax.set_title("Coverage of strict MegaVeridicality matches", loc="left", fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("")
        fig.tight_layout(); fig.savefig(output / "veridicality_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    (output / "README.txt").write_text("Exploratory PNG figures for epistemic-commitment v2. Descriptive only; no PDFs are produced. Generic people responses are reused across dimensions for display and must not be treated as independent in inference.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
