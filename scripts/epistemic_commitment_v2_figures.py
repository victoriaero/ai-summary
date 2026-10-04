"""Standalone PNG figures for the paired epistemic-commitment pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "epistemic_commitment_v2"
DIMENSIONS = ["Race", "Ethnicity", "Gender", "Disability", "Sexual Orientation", "Gender Identity"]
CONDITIONS = ["People", "Minority", "Majority"]
PALETTE = {"People": "#5D6670", "Minority": "#2878B5", "Majority": "#DA7A35"}
NOTE = "Each point is one AIO response/outcome. The same People control is shown in each dimension for description only."


def style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "axes.titleweight": "bold"})


def finish(fig, path: Path, note: str = ""):
    if note:
        fig.text(.5, .009, note, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(.02, .035 if note else .02, .99, .97))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def faceted_responses(data: pd.DataFrame) -> pd.DataFrame:
    explicit = data.loc[data.group_type.isin(["minority", "majority"])].copy()
    people = data.loc[data.group_type.eq("people")].copy()
    repeated = []
    for dimension in DIMENSIONS:
        part = people.copy(); part["dimension"] = dimension; repeated.append(part)
    out = pd.concat([explicit, *repeated], ignore_index=True)
    out["condition_label"] = out.group_type.map({"people": "People", "minority": "Minority", "majority": "Majority"})
    return out


def common_limits(series: pd.Series, include_zero: bool = False):
    values = series.dropna().to_numpy(dtype=float)
    low = min(float(values.min()), 0.) if include_zero else float(values.min())
    high = max(float(values.max()), 0.) if include_zero else float(values.max())
    margin = max(.03 * (high - low), .01)
    return low - margin, high + margin


def box_points_by_dimension(data, metric, ylabel, title, path, covered_only=False, zero_line=False):
    frame = faceted_responses(data)
    if covered_only:
        frame = frame.loc[frame.predicate_coverage.eq(1)]
    limits = common_limits(frame[metric], include_zero=zero_line)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for ax, dimension in zip(axes.flat, DIMENSIONS):
        part = frame.loc[frame.dimension.eq(dimension)]
        sns.boxplot(data=part, x="condition_label", y=metric, order=CONDITIONS,
                    color="#F8F8F8", showfliers=False, width=.55, linewidth=1.3, ax=ax)
        sns.stripplot(data=part, x="condition_label", y=metric, order=CONDITIONS, hue="condition_label",
                      hue_order=CONDITIONS, palette=PALETTE, jitter=.15, size=3.5, alpha=.65, legend=False, ax=ax)
        counts = part.groupby("condition_label")[metric].count().reindex(CONDITIONS, fill_value=0)
        ax.set_title(f"{dimension}  ·  n = {counts['People']}/{counts['Minority']}/{counts['Majority']}", fontsize=11, loc="left")
        ax.set_xlabel(""); ax.set_ylabel(ylabel if ax in axes[:, 0] else "")
        ax.set_ylim(*limits)
        if zero_line:
            ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    fig.suptitle(title + "  |  n order: People / Minority / Majority", fontsize=15, y=.995)
    finish(fig, path, NOTE + " Boxes show median and IQR; points are individual outcomes.")


def means_by_dimension(marginal, model, ylabel, title, path, y_limits=None):
    frame = marginal.loc[marginal.model.eq(model)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, dimension in zip(axes.flat, DIMENSIONS):
        part = pd.concat([frame.loc[frame.condition.eq("People")], frame.loc[frame.dimension.eq(dimension) & frame.condition.ne("People")]])
        for x, condition in enumerate(CONDITIONS):
            row = part.loc[part.condition.eq(condition)].iloc[0]
            ax.errorbar(x, row.estimate, yerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                        fmt="o", color=PALETTE[condition], capsize=3, markersize=7, linewidth=1.5)
            ax.text(x, row.ci_high, f"n={int(row.n_responses) if model != 'veridicality_score' else int(row.n_responses)}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xlim(-.5, 2.5); ax.set_xticks(range(3), CONDITIONS); ax.set_title(dimension, fontsize=11, loc="left")
        ax.set_ylabel(ylabel if ax in axes[:, 0] else "")
        if y_limits:
            ax.set_ylim(*y_limits)
        if model == "veridicality_score":
            ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
    fig.suptitle(title, y=.995, fontsize=15)
    finish(fig, path, "Outcome adjusted estimates and 95% intervals. The People estimate is the same control across dimension panels.")


def coverage_heatmap(data, path):
    expanded = faceted_responses(data)
    groups = expanded.groupby(["dimension", "condition_label"]).predicate_coverage.agg(["sum", "count"]).reset_index()
    groups["percent"] = groups["sum"] / groups["count"] * 100
    matrix = groups.pivot(index="dimension", columns="condition_label", values="percent").reindex(index=DIMENSIONS, columns=CONDITIONS)
    annotations = groups.assign(label=lambda x: x.apply(lambda row: f"{int(row['sum'])}/{int(row['count'])}\n({row['percent']:.0f}%)", axis=1)).pivot(
        index="dimension", columns="condition_label", values="label").reindex(index=DIMENSIONS, columns=CONDITIONS)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=100, linewidths=1, linecolor="white",
                cbar_kws={"label": "Responses with ≥1 strict match (%)"}, ax=ax)
    ax.set_title("Coverage of strict MegaVeridicality-compatible predicates", loc="left", pad=13)
    ax.set_xlabel(""); ax.set_ylabel("")
    finish(fig, path, "Coverage = response contains ≥1 confidently matched predicate. Each cell shows n/N and percent; the People control is reused descriptively.")


def paired_outcomes(paired, metric, ylabel, title, path):
    values = pd.concat([paired[f"{metric}_{c.lower()}"] for c in CONDITIONS], ignore_index=True)
    limits = common_limits(values, include_zero=(metric == "mean_veridicality"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for ax, dimension in zip(axes.flat, DIMENSIONS):
        part = paired.loc[paired.dimension.eq(dimension)]
        for _, row in part.iterrows():
            ys = np.array([row[f"{metric}_{c.lower()}"] for c in CONDITIONS], dtype=float)
            valid = np.isfinite(ys)
            if valid.sum() >= 2:
                ax.plot(np.arange(3)[valid], ys[valid], color="#9AA4AD", alpha=.32, linewidth=.8, zorder=1)
            for x in np.flatnonzero(valid):
                ax.scatter(x, ys[x], s=13, color=PALETTE[CONDITIONS[x]], alpha=.7, zorder=2)
        counts = [int(part[f"{metric}_{c.lower()}"].notna().sum()) for c in CONDITIONS]
        ax.set_title(f"{dimension}  ·  n={counts[0]}/{counts[1]}/{counts[2]}", loc="left", fontsize=11)
        ax.set_xticks(range(3), CONDITIONS); ax.set_xlim(-.3, 2.3); ax.set_ylim(*limits)
        ax.set_ylabel(ylabel if ax in axes[:, 0] else "")
        if metric == "mean_veridicality": ax.axhline(0, color="#777777", linestyle="--", linewidth=.8)
    fig.suptitle(title + "  |  n order: People / Minority / Majority", y=.995, fontsize=15)
    note = "Each line links the same outcome. For veridicality, missing strict coverage leaves partial lines. " + NOTE
    finish(fig, path, note)


def forest(contrasts, title, xlabel, path, comparison_only=True):
    frame = contrasts.loc[contrasts.comparison.isin(["Minority - People", "Majority - People"])].copy() if comparison_only else contrasts.copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    yloc = {d: len(DIMENSIONS)-1-i for i, d in enumerate(DIMENSIONS)}
    for comparison, offset, color in (("Minority - People", .15, PALETTE["Minority"]),
                                      ("Majority - People", -.15, PALETTE["Majority"])):
        subset = frame.loc[frame.comparison.eq(comparison)]
        for _, row in subset.iterrows():
            y = yloc[row.dimension] + offset
            ax.errorbar(row.estimate, y, xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                        fmt="o", capsize=3, color=color, label=comparison if row.dimension == DIMENSIONS[0] else None)
    ax.axvline(0, linestyle="--", color="#444444", linewidth=1)
    ax.set_yticks(list(yloc.values()), list(yloc.keys())); ax.set_xlabel(xlabel); ax.set_title(title, loc="left", pad=12)
    ax.legend(loc="best", frameon=False)
    finish(fig, path, "Outcome-adjusted model contrasts with 95% intervals. People appears only once in model fitting.")


def proximity_plots(rows, summaries, analysis, metric_label, scatter_path, forest_path):
    points = rows.loc[rows.analysis.eq(analysis)]
    overview = summaries.loc[summaries.analysis.eq(analysis)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(data=points, x="dimension", y="proximity_difference", order=DIMENSIONS, color="#386FA4", jitter=.22, alpha=.65, size=4, ax=ax)
    sns.boxplot(data=points, x="dimension", y="proximity_difference", order=DIMENSIONS,
                color="white", width=.45, showfliers=False, boxprops={"facecolor": "none"}, ax=ax)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=18)
    ax.set_xlabel(""); ax.set_ylabel("|People − Majority| − |People − Minority|")
    ax.set_title(f"People proximity by outcome · {metric_label}", loc="left")
    finish(fig, scatter_path, "Positive: People closer to Minority. Negative: People closer to Majority. Veridicality requires all three scores observed.")
    forest(overview.assign(comparison="Minority - People"), f"Mean People proximity · {metric_label}",
           "Mean distance difference (95% outcome bootstrap interval)", forest_path)


def two_stage(marginal, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, model, title, ylabel in ((axes[0], "coverage", "A. Probability of ≥1 strict match", "Predicted probability"),
                                     (axes[1], "veridicality_score", "B. Mean score conditional on coverage", "Predicted score")):
        frame = marginal.loc[marginal.model.eq(model)]
        for j, condition in enumerate(CONDITIONS):
            values = []
            for dimension in DIMENSIONS:
                part = frame.loc[frame.condition.eq("People")] if condition == "People" else frame.loc[frame.dimension.eq(dimension) & frame.condition.eq(condition)]
                values.append(part.iloc[0])
            x = np.arange(len(DIMENSIONS)) + (j-1)*.22
            y = np.array([row.estimate for row in values])
            err = np.array([[row.estimate-row.ci_low for row in values], [row.ci_high-row.estimate for row in values]])
            ax.errorbar(x, y, yerr=err, fmt="o", capsize=2, color=PALETTE[condition], label=condition, markersize=5)
        ax.set_xticks(range(len(DIMENSIONS)), DIMENSIONS, rotation=23, ha="right")
        ax.set_title(title, loc="left", fontsize=12); ax.set_ylabel(ylabel)
        if model == "coverage": ax.set_ylim(0, 1)
        else: ax.axhline(0, linestyle="--", color="#555555", linewidth=.9)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Two-stage MegaVeridicality analysis", y=.995, fontsize=15)
    finish(fig, path, "Outcome-adjusted estimates and 95% intervals. Score panel includes only covered responses; no scores are imputed.")


def lexical_supplement(comparison, summary, output):
    tidy = comparison.melt(id_vars="response_id", value_vars=["hedge_per_100_tokens", "contextual_hedges_per_100_tokens"],
                           var_name="method", value_name="rate")
    tidy.method = tidy.method.map({"hedge_per_100_tokens": "Lexical BioScope overlap", "contextual_hedges_per_100_tokens": "Contextual hedging"})
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=tidy, x="method", y="rate", order=["Lexical BioScope overlap", "Contextual hedging"],
                color="#F8F8F8", showfliers=False, ax=ax)
    sns.stripplot(data=tidy, x="method", y="rate", hue="method", palette={"Lexical BioScope overlap": "#777777", "Contextual hedging": PALETTE["Minority"]},
                  jitter=.16, alpha=.26, size=2.5, legend=False, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("Occurrences per 100 AIO tokens")
    ax.set_title("Lexical overlap and contextual detections", loc="left")
    finish(fig, output / "lexical_contextual_comparable_rates.png",
           f"Median lexical/contextual ratio where contextual > 0: {summary['ratio_median']:.2f} "
           f"(IQR {summary['ratio_q1']:.2f}–{summary['ratio_q3']:.2f}); both rates use the same token denominator.")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(comparison.hedge_per_100_tokens, comparison.contextual_hedges_per_100_tokens,
               s=20, alpha=.45, color=PALETTE["Minority"])
    ax.set_xlabel("Lexical BioScope matches per 100 tokens"); ax.set_ylabel("Contextual hedges per 100 tokens")
    ax.set_title("Lexical overlap versus contextual hedging", loc="left")
    finish(fig, output / "lexical_contextual_scatter_spearman.png",
           f"Each point is one unique AIO response. Spearman ρ = {summary['spearman_rho']:.2f} (n={summary['n_responses']}).")


def predicate_occurrence_plot(scored, path):
    frame = faceted_responses(scored)
    limits = common_limits(frame.veridicality_score, include_zero=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for ax, dimension in zip(axes.flat, DIMENSIONS):
        part = frame.loc[frame.dimension.eq(dimension)]
        sns.boxplot(data=part, x="condition_label", y="veridicality_score", order=CONDITIONS,
                    color="#F8F8F8", showfliers=False, width=.55, ax=ax)
        sns.stripplot(data=part, x="condition_label", y="veridicality_score", order=CONDITIONS, hue="condition_label",
                      palette=PALETTE, jitter=.18, alpha=.4, size=2.5, legend=False, ax=ax)
        ax.axhline(0, linestyle="--", color="#555555", linewidth=.8)
        ax.set_ylim(*limits); ax.set_title(dimension, loc="left", fontsize=11); ax.set_xlabel("")
        ax.set_ylabel("Resource score" if ax in axes[:, 0] else "")
    fig.suptitle("Strict matched predicate scores at occurrence level", y=.995, fontsize=15)
    finish(fig, path, "Each point is one matched predicate occurrence; responses can contribute more than one point. People occurrences repeat only for display.")


def frequent_predicates(scored, unique_responses, path, top_n=8):
    top = scored.predicate_lemma.value_counts().head(top_n).index.tolist()
    frame = faceted_responses(scored)
    counts = frame.loc[frame.predicate_lemma.isin(top)].groupby(["dimension", "condition_label", "predicate_lemma"]).size().rename("n_occurrences").reset_index()
    denominator = faceted_responses(unique_responses).groupby(["dimension", "condition_label"]).response_id.nunique().rename("n_responses").reset_index()
    counts = counts.merge(denominator, on=["dimension", "condition_label"], validate="many_to_one")
    counts["per_100_responses"] = 100 * counts.n_occurrences / counts.n_responses
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), sharex=True)
    for ax, dimension in zip(axes.flat, DIMENSIONS):
        part = counts.loc[counts.dimension.eq(dimension)]
        sns.barplot(data=part, x="per_100_responses", y="predicate_lemma", hue="condition_label", order=top,
                    hue_order=CONDITIONS, palette=PALETTE, ax=ax)
        ax.set_title(dimension, loc="left", fontsize=11); ax.set_xlabel("Occurrences per 100 AIO responses")
        ax.set_ylabel("")
        legend = ax.get_legend()
        if legend and dimension != DIMENSIONS[0]: legend.remove()
    fig.suptitle("Most frequent strict-matched predicates", y=.995, fontsize=15)
    finish(fig, path, "Denominator is unique AIO responses in each condition. The same People control is shown in each dimension for reference.")


def robustness_plot(frame, item_col, count_col, title, path, change_col="change_from_baseline"):
    frame = frame.loc[frame.top_10_frequent.eq(True) & frame.comparison.isin(["Minority - People", "Majority - People"])].copy()
    if frame.empty: return
    names = frame[[item_col, count_col]].drop_duplicates().sort_values(count_col, ascending=False).head(8)[item_col].tolist()
    frame = frame.loc[frame[item_col].isin(names)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 12), sharex=True)
    for ax, dimension in zip(axes.flat, DIMENSIONS):
        part = frame.loc[frame.dimension.eq(dimension)]
        for comparison, offset, color in (("Minority - People", .12, PALETTE["Minority"]),
                                          ("Majority - People", -.12, PALETTE["Majority"])):
            subset = part.loc[part.comparison.eq(comparison)].set_index(item_col)
            for i, name in enumerate(names):
                if name in subset.index and np.isfinite(subset.loc[name, change_col]):
                    ax.scatter(subset.loc[name, change_col], i + offset, color=color, s=28,
                               label=comparison if i == 0 else None)
        ax.set_yticks(range(len(names)), names); ax.invert_yaxis(); ax.axvline(0, linestyle="--", color="#555555", linewidth=.8)
        ax.set_title(dimension, loc="left", fontsize=11); ax.set_xlabel("Change in paired contrast after removal")
    fig.suptitle(title, y=.995, fontsize=15)
    finish(fig, path, "Blue = Minority − People; orange = Majority − People. For predicates, score changes use the same covered pairs before and after removal.")


def generate(output: Path = OUTPUT):
    output = output.resolve(); main_dir = output / "figures" / "main"; supp_dir = output / "figures" / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True); supp_dir.mkdir(parents=True, exist_ok=True)
    style()
    tables = output / "tables"; robustness = output / "robustness"
    data = pd.read_csv(tables / "unique_response_analysis_data.csv")
    paired = pd.read_csv(tables / "paired_outcome_metrics.csv")
    marginal = pd.read_csv(tables / "marginal_condition_estimates.csv")
    hcontrast = pd.read_csv(tables / "hedging_pairwise_contrasts.csv")
    scontrast = pd.read_csv(tables / "veridicality_score_pairwise_contrasts.csv")
    proximity = pd.read_csv(tables / "people_proximity_results.csv")
    proximity_summary = pd.read_csv(tables / "people_proximity_summary.csv")
    scored = pd.read_csv(tables / "strict_predicate_occurrences.csv")
    comparison = pd.read_csv(tables / "lexical_contextual_comparison.csv")
    summary = json.loads((tables / "lexical_contextual_summary.json").read_text())

    box_points_by_dimension(data, "contextual_hedged_sentence_rate", "Hedged sentence rate",
        "Contextual hedging by condition and dimension", main_dir / "contextual_hedging_box_points.png")
    means_by_dimension(marginal, "hedging", "Predicted hedged sentence probability",
        "Contextual hedging · mean and 95% interval", main_dir / "contextual_hedging_mean_ci.png", (0, 1))
    coverage_heatmap(data, main_dir / "veridicality_coverage_heatmap.png")
    box_points_by_dimension(data, "mean_veridicality", "Mean resource score per covered response",
        "Strict MegaVeridicality scores by condition", main_dir / "veridicality_score_box_points.png", covered_only=True, zero_line=True)
    means_by_dimension(marginal, "veridicality_score", "Predicted score among covered responses",
        "Strict MegaVeridicality score · mean and 95% interval", main_dir / "veridicality_score_mean_ci.png")
    paired_outcomes(paired, "contextual_hedged_sentence_rate", "Hedged sentence rate",
        "Paired outcomes · contextual hedging", main_dir / "paired_outcomes_contextual_hedging.png")
    paired_outcomes(paired, "mean_veridicality", "Mean resource score",
        "Paired outcomes · strict MegaVeridicality", main_dir / "paired_outcomes_veridicality.png")
    forest(hcontrast, "Outcome-adjusted paired effects · contextual hedging",
        "Difference in predicted hedged sentence probability", main_dir / "forest_contextual_hedging.png")
    forest(scontrast, "Outcome-adjusted paired effects · veridicality score",
        "Difference in mean resource score among covered responses", main_dir / "forest_veridicality_score.png")
    proximity_plots(proximity, proximity_summary, "contextual_hedging", "contextual hedging",
        main_dir / "people_proximity_contextual_points.png", main_dir / "people_proximity_contextual_forest.png")
    proximity_plots(proximity, proximity_summary, "veridicality_score", "veridicality score",
        main_dir / "people_proximity_veridicality_points.png", main_dir / "people_proximity_veridicality_forest.png")
    two_stage(marginal, main_dir / "veridicality_two_stage.png")

    lexical_supplement(comparison, summary, supp_dir)
    predicate_occurrence_plot(scored, supp_dir / "predicate_occurrence_distribution.png")
    frequent_predicates(scored, data, supp_dir / "frequent_predicates_per_100_responses.png")
    cue_loo = pd.read_csv(robustness / "leave_one_cue_out.csv")
    pred_loo = pd.read_csv(robustness / "leave_one_predicate_out.csv")
    robustness_plot(cue_loo, "omitted_cue", "cue_occurrences", "Detected-cue removal sensitivity",
                    supp_dir / "robustness_leave_one_cue_out.png")
    change_col = "effect_change_same_pairs" if "effect_change_same_pairs" in pred_loo else "change_from_baseline"
    robustness_plot(pred_loo, "omitted_predicate", "predicate_occurrences", "Strict-predicate removal sensitivity",
                    supp_dir / "robustness_leave_one_predicate_out.png", change_col=change_col)
    print(f"Figures written to {output / 'figures'}")


def main():
    parser = argparse.ArgumentParser(description="Generate main and supplementary epistemic-commitment PNG figures.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(); generate(args.output_dir)


if __name__ == "__main__":
    main()
