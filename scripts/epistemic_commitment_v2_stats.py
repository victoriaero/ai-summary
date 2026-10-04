from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import fisher_exact, norm, spearmanr
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "epistemic_commitment"
OUTPUT = ROOT / "results" / "epistemic_commitment_v2"
SEED = 20260930
BOOTSTRAPS = 4000
DRAWS = 4000
DIMENSIONS = ["Race", "Ethnicity", "Gender", "Disability", "Sexual Orientation", "Gender Identity"]
CONDITIONS = ["People", "Minority", "Majority"]
COMPARISONS = [("Minority", "People"), ("Majority", "People"), ("Minority", "Majority")]
META = ["response_id", "query_id", "domain", "outcome", "dimension", "demographic_dimension",
        "condition", "group", "group_type", "location", "replica_id"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(source: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "contextual": source / "contextual_hedging" / "contextual_hedging_response_metrics.csv",
        "hedge_occurrences": source / "contextual_hedging" / "contextual_hedge_occurrences.csv",
        "lexical": source / "lexical_bioscope_overlap" / "hedging_response_metrics.csv",
        "veridicality": source / "veridicality_v2" / "veridicality_response_metrics_v2.csv",
        "predicate_occurrences": source / "veridicality_v2" / "predicate_occurrences_v2.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required upstream outputs are missing:\n" + "\n".join(missing))
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    for name in ("contextual", "lexical", "veridicality"):
        if frames[name].response_id.duplicated().any():
            raise ValueError(f"Duplicate response_id in {name}")
    frames["paths"] = paths
    return frames


def balanced_responses(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = frames["contextual"].copy()
    v = frames["veridicality"].copy()
    l = frames["lexical"].copy()
    h["item_id"] = h.domain.astype(str) + " :: " + h.outcome.astype(str)
    v["item_id"] = v.domain.astype(str) + " :: " + v.outcome.astype(str)
    l["item_id"] = l.domain.astype(str) + " :: " + l.outcome.astype(str)
    expected = {"People"} | {f"{d}::{g}" for d in DIMENSIONS for g in ("Minority", "Majority")}
    h["condition_cell"] = np.where(h.group_type.eq("people"), "People", h.dimension.astype(str) + "::" + h.group_type.str.title())
    valid_replica_items = []
    inventory = []
    for replica, part in h.groupby("replica_id"):
        complete = part.groupby("item_id").condition_cell.agg(lambda x: set(x) == expected and len(x) == len(expected))
        n_complete = int(complete.sum())
        inventory.append({"replica_id": replica, "n_responses": len(part), "n_outcomes": part.item_id.nunique(),
                          "n_complete_outcomes": n_complete, "included_in_models": bool(n_complete == 21),
                          "reason": "all 21 outcomes have all 13 unique conditions" if n_complete == 21 else "incomplete paired collection"})
        if n_complete == 21:
            valid_replica_items.extend((replica, item) for item in complete.index[complete])
    if not valid_replica_items:
        raise ValueError("No fully paired replica has all 13 conditions for 21 outcomes")
    selected = h.set_index(["replica_id", "item_id"]).loc[valid_replica_items].reset_index()
    selected["condition_cell"] = pd.Categorical(selected.condition_cell, categories=["People"] +
        [f"{d}::{g}" for d in DIMENSIONS for g in ("Minority", "Majority")])
    vcols = ["response_id", "predicate_coverage", "n_veridical_predicates", "mean_veridicality", "median_veridicality"]
    lcols = ["response_id", "hedged_sentence_rate", "hedge_per_100_tokens", "n_hedge_occurrences"]
    selected = selected.merge(v[vcols], on="response_id", validate="one_to_one")
    selected = selected.merge(l[lcols], on="response_id", validate="one_to_one")
    selected["group_label"] = selected.group_type.map({"people": "People", "minority": "Minority", "majority": "Majority"})
    selected["n_unhedged_sentences"] = selected.n_sentences - selected.n_hedged_sentences
    if (selected.n_unhedged_sentences < 0).any():
        raise ValueError("Hedged sentence count exceeds n_sentences")
    return selected, pd.DataFrame(inventory)


def paired_table(data: pd.DataFrame) -> pd.DataFrame:
    measures = ["response_id", "contextual_hedged_sentence_rate", "contextual_hedges_per_100_tokens",
                "hedged_sentence_rate", "hedge_per_100_tokens", "predicate_coverage", "mean_veridicality",
                "median_veridicality", "n_veridical_predicates", "n_sentences"]
    people = data.loc[data.group_label.eq("People")].set_index(["replica_id", "item_id"])
    chunks = []
    for dimension in DIMENSIONS:
        subset = data.loc[data.dimension.eq(dimension) & data.group_label.ne("People")]
        minority = subset.loc[subset.group_label.eq("Minority")].set_index(["replica_id", "item_id"])
        majority = subset.loc[subset.group_label.eq("Majority")].set_index(["replica_id", "item_id"])
        keys = people.index.intersection(minority.index).intersection(majority.index)
        if len(keys) == 0:
            continue
        base = people.loc[keys, ["domain", "outcome"]].copy()
        for condition, frame in (("People", people), ("Minority", minority), ("Majority", majority)):
            for col in measures:
                base[f"{col}_{condition.lower()}"] = frame.loc[keys, col].to_numpy()
        base["dimension"] = dimension
        chunks.append(base.reset_index())
    result = pd.concat(chunks, ignore_index=True)
    result.to_csv  # Ensure this remains a plain, auditable matched table.
    return result


def bootstrap_mean(values, seed_offset: int = 0, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(SEED + seed_offset)
    draws = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return mean, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def paired_effects(paired: pd.DataFrame, metric: str, analysis: str) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSIONS:
        part = paired.loc[paired.dimension.eq(dimension)]
        for left, right in COMPARISONS:
            a = part[f"{metric}_{left.lower()}"]
            b = part[f"{metric}_{right.lower()}"]
            eligible = part.loc[a.notna() & b.notna()].copy()
            eligible["difference"] = eligible[f"{metric}_{left.lower()}"] - eligible[f"{metric}_{right.lower()}"]
            by_item = eligible.groupby("item_id").difference.mean()
            estimate, lo, hi = bootstrap_mean(by_item.to_numpy(), seed_offset=hash((analysis, dimension, left, right)) % 100000)
            rows.append({"analysis": analysis, "metric": metric, "dimension": dimension, "comparison": f"{left} - {right}",
                         "estimate": estimate, "ci_low": lo, "ci_high": hi, "n_paired_responses": len(eligible),
                         "n_outcomes": len(by_item), "method": "outcome-cluster percentile bootstrap"})
    return pd.DataFrame(rows)


def _positive_semidefinite(cov: np.ndarray) -> np.ndarray:
    cov = (cov + cov.T) / 2
    eigval, eigvec = np.linalg.eigh(cov)
    return (eigvec * np.maximum(eigval, 0)) @ eigvec.T


def fit_cell_model(data: pd.DataFrame, outcome: str, kind: str, model_name: str):
    formula = 'C(condition_cell, Treatment(reference="People")) + C(item_id)'
    if data.replica_id.nunique() > 1:
        formula += ' + C(replica_id)'
    design = patsy.dmatrix(formula, data, return_type="dataframe")
    groups = data.item_id.astype(str)
    if kind == "binomial":
        successes = data[outcome].to_numpy(dtype=float)
        trials = data.n_sentences.to_numpy(dtype=float) if outcome == "n_hedged_sentences" else np.ones(len(data))
        endog = np.column_stack([successes, trials - successes])
        model = sm.GLM(endog, design, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
        scale = "log_odds"
    else:
        model = sm.OLS(data[outcome].to_numpy(dtype=float), design).fit(cov_type="cluster", cov_kwds={"groups": groups})
        scale = "score"
    model_rows = []
    for term, estimate in model.params.items():
        se = float(model.bse[term])
        p = float(model.pvalues[term])
        model_rows.append({"model": model_name, "term": term, "estimate": float(estimate), "std_error": se,
                           "ci_low": float(estimate - 1.96 * se), "ci_high": float(estimate + 1.96 * se),
                           "p_value": p, "scale": scale, "odds_ratio": float(np.exp(estimate)) if scale == "log_odds" else np.nan,
                           "n_responses": len(data), "n_outcomes": data.item_id.nunique(),
                           "model_specification": "outcome fixed effects; condition by dimension cell effects; outcome-cluster sandwich SE"})
    return model, design.design_info, pd.DataFrame(model_rows), formula


def marginal_cell_results(model, design_info, data: pd.DataFrame, model_name: str, link: str):
    cells = ["People"] + [f"{d}::{g}" for d in DIMENSIONS for g in ("Minority", "Majority")]
    base = data[["item_id", "replica_id", "condition_cell"]].drop_duplicates(["item_id", "replica_id"]).copy()
    beta = np.asarray(model.params, dtype=float)
    cov = _positive_semidefinite(np.asarray(model.cov_params(), dtype=float))
    rng = np.random.default_rng(SEED + (1 if model_name == "hedging" else 2 if model_name == "coverage" else 3))
    draws = rng.multivariate_normal(beta, cov, size=DRAWS, check_valid="ignore")
    rows, means, simulated, matrices = [], {}, {}, {}
    for cell in cells:
        prediction_frame = base.copy()
        prediction_frame["condition_cell"] = cell
        X = np.asarray(patsy.build_design_matrices([design_info], prediction_frame)[0], dtype=float)
        matrices[cell] = X
        linear = X @ beta
        samples = X @ draws.T
        if link == "logit":
            point = float(expit(linear).mean())
            distribution = expit(samples).mean(axis=0)
        else:
            point = float(linear.mean())
            distribution = samples.mean(axis=0)
        means[cell] = point
        simulated[cell] = distribution
        dimension, condition = ("Control", "People") if cell == "People" else cell.split("::", 1)
        observed = data.loc[data.condition_cell.astype(str).eq(cell)]
        rows.append({"model": model_name, "dimension": dimension, "condition": condition, "condition_cell": cell,
                     "estimate": point, "ci_low": float(np.quantile(distribution, .025)), "ci_high": float(np.quantile(distribution, .975)),
                     "scale": "probability" if link == "logit" else "score", "n_responses": len(observed),
                     "n_covered": int(observed.predicate_coverage.sum()) if "predicate_coverage" in observed else np.nan})
    contrasts = []
    for dimension in DIMENSIONS:
        for left, right in COMPARISONS:
            left_cell = "People" if left == "People" else f"{dimension}::{left}"
            right_cell = "People" if right == "People" else f"{dimension}::{right}"
            vector = matrices[left_cell][0] - matrices[right_cell][0]
            log_or_score = float(vector @ beta)
            se = float(np.sqrt(max(0.0, vector @ cov @ vector)))
            z = log_or_score / se if se > 0 else np.nan
            dist = simulated[left_cell] - simulated[right_cell]
            contrasts.append({"model": model_name, "dimension": dimension, "comparison": f"{left} - {right}",
                "estimate": means[left_cell] - means[right_cell], "ci_low": float(np.quantile(dist, .025)),
                "ci_high": float(np.quantile(dist, .975)), "scale": "probability_difference" if link == "logit" else "score_difference",
                "linear_estimate": log_or_score, "linear_ci_low": log_or_score - 1.96 * se,
                "linear_ci_high": log_or_score + 1.96 * se, "linear_scale": "log_odds" if link == "logit" else "score",
                "odds_ratio": float(np.exp(log_or_score)) if link == "logit" else np.nan,
                "p_value": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                "n_outcomes": data.item_id.nunique()})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def fit_occurrence_model(data: pd.DataFrame, occurrences: pd.DataFrame):
    scored = occurrences.loc[occurrences.match_status.eq("matched_strict") & occurrences.veridicality_score.notna()].copy()
    scored = scored.merge(data[["response_id", "condition_cell", "item_id", "replica_id"]], on="response_id", validate="many_to_one", suffixes=("", "_model"))
    scored["condition_cell"] = scored["condition_cell"].astype(str)
    scored["item_id"] = scored["item_id"].astype(str)
    formula = 'veridicality_score ~ C(condition_cell, Treatment(reference="People")) + C(item_id)'
    if scored.replica_id.nunique() > 1:
        formula += ' + C(replica_id)'
    method = "response random-intercept mixed model"
    note = ""
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = smf.mixedlm(formula, scored, groups=scored.response_id).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        if not fitted.converged or not np.isfinite(fitted.fe_params).all():
            raise RuntimeError("mixed model did not converge")
        params = fitted.fe_params; se = fitted.bse_fe; pvals = fitted.pvalues.reindex(params.index)
        note = "; ".join(str(w.message) for w in captured)[:500]
        response_intercept_variance = float(fitted.cov_re.iloc[0, 0])
    except Exception as exc:
        fitted = smf.ols(formula, data=scored).fit(cov_type="cluster", cov_kwds={"groups": scored.item_id})
        method = "occurrence OLS with outcome-cluster robust SE (mixed-model fallback)"
        note = f"Mixed model failed: {type(exc).__name__}: {exc}"
        params = fitted.params; se = fitted.bse; pvals = fitted.pvalues
        response_intercept_variance = np.nan
    rows = [{"model": "predicate_occurrence", "term": term, "estimate": float(value),
             "std_error": float(se[term]), "ci_low": float(value - 1.96 * se[term]),
             "ci_high": float(value + 1.96 * se[term]), "p_value": float(pvals[term]),
             "scale": "score", "n_occurrences": len(scored), "n_responses": scored.response_id.nunique(),
             "method": method, "response_intercept_variance": response_intercept_variance, "warning": note}
            for term, value in params.items()]
    return scored, pd.DataFrame(rows), {"method": method, "n_occurrences": len(scored),
            "n_responses": scored.response_id.nunique(), "response_intercept_variance": response_intercept_variance, "warning": note}


def proximity_tables(paired: pd.DataFrame):
    outcome_rows, summaries = [], []
    for analysis, metric in (("contextual_hedging", "contextual_hedged_sentence_rate"),
                             ("veridicality_score", "mean_veridicality")):
        for dimension in DIMENSIONS:
            part = paired.loc[paired.dimension.eq(dimension)]
            values = []
            for _, row in part.iterrows():
                trio = [row[f"{metric}_{c.lower()}"] for c in CONDITIONS]
                if not all(np.isfinite(trio)):
                    continue
                people, minority, majority = trio
                dmin = abs(people - minority); dmaj = abs(people - majority)
                diff = dmaj - dmin
                values.append((row.item_id, diff))
                outcome_rows.append({"analysis": analysis, "metric": metric, "replica_id": row.replica_id,
                    "item_id": row.item_id, "domain": row.domain, "outcome": row.outcome, "dimension": dimension,
                    "distance_to_minority": dmin, "distance_to_majority": dmaj, "proximity_difference": diff})
            by_item = pd.DataFrame(values, columns=["item_id", "difference"]).groupby("item_id").difference.mean() if values else pd.Series(dtype=float)
            estimate, lo, hi = bootstrap_mean(by_item.to_numpy(), seed_offset=100 + DIMENSIONS.index(dimension))
            summaries.append({"analysis": analysis, "metric": metric, "dimension": dimension, "estimate": estimate,
                "ci_low": lo, "ci_high": hi, "n_triplets": len(values), "n_outcomes": len(by_item),
                "interpretation": "positive means People is closer to Minority; negative means closer to Majority"})
    return pd.DataFrame(outcome_rows), pd.DataFrame(summaries)


def paired_variant_effects(paired: pd.DataFrame, response_values: pd.Series, analysis: str, variant: str,
                           metric: str, eligible_responses: set[str] | None = None) -> list[dict]:
    rows = []
    for dimension in DIMENSIONS:
        part = paired.loc[paired.dimension.eq(dimension)]
        for left, right in COMPARISONS:
            left_id = part[f"response_id_{left.lower()}"]
            right_id = part[f"response_id_{right.lower()}"]
            left_values = left_id.map(response_values)
            right_values = right_id.map(response_values)
            valid = left_values.notna() & right_values.notna()
            if eligible_responses is not None:
                valid &= left_id.isin(eligible_responses) & right_id.isin(eligible_responses)
            differences = (left_values[valid] - right_values[valid]).to_numpy(dtype=float)
            item_ids = part.loc[valid, "item_id"].to_numpy()
            grouped = pd.Series(differences).groupby(item_ids).mean() if len(differences) else pd.Series(dtype=float)
            estimate, lo, hi = bootstrap_mean(grouped.to_numpy(), seed_offset=123 + DIMENSIONS.index(dimension))
            rows.append({"analysis": analysis, "variant": variant, "metric": metric, "dimension": dimension,
                         "comparison": f"{left} - {right}", "estimate": estimate, "ci_low": lo, "ci_high": hi,
                         "n_pairs": int(valid.sum()), "n_outcomes": len(grouped),
                         "method": "paired difference with outcome-cluster percentile bootstrap"})
    return rows


def hedging_sensitivity(data: pd.DataFrame, paired: pd.DataFrame, occurrences: pd.DataFrame):
    rows = []
    by_id = data.set_index("response_id")
    variants = {
        "contextual_primary_sentence_rate": (by_id.contextual_hedged_sentence_rate, None, "hedged_sentence_rate"),
        "lexical_bioscope_sentence_rate": (by_id.hedged_sentence_rate, None, "hedged_sentence_rate"),
        "contextual_per_100_tokens": (by_id.contextual_hedges_per_100_tokens, None, "hedges_per_100_tokens"),
        "lexical_per_100_tokens": (by_id.hedge_per_100_tokens, None, "hedges_per_100_tokens"),
    }
    p05 = int(np.ceil(np.quantile(data.n_sentences, .05)))
    p10 = int(np.ceil(np.quantile(data.n_sentences, .10)))
    variants[f"at_least_{p05}_sentences_p05"] = (by_id.contextual_hedged_sentence_rate,
        set(data.loc[data.n_sentences.ge(p05), "response_id"]), "hedged_sentence_rate")
    variants[f"at_least_{p10}_sentences_p10"] = (by_id.contextual_hedged_sentence_rate,
        set(data.loc[data.n_sentences.ge(p10), "response_id"]), "hedged_sentence_rate")
    for variant, (values, eligible, metric) in variants.items():
        rows += paired_variant_effects(paired, values, "contextual_hedging", variant, metric, eligible)
    occ = occurrences.loc[occurrences.response_id.isin(by_id.index)].copy()
    for threshold in (.95, .99):
        selected = occ.loc[occ.cue_probability.ge(threshold)]
        hedged_sentences = selected.groupby("response_id").sentence_id.nunique()
        rate = hedged_sentences.reindex(by_id.index, fill_value=0) / by_id.n_sentences
        rows += paired_variant_effects(paired, rate, "contextual_hedging", f"cue_probability_at_least_{threshold}", "hedged_sentence_rate")
    return pd.DataFrame(rows), {"extremely_short_cutoff": p05, "minimum_sentences_cutoff": p10,
        "cutoff_rule": "ceil of 5th and 10th percentile of n_sentences over unique complete-replica responses",
        "confidence_thresholds": [.95, .99], "threshold_rule": "fixed before outcome comparison; only stricter than the original detector cutoff"}


def leave_one_cue_out(data: pd.DataFrame, paired: pd.DataFrame, occurrences: pd.DataFrame):
    selected_ids = set(data.response_id)
    occ = occurrences.loc[occurrences.response_id.isin(selected_ids)].copy()
    occ["cue"] = occ.cue_surface.astype(str).str.casefold().str.split().str.join(" ")
    counts = occ.cue.value_counts()
    distinct = occ.drop_duplicates(["response_id", "sentence_id", "cue"])
    sentence_cue_count = distinct.groupby(["response_id", "sentence_id"]).cue.transform("nunique")
    sole_cue = distinct.loc[sentence_cue_count.eq(1)].groupby(["response_id", "cue"]).size()
    base = data.set_index("response_id")
    rows = []
    for cue, n_occ in counts.items():
        losses = sole_cue.xs(cue, level="cue") if cue in sole_cue.index.get_level_values("cue") else pd.Series(dtype=int)
        new_rate = (base.n_hedged_sentences - losses.reindex(base.index, fill_value=0)) / base.n_sentences
        variant = paired_variant_effects(paired, new_rate, "contextual_hedging", "detected_cue_removed", "hedged_sentence_rate")
        for row in variant:
            row.update({"omitted_cue": cue, "cue_occurrences": int(n_occ), "top_10_frequent": bool(cue in set(counts.head(10).index)),
                        "removal_definition": "remove detected cue occurrences from AIO predictions; retain a sentence if another detected cue remains"})
        rows.extend(variant)
    result = pd.DataFrame(rows)
    if not result.empty:
        baseline = paired_variant_effects(paired, base.contextual_hedged_sentence_rate, "contextual_hedging", "baseline", "hedged_sentence_rate")
        base_lookup = {(x["dimension"], x["comparison"]): x["estimate"] for x in baseline}
        result["baseline_estimate"] = [base_lookup[(r.dimension, r.comparison)] for r in result.itertuples()]
        result["change_from_baseline"] = result.estimate - result.baseline_estimate
    return result


def veridicality_sensitivity(data: pd.DataFrame, paired: pd.DataFrame, occurrences: pd.DataFrame):
    rows = []
    by_id = data.set_index("response_id")
    variants = {
        "strict_mean_primary": (by_id.mean_veridicality, None),
        "strict_median": (by_id.median_veridicality, None),
        "strict_mean_at_least_two_matches": (by_id.mean_veridicality,
            set(data.loc[data.n_veridical_predicates.ge(2), "response_id"])),
    }
    matched = occurrences.loc[occurrences.response_id.isin(by_id.index) & occurrences.match_status.eq("matched_strict") & occurrences.veridicality_score.notna()].copy()
    for polarity in ("positive", "negative"):
        variant_values = matched.loc[matched.polarity.eq(polarity)].groupby("response_id").veridicality_score.mean()
        variants[f"strict_{polarity}_polarity_only"] = (variant_values, None)
    counts = matched.predicate_lemma.value_counts()
    if len(counts):
        cutoff = float(counts.median() + 3 * (counts - counts.median()).abs().median())
        dominant = set(counts[counts > cutoff].index)
        variants["exclude_dominant_predicates"] = (matched.loc[~matched.predicate_lemma.isin(dominant)].groupby("response_id").veridicality_score.mean(), None)
    else:
        cutoff = np.nan; dominant = set()
    for variant, (values, eligible) in variants.items():
        rows += paired_variant_effects(paired, values, "veridicality_score", variant, "mean_veridicality", eligible)
    result = pd.DataFrame(rows)
    unavailable = pd.DataFrame([{"analysis": "veridicality_score", "variant": "exclude_low_parser_confidence",
        "metric": "mean_veridicality", "dimension": d, "comparison": f"{a} - {b}",
        "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_pairs": 0, "n_outcomes": 0,
        "method": "not estimable: spaCy parser does not emit calibrated per-occurrence confidence"}
        for d in DIMENSIONS for a, b in COMPARISONS])
    result = pd.concat([result, unavailable], ignore_index=True)
    return result, {"dominant_threshold": cutoff, "dominant_predicates": sorted(dominant),
        "dominant_rule": "occurrence count > median + 3 median absolute deviations among strict-matched predicate lemmas",
        "ambiguous_cases": "already excluded from strict matches; no additional strict-score subset needed",
        "parser_confidence": "all calibrated confidence values missing; low-confidence exclusion not estimable"}


def leave_one_predicate_out(data: pd.DataFrame, paired: pd.DataFrame, occurrences: pd.DataFrame):
    by_id = data.set_index("response_id")
    matched = occurrences.loc[occurrences.response_id.isin(by_id.index) & occurrences.match_status.eq("matched_strict") & occurrences.veridicality_score.notna()].copy()
    counts = matched.predicate_lemma.value_counts()
    total_sum = matched.groupby("response_id").veridicality_score.sum()
    total_n = matched.groupby("response_id").size()
    rows = []
    for predicate, n_occ in counts.items():
        removed = matched.loc[matched.predicate_lemma.eq(predicate)].groupby("response_id").veridicality_score.agg(["sum", "count"])
        remaining_n = total_n - removed["count"].reindex(total_n.index, fill_value=0)
        remaining_sum = total_sum - removed["sum"].reindex(total_sum.index, fill_value=0)
        new_means = (remaining_sum / remaining_n.where(remaining_n.gt(0))).reindex(by_id.index)
        variants = paired_variant_effects(paired, new_means, "veridicality_score", "strict_predicate_removed", "mean_veridicality")
        for row in variants:
            row.update({"omitted_predicate": predicate, "predicate_occurrences": int(n_occ),
                        "top_10_frequent": bool(predicate in set(counts.head(10).index))})
        rows.extend(variants)
    result = pd.DataFrame(rows)
    if not result.empty:
        baseline = paired_variant_effects(paired, by_id.mean_veridicality, "veridicality_score", "baseline", "mean_veridicality")
        lookup = {(x["dimension"], x["comparison"]): x["estimate"] for x in baseline}
        result["baseline_estimate"] = [lookup[(r.dimension, r.comparison)] for r in result.itertuples()]
        result["change_from_baseline"] = result.estimate - result.baseline_estimate
    return result


def lexical_contextual_comparison(data: pd.DataFrame):
    result = data[META + ["n_tokens", "n_sentences", "hedge_per_100_tokens", "contextual_hedges_per_100_tokens",
                             "hedged_sentence_rate", "contextual_hedged_sentence_rate"]].copy()
    result["lexical_to_contextual_rate_ratio"] = result.hedge_per_100_tokens / result.contextual_hedges_per_100_tokens.replace(0, np.nan)
    result["ratio_defined"] = result.contextual_hedges_per_100_tokens.gt(0)
    rho, p = spearmanr(result.hedge_per_100_tokens, result.contextual_hedges_per_100_tokens, nan_policy="omit")
    descriptive = {"spearman_rho": float(rho), "spearman_p_value": float(p), "n_responses": len(result),
                   "ratio_n_defined": int(result.ratio_defined.sum()), "ratio_median": float(result.lexical_to_contextual_rate_ratio.median()),
                   "ratio_q1": float(result.lexical_to_contextual_rate_ratio.quantile(.25)),
                   "ratio_q3": float(result.lexical_to_contextual_rate_ratio.quantile(.75))}
    return result, descriptive


def validation_tables(source: Path, output: Path, data: pd.DataFrame):
    validation_dir = output / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    hedge_blind_path = source / "contextual_hedging" / "human_validation_blind.csv"
    hedge_key_path = source / "contextual_hedging" / "human_validation_key.csv"
    ver_blind_path = source / "veridicality_v2" / "validation_blind.csv"
    ver_key_path = source / "veridicality_v2" / "validation_key.csv"
    hedge_blind = pd.read_csv(hedge_blind_path).fillna("")
    hedge_key = pd.read_csv(hedge_key_path).fillna("")
    ver_blind = pd.read_csv(ver_blind_path).fillna("")
    ver_key = pd.read_csv(ver_key_path).fillna("")
    # Copies remain blind. Existing labels are preserved on reruns.
    hedge_blind.to_csv(validation_dir / "hedging_annotation_sheet.csv", index=False)
    for field in ("negation_correct", "strict_decision_correct"):
        for suffix in ("annotator_1", "annotator_2", "adjudicated"):
            col = f"{field}_{suffix}"
            if col not in ver_blind:
                ver_blind[col] = ""
    ver_blind.to_csv(validation_dir / "veridicality_annotation_sheet.csv", index=False)
    hedge_key.to_csv(validation_dir / "hedging_key_restricted.csv", index=False)
    ver_key.to_csv(validation_dir / "veridicality_key_restricted.csv", index=False)
    lookup = data[["response_id", "group_type", "dimension"]].drop_duplicates()
    hedge = hedge_blind.merge(hedge_key, on="annotation_id", validate="one_to_one").merge(lookup, on="response_id", how="left")
    ver = ver_blind.merge(ver_key, on="annotation_id", validate="one_to_one").merge(lookup, on="response_id", how="left")
    rows, differential = [], []
    for condition in ("all", "people", "minority", "majority"):
        part = hedge if condition == "all" else hedge.loc[hedge.group_type.eq(condition)]
        for band in ("all", "frequent", "rare"):
            subset = part if band == "all" else part.loc[part.cue_frequency_band.eq(band)]
            labeled = subset.loc[subset.adjudicated_label.isin(["yes", "no"])]
            summary = {"analysis": "contextual_hedging", "stage": "cue_detection", "group_type": condition,
                "cue_frequency_band": band, "n_sampled": len(subset), "n_labeled": len(labeled), "status": "complete" if len(labeled) else "pending_annotation"}
            if len(labeled):
                truth = labeled.adjudicated_label.eq("yes").astype(int)
                prediction = labeled.model_prediction.eq("yes").astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(truth, prediction, average="binary", zero_division=0)
                summary.update({"precision": float(precision), "recall": float(recall), "f1": float(f1),
                                "error_rate": float((truth != prediction).mean())})
                double = subset.loc[subset.human_label_1.isin(["yes", "no", "uncertain"]) & subset.human_label_2.isin(["yes", "no", "uncertain"])]
                if len(double) >= 2:
                    summary["human_human_agreement"] = float((double.human_label_1 == double.human_label_2).mean())
                    summary["human_human_kappa"] = float(cohen_kappa_score(double.human_label_1, double.human_label_2))
            rows.append(summary)
    for condition in ("people", "minority", "majority"):
        subset = ver.loc[ver.group_type.eq(condition)]
        for field in ("predicate_correct", "introduces_clause", "frame_correct", "polarity_correct", "negation_correct",
                      "resource_mapping_appropriate", "strict_decision_correct"):
            col = f"{field}_adjudicated"
            labeled = subset.loc[subset[col].isin(["yes", "no"])]
            if field == "resource_mapping_appropriate":
                labeled = labeled.loc[labeled.validation_class.eq("matched")]
            rows.append({"analysis": "veridicality", "stage": field, "group_type": condition, "cue_frequency_band": "",
                         "n_sampled": len(subset), "n_labeled": len(labeled),
                         "accuracy_or_precision": float(labeled[col].eq("yes").mean()) if len(labeled) else np.nan,
                         "status": "complete" if len(labeled) else "pending_annotation"})
    measurement = pd.DataFrame(rows)
    measurement.to_csv(validation_dir / "measurement_validation.csv", index=False)
    for analysis, frame, label_col, predicted_col in (("contextual_hedging", hedge, "adjudicated_label", "model_prediction"),):
        base = frame.loc[frame[label_col].isin(["yes", "no"])].copy()
        if not base.empty:
            base["error"] = base[label_col].ne(base[predicted_col])
            people = base.loc[base.group_type.eq("people")]
            for other in ("minority", "majority"):
                target = base.loc[base.group_type.eq(other)]
                if len(people) and len(target):
                    table = [[int(target.error.sum()), int(len(target) - target.error.sum())],
                             [int(people.error.sum()), int(len(people) - people.error.sum())]]
                    differential.append({"analysis": analysis, "comparison": f"{other} - people",
                        "error_rate_difference": float(target.error.mean() - people.error.mean()),
                        "fisher_exact_p_value": float(fisher_exact(table)[1]), "n_other": len(target), "n_people": len(people),
                        "status": "observed; descriptive test"})
    if not differential:
        differential = [{"analysis": "contextual_hedging", "comparison": "minority - people", "status": "pending_annotation"},
                        {"analysis": "contextual_hedging", "comparison": "majority - people", "status": "pending_annotation"},
                        {"analysis": "veridicality", "comparison": "strict-match precision by condition", "status": "pending_annotation"}]
    pd.DataFrame(differential).to_csv(validation_dir / "differential_measurement_error.csv", index=False)
    return {"hedge_sample_n": len(hedge), "veridicality_sample_n": len(ver),
            "hedge_adjudicated_n": int(hedge.adjudicated_label.isin(["yes", "no"]).sum()),
            "veridicality_resource_mapping_adjudicated_n": int(ver.resource_mapping_appropriate_adjudicated.isin(["yes", "no"]).sum())}


def write_model_text(path: Path, model, formula: str, extra: str) -> None:
    path.write_text(f"Formula/design: {formula}\n{extra}\n\n{model.summary()}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired, outcome-adjusted epistemic-commitment analysis (hedging and veridicality separate).")
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = args.source_dir.resolve(); output = args.output_dir.resolve()
    for subdir in ("tables", "models", "robustness", "validation", "figures/main", "figures/supplementary"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    frames = load_inputs(source)
    data, inventory = balanced_responses(frames)
    paired = paired_table(data)
    inventory.to_csv(output / "tables" / "replica_inventory.csv", index=False)
    data.to_csv(output / "tables" / "unique_response_analysis_data.csv", index=False)
    paired.to_csv(output / "tables" / "paired_outcome_metrics.csv", index=False)

    hmodel, hdesign, hcoef, hformula = fit_cell_model(data, "n_hedged_sentences", "binomial", "hedging")
    hmean, hcontrast = marginal_cell_results(hmodel, hdesign, data, "hedging", "logit")
    hcoef.to_csv(output / "tables" / "hedging_model_results.csv", index=False)
    hcontrast.to_csv(output / "tables" / "hedging_pairwise_contrasts.csv", index=False)
    write_model_text(output / "models" / "hedging_binomial_glm.txt", hmodel, hformula,
        "Successes = n_hedged_sentences, failures = n_sentences - successes; 13 observed condition cells; unique People responses; cluster robust SE by outcome.")

    cmodel, cdesign, ccoef, cformula = fit_cell_model(data, "predicate_coverage", "binomial", "coverage")
    cmean, ccontrast = marginal_cell_results(cmodel, cdesign, data, "coverage", "logit")
    ccoef.to_csv(output / "tables" / "veridicality_coverage_model_results.csv", index=False)
    ccontrast.to_csv(output / "tables" / "veridicality_coverage_pairwise_contrasts.csv", index=False)
    write_model_text(output / "models" / "veridicality_coverage_logistic_glm.txt", cmodel, cformula,
        "Successes = predicate_coverage, failures = 1 - coverage; unique People responses; cluster robust SE by outcome.")

    covered = data.loc[data.predicate_coverage.eq(1) & data.mean_veridicality.notna()].copy()
    smodel, sdesign, scoef, sformula = fit_cell_model(covered, "mean_veridicality", "gaussian", "veridicality_score")
    smean, scontrast = marginal_cell_results(smodel, sdesign, covered, "veridicality_score", "identity")
    scoef.to_csv(output / "tables" / "veridicality_score_model_results.csv", index=False)
    scontrast.to_csv(output / "tables" / "veridicality_score_pairwise_contrasts.csv", index=False)
    write_model_text(output / "models" / "veridicality_score_response_ols.txt", smodel, sformula,
        "Only covered responses; one mean score per response; outcome fixed effects; cluster robust SE by outcome.")
    marginal = pd.concat([hmean, cmean, smean], ignore_index=True)
    marginal.to_csv(output / "tables" / "marginal_condition_estimates.csv", index=False)

    scored_occ, occ_model, occ_diagnostics = fit_occurrence_model(data, frames["predicate_occurrences"])
    scored_occ.to_csv(output / "tables" / "strict_predicate_occurrences.csv", index=False)
    occ_model.to_csv(output / "tables" / "veridicality_occurrence_model_results.csv", index=False)
    (output / "models" / "occurrence_model_diagnostics.json").write_text(json.dumps(occ_diagnostics, indent=2) + "\n")

    paired_bootstrap = pd.concat([paired_effects(paired, "contextual_hedged_sentence_rate", "contextual_hedging"),
                                  paired_effects(paired, "mean_veridicality", "veridicality_score"),
                                  paired_effects(paired, "predicate_coverage", "veridicality_coverage")], ignore_index=True)
    paired_bootstrap.to_csv(output / "tables" / "paired_bootstrap_contrasts.csv", index=False)
    proximity, proximity_summary = proximity_tables(paired)
    proximity.to_csv(output / "tables" / "people_proximity_results.csv", index=False)
    proximity_summary.to_csv(output / "tables" / "people_proximity_summary.csv", index=False)
    lexical_comparison, lexical_summary = lexical_contextual_comparison(data)
    lexical_comparison.to_csv(output / "tables" / "lexical_contextual_comparison.csv", index=False)
    (output / "tables" / "lexical_contextual_summary.json").write_text(json.dumps(lexical_summary, indent=2) + "\n")

    hsens, hdefinitions = hedging_sensitivity(data, paired, frames["hedge_occurrences"])
    hsens.to_csv(output / "robustness" / "hedging_sensitivity.csv", index=False)
    cue_loo = leave_one_cue_out(data, paired, frames["hedge_occurrences"])
    cue_loo.to_csv(output / "robustness" / "leave_one_cue_out.csv", index=False)
    vsens, vdefinitions = veridicality_sensitivity(data, paired, frames["predicate_occurrences"])
    vsens.to_csv(output / "robustness" / "veridicality_sensitivity.csv", index=False)
    pred_loo = leave_one_predicate_out(data, paired, frames["predicate_occurrences"])
    pred_loo.to_csv(output / "robustness" / "leave_one_predicate_out.csv", index=False)
    validation = validation_tables(source, output, data)

    manifest = {"seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "coefficient_draws": DRAWS,
        "included_replicas": inventory.loc[inventory.included_in_models, "replica_id"].tolist(),
        "excluded_incomplete_replicas": inventory.loc[~inventory.included_in_models, "replica_id"].tolist(),
        "n_unique_responses": len(data), "n_items": data.item_id.nunique(),
        "model_design": "13 observed condition cells (one People control + Minority and Majority in each of six dimensions), outcome fixed effects, outcome-cluster robust SE; no duplicated People in inference",
        "hedging_sensitivity_definitions": hdefinitions, "veridicality_sensitivity_definitions": vdefinitions,
        "validation": validation, "versions": {package: metadata.version(package) for package in
            ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "scikit-learn", "spacy", "patsy")},
        "input_sha256": {name: file_hash(path) for name, path in frames["paths"].items()}}
    (output / "models" / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Epistemic commitment v2 tables, models, robustness and validation written to {output}")


if __name__ == "__main__":
    main()
