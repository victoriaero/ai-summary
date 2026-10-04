"""Reproducible Dallas–NY contextual-hedging analysis using literal claims.

The primary outcome is the grouped-binomial rate in claim-bearing sentences.
No claim text generated or normalized by an LLM is measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import patsy
import spacy
import statsmodels.api as sm

from analyze_contextual_hedging import collect_response_predictions
from analyze_epistemic_commitment import discover_responses


ROOT = Path(__file__).resolve().parents[1]
REPLICAS = {"v1_dallas": "Dallas", "v2_ny": "New York"}
CONTRASTS = (("minority", "people"), ("majority", "people"), ("minority", "majority"))
UNITS = {"claim_sentence": ("n_hedged_claim_sentences", "n_claim_sentences"),
         "response": ("n_hedged_sentences", "n_sentences"),
         "claim_span": ("n_claim_spans_with_hedge", "n_valid_claim_spans")}
FORMULA = "C(cell, Treatment(reference='people')) * C(replica_id, Treatment(reference='v1_dallas')) + C(outcome)"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_literal_claims(paths: list[Path], responses: pd.DataFrame) -> pd.DataFrame:
    if len(paths) != 2 or any(not (p / "valid_claims.csv").exists() for p in paths):
        raise FileNotFoundError("Provide both fresh literal Dallas and NY claim outputs; old rewritten claims are forbidden")
    frames = [pd.read_csv(path / "valid_claims.csv") for path in paths]
    data = pd.concat(frames, ignore_index=True)
    required = {"absolute_source_spans", "source_span_texts", "prompt_version", "sentence_start_char",
                "sentence_end_char", "original_sentence", "claim_id", "response_id", "replica_id"}
    if not required.issubset(data.columns) or set(data.prompt_version) != {"aio-literal-claims-v2"}:
        raise ValueError("Inputs must be newly extracted literal-v2 claims, never the rewritten Dallas pilot")
    if data.empty or data.claim_id.duplicated().any():
        raise ValueError("Missing or duplicated literal claim IDs")
    if set(data.replica_id) != set(REPLICAS):
        raise ValueError("Literal claims must cover Dallas and New York")
    if not set(data.response_id).issubset(set(responses.response_id)):
        raise ValueError("Claims refer to responses outside the selected experiment")
    lookup = responses.set_index("response_id").aio_text
    for row in data.itertuples(index=False):
        original = lookup[row.response_id]
        sentence = str(row.original_sentence)
        a, b = int(row.sentence_start_char), int(row.sentence_end_char)
        if original[a:b] != sentence:
            raise ValueError(f"Sentence offset mismatch: {row.claim_id}")
        spans = json.loads(row.absolute_source_spans)
        if not spans or any(original[int(s["start_char"]):int(s["end_char"])] != s["text"] for s in spans):
            raise ValueError(f"Source span mismatch: {row.claim_id}")
        if json.loads(row.source_span_texts) != [s["text"] for s in spans]:
            raise ValueError(f"Literal fragment mismatch: {row.claim_id}")
    return data


def inventory(raw: pd.DataFrame, claims: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for replica, city in REPLICAS.items():
        part = raw.loc[raw.replica_id.eq(replica)]
        cells = part.assign(cell=np.where(part.group_type.eq("people"), "people",
                                         part.dimension.astype(str) + "|" + part.group_type.astype(str)))
        rows.append({"replica_id": replica, "location": city, "n_files": len(part),
                     "n_responses_with_aio": int(part.has_aio_text.sum()),
                     "n_missing_aio": int((~part.has_aio_text).sum()),
                     "n_outcomes": part.outcome.nunique(), "n_conditions": cells.cell.nunique(),
                     "n_dimensions_excluding_control": part.loc[part.group_type.ne("people"), "dimension"].nunique(),
                     "n_duplicate_response_ids": int(part.response_id.duplicated().sum()),
                     "n_people_controls": int(part.group_type.eq("people").sum()),
                     "n_claims": int(claims.replica_id.eq(replica).sum()) if claims is not None else np.nan,
                     "vpn_location_nonblank": int(part.vpn_location.fillna("").astype(str).str.strip().ne("").sum()),
                     "public_ip_nonblank": int(part.public_ip.fillna("").astype(str).str.strip().ne("").sum()),
                     "location_basis": "collection folder; NY VPN/IP confirmed by team, not independently verified in files"})
        if len(part) != 273 or int(part.has_aio_text.sum()) != 273 or cells.cell.nunique() != 13 or part.outcome.nunique() != 21:
            raise ValueError(f"Incomplete design for {replica}: {rows[-1]}")
        if part.response_id.duplicated().any() or int(part.group_type.eq("people").sum()) != 21:
            raise ValueError(f"Duplicate response or People control in {replica}")
        if not cells.groupby(["outcome", "cell"]).size().eq(1).all() or len(cells.groupby(["outcome", "cell"])) != 273:
            raise ValueError(f"Missing or duplicated outcome-condition cell in {replica}")
    return pd.DataFrame(rows)


def span_inside(cue_start: int, cue_end: int, intervals: list[dict]) -> bool:
    return any(int(s["start_char"]) <= cue_start < cue_end <= int(s["end_char"]) for s in intervals)


def measure_claims(claims: pd.DataFrame, occurrences: pd.DataFrame, responses: pd.DataFrame):
    by_response = {key: sub.to_dict("records") for key, sub in occurrences.groupby("response_id")}
    text_by_id = responses.set_index("response_id").aio_text.to_dict()
    claim_rows, sentence_flags, links = [], {}, []
    for row in claims.itertuples(index=False):
        original = text_by_id[row.response_id]
        intervals = json.loads(row.absolute_source_spans)
        inside = []
        sentence_key = (row.response_id, row.sentence_id)
        sentence_flags.setdefault(sentence_key, False)
        for cue in by_response.get(row.response_id, []):
            start, end = int(cue["cue_start"]), int(cue["cue_end"])
            if original[start:end] != cue["cue_surface"]:
                raise ValueError(f"Cue offset mismatch in {row.response_id}")
            if int(row.sentence_start_char) <= start < end <= int(row.sentence_end_char):
                sentence_flags[sentence_key] = True
                if span_inside(start, end, intervals):
                    inside.append(cue)
                    links.append({"claim_id": row.claim_id, "response_id": row.response_id,
                                  "sentence_id": row.sentence_id, "cue_surface": cue["cue_surface"],
                                  "cue_start": start, "cue_end": end,
                                  "cue_probability": cue.get("cue_probability"),
                                  "semantic_scope_verified": False})
        claim_rows.append({"claim_id": row.claim_id, "response_id": row.response_id,
                           "query_id": row.query_id, "replica_id": row.replica_id,
                           "location": REPLICAS[row.replica_id], "outcome": row.outcome,
                           "demographic_dimension": row.demographic_dimension,
                           "group_type": row.group_type, "sentence_id": row.sentence_id,
                           "original_sentence": row.original_sentence,
                           "sentence_start_char": int(row.sentence_start_char),
                           "sentence_end_char": int(row.sentence_end_char),
                           "absolute_source_spans": row.absolute_source_spans,
                           "source_span_texts": row.source_span_texts,
                           "claim_span_contains_contextual_hedge": bool(inside),
                           "n_cues_in_source_spans": len(inside), "semantic_scope_verified": False})
    claim_table = pd.DataFrame(claim_rows)
    sentence_rows = []
    for (response_id, sentence_id), hedged in sentence_flags.items():
        sentence_rows.append({"response_id": response_id, "sentence_id": sentence_id,
                              "has_contextual_hedge": hedged})
    sentence_table = pd.DataFrame(sentence_rows)
    counts_s = sentence_table.groupby("response_id").has_contextual_hedge.agg(["sum", "count"]).rename(
        columns={"sum": "n_hedged_claim_sentences", "count": "n_claim_sentences"})
    counts_c = claim_table.groupby("response_id").claim_span_contains_contextual_hedge.agg(["sum", "count"]).rename(
        columns={"sum": "n_claim_spans_with_hedge", "count": "n_valid_claim_spans"})
    return claim_table, pd.DataFrame(links), counts_s, counts_c


def add_cells(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["cell"] = np.where(data.group_type.eq("people"), "people",
                             data.dimension.astype(str) + "|" + data.group_type.astype(str))
    return data


def fit_binomial(data: pd.DataFrame, success: str, total: str):
    used = data.loc[data[total].gt(0)].copy()
    if used.empty:
        raise ValueError(f"No observations for {total}")
    design = patsy.dmatrix(FORMULA, used, return_type="dataframe")
    endog = np.column_stack([used[success], used[total] - used[success]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.GLM(endog, design, family=sm.families.Binomial()).fit(maxiter=100)
    return used, design.design_info, result, [str(w.message) for w in caught]


def probability(result, design_info, outcomes: list[str], cell: str, replica: str) -> float:
    target = pd.DataFrame({"cell": [cell] * len(outcomes), "replica_id": [replica] * len(outcomes),
                           "outcome": outcomes})
    matrix = patsy.build_design_matrices([design_info], target)[0]
    return float(np.asarray(result.predict(matrix)).mean())


def contrasts(result, design_info, dimensions: list[str], outcomes: list[str]) -> list[dict]:
    rows = []
    for dimension in dimensions:
        for a, b in CONTRASTS:
            cell_a = "people" if a == "people" else f"{dimension}|{a}"
            cell_b = "people" if b == "people" else f"{dimension}|{b}"
            for replica in (*REPLICAS, "pooled"):
                places = list(REPLICAS) if replica == "pooled" else [replica]
                p_a = float(np.mean([probability(result, design_info, outcomes, cell_a, loc) for loc in places]))
                p_b = float(np.mean([probability(result, design_info, outcomes, cell_b, loc) for loc in places]))
                rows.append({"demographic_dimension": dimension, "comparison": f"{a}-{b}",
                             "replica_id": replica, "location": REPLICAS.get(replica, "Pooled"),
                             "predicted_probability_a": p_a, "predicted_probability_b": p_b,
                             "estimate": p_a - p_b})
    return rows


def bootstrap_model(data: pd.DataFrame, unit: str, bootstraps: int, seed: int) -> tuple[pd.DataFrame, dict]:
    success, total = UNITS[unit]
    used, design_info, fitted, messages = fit_binomial(data, success, total)
    outcomes = sorted(data.outcome.unique())
    dimensions = sorted(data.loc[data.group_type.ne("people"), "dimension"].unique())
    base = pd.DataFrame(contrasts(fitted, design_info, dimensions, outcomes))
    keys = ["demographic_dimension", "comparison", "replica_id"]
    replicates = []
    rng = np.random.default_rng(seed)
    for _ in range(bootstraps):
        chosen = rng.choice(outcomes, size=len(outcomes), replace=True)
        # Duplicate entire outcome clusters, including both locations and the
        # shared People control; never resample individual claims/sentences.
        sampled = pd.concat([used.loc[used.outcome.eq(outcome)] for outcome in chosen], ignore_index=True)
        try:
            boot_used, boot_design, boot_fit, _ = fit_binomial(sampled, success, total)
            boot_rows = contrasts(boot_fit, boot_design, dimensions, list(chosen))
            replicates.append(pd.DataFrame(boot_rows)[keys + ["estimate", "predicted_probability_a",
                                                         "predicted_probability_b"]].rename(
                                                             columns={"estimate": "bootstrap_estimate"}))
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue
    if len(replicates) < max(1, int(np.ceil(.8 * bootstraps))):
        raise RuntimeError(f"Only {len(replicates)}/{bootstraps} outcome-bootstrap fits succeeded for {unit}; "
                           "do not report unstable confidence intervals")
    if replicates:
        boot = pd.concat(replicates, keys=range(len(replicates)), names=["bootstrap_id", "row"]).reset_index(level=0)
        for column, low_name, high_name in (("bootstrap_estimate", "ci_low", "ci_high"),
                                            ("predicted_probability_a", "probability_a_ci_low", "probability_a_ci_high"),
                                            ("predicted_probability_b", "probability_b_ci_low", "probability_b_ci_high")):
            ci = boot.groupby(keys)[column].quantile([.025, .975]).unstack().reset_index().rename(
                columns={.025: low_name, .975: high_name})
            base = base.merge(ci, on=keys, how="left")
    else:
        for name in ("ci_low", "ci_high", "probability_a_ci_low", "probability_a_ci_high",
                     "probability_b_ci_low", "probability_b_ci_high"):
            base[name] = np.nan
    base["unit"] = unit
    base["n_responses_used"] = len(used)
    return base, {"unit": unit, "model": "grouped-binomial GLM; outcome fixed effects; location fixed effect",
                  "random_intercept_attempt": "not available as a stable frequentist grouped-binomial fit in installed statsmodels; fixed-effect fallback used",
                  "converged": bool(fitted.converged), "warnings": messages,
                  "n_outcomes": len(outcomes), "bootstrap_requested": bootstraps,
                  "bootstrap_succeeded": len(replicates), "seed": seed,
                  "coefficients": {name: float(value) for name, value in fitted.params.items()}}


def scope_sample(claim_table: pd.DataFrame, links: pd.DataFrame, claims: pd.DataFrame, n: int = 250) -> pd.DataFrame:
    positive = claim_table.loc[claim_table.claim_span_contains_contextual_hedge].copy()
    if positive.empty:
        return pd.DataFrame(columns=["annotation_id", "claim_id", "response_id", "replica_id", "location",
                                     "group_type", "demographic_dimension", "outcome", "original_query",
                                     "surrounding_context", "original_sentence", "highlighted_sentence",
                                     "source_span_texts", "cue_surface", "cue_start", "cue_end",
                                     "human_label_1", "human_label_2", "adjudicated_label", "notes"])
    first_link = links.sort_values(["claim_id", "cue_start"]).drop_duplicates("claim_id")
    positive = positive.merge(first_link[["claim_id", "cue_surface", "cue_start", "cue_end"]], on="claim_id")
    context = claims[["claim_id", "original_query", "preceding_sentences", "following_sentences"]]
    positive = positive.merge(context, on="claim_id", validate="one_to_one")
    positive["highlighted_sentence"] = positive.apply(
        lambda r: r.original_sentence[:int(r.cue_start) - int(r.sentence_start_char)] + "[[" +
        r.original_sentence[int(r.cue_start) - int(r.sentence_start_char):
                            int(r.cue_end) - int(r.sentence_start_char)] + "]]" +
        r.original_sentence[int(r.cue_end) - int(r.sentence_start_char):], axis=1)
    positive["surrounding_context"] = positive.apply(lambda r: json.dumps({"preceding": json.loads(r.preceding_sentences),
        "following": json.loads(r.following_sentences)}, ensure_ascii=False), axis=1)
    # Stratify without looking at group effects; cap to one cue per claim in
    # this first audit, retaining all cue links in the occurrence table.
    strata = positive.groupby(["replica_id", "group_type", "demographic_dimension"], dropna=False)
    core = pd.concat([
        strata.sample(n=1, random_state=20261003),
        positive.groupby("cue_surface", dropna=False).sample(n=1, random_state=20261004),
    ]).drop_duplicates("claim_id")
    if len(core) > n:
        core = core.sample(n=n, random_state=20261003)
    rest = positive.loc[~positive.claim_id.isin(core.claim_id)]
    sample = pd.concat([core, rest.sample(n=min(max(0, n - len(core)), len(rest)), random_state=20261003)])
    sample = sample.head(n).copy()
    sample.insert(0, "annotation_id", [f"scope_{i:04d}" for i in range(1, len(sample) + 1)])
    for field in ("human_label_1", "human_label_2", "adjudicated_label", "notes"):
        sample[field] = ""
    return sample[["annotation_id", "claim_id", "response_id", "replica_id", "location", "group_type",
                   "demographic_dimension", "outcome", "original_query", "surrounding_context",
                   "original_sentence", "highlighted_sentence", "source_span_texts", "cue_surface", "cue_start", "cue_end",
                   "human_label_1", "human_label_2", "adjudicated_label", "notes"]]


def paired_sensitivity(data: pd.DataFrame, variant: str, unit: str, rate: str) -> list[dict]:
    rows = []
    for dimension in sorted(data.loc[data.group_type.ne("people"), "dimension"].unique()):
        for a, b in CONTRASTS:
            ca = "people" if a == "people" else f"{dimension}|{a}"
            cb = "people" if b == "people" else f"{dimension}|{b}"
            left = data.loc[data.cell.eq(ca), ["replica_id", "outcome", rate]].rename(columns={rate: "a"})
            right = data.loc[data.cell.eq(cb), ["replica_id", "outcome", rate]].rename(columns={rate: "b"})
            paired = left.merge(right, on=["replica_id", "outcome"], validate="one_to_one").dropna(subset=["a", "b"])
            for replica in (*REPLICAS, "pooled"):
                part = paired if replica == "pooled" else paired.loc[paired.replica_id.eq(replica)]
                rows.append({"variant": variant, "unit": unit, "comparison": f"{a}-{b}",
                             "demographic_dimension": dimension, "replica_id": replica,
                             "n_paired_outcome_locations": len(part),
                             "mean_paired_rate_difference": float((part.a - part.b).mean()) if len(part) else np.nan})
    return rows


def robustness(data: pd.DataFrame, claims: pd.DataFrame, occurrences: pd.DataFrame,
               responses: pd.DataFrame) -> pd.DataFrame:
    variants = []
    rate_names = {"response": "hedged_sentence_rate", "claim_sentence": "claim_sentence_hedge_rate",
                  "claim_span": "claim_source_span_hedge_rate"}
    for unit, rate in rate_names.items():
        variants += paired_sensitivity(data, "baseline", unit, rate)
    # Threshold is fixed from the outcome-blind pooled response distribution.
    min_sentences = max(1, int(data.n_sentences.quantile(.05)))
    filtered = data.loc[data.n_sentences.ge(min_sentences)]
    for unit, rate in rate_names.items():
        variants += paired_sensitivity(filtered, f"exclude_short_lt_{min_sentences}_sentences", unit, rate)
    variants += paired_sensitivity(data, "per_100_tokens", "response", "contextual_hedges_per_100_tokens")
    for outcome in sorted(data.outcome.unique()):
        reduced = data.loc[data.outcome.ne(outcome)]
        for unit, rate in rate_names.items():
            variants += paired_sensitivity(reduced, f"leave_outcome_out:{outcome}", unit, rate)
    top_cues = occurrences.cue_surface.fillna("").str.casefold().str.strip().value_counts().head(10).index
    for cue in top_cues:
        remaining = occurrences.loc[occurrences.cue_surface.fillna("").str.casefold().str.strip().ne(cue)]
        sentence_counts = remaining.groupby("response_id").sentence_id.nunique()
        cue_counts = remaining.groupby("response_id").size()
        varied = data.copy()
        varied["hedged_sentence_rate"] = varied.response_id.map(sentence_counts).fillna(0) / varied.n_sentences.replace(0, np.nan)
        varied["contextual_hedges_per_100_tokens"] = 100 * varied.response_id.map(cue_counts).fillna(0) / varied.n_tokens.replace(0, np.nan)
        _, _, counts_s, counts_c = measure_claims(claims, remaining, responses)
        varied["claim_sentence_hedge_rate"] = varied.response_id.map(counts_s.n_hedged_claim_sentences).fillna(0) / varied.n_claim_sentences.replace(0, np.nan)
        varied["claim_source_span_hedge_rate"] = varied.response_id.map(counts_c.n_claim_spans_with_hedge).fillna(0) / varied.n_valid_claim_spans.replace(0, np.nan)
        for unit, rate in rate_names.items():
            variants += paired_sensitivity(varied, f"leave_detected_cue_out:{cue}", unit, rate)
    result = pd.DataFrame(variants)
    result["interpretation"] = "descriptive paired-outcome sensitivity; not model-refitted inference"
    return result


def write_report(output: Path, inv: pd.DataFrame, response: pd.DataFrame, effects: pd.DataFrame,
                 scope_n: int, match_n: int) -> None:
    def percent(value: float) -> str:
        return f"{100 * value:+.1f}" if pd.notna(value) else "NA"

    lines = ["# Contextual hedging: Dallas and New York", "",
             "This is an exploratory observational analysis, not a formal replication or a measure of general epistemic commitment.",
             "The primary outcome is contextual hedging in sentences that produced at least one automatically accepted, literal-source claim.", "",
             "## Data inventory", "",
             "| Location | Responses with AIO | Outcomes | Conditions | People controls | Literal claims |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in inv.itertuples():
        lines.append(f"| {row.location} | {row.n_responses_with_aio} | {row.n_outcomes} | {row.n_conditions} | {row.n_people_controls} | {row.n_claims} |")
    lines += ["", "The research team confirmed that all NY responses were collected in New York using the same IP. "
              "The collection files do not independently verify that assertion: VPN location and IP metadata are incomplete. "
              "People is counted once per outcome/location in the models.", "",
              "## Operational definitions", "",
              "- All-response rate: hedged AIO sentences / all AIO sentences.",
              "- Primary claim-bearing-sentence rate: hedged sentences originating ≥1 accepted literal claim / distinct claim-bearing sentences.",
              "- Claim-source-span rate: accepted claims with a contextual cue inside an original source span / accepted claims. "
              "This is an unverified semantic-scope proxy, not a claim-level hedge judgment.",
              "- Claims are selected literal fragments with checked offsets. No normalized or rewritten claim is measured.", "",
              "## Model specification", "",
              "Grouped-binomial GLM on response counts, with 13 observed condition cells × location and outcome fixed effects. "
              "Location is fixed; the unique People control is shared across demographic dimensions. "
              "Contrasts are standardized equally over the 21 outcomes. Intervals are percentile CIs from outcome-cluster "
              "bootstrap resampling both locations and all conditions together. Installed statsmodels does not provide "
              "a stable frequentist grouped-binomial random-intercept fit here, so this prespecified fallback is used. "
              "See `analysis_manifest.json` for convergence, warnings, seeds and successful replicates.", "",
              "## Descriptive observations", "",
              "The following rates pool counts within each location/condition; their denominators differ by unit and "
              "they are not model-adjusted.", "",
              "| Location | Condition | All sentences | Claim-bearing sentences | Claim source spans |",
              "|---|---|---:|---:|---:|"]
    for (replica, group), part in response.groupby(["replica_id", "group_type"]):
        rates = []
        for unit in ("response", "claim_sentence", "claim_span"):
            success, total = UNITS[unit]
            rates.append(f"{100 * part[success].sum() / part[total].sum():.1f}%" if part[total].sum() else "NA")
        lines.append(f"| {REPLICAS[replica]} | {group} | {' | '.join(rates)} |")
    lines += ["", "## Model-based inference and cross-location consistency", "",
              "Estimates below are adjusted probability differences in percentage points; CIs come from the "
              "outcome-cluster bootstrap. Examine signs, magnitudes and interval overlap rather than isolated p-values.", "",
              "| Dimension | Contrast | Dallas [95% CI] | New York [95% CI] | Pooled [95% CI] |",
              "|---|---|---:|---:|---:|"]
    primary = effects.loc[effects.unit.eq("claim_sentence")]
    for (dimension, comparison), part in primary.groupby(["demographic_dimension", "comparison"]):
        values = part.set_index("replica_id")
        cells = []
        for replica in (*REPLICAS, "pooled"):
            item = values.loc[replica]
            cells.append(f"{percent(item.estimate)} [{percent(item.ci_low)}, {percent(item.ci_high)}]")
        lines.append(f"| {dimension} | {comparison} | {' | '.join(cells)} |")
    lines += ["", "## Claim-bearing and claim-span sensitivity", "",
              "The all-response and claim-source-span model contrasts are in `hedging_model_results.csv`; "
              "Figure B displays descriptive rates with different denominators. "
              "`hedging_robustness.csv` reports paired-outcome sensitivity to detected-cue removal, "
              "one-outcome removal, short responses, and per-100-token scaling. These are descriptive checks, "
              "not refitted model CIs; they are not used to select a favorable specification.", "",
              "## Measurement validation", "",
              f"A blind sample of {scope_n} positive cue-in-span cases is in `hedge_scope_validation_sample.csv`. "
              "Cue-to-proposition scope and claim-extraction quality remain unvalidated by humans. "
              "Run the separate evaluator only after both annotators and adjudication have filled the labels; "
              "until then, `hedge_scope_validation_results.csv` says `pending_annotation`.", "",
              "## Matched-claim feasibility", "",
              f"{match_n} embedding-generated, Phi-4-classified candidate pairs are provisional. "
              "`matched_claim_validated.csv` remains empty until human adjudication. No matched-claim inference "
              "is produced below 30 validated pairs per contrast/location spanning at least 10 outcomes.", "",
              "## Limitations", "",
              "The primary analysis restricts to claim-bearing sentences but does not guarantee identical factual content "
              "between conditions. The claim detector and acceptance checks use the same model; literal offsets "
              "ensure textual fidelity, not factual truth or semantic-scope validity. Shared People controls, "
              "claims from the same response, 21 outcome clusters and observational Google outputs limit inference. "
              "Higher hedging probability is not evidence of a guardrail, intention or generalized epistemic caution. "
              "MegaVeridicality is a separate supplementary analysis.", "",
              "## Reproduce", "", "```bash",
              "python scripts/run_literal_claim_extraction.py --input-dir annotations/v1_dallas/google_aio_collection",
              "python scripts/run_literal_claim_extraction.py --input-dir annotations/v2_ny/google_aio_collection",
              "python scripts/analyze_hedging_ny_dallas.py", "```", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collections-root", type=Path, default=ROOT / "annotations")
    parser.add_argument("--dallas-claims", type=Path, default=ROOT / "results/claim_extraction_literal_v2/microsoft-phi-4/v1_dallas")
    parser.add_argument("--ny-claims", type=Path, default=ROOT / "results/claim_extraction_literal_v2/microsoft-phi-4/v2_ny")
    parser.add_argument("--hedge-model", type=Path, default=ROOT / "results/epistemic_commitment/contextual_hedging/contextual_hedging_model.joblib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/hedging_ny_dallas")
    parser.add_argument("--bootstraps", type=int, default=300)
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error("--bootstraps must be positive")
    all_responses, all_inventory = discover_responses(args.collections_root)
    selected = all_responses.loc[all_responses.replica_id.isin(REPLICAS)].copy()
    raw = all_inventory.loc[all_inventory.replica_id.isin(REPLICAS)].copy()
    claims = read_literal_claims([args.dallas_claims, args.ny_claims], selected)
    inv = inventory(raw, claims)
    bundle = joblib.load(args.hedge_model)
    nlp = spacy.load(bundle["spacy_model"])
    occ, response_metrics, _ = collect_response_predictions(selected, nlp, bundle["pipeline"], bundle["threshold"])
    if len(response_metrics) != 546:
        raise ValueError("Expected 546 unique Dallas–NY response metrics")
    claim_table, links, counts_s, counts_c = measure_claims(claims, occ, selected)
    meta = selected.drop(columns=["aio_text", "query", "notes"], errors="ignore")
    response = meta.merge(response_metrics[["response_id", "n_sentences", "n_tokens", "n_hedged_sentences",
                                            "n_contextual_hedges", "contextual_hedges_per_100_tokens"]],
                          on="response_id", validate="one_to_one")
    response = response.merge(counts_s, on="response_id", how="left", validate="one_to_one")
    response = response.merge(counts_c, on="response_id", how="left", validate="one_to_one")
    for col in ("n_hedged_claim_sentences", "n_claim_sentences", "n_claim_spans_with_hedge", "n_valid_claim_spans"):
        response[col] = response[col].fillna(0).astype(int)
    response = add_cells(response)
    response["hedged_sentence_rate"] = response.n_hedged_sentences / response.n_sentences.replace(0, np.nan)
    response["claim_sentence_hedge_rate"] = response.n_hedged_claim_sentences / response.n_claim_sentences.replace(0, np.nan)
    response["claim_source_span_hedge_rate"] = response.n_claim_spans_with_hedge / response.n_valid_claim_spans.replace(0, np.nan)
    model_results, diagnostics = [], []
    for index, unit in enumerate(UNITS):
        table, info = bootstrap_model(response, unit, args.bootstraps, seed=20261003 + index)
        model_results.append(table)
        diagnostics.append(info)
    effects = pd.concat(model_results, ignore_index=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inv.to_csv(output / "hedging_inventory_by_location.csv", index=False)
    response[[*meta.columns, "cell", "n_sentences", "n_tokens", "n_hedged_sentences",
              "n_contextual_hedges", "hedged_sentence_rate", "contextual_hedges_per_100_tokens"]].to_csv(
                  output / "response_level_hedging.csv", index=False)
    response[[*meta.columns, "cell", "n_claim_sentences", "n_hedged_claim_sentences",
              "claim_sentence_hedge_rate"]].to_csv(output / "claim_sentence_hedging.csv", index=False)
    claim_table.to_csv(output / "claim_span_hedging.csv", index=False)
    response[["response_id", "replica_id", "outcome", "cell", "n_valid_claim_spans",
              "n_claim_spans_with_hedge", "claim_source_span_hedge_rate"]].to_csv(
                  output / "claim_span_response_rates.csv", index=False)
    occ.to_csv(output / "contextual_hedge_occurrences.csv", index=False)
    links.to_csv(output / "claim_cue_links.csv", index=False)
    effects.to_csv(output / "hedging_model_results.csv", index=False)
    effects.loc[effects.replica_id.ne("pooled")].to_csv(output / "hedging_location_contrasts.csv", index=False)
    effects.loc[effects.replica_id.eq("pooled")].to_csv(output / "hedging_pooled_contrasts.csv", index=False)
    robustness(response, claims, occ, selected).to_csv(output / "hedging_robustness.csv", index=False)
    sample_path = output / "hedge_scope_validation_sample.csv"
    proposed_sample = scope_sample(claim_table, links, claims)
    key_path = output / "hedge_scope_validation_key.csv"
    key_columns = ["annotation_id", "claim_id", "response_id", "replica_id", "location",
                   "group_type", "demographic_dimension", "outcome", "cue_surface"]
    blind_columns = ["annotation_id", "original_query", "surrounding_context", "original_sentence",
                     "highlighted_sentence", "source_span_texts", "cue_surface", "cue_start", "cue_end",
                     "human_label_1", "human_label_2", "adjudicated_label", "notes"]
    proposed_key = proposed_sample[key_columns]
    proposed_blind = proposed_sample[blind_columns]
    if sample_path.exists():
        prior = pd.read_csv(sample_path).fillna("")
        prior_key = pd.read_csv(key_path).fillna("") if key_path.exists() else pd.DataFrame()
        if prior[["annotation_id", "original_sentence", "cue_start", "cue_end"]].to_dict("records") != proposed_blind[["annotation_id", "original_sentence", "cue_start", "cue_end"]].to_dict("records") or \
                prior_key[["annotation_id", "claim_id"]].to_dict("records") != proposed_key[["annotation_id", "claim_id"]].to_dict("records"):
            raise RuntimeError("Scope sample changed; preserve prior human annotations and use a new output directory")
    else:
        proposed_blind.to_csv(sample_path, index=False)
        proposed_key.to_csv(key_path, index=False)
    validation_path = output / "hedge_scope_validation_results.csv"
    if not validation_path.exists():
        pd.DataFrame([{"status": "pending_annotation", "n_sampled": len(proposed_sample),
                       "human_human_agreement": np.nan, "precision_yes": np.nan}]).to_csv(
                           validation_path, index=False)
    candidates = pd.concat([pd.read_csv(path / "matched_claim_candidates.csv") for path in (args.dallas_claims, args.ny_claims)], ignore_index=True)
    candidates.to_csv(output / "matched_claim_candidates.csv", index=False)
    validated_path = output / "matched_claim_validated.csv"
    if not validated_path.exists():
        pd.DataFrame(columns=["claim_a_id", "claim_b_id", "replica_id", "demographic_dimension",
                              "outcome", "adjudicated_label"]).to_csv(validated_path, index=False)
    old_path = ROOT / "results/claim_extraction_pilot/microsoft-phi-4/valid_claims.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        new = claims.loc[claims.replica_id.eq("v1_dallas")]
        diagnostic = pd.DataFrame([
            {"pipeline": "old_rewritten_pilot", "n_claims": len(old),
             "n_responses_with_claim": old.response_id.nunique(), "comparable_measure": False},
            {"pipeline": "new_literal_v2", "n_claims": len(new),
             "n_responses_with_claim": new.response_id.nunique(), "comparable_measure": False},
        ])
        diagnostic.to_csv(output / "dallas_old_vs_literal_inventory.csv", index=False)
    manifest = {"analysis": "contextual hedging; literal AIO claims", "primary_unit": "claim_sentence",
                "response_rows": len(response), "claim_rows": len(claim_table),
                "detector_version": bundle.get("model_version"), "detector_threshold": bundle["threshold"],
                "model_diagnostics": diagnostics,
                "inputs_sha256": {str(p): digest(p) for p in [args.hedge_model,
                    args.dallas_claims / "valid_claims.csv", args.ny_claims / "valid_claims.csv"]},
                "ny_location_note": "Team confirmed all NY collections used NY and the same IP; file-level VPN/IP metadata incomplete"}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from plot_hedging_ny_dallas import generate_figures
    generate_figures(output)
    write_report(output, inv, response, effects, len(proposed_sample), len(candidates))
    print(f"Dallas–NY contextual hedging outputs written to {output}")


if __name__ == "__main__":
    main()
