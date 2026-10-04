# Google AI Overviews audit: project context and research history

Prepared 29 September 2026 for the repository-based Codex audit, before drafting the remaining methodology.

## 1. Purpose and authority of this document

The author is preparing an IASEAI 2027 paper about identity-conditioned information mediation in Google AI Overviews (AIOs). A separate Codex session has the actual project workspace, scripts, experiments, and results. This document transfers the motivation, design, analysis history, and unresolved issues to that session. Its next task is to establish what was actually implemented and executed, not to write the Methods section yet.

This is a reconstruction from the conversation visible in this session, retrieved earlier project context, the entire four-page `iaseai_2.pdf`, and selected metadata, result tables, and examples in three earlier exported JSON-like text files. It is not a verbatim archive of the entire chat: some earlier messages were truncated and cannot be claimed to have been reread. The source repository was not available to the assistant preparing this handoff.

Use these evidence distinctions throughout the audit:

| Label | Meaning |
|---|---|
| Draft-reported | Stated in the current paper; execution still requires verification. |
| Export-observed | Present in an inspected historical result export; not necessarily the current or valid final analysis. |
| Code-visible | Implemented in the code excerpt supplied in the conversation; successful execution not established. |
| Discussed | A prior intention, proposal, interpretation, or troubleshooting diagnosis. |
| Unresolved | Must be established from code, data, logs, or the author. |

Resolve discrepancies by tracing raw records, exact code/configuration, and the outputs of the corresponding run. A newer script does not prove an older result was generated with it. A filename containing `v2` does not establish a new collection. Keep documentary claims and execution evidence separate.

## 2. Motivation and scientific problem

Generative search changes information access by selecting visible sources and producing a synthesized account. The project asks whether changing only the social group mentioned in an otherwise matched information request changes the sources surfaced and the information in the final answer.

The motivating issue is broader than offensive language or factual accuracy. An answer can be readable and apparently reasonable while presenting different sources, emphasizing different explanations, or having a different relationship to its cited evidence. The paper frames this as information allocation or information governance.

The normative premise is deliberately limited: different groups can legitimately require different answers because evidence, institutions, risks, and circumstances differ. Difference alone does not demonstrate unfairness, discrimination, intentional bias, or harm. The audit seeks systematic patterns and their relationship to visible evidence. The generic condition is a reference point, not a ground-truth fair answer.

The current introduction also raises a motivation about obtaining relevant information without unnecessarily disclosing personal information. This study varies the population named in the query; it does not establish the searcher's actual identity, measure disclosure behavior, or audit downstream privacy outcomes.

We observe public AIO outputs and their displayed/cited links. We do not observe Google's complete candidate retrieval pool, internal prompts, reasoning, or actual passages consumed during generation. Fetched cited pages are externally recovered evidence, not a recording of Google's internal context. Associations between measured layers are not causal mediation estimates.

## 3. Evolution of the project

1. Earlier planning considered roughly 500–520 queries, SerpApi, a standardized Chicago configuration, and temporal snapshots. These are historical proposals, not the current protocol. Temporal snapshots were later dropped in discussion.
2. An early conceptual plan proposed source-type classification, atomic claim extraction, claim equivalence, claim survival, explanation categories, polarity, and regression models. These ideas explain the framing, but their implementation is unverified. Do not copy them into Methods as completed work.
3. The current draft reports a human-reviewed set of 21 outcomes, 13 conditions, and manual collection from three U.S. VPN exit locations.
4. Work proceeded through collection validation and citation counts, source-set overlap, answer semantic displacement relative to the generic condition, and comparison of recovered cited evidence with the AIO.
5. Evidence-to-synthesis V1 produced outputs but exposed coverage, language, and segmentation problems. Its terminology also overstated what cosine similarity measures.
6. V2 was supplied to address these problems. The user started running it, encountered an environment error, then reported progress stopping visibly at `Processed 250/273 queries`. Successful V2 completion has not been established here.
7. The current task is an implementation audit followed by a report exchanged with the writing assistant. The methodology will be written after that reconciliation.

## 4. Current query design

### Domains and outcomes

The draft reports seven domains and three retained outcomes per domain. The following roster is observed in the Dallas source-overlap export; verify it against the authoritative query manifest and annotations.

| Domain | Outcomes observed in the export |
|---|---|
| Healthcare | Insurance coverage; Preventive care; Patient safety |
| Employment | Hiring; Career advancement; Workplace harassment |
| Education | College access/admission; Educational opportunity/access; Harassment/bullying |
| Housing | Access to housing; Housing stability; Homelessness |
| Credit & Financial Services | Credit/loan approval; Access to credit; Access to banking services |
| Criminal Justice | Incarceration; Police use of force; Arrest / detention by police |
| Government Benefits | Benefit retention / renewal; Application/approval; Access to government benefits |

According to the draft, a researcher with a Responsible AI background proposed candidate outcomes. Two independent annotators, one with a Human Rights Law background and one with a Computer Science/Responsible AI background, assessed relevance on an ordinal 0–5 scale and flagged binary cross-group comparability concerns. The draft names quadratically weighted Gwet's AC2 for relevance and Gwet's AC1 for comparability.

Missing from the draft: initial candidate count, exact annotation instructions and scale anchors, actual agreement coefficients, uncertainty estimates if computed, selection rule, treatment of disagreements/ties/missing ratings, and any adjudication. Do not invent a threshold or claim that a fixed rule was prespecified. The earlier files named `Anotator1/2 - Google AI Overview Study - Outcomes.csv` and `... - Guidelines.csv` may help locate these materials in the repository.

### Conditions

| Dimension | Focal/minority-marked label | Comparison/majority-marked label |
|---|---|---|
| Race | Black people | White people |
| Ethnicity | Latino people | non-Latino people |
| Gender | women | men |
| Disability | people with disabilities | people without disabilities |
| Sexual orientation | homosexual people | heterosexual people |
| Gender identity | transgender people | cisgender people |

The shared generic condition is `people`. This gives twelve explicit conditions plus one generic condition, not six independently collected generic baselines. Preserve the original labels in reporting and audit case/alias differences in files. The code uses `minority`, `majority`, and `control`; in prose, focal/comparison can avoid treating these labels as universal numerical population categories.

The template in the draft is:

> What factors influence [OUTCOME] for [GROUP] in [DOMAIN]?

Within an outcome, only the group slot changes. “What factors influence” was chosen to avoid explicitly presupposing barriers or disadvantage and to allow different kinds of explanations to emerge. This is one template, not established paraphrase robustness. The design does not cover intersectional combinations.

Design arithmetic: 21 × 13 = 273 unique query strings; 21 × 12 = 252 explicit-versus-generic comparisons per complete location; 21 paired focal-versus-comparison outcomes per dimension. These comparison rows are not 252 independent experimental units.

## 5. Collection as described in the draft

Collection used the public Google Search interface manually. For each query, the draft reports recording AIO presence, the full displayed AIO text without rewriting, and all links cited or displayed as supporting sources. It reports repeating the complete set from Dallas, Texas; New York City, New York; and Los Angeles, California using separate VPN exit points.

The intended role of these locations is replication across access points, not estimating regional differences. Three complete collections would yield 819 search observations, but this is not evidence of 819 nonempty answers or 819 observations eligible for every analysis. The draft says “up to 819.” Establish actual counts and missingness separately.

The draft lists retained query, condition, domain, outcome, location, full answer text, and cited URLs. The conversation discussed validation fields such as `aio_chars`, `n_links`, duplicate links, AIO presence, status, errors, and warnings. A blank presence field or zero links cannot alone establish no AIO. One historical Dallas semantic export reports zero empty answers; the overlap export reports ten documents with zero sources. These are compatible and must not be conflated.

Verify collection dates/times, ordering, retries, browser/session/account state, cookies/personalization, language and locale settings, whether content was expanded, whether translation occurred, and exact link-capture rules. None should be filled in from earlier recommendations. U.S. VPN exits do not by themselves establish English-language output or isolate geography from time/session variation.

The supplied examples directly contain English source passages and Portuguese AIO text; another example contains an English AIO. The language distribution and origin of Portuguese text remain unresolved. Do not assume all queries, answers, and sources are in one language, and do not assert that translation was performed without evidence.

## 6. Known workspace and version clues

These are historical locations to search, not paths inspected in this session:

| Item | Historical path or identifier |
|---|---|
| Project root | `/scratch/victoria.estanislau/ai-summary` |
| Dallas collection | `annotations/v1_dallas/google_aio_collection` |
| Source overlap results | `results/source_overlap_analysis_dallas` |
| Answer embedding results | `results/semantic_embedding_analysis_dallas` |
| Evidence V2 results | `results/evidence_synthesis_analysis_dallas_v2` |
| Evidence script | `scripts/evidence_synthesis.py` |
| Python environment in traceback | `/scratch/arthurbuzelin/env` |

Both inspected source-overlap and semantic-displacement exports identify collection `v1_dallas`. The inspected V1 evidence export also uses `v1_dallas`. `evidence_synthesis_analysis_dallas_v2` is an analysis-version clue, not evidence of recollection. Other location datasets and analysis outputs must be located independently. Collection in three locations does not prove that all analysis stages ran on all three.

## 7. Analysis A: availability, size, and source counts

Earlier work requested summaries by group, dimension, domain, and outcome, together with statistical comparisons and both CSV outputs and one comprehensive JSON. Recover the actual scripts and results for AIO presence, answer length, number of links, unique URLs, and unique hosts/domains. Distinguish raw citation occurrences from deduplicated source counts.

Retrieved conversation history contains an earlier assistant summary of Dallas results: focal/minority conditions about 9.05 links versus 6.36 for comparison conditions, generic `people` about 8.86; Friedman statistic about 70.73. These are historical discussion clues, not verified final numbers. Reproduce the exact count definition, denominators, comparison unit, and tests before using them. “Majority marking” was an interpretation under discussion, not an established mechanism.

## 8. Analysis B: source-set displacement from the generic answer

An inspected export describes source overlap for 273 Dallas documents, 252 explicit-versus-generic pairs, and ten zero-source documents. Within each outcome, compare the explicit condition's source set with the generic answer's source set.

The core measure is Jaccard distance:

\[
D^{source}_{og}=1-\frac{|U_{og}\cap U_{o0}|}{|U_{og}\cup U_{o0}|}.
\]

Here `o` is the matched outcome, `g` an explicit group, and `0` the generic condition. The same construction was applied to URL and host/domain sets. Directional coverage measures, shared-source counts, and explicit-only/generic-only lists were also exported.

Export metadata states that fragments and obvious tracking parameters are removed while substantive query parameters are preserved. Both-empty source sets produce undefined/null Jaccard values; exactly one empty set gives similarity zero/distance one. Audit the exact normalization, redirect handling, and deduplication rules.

**Verified historical caveat:** metadata says registrable domains are used when `tldextract` is available, otherwise hostnames. This run explicitly reports `tldextract_available: false`. Thus its `domain_*` fields used the hostname fallback. Do not call these results registrable-domain or publisher-level comparisons without verifying a corrected run.

The historical URL-distance contrast is focal-minus-comparison distance from their shared generic baseline. Exported differences and Holm-adjusted sign-flip p-values were:

| Dimension | Mean difference | Adjusted p |
|---|---:|---:|
| Disability | 0.07822 | 0.015625 |
| Ethnicity | 0.02449 | 0.09375 |
| Gender | 0.14310 | 0.0007324 |
| Gender identity | 0.02231 | 0.09375 |
| Race | 0.02156 | 0.09375 |
| Sexual orientation | 0.05209 | 0.0097656 |

These are provenance landmarks, not instructions to preserve significance. Many URL distances are near one, so inspect ceiling effects, ties, zero differences, and whether set size contributes to the pattern. Disjoint URLs do not imply contradictory information, low quality, or unfairness.

## 9. Analysis C: semantic displacement of the answer

The inspected Dallas export compares each explicit-group AIO with the generic AIO for the same outcome. Its primary field is `dense_distance_subject_normalized` (some result labels omit `distance` in the same position).

Historical configuration: `sentence-transformers/all-mpnet-base-v2`, CUDA, maximum sequence length 384, token-aware chunks of at most 320 tokens, and weighted pooling of chunk embeddings. The exact pooling weights, chunk boundaries, normalization order, and any overlap require code inspection. The export says the exact experimental group label is replaced with `people` before embedding. This aims to reduce trivial differences caused by the queried identity string, not erase substantive group-related information.

The main comparison is a difference of distances to the shared generic answer, not necessarily a direct cosine distance between focal and comparison answers:

\[
D^{answer}_{og}=1-\cos(e(\widetilde A_{og}),e(\widetilde A_{o0})),\qquad
\Delta_{od}=D^{answer}_{o,focal(d)}-D^{answer}_{o,comparison(d)}.
\]

The tilde denotes the implemented label normalization. Raw embeddings, normalized TF-IDF distance, sentence-level generic coverage, group novelty, symmetric sentence distance, and length diagnostics were secondary outputs. Recover their exact formulas rather than inferring them from names.

Historical primary results, all with 21 matched outcomes:

| Dimension | Focal distance | Comparison distance | Difference | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Disability | 0.27861 | 0.17572 | 0.10290 | 0.0006250 |
| Ethnicity | 0.30323 | 0.25053 | 0.05269 | 0.0076800 |
| Gender | 0.22945 | 0.18687 | 0.04258 | 0.24485 |
| Gender identity | 0.43001 | 0.27415 | 0.15586 | 0.0002100 |
| Race | 0.27858 | 0.25335 | 0.02523 | 0.24485 |
| Sexual orientation | 0.39468 | 0.30943 | 0.08525 | 0.0116400 |

The export's outcome-aggregated difference is about 0.07742, bootstrap interval [0.05090, 0.10409]. These results require renewed scrutiny if the answer corpus includes Portuguese or mixed-language comparisons. The multilingual V2 evidence change does not automatically repair previously saved answer-displacement metrics. Exact English-label replacement may also fail on translated labels. Request replacement-hit counts and examples by group/language.

TF-IDF and Wilcoxon analyses of the same observations are robustness checks, not independent replications or new data.

## 10. Analysis D: recovered evidence versus AIO

### V1: executed historical analysis with known limitations

The inspected export uses `all-mpnet-base-v2`, three query-relevant passages per source, source passages of at most 180 tokens, a descriptive match threshold of 0.65, and an 80% source-fetch coverage eligibility rule. It reports mean coverage 0.55648 and only 30 of 273 queries eligible; its principal paired-test sections are empty.

Old field names include `evidence_omission_source_balanced`, retention, source survival, contributing sources, and contribution entropy. These names do not establish actual claim omission or survival. One exported match pairs an English source passage with almost a whole Portuguese AIO stored as a single sentence, corroborating the language/segmentation concern.

V1 also exports ordinary Spearman correlations across 252 explicit-versus-generic rows: URL displacement versus answer displacement about 0.40642; host/domain displacement versus answer displacement about 0.25724. These are historical descriptive associations. Their row-independent p-values do not account for the repeated outcomes and shared generic answers. Do not report a rounded `p=0` as an exact probability.

### V2: supplied replacement, execution incomplete/unverified

The discussed replacement uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, token-based chunks for both sources and AIOs, exact experimental-label normalization, source-balanced aggregation, and coverage sensitivity. Verify actual constants, tokenizer limits, runtime truncation, and normalization behavior from the current code.

The intended primary metric is `evidence_semantic_gap_source_balanced`. For each available cited source, select top query-relevant passages; match each selected passage to its most similar AIO chunk; average those similarities within source; then average the source scores equally. Gap is one minus that alignment:

\[
G_q=1-\frac{1}{|S_q|}\sum_{s\in S_q}\frac{1}{|P_{qs}|}
\sum_{p\in P_{qs}}\max_{a\in C_q}\cos(e(p),e(a)).
\]

`S_q` denotes usable recovered sources, `P_qs` the selected passages for one source, and `C_q` the AIO chunks. This formula restates the supplied design; confirm weighting and selection in implementation. Equal weighting prevents sources with more selected passages from automatically dominating, but does not remove duplication between sources or missing-source bias.

The reverse direction is a secondary answer-to-evidence alignment measure, named `answer_support_alignment` / `answer_support_semantic_gap`. Audit whether it searches all source chunks or only selected passages and exactly how it averages AIO chunks. “Support” here is embedding similarity, not verified factual support or entailment.

The discussed reference eligibility is at least 50% recovered-source coverage and at least three usable cited sources, with coverage thresholds 40%, 50%, 60%, 70%, and 80% reported as sensitivity analyses. The visible code references constants rather than supplying their definitions. Confirm the actual minimum-source setting and how zero cited sources and extraction failures enter the denominator.

The reference threshold was revised after sparse eligibility at 80% became apparent. Do not retroactively call it preregistered. The stated rationale was coverage feasibility, not selection for significance. Confirm chronology if relevant. Stable results across thresholds do not prove missingness is ignorable; changing thresholds also changes the analyzed pairs.

A positive focal-minus-comparison gap difference means lower semantic alignment with each condition's own recovered cited evidence. It does not mean a measured proportion of claims was omitted. Unclipped one-minus-cosine is not intrinsically a probability or necessarily confined to [0,1].

### V2 outputs named in the supplied code

`query_evidence_alignment_metrics_v2.csv`, `source_evidence_alignment_metrics_v2.csv`, `selected_query_relevant_evidence_v2.csv`, `aio_chunks_v2.csv`, `evidence_passage_matches_v2.csv`, `aio_chunk_support_v2.csv`, `evidence_alignment_by_group_v2.csv`, `reference_50pct_dimension_tests_v2.csv`, `sensitivity_dimension_tests_v2.csv`, `coverage_threshold_stability_v2.csv`, `aggregate_sensitivity_v2.csv`, `answer_support_sensitivity_v2.csv`, `coverage_bias_tests_v2.csv`, `metric_vs_coverage_diagnostics_v2.csv`, `global_group_test_v2.csv`, `mechanism_map_v2.csv`, `pipeline_blocked_correlations_v2.csv`, `pipeline_naive_correlations_for_comparison.csv`, `weakly_aligned_relevant_evidence.csv`, `strongly_aligned_relevant_evidence.csv`, `all_results_v2.json`, and `README.txt`.

These are expected outputs, not files verified as present or complete. The JSON is intended to include metadata, collection quality, descriptives, reference/sensitivity analyses, secondary alignment, correlations, and audit examples. The user prefers organized results subdirectories, CSV tables, and a comprehensive JSON.

## 11. Statistical structure visible in the supplied code

Per-dimension contrasts pair focal and comparison conditions on `outcome_id`. Earlier exports use paired sign-flip tests, bootstrap intervals, Wilcoxon checks, rank-biserial effects, alpha 0.05, and Holm adjustment across six dimensions. Historical source/answer exports specify up to 200,000 Monte Carlo permutations and 20,000 bootstraps, with exact sign-flip enumeration for some smaller nonzero samples. Recover actual per-stage settings; do not assume one resample count for every script.

The V2 excerpt additionally implements:

- Separate dimension tests at each coverage threshold for the primary gap and secondary answer alignment.
- Aggregate effects formed by averaging available dimension-level paired differences within outcome, requiring a minimum number of dimensions, then analyzing outcome means. The value of `MIN_DIMENSIONS_FOR_AGGREGATE` is not visible here. Changing dimension composition can change the aggregate estimand.
- Focal/comparison source-coverage tests and metric-versus-coverage diagnostics. A nonsignificant coverage test does not establish missing-at-random recovery. Ordinary Spearman diagnostics remain descriptively useful but do not inherit blocked inference merely because another module uses it.
- A complete-case Friedman test at the reference threshold with Kendall's W, if enough outcomes/groups remain. Check which groups and how many complete outcomes actually enter.
- Joins to earlier source-overlap and answer-displacement results by group and outcome, then pipeline correlations. Validate join uniqueness, collection/location compatibility, and old/new model provenance; a V2 label on the combined table does not make upstream fields V2 measurements.

The blocked rank-correlation implementation ranks X and Y separately within each outcome, centers each set of ranks within that outcome, concatenates them, and computes Pearson correlation. It permutes Y within outcomes and bootstraps whole outcomes, assigning new block IDs to repeated bootstrap draws. Blocks need at least three complete observations, and the function needs at least three valid blocks. Holm adjustment is applied across the resulting pipeline tests. It is not ordinary pooled Spearman, not necessarily the mean of per-outcome correlations, and not a causal model.

This approach addresses some repeated-outcome structure, but its exchangeability assumptions still need assessment: group identity, paired dimensions, shared generic baselines, unequal block sizes, and repeated locations are not automatically handled by blocking only on outcome. Outcomes are also nested in seven purposively selected domains. State the scope of uncertainty and population of inference rather than calling the method universally correct.

## 12. Execution issues and current stopping point

The user hit `RuntimeError: operator torchvision::nms does not exist` while importing Sentence Transformers through Transformers. The earlier assistant suggested removing the optional broken `torchvision` installation for this text-only pipeline; whether that change was performed is not established here. No environment changes are requested by this handoff.

Later the matching loop printed 25-query increments through `Processed 250/273 queries`. The earlier assistant suspected matching had completed silently and execution had moved to slow statistical loops using repeated pandas grouping/copying/concatenation. This is a plausible diagnosis, not proof of the process location or completion.

A separate vectorized statistics script reusing saved query metrics was proposed, but its creation or execution is not visible. Do not assert that `stats_v2.py` exists. Verify final row counts, file timestamps, provenance, completion markers, exceptions, and caches; file existence alone may reflect a stale or partial run. Do not stop a live process or rerun expensive stages for this audit without a specific need and authorization.

## 13. What the current paper contains and still promises

The entire supplied PDF consists of an introduction, related work, Methods with `Query and Condition Design` and `Data Collection`, and references. The title and abstract are still AAAI template text. There is no Results or Discussion section in this version, and no completed analytical-methods description.

Related work develops four parts: algorithmic auditing/search exposure; social bias/counterfactual evaluation; generative search/evidence pipelines; identity-conditioned information mediation. It motivates examining both sources and answers.

The introduction and related-work conclusion currently promise source kinds, explanation types, which explanations survive, and tracing effects across selection and synthesis. The observed metric pipeline does not yet establish source taxonomies, coded explanation categories, atomic claim survival, or identifiable internal stage effects. Either the repository contains additional validated analyses or the paper's claims must be narrowed to observed sources, answer displacement, and semantic alignment with recovered evidence. Preserve the ambition while matching the measurement.

The draft's references were read as part of understanding the manuscript, not independently checked against external publications in this task. Bibliographic verification remains separate.

## 14. Next exchange and completion criteria

Repository Codex should return a self-contained, evidence-linked implementation report using the accompanying prompt. It should identify current code, inputs, parameters, execution status, exact formulas, sample counts, discrepancies, and safe versus unsupported manuscript claims. It should distinguish historical findings from valid current results and recommendations from completed work.

Once that report returns, the writing assistant will reconcile it with the paper, settle the analysis scope, finalize terminology and subsection order, and draft clear, objective Methods prose in LaTeX. The section should explain the reader's comparison logic rather than narrating software development or listing scripts.
