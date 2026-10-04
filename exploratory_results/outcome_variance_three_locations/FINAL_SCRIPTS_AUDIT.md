# Audit of analyses retained in `ai_summary_final_scripts`

The final-paper package was compared with the exploratory scripts and outputs
developed earlier in this repository. This audit describes whether the concept
is present in the final package; it does not claim that every earlier figure or
estimand was preserved unchanged.

| Earlier analysis | Status in final scripts | What is retained or changed |
|---|---|---|
| Collection completeness and AIO presence | Retained | `consistency.py` and `link_count_analysis.py` validate the collections and record AIO presence. |
| Number of cited sources | Retained as a core analysis | Three per-location analyses plus a secondary pooled analysis; paired outcome inference, bootstrap CIs, effect sizes, and multiplicity correction are retained. |
| Number of unique domains | Retained descriptively | Unique registrable domains and hostname fallbacks are recorded in the source-overlap outputs. |
| AIO length | Partially retained | Character length remains in the link-count data and semantic diagnostics, but the earlier source-count/length figures are not part of the final figure set. |
| URL/domain Jaccard | Retained with a changed primary estimand | Exact URL and registrable-domain Jaccard are present. The final primary contrast compares each explicit group with the shared generic/People response, then contrasts displacement; it is not simply the earlier direct three-pair overlap analysis. |
| Overlap coefficient | Not retained | Directed explicit/generic coverage is available, but the symmetric overlap coefficient `|A∩B|/min(|A|,|B|)` is not reported. |
| Shared/A-only/B-only and expansion/substitution | Partially retained as raw sets | Shared, explicit-only, and generic-only URL/domain sets are saved. The normalized net-expansion and turnover analysis, especially direct minority-majority decomposition, is not a final-paper analysis. |
| Source-authority composition | Excluded deliberately | The final README excludes institutional source type because the automatic taxonomy was not manually validated. |
| DIF-inspired outcome plots | Not retained | The Dallas item-level pilot remains outside the final package. |
| Hierarchical outcome heterogeneity and variance decomposition | Not retained | The Dallas-only pilot remains outside the final package. A new three-location exploratory implementation is provided alongside this audit. |
| Semantic displacement | Retained and strengthened | Subject-normalized dense distance is primary; raw dense, TF-IDF, generic coverage, group novelty, and symmetric sentence distance are retained as robustness/supporting measures. |
| Evidence-to-synthesis alignment | Retained | Source-balanced evidence semantic gap, threshold sensitivity, and blocked correlations are retained. |
| Formal claim survival | Excluded deliberately | Evidence alignment is explicitly not described as claim survival. |
| Claim extraction, contextual hedging, and MegaVeridicality | Excluded deliberately | These branches are outside the final-paper package. |

The final package therefore centers on citation volume, source-set displacement,
answer semantic displacement, pathway/surface coupling, and evidence alignment.
The new variance analysis is intentionally stored under `exploratory_results`
and does not modify that frozen set.
