# Qualification-preserving Claimify-inspired AIO claim extraction pilot

> Historical Qwen-specific preparation directory. The current model-agnostic
> entrypoint defaults to local Phi-4 and writes to `results/claim_extraction_pilot/microsoft-phi-4/`.
> See [the current instructions](../claim_extraction_pilot/README.md).

This is an experimental adaptation of Metropolitansky & Larson's [Claimify paper](https://aclanthology.org/2025.acl-long.348/), **not** a reproduction of Claimify. It extracts propositions that could in principle be checked, but does not verify their truth. Citation, query relevance and model opinion are not used as factuality tests. It does not change the existing source-evidence claim-survival script.

## Scope and method

The first run covers the 273 Dallas responses with AIO text: 21 shared `People`, 126 minority and 126 majority. The reader preserves query, group, condition, outcome, domain, location and replica metadata. `response_inventory.csv` preserves the complete AIO text. `sentences.csv` has stable response-relative IDs and exact character offsets into that text. Explicit bullet items, paragraphs and headings are preserved. A `structure_warning` flags likely pasted-together words/headings; the script does **not** reconstruct information missing in the original collection file. Heading-only lines do not count as sentence-like units.

Qwen runs separate selection, disambiguation, decomposition, entailment, qualifier-preservation and context-audit prompts. Selection spans must match the original sentence *exactly* at the saved offsets. The context window is up to five sentences on either side, never external evidence. The decontextualized claim retains attribution, uncertainty, modality, negation, quantitative and population scope, conditions and association-versus-causation; added context is bracket-marked and linked to the recorded resolution. Invalid JSON/schema/offset output is logged and retried at most twice, then excluded from downstream stages. An unresolved ambiguity, non-entailed/uncertain claim, missing/uncertain qualifier preservation, or ungrounded/uncertain contextual addition is excluded from `valid_claims.csv` but remains in intermediate artifacts. These Qwen self-checks are **not independent validation**; the blank human-review sheet is necessary to estimate their reliability.

Exact duplicate claims are removed within response. A local embedding model proposes semantic duplicate and cross-condition counterparts (bidirectional top five); Qwen judges whether they are equivalent while checking qualifiers. Neither high cosine similarity nor an LLM verdict is treated as human-validated equivalence. A structured qualifier mismatch vetoes automatic equivalence. Conflicting/nontransitive triplets are marked ambiguous. Singleton `People only` etc. means *no counterpart found by this candidate-generation procedure*, not proof that none exists. The shared People response is processed once and only repeated in dimension-specific comparison views.

## Commands

From the repository root, in a GPU environment with compatible `vllm`, `transformers`, `sentence-transformers`, and `huggingface-hub`:

```bash
python -m pip install -r requirements-claim-extraction.txt
# Install a CUDA-compatible vllm build for your Python/CUDA environment.
python scripts/run_claim_extraction_qwen14b.py --preflight \
  --model-path /path/to/existing/Qwen2.5-14B-Instruct-AWQ \
  --embedding-model-path /path/to/existing/all-mpnet-base-v2
python scripts/run_claim_extraction_qwen14b.py \
  --model-path /path/to/existing/Qwen2.5-14B-Instruct-AWQ \
  --embedding-model-path /path/to/existing/all-mpnet-base-v2
```

If the two checkpoints already exist in the Hugging Face cache visible to that environment, the path flags may be omitted. Resolution uses `local_files_only=True`; **there is no automatic download**. The preflight fails before writing results or allocating GPU if a package/checkpoint is absent. The current local `.m1` environment lacks `vllm` and `sentence-transformers`; thus only preparation and unit tests were executed here:

```bash
python scripts/run_claim_extraction_qwen14b.py --prepare-only
python -m unittest tests/test_claim_extraction_qwen14b.py -v
```

The full command resumes from validated JSONL stage checkpoints using source-record ID, prompt text/version and model path. Changing a prompt/model invalidates the matching checkpoint key. `logs/run_manifest.json` records seed, checkpoints, model config hash, counts and source-file hashes.

## Outputs and interpretation

`selection_outputs.jsonl`, `disambiguation_outputs.jsonl`, `logs/decomposition_outputs.jsonl`, `entailment_validation.jsonl`, `qualification_validation.jsonl` and `context_audit_validation.jsonl` preserve raw responses, parse outcomes and failed retries. `extracted_claims.csv` links each normalized claim to its selected original spans/sentence/response and validation verdicts; `valid_claims.csv` contains only claims passing all quality gates after within-response deduplication. `claim_matching_candidates.csv` includes candidate-pair similarities and provisional Qwen judgments; `claim_presence_patterns.csv` and `claim_matching_triplets.csv` hold provisional People/minority/majority patterns, claim text and three pairwise similarities where available. `human_validation_sample.csv` is stratified for review and leaves all human labels blank.

`response_metrics.csv` has one row per original response, including zero-claim responses. Its claim-sentence rate uses only sentences with a valid selection decision; `n_selection_invalid` is reported separately, never silently treated as no claim. Figures cover factual-claim sentence rate, response claim counts, claim-bearing sentence counts, a sentence-unit funnel with counts and denominators, ambiguity among selected sentences, provisional counterpart coverage and a pairwise matrix. These are descriptive pilot results; they are neither causal estimates nor truth/accuracy measurements. Matching recall and extraction quality require human validation before substantive claims about condition differences. Linguistic commitment analyses should always return to the **original span/sentence**, never analyze only the normalized claim.
