# AIO claim-extraction pilot: local model selection

The pipeline is a qualification-preserving adaptation inspired by [Claimify](https://aclanthology.org/2025.acl-long.348/), not a reproduction or truth-verification system. It preserves the original AIO sentence/span and records all model-generated checks separately. Dallas is the first pilot corpus; the shared `People` control is processed once.

## Default and alternatives

The default local generation model is `/scratch/LLMs/models/microsoft/phi-4`, identified as `microsoft/phi-4`. It uses two visible GPUs by default (`--tensor-parallel-size 2`), `--gpu-memory-utilization 0.65` and an 8,192-token context. These are starting settings, not a guarantee of fit on a busy GPU. The stage prompts, parsing and output schema do not depend on the model family; any **text-generation checkpoint supported by the installed vLLM version** can be supplied via `--model-path`. For a different checkpoint, pass `--model-id` if its `config.json` lacks a trustworthy identifier. Supported inference knobs include `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`, `--batch-size` and `--trust-remote-code` (only when needed and the checkpoint is trusted).

The semantic candidate generator separately requires a local sentence-transformer checkpoint. Default identifier: `sentence-transformers/all-mpnet-base-v2`; supply its filesystem path with `--embedding-model-path`. No model is downloaded automatically. Results default to `results/claim_extraction_pilot/<model-id-slug>/`; a configuration guard prevents resuming a different model or embedding checkpoint in the same output directory. Each stage checkpoint is also keyed by model fingerprint, source item and prompt.

## Run from repository root

```bash
python -m pip install -r requirements-claim-extraction.txt
# Install a CUDA-compatible vllm build separately in this GPU environment.
python scripts/run_claim_extraction.py --preflight \
  --embedding-model-path /path/to/local/all-mpnet-base-v2
python scripts/run_claim_extraction.py \
  --embedding-model-path /path/to/local/all-mpnet-base-v2
```

On the current Medusa `.m1` environment with RTX 5090 GPUs, the working launch
uses its CUDA 13 toolkit and the non-FlashInfer sampler:

```bash
CUDA_HOME=/scratch/victoria.estanislau/.m1/lib/python3.12/site-packages/nvidia/cu13 \
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 \
python scripts/run_claim_extraction.py \
  --embedding-model-path /scratch/victoria.estanislau/models/all-mpnet-base-v2
```

For another vLLM-compatible model:

```bash
python scripts/run_claim_extraction.py \
  --model-path /path/to/local/model \
  --model-id organization/model-name \
  --tensor-parallel-size 2 \
  --embedding-model-path /path/to/local/all-mpnet-base-v2
```

To prepare Dallas inputs without GPU or checkpoints:

```bash
python scripts/run_claim_extraction.py --prepare-only
python -m unittest tests/test_claim_extraction_qwen14b.py -v
```

The model and embedding checkpoint are now present in `.m1`; Dallas inference
has been started there. Check the run manifest and all stage checkpoints before
treating an output directory as complete. If the process stops, rerun the same
command to resume from valid checkpoints.

## Outputs and safeguards

Each model-specific output directory contains `response_inventory.csv`, `sentences.csv`, stage JSONL checkpoints, `extracted_claims.csv`, `valid_claims.csv`, `response_metrics.csv`, provisional matching tables, a blank human-validation sample, figures, and a run manifest. `valid_claims.csv` requires successful selection, resolvable context, entailment, qualifier preservation and grounded context. The same model performs those automatic checks, so they are not independent validation. Similarity alone never declares claim equivalence. No causal or truth conclusions follow from this pilot.

Selection spans must be literal substrings of the original sentence. If a model
returns the exact text with wrong numeric offsets, the validator realigns it
only when that substring occurs exactly once. Recovered checkpoint entries are
appended with `recovery_method=deterministic_unique_literal_match`; original
invalid attempts remain untouched for audit. Paraphrases and ambiguous repeated
substrings remain invalid.
