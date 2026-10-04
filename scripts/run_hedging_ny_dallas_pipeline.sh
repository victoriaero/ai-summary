#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_python="${PROJECT_PYTHON:-/scratch/victoria.estanislau/.m1/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/scratch/victoria.estanislau/.m1/lib/python3.12/site-packages/nvidia/cu13}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$repo_dir"
"$project_python" scripts/run_literal_claim_extraction.py \
  --input-dir annotations/v1_dallas/google_aio_collection --batch-size 16
"$project_python" scripts/run_literal_claim_extraction.py \
  --input-dir annotations/v2_ny/google_aio_collection --batch-size 16
"$project_python" scripts/analyze_hedging_ny_dallas.py
