#!/bin/bash
# Single-GPU overfit sanity check for UniScene3D-LLM on a tiny SQA3D subset.
# No DeepSpeed, no wandb (hard_debug). Use this to validate the end-to-end pipeline
# (Phase 4: single-batch overfit) before launching the full ZeRO-3 run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

set -e
set -o pipefail

export TOKENIZERS_PARALLELISM=false

python run.py \
  --config-path configs/finetune \
  --config-name sqa3d_llm.yaml \
  num_gpu=1 \
  name="SQA3D_LLM" note="debug_overfit" \
  debug.flag=True debug.hard_debug=True debug.debug_size=8 \
  dataloader.batchsize=1 dataloader.num_workers=0 \
  solver.gradient_accumulation_steps=1 \
  solver.epochs=50 solver.epochs_per_eval=10 solver.sched.args.warmup_steps=10 \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled
