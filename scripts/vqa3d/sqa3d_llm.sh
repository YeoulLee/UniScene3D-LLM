#!/bin/bash
# UniScene3D-LLM full fine-tune on SQA3D with DeepSpeed ZeRO-3 across 4x H100.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

set -e
set -o pipefail

export TOKENIZERS_PARALLELISM=false

# ==== USER SETTINGS ====
NAME="SQA3D_LLM"
NOTE="sqa3d_llm_world_run1"      # change per experiment (e.g. ..._ego_run1)
COORD_FRAME="world"               # world | ego
VISION_FEATURE="projected"        # projected | penultimate

echo "[INFO] Launching ${NAME} (${NOTE}) | frame=${COORD_FRAME} feature=${VISION_FEATURE}"

# launch.py does not wire DeepSpeed, so call accelerate directly with the ZeRO-3 config.
accelerate launch --config_file configs/accelerate/zero3_h100x4.yaml \
  run.py \
  --config-path configs/finetune \
  --config-name sqa3d_llm.yaml \
  num_gpu=4 \
  name="$NAME" note="$NOTE" \
  model.coord_frame="$COORD_FRAME" \
  model.vision_feature="$VISION_FEATURE" \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled
