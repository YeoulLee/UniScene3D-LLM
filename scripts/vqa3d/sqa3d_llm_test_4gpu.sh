#!/bin/bash
# Evaluate a trained UniScene3D-LLM checkpoint on SQA3D (generative EM) across 4x H100.
# Faster than the single-GPU test: the val set is sharded over 4 GPUs and predictions are
# gathered for EM. Loads the CONSOLIDATED model dir (best_model) via +test_state_dict, so it
# works regardless of the #GPUs used for training (no zero_to_fp32, no sharded-ckpt reload).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

set -e
set -o pipefail

export TOKENIZERS_PARALLELISM=false

# ==== USER SETTINGS ====
# Consolidated model directory saved during training (sibling of best.pth).
# e.g. results/SQA3D_LLM_.../<timestamp>/ckpt/best_model  (or final_model)
STATE_DICT="results/SQA3D_LLM_.../ckpt/best_model"
NOTE="sqa3d_llm_eval_4gpu"

# Must match the trained run's model switches:
USE_VISION="True"               # True | False
VISION_FEATURE="projected"      # projected | penultimate
COORD_FRAME="world"             # world | ego
POS_EMBED_ENABLED="True"        # True | False
POS_EMBED_NORMALIZE="none"      # none | scene_bbox | fixed_scale
ENCODER_TUNE="frozen"           # frozen | partial | full
ENCODER_UNFREEZE_LAST_N="4"     # used when ENCODER_TUNE=partial

if [ ! -e "${STATE_DICT}" ]; then
  echo "[ERROR] STATE_DICT does not exist: ${STATE_DICT}"
  echo "        Point it at the consolidated model dir saved during training (best_model/final_model)."
  echo "        Use +test_state_dict (a consolidated dir), NOT +ckpt_path (sharded best.pth)."
  exit 1
fi

echo "[INFO] Testing state_dict on 4 GPUs: ${STATE_DICT}"
echo "[INFO] switches: vision=${USE_VISION} feature=${VISION_FEATURE} frame=${COORD_FRAME} pe=${POS_EMBED_ENABLED} norm=${POS_EMBED_NORMALIZE} encoder=${ENCODER_TUNE}"

# ZeRO-3 launch: consolidated weights are loaded into the sharded model via GatheredParameters.
accelerate launch --config_file configs/accelerate/zero3_h100x4.yaml \
  run.py \
  --config-path configs/finetune \
  --config-name sqa3d_llm.yaml \
  num_gpu=4 \
  name="SQA3D_LLM" note="$NOTE" \
  mode=test \
  +test_state_dict="$STATE_DICT" \
  eval.save=True \
  model.use_vision="$USE_VISION" \
  model.vision_feature="$VISION_FEATURE" \
  model.coord_frame="$COORD_FRAME" \
  model.pos_embed.enabled="$POS_EMBED_ENABLED" \
  model.pos_embed.normalize="$POS_EMBED_NORMALIZE" \
  model.encoder_tune="$ENCODER_TUNE" \
  model.encoder_unfreeze_last_n="$ENCODER_UNFREEZE_LAST_N" \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled
