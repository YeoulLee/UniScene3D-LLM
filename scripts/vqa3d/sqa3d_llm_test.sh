#!/bin/bash
# Evaluate a trained UniScene3D-LLM checkpoint on SQA3D (generative EM), 4x H100 ZeRO-3.
#
# This loads the TRAINED projector + Qwen weights via resume (load_pretrain only restores the
# frozen FG-CLIP encoder). The model-architecture switches below MUST match the trained run,
# otherwise the checkpoint state_dict will not align (e.g. projected vs penultimate changes the
# projector input dim). Run with the same #GPUs / ZeRO config used for training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

set -e
set -o pipefail

export TOKENIZERS_PARALLELISM=false

# ==== USER SETTINGS ====
# Path to the trained checkpoint DIRECTORY saved by save_state (e.g. .../ckpt/best.pth).
CKPT_PATH="results/SQA3D_LLM_.../ckpt/best.pth"
NOTE="sqa3d_llm_eval"

# Must match the trained run's model switches:
USE_VISION="True"               # True | False
VISION_FEATURE="projected"      # projected | penultimate
COORD_FRAME="world"             # world | ego
POS_EMBED_ENABLED="True"        # True | False
POS_EMBED_NORMALIZE="none"      # none | scene_bbox | fixed_scale
ENCODER_TUNE="frozen"           # frozen | partial | full
ENCODER_UNFREEZE_LAST_N="4"     # used when ENCODER_TUNE=partial

if [ ! -e "${CKPT_PATH}" ]; then
  echo "[ERROR] CKPT_PATH does not exist: ${CKPT_PATH}"
  echo "        Point it at the trained best.pth directory (saved by save_state)."
  exit 1
fi

echo "[INFO] Testing checkpoint: ${CKPT_PATH}"
echo "[INFO] switches: vision=${USE_VISION} feature=${VISION_FEATURE} frame=${COORD_FRAME} pe=${POS_EMBED_ENABLED} norm=${POS_EMBED_NORMALIZE} encoder=${ENCODER_TUNE}"

accelerate launch --config_file configs/accelerate/zero3_h100x4.yaml \
  run.py \
  --config-path configs/finetune \
  --config-name sqa3d_llm.yaml \
  num_gpu=4 \
  name="SQA3D_LLM" note="$NOTE" \
  mode=test \
  resume=True \
  +ckpt_path="$CKPT_PATH" \
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
