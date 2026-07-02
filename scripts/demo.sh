#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Quick single-trajectory demo (minimal runnable example).
#
# Usage:
#   bash scripts/demo.sh <input_trajectory.hdf5> <output_dir> [task_name] [camera_name]
#
# Model paths are configured in src/config.py or overridden via environment
# variables, e.g.:
#   export SAM2_CHECKPOINT=/your/path/sam2.1_hiera_large.pt
# ---------------------------------------------------------------------------
set -e

# Optional: HuggingFace / Torch cache directories
# export HF_HOME=/your/huggingface_cache
# export TORCH_HOME=/your/torch_cache

# SAM 2 hydra config name (package-relative, not a file path).
export SAM2_MODEL_CFG="${SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

INPUT="${1:?Usage: bash scripts/demo.sh <input.hdf5> <output_dir> [task_name] [camera_name]}"
OUTPUT="${2:?Usage: bash scripts/demo.sh <input.hdf5> <output_dir> [task_name] [camera_name]}"
TASK_NAME="${3:-agilex_1_upright_mug}"
CAMERA_NAME="${4:-camera_top}"

# Run from the src directory (relative imports: sam2 / groundingdino / read_h5 / config)
cd "$(dirname "$0")/../src"

mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=0 python sam2_region_augment.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --camera_name "$CAMERA_NAME" \
    --pipe sd3 \
    --sd_guidance_scale 10 \
    --task_name "$TASK_NAME" \
    --selected_labels "table" \
    --text_prompt "surface,table" \
    --bgr False \
    --stride 3
