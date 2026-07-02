#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch processing: iterate over the trajectories under a dataset directory and
# run data augmentation on each one (optionally multiple times).
#
# Usage:
#   bash scripts/batch_run.sh <base_input_path> <base_output_path> <start_number> <run_time> [task_name] [camera_name]
#
#   base_input_path    root directory containing trajectory subdirs (e.g. 1121_205701)
#   base_output_path   root directory for augmented outputs
#   start_number       offset into the sorted trajectory list
#   run_time           how many times to repeat augmentation per trajectory
#   task_name          annotation task name (default: agilex_1_upright_mug)
#   camera_name        camera to process (default: camera_top)
# ---------------------------------------------------------------------------
set -e

# Optional: HuggingFace / Torch cache directories
# export HF_HOME=/your/huggingface_cache
# export TORCH_HOME=/your/torch_cache

# SAM 2 hydra config name (package-relative, not a file path).
export SAM2_MODEL_CFG="${SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

base_input_path="${1:?missing base_input_path}"
base_output_path="${2:?missing base_output_path}"
start_number="${3:-0}"
run_time="${4:-1}"
task_name="${5:-agilex_1_upright_mug}"
camera_name="${6:-camera_top}"

SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"

# Find all trajectory dirs shaped like 0000_ and sort them numerically
all_trajectory_dirs=($(find "$base_input_path" -maxdepth 1 -type d | grep -E '/[0-9]{4}_' | sort -t '_' -k 2 -n))

if [ ${#all_trajectory_dirs[@]} -lt 5 ]; then
    echo "Error: fewer than 5 valid trajectory dirs, aborting."
    exit 1
fi

# Take 10 trajectories starting from start_number
trajectory_dirs=("${all_trajectory_dirs[@]:$start_number:10}")

cd "$SRC_DIR"

for run in $(seq 1 "$run_time"); do
    logger_file="$base_output_path/script_$start_number.log"
    for input_dir in "${trajectory_dirs[@]}"; do
        timestamp=$(basename "$input_dir")
        full_input_path="$input_dir/data/trajectory.hdf5"
        full_output_path="$base_output_path/$timestamp/run_$run"
        echo "full input path is $full_input_path" >> "$logger_file"
        mkdir -p "$full_output_path"

        CUDA_VISIBLE_DEVICES=0 python sam2_region_augment.py \
            --input "$full_input_path" \
            --output "$full_output_path" \
            --camera_name "$camera_name" \
            --pipe sd3 \
            --sd_guidance_scale 10 \
            --task_name "$task_name" \
            --selected_labels "table" \
            --text_prompt "surface,table" \
            --bgr False \
            --stride 3

        echo "finish $run time processing of $full_input_path" >> "$logger_file"
    done
done
