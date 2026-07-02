"""Centralized configuration for all model weight / data paths.

Every path can be overridden with an environment variable, so the project can be
run on a new machine without editing the source, e.g.:
    export SAM2_CHECKPOINT=/your/path/...

Defaults follow the original project paths so existing setups keep working.
"""

import os


def _env(key: str, default: str) -> str:
    """Return the environment variable if set, otherwise the default."""
    return os.environ.get(key, default)


# --------------------------------------------------------------------------- #
# SAM2 (Segment Anything 2) video/image segmentation model
# --------------------------------------------------------------------------- #
SAM2_CHECKPOINT = _env(
    "SAM2_CHECKPOINT",
    "/media/gongda/huggingface_model/sam2/sam2.1_hiera_large.pt",
)
# This is the sam2 built-in hydra config name (relative to the sam2 package's
# `configs/` search path), not a filesystem path. The official sam2 package
# expects the full `configs/sam2.1/...` prefix.
SAM2_MODEL_CFG = _env("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")


# --------------------------------------------------------------------------- #
# GroundingDINO open-vocabulary detection model
# --------------------------------------------------------------------------- #
GROUNDING_DINO_CONFIG = _env(
    "GROUNDING_DINO_CONFIG",
    "/media/gongda/huggingface_model/GroundingDINO/GroundingDINO_SwinT_OGC.py",
)
GROUNDING_DINO_CHECKPOINT = _env(
    "GROUNDING_DINO_CHECKPOINT",
    "/media/gongda/huggingface_model/GroundingDINO/groundingdino_swint_ogc.pth",
)


# --------------------------------------------------------------------------- #
# Stable Diffusion 3 Medium (text-to-image)
# --------------------------------------------------------------------------- #
# pipe == "sd3"
SD3_MEDIUM_PATH = _env(
    "SD3_MEDIUM_PATH",
    "/media/gongda/huggingface_model/stable-diffusion-3-medium-diffusers",
)


# --------------------------------------------------------------------------- #
# DINOv2 feature matching (used by dinov2_matching)
# --------------------------------------------------------------------------- #
DINOV2_PATH = _env(
    "DINOV2_PATH",
    "/media/will-wang/pretrained_model/dinov2-base",
)


# --------------------------------------------------------------------------- #
# Human annotation root directory
# Expected layout: {ANNOTATION_BASE_PATH}/{task_name}/{camera_name}_keyframe_0.json (.jpg)
# --------------------------------------------------------------------------- #
ANNOTATION_BASE_PATH = _env(
    "ANNOTATION_BASE_PATH",
    "/media/will-wang/dataset/augvla_dataset/human_annotation",
)
