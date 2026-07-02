# RoboAug

**RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data
Augmentation for Robotic Manipulation**

[Project page](https://x-roboaug.github.io) · [arXiv:2602.14032](https://arxiv.org/abs/2602.14032)

RoboAug is a **region-contrastive data augmentation framework** that improves the
generalization of visuomotor / VLA policies against out-of-distribution (OOD)
interference — complex background variations, drastic lighting changes, and
task-irrelevant distractors — while requiring only the bounding-box annotation of
a **single reference frame per task**.

RoboAug decomposes a visual observation into **task-relevant regions** (e.g., the
robot arm and manipulated objects) and **task-irrelevant scenario factors**
(background, distractors, lighting), and proceeds in three stages:

1. **Task-Relevant Region Extraction** — from a single annotated reference frame,
   a training-free *one-shot region matching* strategy localizes task-relevant
   elements in each trajectory's anchor frame, and *semantic mask propagation*
   then extends them into dense, pixel-level masks with spatiotemporal
   consistency across all frames.
2. **Semantic Data Augmentation** — guided by these masks, diverse full-scene
   backgrounds are synthesized with pre-trained generative models and the
   preserved foreground regions are composited onto them, expanding the dataset
   by orders of magnitude while keeping task-critical regions intact.
3. **Region-Contrastive Policy Learning** — a plug-and-play region-contrastive
   loss is injected into the visual encoder (no architectural change), clustering
   feature representations of the same semantic class and repelling different
   classes, so the policy attends to task-relevant regions under visual
   interference.

**This repository implements the data-generation part of RoboAug — stages (1)
and (2)** — turning an HDF5 trajectory plus a single
keyframe bounding-box annotation into visually diverse augmented HDF5 data, and
also hosts the documentation for the released datasets.

---

## TODO List

- [x] Dataset release (annotation dataset + real-robot datasets)
- [x] Stage 1 — Task-Relevant Region Extraction
- [x] Stage 2 — Semantic Data Augmentation
- [ ] Stage 3 — Region-Contrastive Policy Learning (RCL) — *to be open-sourced*

---

## Repository structure

```
roboaug/
├── README.md
├── LICENSE                  # Apache License 2.0
├── requirements.txt
├── dataset/                 # released datasets: docs, format & download (see below)
│   ├── README.md
│   ├── requirements.txt
│   ├── annotation_dataset/README.md
│   └── robot_dataset/README.md
├── scripts/
│   ├── demo.sh              # quick single-trajectory run
│   └── batch_run.sh         # batch over many trajectories
└── src/
    ├── config.py            # central model / data path config (env-var overridable)
    ├── sam2_region_augment.py             # main entry point (region extraction + augmentation)
    └── read_h5.py           # HDF5 trajectory read/write (ReadH5Files) used by the pipeline and the released datasets
```

> The open-set detector (GroundingDINO) and the tracking-and-segmentation model
> (SAM 2) are **not** bundled — install them from their upstream repositories
> (see [Installation](#installation)).

---

## Datasets

The datasets accompanying the paper are documented under [`dataset/`](dataset/)
and hosted on the Hugging Face Hub at
[**X-Humanoid/RoboAug-Datasets**](https://huggingface.co/datasets/X-Humanoid/RoboAug-Datasets).
They include:

- **`annotation_images`** — bounding-box annotation images (LabelMe format).
- **`Single_Arm_UR_5e`** — UR-5e single-arm real-robot demonstrations (HDF5).
- **`AgileX_Cobot_Magic_V2.0`** — AgileX dual-arm demonstrations (HDF5).
- **`Tien_Kung_2.0`** — Tien Kung 2.0 humanoid dual-arm demonstrations (HDF5).

See [`dataset/README.md`](dataset/README.md) for the Hugging Face download
instructions, directory layout, HDF5 field descriptions and data loader usage.

---

## Pipeline

1. Read the multi-camera RGB / depth sequences from an HDF5 trajectory.
2. Load the one-shot reference annotation for the task
   (`{ANNOTATION_BASE_PATH}/{task_name}/{camera}_keyframe_0.json`) — the
   bounding boxes and labels of task-relevant regions in a single frame.
3. **One-shot region matching**: an open-set detector proposes candidate regions
   on the anchor frame, and a vision foundation model establishes category
   correspondence to the reference templates via embedding similarity — a
   training-free alignment that filters out background clutter.
4. **Semantic mask propagation**: a tracking-and-segmentation model turns the
   sparse boxes into dense pixel-level masks and propagates them across the whole
   trajectory with spatiotemporal consistency.
5. **Semantic data augmentation**: a pre-trained text-to-image generative model
   synthesizes diverse full-scene backgrounds (prompts sampled from a background
   template library), and the preserved foreground (task-relevant) regions are
   composited onto the generated background using the masks.
6. Write the augmented images and masks back into HDF5.

---

## Installation

Python 3.10 + a CUDA environment is recommended.

```bash
# 1. Create the environment
conda create -n dataaug python=3.10 -y
conda activate dataaug

# 2. Install torch / torchvision matching your CUDA version (see https://pytorch.org)
#    Verified with torch 2.5.1 + CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining dependencies (transformers / diffusers are pinned;
#    see the note below)
pip install -r requirements.txt

# 4. Install the open-set detector and the tracking-and-segmentation model
#    from their upstream repositories. Both build CUDA C++ extensions, so make
#    sure CUDA_HOME points to a toolkit matching your torch build, e.g.:
export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git
```

> **GroundingDINO / SAM 2**
> These two libraries are used as-is from their original repositories and are not
> vendored here. Install them as shown above (or clone and `pip install -e .`).
> After installation they are importable as the `groundingdino` and `sam2`
> packages, which is what `src/sam2_region_augment.py` expects.

> **Pinned versions (important)**
> `requirements.txt` pins `transformers==4.44.2` and `diffusers==0.30.0`.
> GroundingDINO relies on an older transformers API (newer releases raise
> `BertModel has no attribute 'get_head_mask'`), while the newest diffusers
> requires a newer transformers than GroundingDINO tolerates — this pair is a
> verified compromise that satisfies both GroundingDINO and Stable Diffusion 3.
> `sentencepiece` (required by the SD3 T5 tokenizer) and `accelerate` are also
> installed from `requirements.txt`.

---

## Model weights

The pipeline relies on several pre-trained models. Weights are not bundled;
download them, then set the paths in `src/config.py` or override them via
environment variables (recommended). Each variable is described by its role in
the pipeline:

| Environment variable | Role in the pipeline | When needed |
| --- | --- | --- |
| `GROUNDING_DINO_CONFIG` | open-set detector config (region proposals) | required |
| `GROUNDING_DINO_CHECKPOINT` | open-set detector weights | required |
| `DINOV2_PATH` | vision foundation model for category correspondence | required |
| `SAM2_CHECKPOINT` | tracking-and-segmentation model weights (mask propagation) | required |
| `SAM2_MODEL_CFG` | tracking-and-segmentation hydra config **name** (not a file path) | required |
| `SD3_MEDIUM_PATH` | text-to-image generative model for background synthesis | `--pipe sd3` |
| `ANNOTATION_BASE_PATH` | root directory of the human annotations | required |

`SAM2_MODEL_CFG` is resolved by the `sam2` package's internal hydra search path,
so it must be given as a package-relative config name (with the `configs/sam2.1/`
prefix), **not** an absolute filesystem path — e.g.
`configs/sam2.1/sam2.1_hiera_l.yaml` for the `sam2.1_hiera_large.pt` checkpoint.

Example:

```bash
export GROUNDING_DINO_CONFIG=/path/to/detector/GroundingDINO_SwinT_OGC.py
export GROUNDING_DINO_CHECKPOINT=/path/to/detector/groundingdino_swint_ogc.pth
export DINOV2_PATH=/path/to/feature_matching_model
export SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt
export SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
export SD3_MEDIUM_PATH=/path/to/stable-diffusion-3-medium-diffusers
export ANNOTATION_BASE_PATH=/path/to/human_annotation
```

---

## Annotation data format

For each task and each camera, provide one keyframe annotation (LabelMe-style
JSON) plus the corresponding image:

```
{ANNOTATION_BASE_PATH}/{task_name}/{camera_name}_keyframe_0.json
{ANNOTATION_BASE_PATH}/{task_name}/{camera_name}_keyframe_0.jpg
```

Each element of the `shapes` list contains a `label` and a two-point rectangle
`points: [[x0, y0], [x1, y1]]`. See
[`dataset/annotation_dataset/README.md`](dataset/annotation_dataset/README.md)
for the full field reference.

### Input trajectory HDF5 layout

The pipeline reads RGB frames from `observations/rgb_images/{camera_name}`
(JPEG-encoded byte arrays), which is the layout of the released
`Single_Arm_UR_5e` / `AgileX_Cobot_Magic_V2.0` datasets. If your data stores
images under a different group or key, adjust the `robot_infor` dict in
`src/sam2_region_augment.py` accordingly.

---

## Usage

### Quick single-trajectory run

```bash
bash scripts/demo.sh /path/to/trajectory.hdf5 /path/to/output_dir
```

### Batch processing

```bash
# bash scripts/batch_run.sh <base_input_path> <base_output_path> <start_number> <run_time>
bash scripts/batch_run.sh /data/episodes /data/aug_output 0 1
```

### Calling the main script directly

```bash
cd src
export SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
CUDA_VISIBLE_DEVICES=0 python sam2_region_augment.py \
    --input  /path/to/trajectory.hdf5 \
    --output /path/to/output_dir \
    --camera_name "camera_top" \
    --pipe sd3 \
    --sd_guidance_scale 10 \
    --task_name "agilex_1_upright_mug" \
    --selected_labels "table" \
    --text_prompt "surface,table" \
    --bgr False \
    --stride 3
```

The `--task_name` must have a matching annotation directory under
`ANNOTATION_BASE_PATH`, and `--camera_name` must exist both as an annotated
keyframe and as an `observations/rgb_images/{camera_name}` group in the input
HDF5.

### Key arguments

| Argument | Description |
| --- | --- |
| `--input` | input HDF5 trajectory path |
| `--output` | output directory |
| `--pipe` | generation pipeline variant |
| `--sd_guidance_scale` | text guidance strength |
| `--camera_name` | camera name(s) to process (comma-separated for multiple) |
| `--task_name` | task name, used to locate the annotation directory |
| `--selected_labels` | task-relevant labels to preserve/augment, comma-separated, e.g. `"table"` |
| `--text_prompt` | background generation text prompt, e.g. `"surface,table"` |
| `--bgr` | whether the input images are in BGR channel order |
| `--stride` | process one frame every N frames (default 3) |

---

## License

Released under the **Apache License 2.0**; see [`LICENSE`](LICENSE).

This project depends on third-party open-source components (installed
separately) under their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attributions and terms.

## Citation

If you use this code or the datasets in your research, please cite:

```bibtex
@misc{wang2026roboaugannotationhundredsscenes,
      title={RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation}, 
      author={Xinhua Wang and Kun Wu and Zhen Zhao and Hu Cao and Yinuo Zhao and Zhiyuan Xu and Meng Li and Shichao Fan and Di Wu and Yixue Zhang and Ning Liu and Zhengping Che and Jian Tang},
      year={2026},
      eprint={2602.14032},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.14032}, 
}
```
