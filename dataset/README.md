# Robot Manipulation Datasets (Open-Source Release)

These datasets accompany the paper **RoboAug: One Annotation to Hundreds of
Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation**
([arXiv:2602.14032](https://arxiv.org/abs/2602.14032)). RoboAug is a generative
data-augmentation framework that needs only the bounding-box annotation of a
single image during training, and was evaluated on three real robots — UR-5e,
AgileX, and Tien Kung 2.0. The datasets below correspond to those annotations
and real-robot demonstrations.

This repository open-sources two kinds of data:

1. **Bounding-box annotation image dataset (`annotation_images`)** — keyframe
   images from multiple tasks and camera views, annotated with bounding boxes in
   the LabelMe format.
2. **Real-robot manipulation datasets** — teleoperated demonstrations stored as
   per-episode HDF5 files (multi-camera RGB/depth images + robot proprioceptive
   states). Three robot platforms are included:
   - `Single_Arm_UR_5e` — Universal Robots UR5e, single arm.
   - `AgileX_Cobot_Magic_V2.0` — AgileX Cobot Magic V2.0, dual arm.
   - `Tien_Kung_2.0` — Tien Kung 2.0 humanoid, dual arm.

   A data loader is provided for `Single_Arm_UR_5e` (the `ReadH5Files` class in
   [`../src/read_h5.py`](../src/read_h5.py)); the other two share the same outer
   structure and image encoding but differ in their HDF5 fields (documented in
   the robot dataset README).

---

## Download the data

The code and documentation live in this GitHub repository, while the datasets
themselves are hosted on the Hugging Face Hub:

**https://huggingface.co/datasets/X-Humanoid/RoboAug-Datasets**

Download everything (or a single subset) with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[hf_transfer]"

# Download the whole dataset into ./data
hf download X-Humanoid/RoboAug-Datasets --repo-type dataset --local-dir ./data

# Or download only one subset, e.g. the UR5e demonstrations
hf download X-Humanoid/RoboAug-Datasets --repo-type dataset \
    --include "Single_Arm_UR_5e/*" --local-dir ./data
```

After downloading, the layout under `./data` is:

```
data/
├── annotation_images/            # bounding-box annotation images (JPEG + LabelMe JSON)
├── Single_Arm_UR_5e/             # UR5e demonstrations (HDF5)
├── AgileX_Cobot_Magic_V2.0/      # AgileX demonstrations (HDF5)
└── Tien_Kung_2.0/                # Tien Kung 2.0 demonstrations (HDF5)
```

---

## Documentation layout

These docs live under `dataset/` in the project repository; the UR5e data loader
is the `ReadH5Files` class in the project's `src/read_h5.py`.

```
roboaug/
├── LICENSE                         # Apache License 2.0 (project-wide)
├── src/
│   └── read_h5.py                  # HDF5 loader for Single_Arm_UR_5e (ReadH5Files)
└── dataset/
    ├── README.md                   # this file (overview & data download)
    ├── requirements.txt            # minimal deps to read the datasets
    ├── annotation_dataset/
    │   └── README.md               # annotation dataset format & fields
    └── robot_dataset/
        └── README.md               # real-robot dataset details (all 3 platforms)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `h5py`, `opencv-python`, `Pillow`.

### 2. Bounding-box annotation dataset

The images are plain `.jpg` files, each paired with a LabelMe-style `.json`
annotation of the same base name. The complete description of the directory
layout, JSON fields and bounding-box coordinate convention is in
[`annotation_dataset/README.md`](annotation_dataset/README.md).

### 3. Real-robot datasets

The provided loader (`ReadH5Files` in `src/read_h5.py`) targets
the `Single_Arm_UR_5e` dataset; it decodes the per-episode HDF5 files and returns
per-camera RGB/depth arrays. The `AgileX_Cobot_Magic_V2.0` and `Tien_Kung_2.0`
datasets use the same directory structure and image encoding but different HDF5
fields — see [`robot_dataset/README.md`](robot_dataset/README.md) for the full
field-by-field comparison.

```bash
# Decode one episode and dump the frames as PNG images (run from the repo root)
python src/read_h5.py \
    --camera_names "camera_front,camera_left,camera_top,camera_wrist_left" \
    --file_path data/Single_Arm_UR_5e/ur_put_corn_into_the_pot/success_episodes/train/0307_154247/data/trajectory.hdf5 \
    --save_dir ./out
```

```python
from src.read_h5 import ReadH5Files

robot_infor = {
    "camera_names": ["camera_front", "camera_left", "camera_top", "camera_wrist_left"],
    "camera_sensors": ["rgb_images", "depth_images"],
    "arms": ["master", "puppet"],
    "controls": ["joint_position"],
}
reader = ReadH5Files(robot_infor)
rgb, depth, masked = reader.execute(file_path="<path>/trajectory.hdf5")
print(rgb["camera_front"].shape)   # (T, H, W, 3)
```

See [`robot_dataset/README.md`](robot_dataset/README.md) for details.

---

## Dataset overview

| Dataset | Tasks | Scale | Annotations / labels | Format |
|---------|-------|-------|----------------------|--------|
| annotation_images | 35 tasks | ~76k images / ~73k annotation files | 47 object classes, all rectangular bboxes | JPEG + LabelMe JSON |
| Single_Arm_UR_5e | 5 tasks | 799 demonstrations (train 719 / val 80) | language instruction + proprioception | per-episode HDF5 |
| AgileX_Cobot_Magic_V2.0 | 5 tasks | ~275 demonstrations (train ~248 / val ~27) | language instruction + dual-arm proprioception | per-episode HDF5 |
| Tien_Kung_2.0 | 5 tasks | ~280 demonstrations (train ~252 / val ~28) | dual-arm proprioception (no language) | per-episode HDF5 |

---

## License and citation

The dataset and code are released under the **Apache License 2.0**; see
[`../LICENSE`](../LICENSE).

If you use this dataset in your research, please cite:

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
