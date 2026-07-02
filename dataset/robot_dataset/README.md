# Real-Robot Manipulation Datasets

This part contains teleoperated demonstrations collected on **three** different
robot platforms. Each demonstration (episode) is a standalone HDF5 file
(`trajectory.hdf5`) and all datasets share the same on-disk directory layout,
but the **internal HDF5 fields differ** between platforms (different number of
arms, gripper representation, camera group names, etc.).

| Dataset | Platform | Arms | Cameras (in samples) | Language instruction |
|---------|----------|------|----------------------|----------------------|
| `Single_Arm_UR_5e` | Universal Robots UR5e | single 6-DoF arm + gripper | up to 6: front / left / right / top / wrist_left / wrist_right | yes |
| `AgileX_Cobot_Magic_V2.0` | AgileX Cobot Magic V2.0 (dual-arm) | dual 7-DoF arms (joint+gripper packed) | camera_top | yes |
| `Tien_Kung_2.0` | Tien Kung 2.0 (humanoid, dual-arm) | dual 7-DoF arms + grippers | camera_head | no |

> The included **data loader (`src/read_h5.py`, the `ReadH5Files` class) targets
> the `Single_Arm_UR_5e` dataset only.** The AgileX and Tien Kung datasets share
> the same outer structure and the same JPEG/PNG image encoding, so the same
> decoding logic applies; only the dataset keys differ (documented below).

## Common directory layout

```
<dataset>/
└── <task_name>/
    └── success_episodes/
        ├── train/
        │   └── <episode_id>/                      # e.g. 0307_154247
        │       └── data/trajectory.hdf5
        └── val/
            └── <episode_id>/
                └── data/trajectory.hdf5
```

## Common conventions

- Images are stored as **encoded byte strings** (HDF5 `object` dtype), one per
  time step. RGB/color frames are **JPEG**-encoded (decode to `(H, W, 3)` uint8)
  and depth frames are **PNG**-encoded (decode to `(H, W)` uint16, unit:
  millimeters). Decode with `cv2.imdecode` (see `src/read_h5.py`).
- `master/*` are the leader (teleoperation) signals; `puppet/*` are the
  follower (actually executed) signals — use `puppet/*` as the action / state
  targets for policy learning.
- Camera resolutions are not uniform across cameras or datasets; always read
  the shape from the decoded array rather than assuming a fixed size.

---

## 1. `Single_Arm_UR_5e` (Universal Robots UR5e, single arm)

5 tasks, 799 demonstrations (train 719 / val 80).

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `language_instruction` | scalar | string | natural-language task instruction |
| `is_intervene` | `(T,)` | int | human-intervention flag (1 = teleoperated frame) |
| `master/arm_joint_position` | `(T, 6)` | float64 | leader 6-axis joint angles |
| `master/hand_joint_position` | `(T, 1)` | float64 | leader gripper |
| `puppet/arm_joint_position` | `(T, 6)` | float32 | follower 6-axis joint angles |
| `puppet/hand_joint_position` | `(T, 1)` | float32 | follower gripper |
| `puppet/end_effector` | `(T, 6)` or `(T, 7)` | float32 | follower TCP pose (6 = position + Euler, 7 = position + quaternion) |
| `observations/rgb_images/<camera>` | `(T,)` | object | JPEG-encoded RGB, e.g. `camera_front` 720x1280, `camera_left` 480x640 |
| `observations/depth_images/<camera>` | `(T,)` | object | PNG-encoded uint16 depth |

Per-task example counts and camera sets:

| Task | train | val | Cameras |
|------|------:|----:|---------|
| ur_move_lemon_from_plate_to_bowl | 265 | 30 | front, left, top, wrist_left |
| ur_put_corn_into_the_pot | 264 | 30 | front, left, top, wrist_left |
| ur_stack_bowl_250523 | 90 | 10 | front, left, right, top, wrist_left, wrist_right |
| ur_open_drawer_and_put_corn | 50 | 5 | front, left, right, top, wrist_left, wrist_right |
| ur_put_carrot_and_close_drawer | 50 | 5 | front, left, right, top, wrist_left, wrist_right |

---

## 2. `AgileX_Cobot_Magic_V2.0` (AgileX Cobot Magic V2.0, dual arm)

5 tasks, ~248 train / ~27 val demonstrations. Two arms, with each arm's gripper
**packed into the joint vector** (no separate `hand_joint_position`).

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `language_instruction` | scalar | string | natural-language task instruction |
| `is_intervene` | `(T,)` | int | human-intervention flag |
| `master/arm_joint_position` | `(T, 14)` | float64 | leader, both arms: 2 x (6 joints + 1 gripper) = 14 |
| `puppet/arm_joint_position` | `(T, 14)` | float32 | follower, both arms (same layout as above) |
| `puppet/end_effector` | `(T, 12)` | float32 | follower, both arms TCP pose: 2 x 6 (position + Euler) = 12 |
| `observations/rgb_images/camera_top` | `(T,)` | object | JPEG-encoded RGB, 480x640 |
| `observations/depth_images/camera_top` | `(T,)` | object | PNG-encoded uint16 depth, 400x640 |

**Differences vs `Single_Arm_UR_5e`:** dual-arm so `arm_joint_position` is
14-dim and `end_effector` is 12-dim; the gripper is part of the joint vector (no
`hand_joint_position`); `master` has no `end_effector`; RGB and depth have
different resolutions (480x640 vs 400x640).

---

## 3. `Tien_Kung_2.0` (Tien Kung 2.0 humanoid, dual arm)

5 tasks, ~252 train / ~28 val demonstrations. This dataset uses a **different
schema**: camera data lives under `camera_observations` (with `color_images`
instead of `rgb_images`), the per-arm signals are split into left/right groups,
each wrapped in a `data` dataset, and there is **no** `language_instruction`.

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `camera_observations/color_images/camera_head` | `(T,)` | object | JPEG-encoded RGB, 720x1280 |
| `camera_observations/depth_images/camera_head` | `(T,)` | object | PNG-encoded uint16 depth, 720x1280 |
| `camera_observations/is_intervene` | `(T,)` | bool | human-intervention flag |
| `camera_observations/timestamp` | `(T,)` | float64 | Unix timestamp per frame |
| `master/arm_left_position_align/data` | `(T, 7)` | float64 | leader left arm 7-DoF joints |
| `master/arm_right_position_align/data` | `(T, 7)` | float64 | leader right arm 7-DoF joints |
| `master/end_effector_left_position_align/data` | `(T, 1)` | float64 | leader left gripper |
| `master/end_effector_right_position_align/data` | `(T, 1)` | float64 | leader right gripper |
| `puppet/arm_left_position_align/data` | `(T, 7)` | float32 | follower left arm 7-DoF joints |
| `puppet/arm_right_position_align/data` | `(T, 7)` | float32 | follower right arm 7-DoF joints |
| `puppet/end_effector_left_position_align/data` | `(T, 1)` | float32 | follower left gripper |
| `puppet/end_effector_right_position_align/data` | `(T, 1)` | float32 | follower right gripper |

**Differences vs `Single_Arm_UR_5e`:** images are under
`camera_observations/color_images` (not `observations/rgb_images`);
`is_intervene` and `timestamp` live under `camera_observations`; left/right arms
and grippers are stored as separate groups, each with a nested `data` dataset;
no end-effector pose and no language instruction are provided.

---

## Usage of the UR5e loader (`src/read_h5.py`)

The loader is the `ReadH5Files` class, configured with a `robot_infor` dict.
Run the commands below from the project repository root.

Command line (decode one episode and save the frames as PNG images):

```bash
python src/read_h5.py \
    --camera_names "camera_front,camera_left,camera_top,camera_wrist_left" \
    --file_path /path/to/Single_Arm_UR_5e/<task>/success_episodes/train/<episode>/data/trajectory.hdf5 \
    --save_dir ./out
```

Python API:

```python
from src.read_h5 import ReadH5Files

robot_infor = {
    "camera_names": ["camera_front", "camera_left", "camera_top", "camera_wrist_left"],
    "camera_sensors": ["rgb_images", "depth_images"],
    "arms": ["master", "puppet"],
    "controls": ["joint_position"],
}
reader = ReadH5Files(robot_infor)
rgb_images_dict, depth_images_dict, masked_rgb_dict = reader.execute(
    file_path="/path/to/trajectory.hdf5"
)
print(rgb_images_dict["camera_front"].shape)   # (T, H, W, 3) uint8
```

To read the proprioceptive states (`master/*`, `puppet/*`, `is_intervene`,
`language_instruction`), open the file directly with `h5py`:

```python
import h5py

with h5py.File("/path/to/trajectory.hdf5", "r") as f:
    instr = f["language_instruction"][()].decode("utf-8")
    arm = f["puppet/arm_joint_position"][:]      # (T, 6)
    hand = f["puppet/hand_joint_position"][:]    # (T, 1)
    ee = f["puppet/end_effector"][:]             # (T, 6|7)
```

For the `AgileX_Cobot_Magic_V2.0` and `Tien_Kung_2.0` datasets, reuse the same
`cv2.imdecode` logic for the images and read the proprioceptive keys listed in
their tables above directly with `h5py`.
