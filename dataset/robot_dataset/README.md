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
        └── <episode_id>/                          # e.g. 0307_154247
            └── data/trajectory.hdf5
```

No train/val split is shipped: every episode of a task sits directly under
`success_episodes/`, each episode appears exactly once, and episode ids are
unique within a task. Build your own split if you need one.

## Common conventions

- Images are stored as **encoded byte strings** (HDF5 `object` dtype), one per
  time step. RGB/color frames are **JPEG**-encoded (decode to `(H, W, 3)` uint8)
  and depth frames are **PNG**-encoded (decode to `(H, W)` uint16, unit:
  millimeters). Decode with `cv2.imdecode` (see `src/read_h5.py`).
- All JPEGs use the standard color order: they display correctly in any image
  viewer and PIL decodes them to RGB. `cv2.imdecode` returns **BGR**, so convert
  with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`, or pass `"to_rgb": True` to the
  loader (see below).
- `master/*` are the leader (teleoperation) signals; `puppet/*` are the
  follower (actually executed) signals — use `puppet/*` as the action / state
  targets for policy learning.
- Camera resolutions are not uniform across cameras or datasets; always read
  the shape from the decoded array rather than assuming a fixed size.

---

## 1. `Single_Arm_UR_5e` (Universal Robots UR5e, single arm)

5 tasks, 789 demonstrations, 105,704 frames.

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `language_instruction` | scalar | string | natural-language task instruction |
| `is_intervene` | `(T,)` | int | human-intervention flag (1 = teleoperated frame) |
| `master/arm_joint_position` | `(T, 6)` | float64 | leader 6-axis joint angles |
| `master/hand_joint_position` | `(T, 1)` | float64 | leader gripper |
| `puppet/arm_joint_position` | `(T, 6)` | float32 | follower 6-axis joint angles |
| `puppet/hand_joint_position` | `(T, 1)` | float32 | follower gripper |
| `puppet/end_effector` | `(T, 6)` or `(T, 7)` | float32 | follower TCP pose (6 = position + Euler, 7 = position + quaternion) |
| `observations/rgb_images/<camera>` | `(T,)` | object | JPEG-encoded RGB: `camera_front` / `camera_top` 720x1280, other cameras 480x640 |
| `observations/depth_images/<camera>` | `(T,)` | object | PNG-encoded uint16 depth, same resolution as the RGB stream |

Per-task episode counts and camera sets:

| Task | Episodes | Frames | Cameras |
|------|---------:|-------:|---------|
| ur_move_lemon_from_plate_to_bowl | 295 | 46,766 | front, left, top, wrist_left |
| ur_put_corn_into_the_pot | 294 | 33,725 | front, left, top, wrist_left |
| ur_stack_bowl_250523 | 100 | 11,609 | front, left, right, top, wrist_left, wrist_right |
| ur_open_drawer_and_put_corn | 50 | 7,289 | front, left, right, top, wrist_left, wrist_right |
| ur_put_carrot_and_close_drawer | 50 | 6,315 | front, left, right, top, wrist_left, wrist_right |

> **Known issue:** in `ur_put_carrot_and_close_drawer`, `camera_wrist_left`
> carries no usable signal — every frame of every episode is the same still
> image. Use `camera_wrist_right` (or the other views) for this task.

---

## 2. `AgileX_Cobot_Magic_V2.0` (AgileX Cobot Magic V2.0, dual arm)

5 tasks, 254 demonstrations, 97,359 frames. Two arms, with each arm's gripper
**packed into the joint vector** (no separate `hand_joint_position`).

| Task | Episodes | Frames |
|------|---------:|-------:|
| agilex_1_open_pot_and_put_corn | 50 | 22,276 |
| agilex_1_pick_banana_and_close_drawer | 49 | 17,311 |
| agilex_1_put_corn_into_plate | 55 | 18,497 |
| agilex_1_stack_bowl_250524 | 51 | 18,669 |
| agilex_1_upright_mug | 49 | 20,606 |

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

5 tasks, 258 demonstrations, 45,670 frames. This dataset uses a **different
schema**: camera data lives under `camera_observations` (with `color_images`
instead of `rgb_images`), the per-arm signals are split into left/right groups,
each wrapped in a `data` dataset, and there is **no** `language_instruction`.

| Task | Episodes | Frames |
|------|---------:|-------:|
| tienkung_16_collect_balls_251016 | 51 | 6,840 |
| tienkung_16_lay_bowl_plate_251031 | 55 | 9,570 |
| tienkung_16_oven | 50 | 8,982 |
| tienkung_16_select_yellow_button | 52 | 9,676 |
| tienkung_16_weight_apple | 50 | 10,602 |

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
    --file_path /path/to/Single_Arm_UR_5e/<task>/success_episodes/<episode>/data/trajectory.hdf5 \
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
    "to_rgb": True,   # default False returns OpenCV BGR arrays
}
reader = ReadH5Files(robot_infor)
rgb_images_dict, depth_images_dict, masked_rgb_dict = reader.execute(
    file_path="/path/to/trajectory.hdf5"
)
print(rgb_images_dict["camera_front"].shape)   # (T, H, W, 3) uint8, RGB
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
their tables above directly with `h5py`:

```python
import cv2
import h5py
import numpy as np

with h5py.File("/path/to/Tien_Kung_2.0/<task>/success_episodes/<episode>/data/trajectory.hdf5", "r") as f:
    buf = f["camera_observations/color_images/camera_head"][0]
    rgb = cv2.cvtColor(cv2.imdecode(np.asarray(buf), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth = cv2.imdecode(np.asarray(f["camera_observations/depth_images/camera_head"][0]), cv2.IMREAD_UNCHANGED)
    left_arm = f["puppet/arm_left_position_align/data"][:]   # (T, 7)
```
