# Bounding-Box Annotation Image Dataset (`annotation_images`)

Keyframe images of robot manipulation scenes from multiple tasks and camera
views. Each image is paired with one LabelMe-style bounding-box annotation file.
The dataset can be used directly for object detection training, scene
understanding, affordance research, and similar tasks.

## Directory layout

```
annotation_images/
└── <task_name>/                         # 35 tasks, e.g. ur_03_classify_cubes
    └── <episode_id>/                     # collection timestamp, e.g. 0409_194401
        └── rgb/
            ├── camera_front_keyframe_0.jpg
            ├── camera_front_keyframe_0.json
            ├── camera_left_keyframe_19.jpg
            ├── camera_left_keyframe_19.json
            └── ...
```

- **Task name prefixes** correspond to different robot platforms: `ur_*` (UR),
  `franka_*` / `franka2_*` (Franka), `agilex_*` (Agilex dual-arm), etc.
- **Cameras**: commonly `camera_front`, `camera_left`, `camera_right`,
  `camera_top`, `camera_wrist_left`, etc. (the set of cameras can differ between
  tasks).
- **File names**: `<camera>_keyframe_<frame_idx>.jpg`; some tasks omit the
  `keyframe` token, e.g. `camera_top_0.jpg`. Every `.jpg` has a matching `.json`
  annotation file with the same base name (a small number of images have no
  annotation file).

## Scale

- ~35 tasks
- ~76k images (`.jpg`)
- ~73k annotation files (`.json`)
- ~47 unique object labels, all annotated as axis-aligned rectangles (bboxes)

## Annotation format (LabelMe)

Each `.json` follows the LabelMe schema (version `1.1.0`). The full set of
fields is described below.

```json
{
  "version": "1.1.0",
  "flags": {},
  "shapes": [
    {
      "label": "yellow cube",
      "text": "",
      "points": [[252.96, 352.06], [341.61, 436.01]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "imagePath": "camera_front_keyframe_0.jpg",
  "imageData": null,
  "imageHeight": 720,
  "imageWidth": 1280
}
```

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | LabelMe schema version (`1.1.0`). |
| `flags` | object | Image-level flags; empty `{}` in this dataset. |
| `shapes` | array | List of annotated objects (one entry per bounding box). See below. |
| `imagePath` | string | File name of the paired image (same base name as the JSON). |
| `imageData` | null | Always `null`; the image is **not** embedded, load it from `imagePath`. |
| `imageHeight` | int | Image height in pixels (e.g. `720`). |
| `imageWidth` | int | Image width in pixels (e.g. `1280`). |

### Per-shape fields (each item in `shapes`)

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Object class name, free text (e.g. `table`, `corn`, `yellow cube`). |
| `text` | string | Optional note; empty `""` in this dataset. |
| `points` | array | Exactly two points `[[x1, y1], [x2, y2]]` in **pixel** coordinates: the two opposite corners of the box. |
| `group_id` | int / null | Optional instance group id; `null` in this dataset. |
| `shape_type` | string | Always `"rectangle"` (axis-aligned bounding box). |
| `flags` | object | Per-shape flags; empty `{}` in this dataset. |

### Bounding-box coordinate convention

- `points` stores **two opposite corners** of the rectangle, typically
  top-left and bottom-right, in absolute pixel coordinates.
- The corners are **not guaranteed to be ordered**, so to obtain a canonical
  `(x_min, y_min, x_max, y_max)` box, sort the two x values and the two y
  values:

  ```python
  (xa, ya), (xb, yb) = shape["points"]
  x_min, x_max = sorted((xa, xb))
  y_min, y_max = sorted((ya, yb))
  ```

- To convert to COCO `[x, y, w, h]`: `x = x_min`, `y = y_min`,
  `w = x_max - x_min`, `h = y_max - y_min`.
- Coordinates may be floating point and can sit on the image border
  (e.g. `0` or `imageWidth` / `imageHeight`); clamp to the image bounds if your
  pipeline requires it.

## Labels

Labels are free-form strings. They include scene elements (e.g. `table`,
`robot arm`, `left robot arm`, `right robot arm`, `unused robot arm`) and
manipulated objects (e.g. `corn`, `pot`, `pot lid`, `white bowl`, `apple`, and
various colored `cube` / `bowl` / `plate` / `cup`). A few spelling variants
exist in the raw data (e.g. `unused robot arn`), so it is recommended to build a
label vocabulary first and apply a normalization mapping if your task needs a
clean, fixed category set.
