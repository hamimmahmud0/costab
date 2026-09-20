# Outputs and coordinates

[← Python API](python-api.md) · [Examples →](examples.md)

Unless overridden, files are written below:

```text
<runs-dir>/<run-name>/
├── stabilized.mp4
├── motion_inliers_outliers.mp4
├── stabilization.npz
├── stabilization.yaml
└── tracks.mp4                    # only with --tracks
```

## Video outputs

### `stabilized.mp4`

The full-frame-rate stabilized result. Each raw input frame is warped exactly
once by its absolute current-to-reference transform and then mapped through the
selected output crop/resize transform. Source audio is copied by default when
present.

### `motion_inliers_outliers.mp4`

Enabled by default. This sampled-rate diagnostic places reference and current
views side by side and shows accepted/rejected registration state, inlier and
outlier correspondences, motion estimates, worker information, and quality
statistics. Disable it with `--no-diagnostic`.

### `tracks.mp4`

Created only with `--tracks`. It visualizes persistent reference feature IDs
and their sampled-observation histories. It is intended for inspecting
background tracking quality, not for exporting object trajectories.

## `stabilization.npz`

Load the compressed archive with:

```python
import numpy as np

data = np.load("runs/intersection-01/stabilization.npz")
print(data.files)
```

Here `T` is the decoded frame count, `S` is the sampled-frame count, and `P` is
the vertex count of the common polygon.

### Per-frame geometry

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `frame_indices` | `[T] int32` | Zero-based frame numbers. |
| `transforms_current_to_reference` | `[T,3,3] float64` | Absolute transform from raw coordinates in frame `t` to fixed-reference coordinates. |
| `warped_corners` | `[T,4,2] float32` | Input image corners after interpolation/smoothing in reference coordinates. |
| `raw_to_output` | `[T,3,3] float64` | Ready-to-use transform from a raw point in frame `t` directly to its coordinate in the saved stabilized video. |
| `reference_to_output` | `[3,3] float64` | Crop and optional resize mapping from reference coordinates to output-video coordinates. |
| `output_to_reference` | `[3,3] float64` | Inverse mapping from output-video coordinates to reference coordinates. |

The matrices satisfy:

```text
raw_to_output[t] = reference_to_output @ transforms_current_to_reference[t]
```

### Crop geometry

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `crop_rect` | `[4] int32` | Stable crop `(x, y, width, height)` in reference coordinates. |
| `common_polygon` | `[P,2] float32` | Convex region valid across all final per-frame transforms. |
| `width`, `height` | scalar int32 | Raw input dimensions. |
| `output_width`, `output_height` | scalar int32 | Encoded stabilized-video dimensions. |

In black-border mode, the crop covers the source canvas and
`reference_to_output` is identity. With crop-and-resize, the mapping includes
both crop translation and resizing.

### Sample registration diagnostics

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `sample_indices` | `[S] integer` | Frame index for every directly matched sample, including the reference identity sample. |
| `sample_valid` | `[S] bool` | Whether each direct registration passed every enabled gate. |
| `sample_transforms_current_to_reference` | `[S,3,3]` | Direct sampled transforms. Invalid samples contain NaN matrices. |
| `sample_matches` | `[S] integer` | Accepted LightGlue correspondence count before RANSAC rejection. |
| `sample_inliers` | `[S] integer` | RANSAC inlier count. |
| `sample_inlier_ratio` | `[S] floating` | `inliers / matches`. |
| `sample_reprojection_rms` | `[S] floating` | Inlier RMS reprojection error in original-resolution pixels. |
| `sample_reasons` | `[S] string` | `ok`, `reference identity`, or a rejection reason. |
| `sample_worker_ids` | `[S] integer` | Persistent worker that handled the sample. |
| `sample_worker_devices` | `[S] string` | Torch device used by that worker. |

### Run scalars

| Key | Type | Meaning |
| --- | --- | --- |
| `fps` | float64 | Input and stabilized-output FPS. |
| `frames` | int32 | Decoded frame count `T`. |
| `reference_frame` | int32 | Fixed zero-based reference index. |
| `step` | int32 | Requested direct-registration interval. |
| `inference_scale` | float32 | Requested matching scale. |
| `smooth_radius` | int32 | Absolute corner-trajectory smoothing radius. |

## Coordinate conversion

Points use homogeneous column vectors. To convert raw-frame point `(x, y)` at
frame `t` to the saved output video:

```python
import numpy as np

def transform_point(matrix: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    point_h = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    mapped_h = matrix @ point_h
    if abs(mapped_h[2]) < 1e-12:
        raise ZeroDivisionError("point mapped to infinity")
    return mapped_h[:2] / mapped_h[2]

data = np.load("runs/intersection-01/stabilization.npz")
output_xy = transform_point(data["raw_to_output"][120], (843.2, 417.8))
```

To compare detections from different raw frames in the fixed reference system,
use `transforms_current_to_reference[t]`. To map an annotation made on the
stabilized video back to the fixed reference coordinate system, use
`output_to_reference`.

For OpenCV batch conversion:

```python
import cv2
import numpy as np

points = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
mapped = cv2.perspectiveTransform(
    points.reshape(1, -1, 2),
    data["raw_to_output"][120],
).reshape(-1, 2)
```

## `stabilization.yaml`

The YAML file is intended for inspection, provenance, and downstream run
catalogs. It contains:

- `run_name` and input file metadata.
- Stabilization reference, sampling, motion model, scale, and smoothing.
- Parallel devices, worker counts, queue settings, and memory behavior.
- SuperPoint and LightGlue settings.
- RANSAC gates plus valid/rejected sample counts and maximum anchor gap.
- Crop rectangle, retained area, output dimensions, and coordinate matrices.
- Encoder configuration.
- Final paths for each deliverable.

Unlike the NPZ, YAML does not contain every per-frame transform; use the NPZ
for numeric processing.

## Reading rejection reasons

```python
import numpy as np

data = np.load("runs/intersection-01/stabilization.npz")
for frame, valid, reason in zip(
    data["sample_indices"],
    data["sample_valid"],
    data["sample_reasons"],
):
    if not valid:
        print(f"frame {int(frame)}: {reason}")
```

Use the [quality-gate options](cli.md#ransac-and-quality-gates) to tune repeated
rejections, and visually verify changes in the diagnostic video.

---

[← Python API](python-api.md) · [Next: Examples →](examples.md)
