# Python API

[← CLI reference](cli.md) · [Outputs and coordinates →](outputs.md)

The installed module is `lgstab.stabilize`. The supported programmatic entry
points are `main()`, `build_parser()`, `validate_args()`, and `run()`. The
remaining functions are documented for advanced integrations and testing, but
they currently expose implementation-level NumPy arrays, dictionaries,
queues, and model objects.

## Run with CLI-style arguments

`main()` accepts a sequence of strings, so callers do not need to mutate
`sys.argv`:

```python
from lgstab.stabilize import main

exit_code = main(
    [
        "--input", "assets/nadir.mp4",
        "--run-name", "python-run",
        "--devices", "cuda:0",
        "--step", "4",
        "--motion-model", "homography",
        "--crop",
    ]
)

if exit_code != 0:
    raise RuntimeError(f"lgstab failed with status {exit_code}")
```

Argument names, defaults, and validation are identical to the
[CLI reference](cli.md).

## Parse, modify, and run a namespace

For configuration assembled by Python code, use the same parser so every
option receives its normal default:

```python
from pathlib import Path

from lgstab.stabilize import build_parser, run, validate_args

parser = build_parser()
args = parser.parse_args(
    [
        "--input", str(Path("assets/nadir.mp4")),
        "--run-name", "configured-in-python",
        "--devices", "cuda:0",
    ]
)

# Values may be changed after parsing.
args.step = 2
args.scale = 0.75
args.tracks = True

# Validate again after programmatic changes.
validate_args(parser, args)
exit_code = run(args)
```

Do not create a small `argparse.Namespace` manually: `run()` reads all fields
defined by `build_parser()`. Parsing a base list first guarantees a complete
namespace.

## Load transforms in Python

```python
from pathlib import Path

import numpy as np

archive = np.load(Path("runs/python-run/stabilization.npz"))
raw_to_output = archive["raw_to_output"]

frame_index = 120
raw_xy = np.array([843.2, 417.8, 1.0])
output_h = raw_to_output[frame_index] @ raw_xy
output_xy = output_h[:2] / output_h[2]
print(output_xy)
```

See [Outputs and coordinates](outputs.md#coordinate-conversion) for transform
directions and homogeneous normalization.

## Entry-point functions

### `build_parser() -> argparse.ArgumentParser`

Builds and returns the complete parser. It has no parameters. The parser owns
all defaults, aliases, mutually exclusive groups, choice constraints, and help
text documented in the [CLI reference](cli.md). Range checks that cannot be
expressed as parser types are handled by `validate_args()`.

### `validate_args(parser, args) -> None`

- `parser`: the parser that should report user-facing errors.
- `args`: a namespace returned by `build_parser()` or an equivalent complete
  namespace.

Normalizes `--devices`/`--device`, including comma-separated and quoted device
lists, then validates numeric ranges and model-dependent minimum point counts.
Failures call `parser.error()`, which prints usage and raises `SystemExit(2)`.
The normalized device strings are written to `args.devices`.

### `main(argv=None) -> int`

- `argv`: optional sequence of argument strings without a program name. When
  `None`, `argparse` reads `sys.argv[1:]`.

Builds the parser, parses and validates arguments, delegates to `run()`, and
returns its status code. This function backs the installed `lgstab` command.

### `run(args) -> int`

- `args`: a complete, normalized, validated namespace.

Executes both pipeline passes, writes all configured outputs, prints a summary,
and returns `0` on success. Runtime failures such as missing files, unavailable
devices, invalid video, failed workers, failed FFmpeg, or unusable geometry are
raised as exceptions.

## FFmpeg writer

### `FFmpegRawWriter`

A context-manageable writer that streams contiguous `uint8` BGR frames to an
FFmpeg subprocess over standard input.

Constructor parameters:

- `ffmpeg`: resolved FFmpeg executable.
- `output_path`: destination video path; parent directories are created.
- `width`, `height`: exact expected frame dimensions.
- `fps`: output frame rate.
- `codec`, `crf`, `preset`, `pixel_format`: FFmpeg video settings.
- `loglevel`: FFmpeg log level.
- `source_audio`: optional source video used as a second FFmpeg input.
- `audio_codec`: `none`, `copy`, or an encoder name.
- `audio_bitrate`: bitrate used when audio is re-encoded.

Methods:

- `write(frame)`: validates dimensions and `uint8` dtype, makes the frame
  contiguous, and writes BGR bytes. Raises if the subprocess has failed.
- `close()`: closes stdin, waits for FFmpeg, and raises on a nonzero exit code.
- `__enter__()`: returns the writer.
- `__exit__(...)`: closes normally or kills the subprocess when the managed
  block exits through an exception.

## Geometry functions

### `normalize_homography(H) -> np.ndarray`

Accepts a `3×3` transform, converts it to `float64`, and divides by `H[2,2]`
when that value is safely nonzero. Raises `ValueError` for any other shape.

### `to_homography(affine_2x3) -> np.ndarray`

Embeds a `2×3` affine matrix into the first two rows of a `3×3` identity
matrix. The result uses homogeneous coordinates.

### `scale_matrix(sx, sy) -> np.ndarray`

Returns `diag(sx, sy, 1)` as `float64`. `sx` and `sy` are independent x/y
coordinate scale factors.

### `lift_scaled_homography(H_scaled, sx, sy) -> np.ndarray`

Converts a current-to-reference homography estimated in resized inference
coordinates back to original image coordinates using `S⁻¹ H_scaled S`, where
`S = scale_matrix(sx, sy)`. Returns a normalized `3×3` matrix.

### `image_corners(width, height) -> np.ndarray`

Returns the four pixel-center corners `[(0,0), (w-1,0), (w-1,h-1),
(0,h-1)]` as a `4×2 float32` array.

### `warp_points(points, H) -> np.ndarray`

Applies a `3×3` perspective transform to an `N×2` point array through OpenCV
and returns an `N×2` array.

### `corners_to_transform(source_corners, target_corners, motion_model)`

Fits the requested `homography`, `affine`, or `similarity` transform from four
source corners to target corners. Returns a normalized `3×3` matrix. Similarity
estimation can raise when OpenCV cannot construct a model.

### `transform_motion_stats(H, width, height)`

Measures the transform near the image center and returns `(dx, dy,
rotation_degrees, scale)`. These are visualization/diagnostic approximations,
not a decomposition of the entire projective transform.

### `transform_is_reasonable(H, width, height, max_corner_displacement, max_projective)`

Checks shape/finite values, optional projective-term limits, and optional
corner-displacement limits. Limits of `0` disable their corresponding check.
Returns `(accepted: bool, reason: str)`.

## Video, device, and model functions

### `resolve_ffmpeg(executable) -> str`

Resolves an executable through `PATH`, falling back to an explicit existing
file path. Raises `FileNotFoundError` when neither exists.

### `resolve_device(requested) -> torch.device`

Builds a Torch device. For CUDA, it verifies CUDA availability and that the
requested index exists. Returns the resolved `torch.device` or raises a runtime
error.

### `resolve_devices(requested_devices) -> list[torch.device]`

Resolves every device string with `resolve_device()` and rejects duplicates.

### `get_video_metadata(video_path) -> dict`

Uses OpenCV to read `width`, `height`, `fps`, and `reported_frames`. It raises
if the video cannot be opened or has invalid dimensions/FPS.

### `read_reference_frame(video_path, reference_index) -> np.ndarray`

Decodes sequentially through the zero-based requested frame and returns a copy
in OpenCV BGR format. Raises `IndexError` when the frame cannot be decoded.

### `resize_for_inference(frame, scale)`

Resizes a BGR frame with area interpolation. Returns `(resized, sx, sy)`, using
the actual rounded output/input ratios. At scale `1.0`, it returns the original
frame and ratios `(1.0, 1.0)`.

### `bgr_to_tensor(frame_bgr, device, pin_memory) -> torch.Tensor`

Converts BGR to RGB, transposes HWC to CHW, converts to float in `[0,1]`,
optionally pins host memory for CUDA, and transfers the tensor to `device`.

### `build_models(args, device)`

Creates eval-mode SuperPoint and LightGlue instances from namespace fields,
moves them to the device, enables mixed precision where configured, and
optionally compiles LightGlue. Returns `(extractor, matcher)`.

### `extract_features(extractor, frame_scaled, device, pin_memory)`

Converts a scaled frame with `bgr_to_tensor()` and runs
`extractor.extract(..., resize=None)` under Torch inference mode. Returns the
model's feature dictionary.

## Registration functions

### `estimate_geometry(current_points, reference_points, args, inference_scale)`

Fits the selected motion model with OpenCV RANSAC. The original-pixel threshold
is multiplied by `inference_scale`. Returns `(H_or_none, inlier_mask)`, where H
maps current inference coordinates to reference inference coordinates.

### `match_sample_to_reference(frame, reference_features, extractor, matcher, device, args, width, height)`

Runs the full registration for one sampled BGR frame: resize, feature
extraction, LightGlue matching, RANSAC, original-resolution lifting, RMS
measurement, and quality gates. `width` and `height` describe the original
video. The returned dictionary contains validity/reason fields, transform and
match statistics, and point/mask data used by diagnostics and tracks.

### `registration_result_to_record(frame_index, result, reference_keypoint_count, worker_id, worker_device)`

Converts a detailed registration result to the compact per-sample record saved
later in the NPZ. A `None` result represents the reference frame and becomes an
identity transform with perfect statistics. Failed transforms are represented
by a `3×3` NaN matrix.

## Worker functions and class

### `RegistrationWorker`

A persistent daemon `threading.Thread`. Each instance owns one SuperPoint and
LightGlue pair and its own device-local reference features. Constructor inputs
identify the worker/device, shared task/result/status queues, stop event,
reference image, full argument namespace, and source dimensions. `run()`
initializes models, reports readiness, processes `(frame_index, frame)` tasks,
places result packages on the result queue, and reports tracebacks through the
status queue.

### `start_registration_workers(devices, args, reference_scaled, width, height)`

Creates `args.gpu_workers` workers per device and bounded shared queues, waits
for every worker to initialize, and verifies reference keypoint counts. Returns
`(workers, task_queue, result_queue, stop_event, ready, reference_keypoint_count)`.

### `stop_registration_workers(workers, task_queue, stop_event, graceful)`

For graceful shutdown, places one sentinel per worker; otherwise sets the stop
event and attempts nonblocking sentinels. Joins each worker and raises if a
graceful stop leaves threads alive.

## Visualization functions

### `even_dimension(value, minimum=2) -> int`

Clamps to `minimum` and rounds up to an even integer, supporting codecs and
pixel formats that require even dimensions.

### `visualization_size(width, height, scale) -> tuple[int, int]`

Scales source dimensions, rounds them, and applies `even_dimension()` to both.

### `put_text(image, text, origin, scale=0.55, color=WHITE, thickness=1)`

Draws anti-aliased OpenCV text with the requested origin, scale, color, and
thickness. The image is modified in place.

### `select_visualization_matches(inlier_mask, scores, maximum) -> np.ndarray`

Returns match indices selected for drawing. It preserves an inlier/outlier
representation and prioritizes match scores while enforcing `maximum`; `0`
means no limit.

### `render_registration_diagnostic(...) -> np.ndarray`

Builds the side-by-side reference/current diagnostic frame plus header. Inputs
include both BGR frames, current/reference indices, source and visualization
dimensions, header height, detailed result dictionary, and CLI namespace. It
draws motion, accepted/rejected state, inliers, outliers, and statistics.

### `update_track_histories(histories, allowed_ids, result, maximum_ids, track_length)`

Updates caller-owned feature-history deques using a registration result.
`allowed_ids` fixes the set of persistent reference IDs, `maximum_ids` caps
that set (`0` is unlimited), and `track_length` bounds each trail.

### `render_tracks(frame, histories, frame_index, reference_index, width, height, viz_width, viz_height, args)`

Resizes a BGR frame and draws current points and historical trails from the
track-history mapping. Returns the rendered visualization frame.

## Interpolation and crop functions

### `moving_average_1d(values, radius) -> np.ndarray`

Applies an edge-padded centered moving average with window `2*radius+1`.
Radius `0` or an empty input returns a copy.

### `build_full_absolute_transforms(num_frames, sample_indices, sample_transforms, sample_valid, width, height, motion_model, smooth_radius, reference_frame)`

Warps image corners at valid direct-reference anchors, linearly interpolates
every corner coordinate over all frame numbers, optionally smooths each
trajectory, re-locks the selected reference frame, and projects corners back
to the selected motion family. Returns `(transforms, full_corners)` with shapes
`[T,3,3]` and `[T,4,2]`. It raises when no valid anchors exist.

### `maximum_anchor_gap(sample_indices, sample_valid) -> int`

Returns the greatest frame-number difference between adjacent valid anchors,
or `0` when fewer than two valid anchors exist.

### `convex_common_region(transforms, width, height) -> np.ndarray`

Warps the image quadrilateral by every current-to-reference transform and
iteratively intersects their convex hulls. Returns the common polygon or raises
when no valid common area remains.

### `largest_rectangle_in_binary(mask) -> tuple[int, int, int, int]`

Uses a row-histogram/monotonic-stack algorithm to find the largest axis-aligned
rectangle of nonzero mask pixels. Returns `(x, y, width, height)`.

### `compute_common_crop(transforms, width, height, mask_scale, safety)`

Computes the common polygon, rasterizes it at `mask_scale`, finds its largest
axis-aligned rectangle, maps conservatively to source coordinates, shrinks it
by `safety`, and enforces even dimensions. Returns
`(x, y, width, height, common_polygon)`.

### `output_coordinate_transforms(border_mode, crop_rect, width, height, crop_resize_original)`

Builds coordinate maps after border handling. Black-border mode returns
identity maps and source dimensions. Crop mode applies crop translation and,
when requested, resize scaling. Returns `(reference_to_output,
output_to_reference, output_width, output_height)`.

---

[← CLI reference](cli.md) · [Next: Outputs and coordinates →](outputs.md)
