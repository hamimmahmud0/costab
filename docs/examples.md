# Examples

[← Outputs and coordinates](outputs.md) · [Home →](index.md)

## CLI examples

### Single-GPU baseline

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name intersection-01 \
  --devices cuda:0
```

This directly registers every fourth frame to frame 0, uses homographies,
crops to the common valid region, writes the diagnostic video, and uses NVENC.

### Two GPUs with denser sampling

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name intersection-dense \
  --devices cuda:0 cuda:1 \
  --step 2 \
  --scale 0.75 \
  --prefetch 6 \
  --tracks
```

Workers share one task queue, so work is dynamically distributed rather than
assigning a fixed frame range to each GPU. `--scale` reduces model input size;
saved coordinates remain in original-resolution space.

### CPU with software video encoding

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name cpu-test \
  --devices cpu \
  --no-amp \
  --no-pin-memory \
  --codec libx264 \
  --viz-codec libx264 \
  --crf 18 \
  --diagnostic-crf 20
```

This avoids CUDA and NVIDIA encoder requirements. It is useful for validation
and short clips but is normally slower than GPU inference.

### Strict quality-controlled registration

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name strict-run \
  --devices cuda:0 \
  --step 2 \
  --strict \
  --min-matches 50 \
  --min-inliers 35 \
  --min-inlier-ratio 0.40 \
  --max-reprojection-rms 2.5 \
  --max-interpolation-gap 12
```

This run aborts on any rejected direct sample. It is appropriate when silent
interpolation over registration failures is unacceptable. Thresholds must be
tuned for scene texture and inference scale.

### Stable affine model with original output dimensions

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name affine-resized \
  --devices cuda:0 \
  --motion-model affine \
  --crop \
  --crop-resize-original \
  --smooth 3
```

Affine motion is less flexible than homography. The common crop is resized to
the input dimensions, and the absolute corner trajectories receive a
seven-frame moving average (`2 * 3 + 1`) while the reference stays fixed.

### Preserve the full canvas

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name full-canvas \
  --devices cuda:0 \
  --black-border \
  --no-diagnostic
```

The output keeps the input dimensions and may contain black areas. Diagnostics
are skipped to reduce extra rendering and encoding work.

### Custom outputs and reference frame

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name midday-reference \
  --reference-frame 900 \
  --devices cuda:0 \
  --output exports/stable.mp4 \
  --transforms-file exports/stable-transforms.npz \
  --metadata-file exports/stable-run.yaml \
  --diagnostic-output exports/stable-diagnostic.mp4
```

The run name is still recorded in metadata, while explicit files take
precedence over the corresponding run-directory defaults.

## Python examples

### Call the CLI entry function

```python
from lgstab.stabilize import main

main(
    [
        "-i", "assets/nadir.mp4",
        "-r", "embedded-run",
        "--devices", "cuda:0",
        "--step", "4",
        "--tracks",
    ]
)
```

This is the shortest Python integration and preserves exactly the CLI parsing
and validation behavior.

### Configure an argument namespace

```python
from lgstab.stabilize import build_parser, run, validate_args

parser = build_parser()
args = parser.parse_args(
    [
        "--input", "assets/nadir.mp4",
        "--run-name", "namespace-run",
        "--devices", "cuda:0",
    ]
)

args.reference_frame = 150
args.motion_model = "affine"
args.step = 2
args.scale = 0.5
args.diagnostic = False

validate_args(parser, args)
run(args)
```

Parsing first supplies all defaults. Calling `validate_args()` after mutations
protects against invalid values.

### Process several videos

```python
from pathlib import Path

from lgstab.stabilize import main

for video in sorted(Path("incoming").glob("*.mp4")):
    main(
        [
            "--input", str(video),
            "--run-name", video.stem,
            "--runs-dir", "runs/batch",
            "--devices", "cuda:0",
            "--no-diagnostic",
        ]
    )
```

Runs are sequential in this example. A single lgstab run already manages its
own persistent device workers; launching concurrent runs on the same GPU can
exhaust device memory.

### Transform a trajectory into stabilized coordinates

```python
from pathlib import Path

import numpy as np

archive = np.load(Path("runs/intersection-01/stabilization.npz"))
matrices = archive["raw_to_output"]

# Columns are frame, raw_x, raw_y.
trajectory = np.array(
    [
        [10, 812.5, 406.2],
        [11, 817.1, 407.9],
        [12, 822.0, 409.4],
    ],
    dtype=np.float64,
)

stabilized = []
for frame_value, x, y in trajectory:
    frame = int(frame_value)
    mapped_h = matrices[frame] @ np.array([x, y, 1.0])
    mapped_xy = mapped_h[:2] / mapped_h[2]
    stabilized.append((frame, *mapped_xy))

print(np.asarray(stabilized))
```

The `raw_to_output` matrix already includes the chosen crop and optional resize.
See [Coordinate conversion](outputs.md#coordinate-conversion) for the other
available coordinate systems.

## Choosing a configuration

| Goal | Starting point |
| --- | --- |
| Highest registration coverage | Lower `--step`; keep `--scale` near `1.0`; consider more keypoints. |
| Faster inference | Increase `--step`, reduce `--scale`, use CUDA/AMP, and consider LightGlue compilation for long videos. |
| Less flexible geometry | Use `affine` or `similarity` instead of `homography`. |
| Detect failures immediately | Enable `--strict` and set a finite `--max-interpolation-gap`. |
| No black borders | Use `--crop` and tune crop mask scale/safety. |
| Preserve input dimensions | Use `--black-border`, or `--crop --crop-resize-original`. |
| Inspect bad matching | Keep diagnostics enabled and inspect NPZ rejection reasons. |
| Portable encoding | Replace both NVENC codec defaults with `libx264`. |

For the exact meaning and range of every option, return to the
[CLI reference](cli.md). For implementation-level integration, see the
[Python API](python-api.md).

---

[← Outputs and coordinates](outputs.md) · [Back to Home →](index.md)
