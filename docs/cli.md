# CLI reference

[← Installation](installation.md) · [Python API →](python-api.md)

## Invocation

```text
lgstab --input VIDEO --run-name NAME [OPTIONS]
```

Short forms are available for the two required options:

```bash
lgstab -i VIDEO -r NAME
```

Options shown as `--flag / --no-flag` use Python's boolean optional action.
The first form enables the feature and the second disables it explicitly.
`lgstab --help` prints the parser-generated reference from the installed
version.

## Required parameters

| Parameter | Type | Purpose |
| --- | --- | --- |
| `-i`, `--input` | path | Input video to decode. The path must name an existing regular file readable by OpenCV. |
| `-r`, `--run-name` | string | Names the run and, unless individual paths are overridden, its output directory under `--runs-dir`. |

## Core stabilization

| Parameter | Default | Purpose and guidance |
| --- | --- | --- |
| `-s`, `--step N` | `4` | Match every Nth frame directly to the reference. Smaller values improve temporal coverage but increase feature extraction and matching work. Must be at least 1. |
| `--reference-frame N` | `0` | Zero-based frame used as the fixed geometric reference. It must exist in the video. A sharp frame with a clear, mostly unobstructed background is best. |
| `--scale FLOAT` | `1.0` | Resize factor used only for model inference. It must be in `(0, 1]`. Geometry is lifted back to original-resolution coordinates before output. |
| `--smooth N` | `0` | Radius, in frames, for moving-average smoothing of absolute warped-corner trajectories. `0` preserves a strict reference lock; larger values trade lock strength for smoother camera correction. |
| `--motion-model MODEL` | `homography` | Registration family: `homography` supports perspective, `affine` supports translation/rotation/scale/shear, and `similarity` restricts the result to translation/rotation/uniform scale. |
| `--crop` | enabled | Compute a stable common rectangle and remove black warp borders. Mutually exclusive with `--black-border`. |
| `--black-border` | disabled | Keep the original canvas dimensions and permit empty black regions introduced by warping. |

All sampled frames are matched to one reference. `--step` does not enable
sequential transform chaining.

## Output paths

The effective run directory is `<runs-dir>/<run-name>`.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--runs-dir PATH` | `runs` | Parent directory for named runs. |
| `-o`, `--output PATH` | `<run-dir>/stabilized.mp4` | Stabilized video path. Its parent directory is created automatically. |
| `--transforms-file PATH` | `<run-dir>/stabilization.npz` | Compressed NumPy archive containing frame transforms, quality measurements, geometry, and video metadata. |
| `--metadata-file PATH` | `<run-dir>/stabilization.yaml` | Human-readable run configuration and result summary. |
| `--diagnostic-output PATH` | `<run-dir>/motion_inliers_outliers.mp4` | Registration visualization path. Used only when diagnostics are enabled. |
| `--tracks-output PATH` | `<run-dir>/tracks.mp4` | Feature-track visualization path. Used only with `--tracks`. |

See [Outputs and coordinates](outputs.md) for schemas and transform usage.

## SuperPoint parameters

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--max-keypoints N` | `4096` | Maximum SuperPoint detections retained per sampled frame. Higher values may improve difficult scenes but consume more memory and matching time. Must be at least 1. |
| `--detection-threshold FLOAT` | `0.0005` | Minimum SuperPoint detector response. Lower values admit weaker points; higher values produce fewer, stronger points. Must be non-negative. |
| `--nms-radius N` | `4` | Non-maximum-suppression radius in inference pixels. Larger values spread detections farther apart. Must be non-negative. |
| `--remove-borders N` | `4` | Width, in inference pixels, excluded around image borders when selecting keypoints. Must be non-negative. |

## LightGlue parameters

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--lightglue-filter-threshold FLOAT` | `0.1` | Minimum LightGlue match confidence accepted for geometric estimation. Must be in `[0, 1]`. |
| `--lightglue-depth-confidence FLOAT` | `0.95` | Confidence for adaptive early stopping across transformer layers. Use `-1` to disable; otherwise use `[0, 1]`. |
| `--lightglue-width-confidence FLOAT` | `0.99` | Confidence for adaptive keypoint pruning. Use `-1` to disable; otherwise use `[0, 1]`. |
| `--compile-lightglue`, `--no-compile-lightglue` | disabled | Enable or disable `torch.compile` for each LightGlue worker. Compilation adds startup cost and can improve repeated inference throughput. |
| `--compile-mode MODE` | `reduce-overhead` | `torch.compile` mode: `default`, `reduce-overhead`, or `max-autotune`. It has no effect unless compilation is enabled. |

## RANSAC and quality gates

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--ransac-threshold FLOAT` | `2.0` | Maximum RANSAC reprojection error measured in original-resolution pixels. The code converts it to inference scale. Must be greater than 0. |
| `--ransac-confidence FLOAT` | `0.999` | Desired probability that RANSAC finds a valid model. Must be strictly between 0 and 1. |
| `--ransac-max-iters N` | `10000` | Maximum robust-estimation iterations. Must be at least 1. |
| `--ransac-refine-iters N` | `10` | Refinement iterations used by affine and similarity estimation. Must be non-negative. |
| `--min-matches N` | `30` | Minimum LightGlue correspondences required before a sampled registration can be accepted. Must also satisfy the selected model's minimum point count. |
| `--min-inliers N` | `20` | Minimum RANSAC inliers required for acceptance. Must be at least 4 for homography or 3 for affine/similarity. |
| `--min-inlier-ratio FLOAT` | `0.25` | Minimum `inliers / matches` ratio. Must be in `[0, 1]`. |
| `--max-reprojection-rms FLOAT` | `5.0` | Maximum RMS error over inliers, in original pixels. Set `0` to disable this gate. |
| `--max-corner-displacement FLOAT` | `0.35` | Reject a transform if any image corner moves farther than this fraction of the image diagonal. Set `0` to disable. |
| `--max-projective FLOAT` | `0.003` | Homography sanity limit for the absolute normalized `h20` or `h21` projective terms. Set `0` to disable. |
| `--max-interpolation-gap N` | `0` | Maximum allowed frame distance between valid direct-reference anchors. `0` disables this post-registration safety gate. |
| `--strict`, `--no-strict` | disabled | When enabled, abort if any sampled registration fails. Otherwise, invalid samples may be filled from neighboring valid absolute registrations. |

Tight gates reduce the risk of accepting moving-object or low-texture matches,
but too-tight gates can leave too few valid anchors.

## Crop parameters

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--crop-mask-scale FLOAT` | `0.25` | Raster scale used while finding the largest rectangle inside the common valid region. It must be in `(0, 1]`; higher values improve crop precision at greater cost. |
| `--crop-safety N` | `4` | Shrink the selected rectangle by N original-resolution pixels per side to avoid edge artifacts. Must be non-negative. |
| `--crop-resize-original`, `--no-crop-resize-original` | disabled | Resize the cropped result back to the input width and height. Without it, output dimensions equal the actual crop dimensions. |

These parameters matter only in `--crop` mode.

## Devices and numerical behavior

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--devices DEVICE [DEVICE ...]` | `cuda:0 cuda:1` | Inference devices, such as `cuda:0`, `cuda:0 cuda:1`, or `cpu`. Duplicate device entries are rejected; use `--gpu-workers` for multiple workers on one device. |
| `--device DEVICE [DEVICE ...]` | none | Backward-compatible alias for `--devices`. It accepts separate, quoted space-separated, or comma-separated device values. Do not combine it with `--devices`. |
| `--gpu-workers N` | `1` | Persistent workers created per device. Every worker owns a SuperPoint model, a LightGlue model, and reference features, so increasing it also increases device memory use. |
| `--prefetch N` | `4` | Maximum queued sampled frames per worker. Higher values can improve utilization while increasing bounded CPU memory use. Must be at least 1. |
| `--pin-memory`, `--no-pin-memory` | enabled | Pin CPU tensors and use non-blocking copies for CUDA. Disable for CPU-only execution or when pinned memory is undesirable. |
| `--amp`, `--no-amp` | enabled | Enable LightGlue mixed precision on CUDA. It is ignored for non-CUDA devices. |
| `--matmul-precision LEVEL` | `high` | PyTorch float32 matrix-multiplication precision: `highest`, `high`, or `medium`. |
| `--torch-seed N` | `0` | Torch random seed used to make supported operations more repeatable. |
| `--opencv-seed N` | `0` | OpenCV RNG seed used by robust geometric estimation. |

## Stabilized-video encoding

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--ffmpeg PATH` | `ffmpeg` | FFmpeg executable name or filesystem path. Resolution occurs before processing begins. |
| `--codec NAME` | `hevc_nvenc` | FFmpeg video encoder for `stabilized.mp4`. Choose an encoder supported by the local FFmpeg build. |
| `--crf N` | `0` | Stabilized-video CRF value passed to FFmpeg. Interpretation and supported range are encoder-specific. Must be non-negative. |
| `--preset NAME` | `medium` | Encoder preset passed to FFmpeg. Supported values depend on `--codec`. |
| `--pixel-format NAME` | `yuv420p` | Output pixel format for the stabilized video. `yuv420p` has broad player compatibility. |
| `--audio-codec NAME` | `copy` | Copy source audio by default. Use `none` to omit audio, or name an FFmpeg audio encoder to re-encode. |
| `--audio-bitrate RATE` | `192k` | Audio bitrate used only when `--audio-codec` selects a re-encoding codec. |
| `--ffmpeg-loglevel LEVEL` | `warning` | FFmpeg verbosity: `quiet`, `panic`, `fatal`, `error`, `warning`, `info`, `verbose`, or `debug`. |

## Registration diagnostics

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--diagnostic`, `--no-diagnostic` | enabled | Generate the sampled motion/inlier/outlier visualization. Disable it to save rendering and encoding work. |
| `--diagnostic-crf N` | `18` | CRF for the diagnostic video. Must be non-negative. |
| `--viz-scale FLOAT` | `0.40` | Spatial resize factor for diagnostic and track visualizations. Must be greater than 0. |
| `--viz-header N` | `88` | Diagnostic header height in output pixels. It is rounded to an even dimension. Must be non-negative. |
| `--viz-max-matches N` | `800` | Maximum match lines drawn per sampled frame. `0` draws all available matches. |
| `--viz-point-radius N` | `2` | Radius of diagnostic match points. Must be at least 1. |
| `--viz-line-thickness N` | `1` | Thickness of diagnostic match lines. Must be at least 1. |
| `--viz-codec NAME` | `hevc_nvenc` | FFmpeg encoder shared by diagnostic and track videos. |
| `--viz-preset NAME` | `medium` | Encoder preset shared by diagnostic and track videos. |
| `--viz-pixel-format NAME` | `yuv420p` | Pixel format shared by diagnostic and track videos. |

The diagnostic frame rate is the input FPS divided by `--step`, with a floor
of 0.1 FPS, because only sampled registrations are visualized.

## Feature-track visualization

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--tracks`, `--no-tracks` | disabled | Generate `tracks.mp4`, showing persistent reference feature IDs across sampled observations. |
| `--tracks-crf N` | `18` | CRF for the track video. Must be non-negative. |
| `--track-length N` | `24` | Maximum trail length measured in sampled observations, not raw frames. Must be at least 2. |
| `--tracks-max-points N` | `1200` | Maximum persistent feature IDs visualized. `0` removes the limit. |
| `--tracks-point-radius N` | `2` | Radius of the current track point. Must be at least 1. |
| `--tracks-line-thickness N` | `1` | Thickness of historical track trails. Must be at least 1. |

## Runtime and preview

| Parameter | Default | Purpose and effect |
| --- | --- | --- |
| `--preview`, `--no-preview` | disabled | Display stabilized frames in an OpenCV window while rendering. This requires a graphical environment. |
| `--preview-scale FLOAT` | `0.5` | Display-only resize factor for the preview window. It does not affect saved video. Must be greater than 0. |
| `--progress-every N` | `100` | Print progress after every N frames. `0` disables periodic messages; completion output is still printed. |
| `-h`, `--help` | — | Print parser-generated help and exit without processing a video. |

## Validation behavior

Invalid ranges are reported as parser errors before model initialization. File,
device, FFmpeg, decoding, and output failures are raised at runtime. In
particular, remember that the default device list expects two CUDA devices and
the default video encoders expect NVENC support.

---

[← Installation](installation.md) · [Next: Python API →](python-api.md)
