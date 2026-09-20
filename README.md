# lgstab

lgstab is a command-line tool for stabilizing stationary nadir video. It uses
SuperPoint and LightGlue to register sampled frames directly against one fixed
reference frame, then writes stabilized video, transform matrices, metadata,
and optional diagnostic visualizations.

## Requirements

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/) available on `PATH`
- A CUDA-capable PyTorch setup for the default configuration, or `--devices cpu`
  for CPU inference

## Install

Install from this checkout in editable mode while developing:

```bash
python -m pip install -e .
```

The LightGlue revision used by this project is installed directly from GitHub,
so Git must also be available during installation.

## Usage

```bash
lgstab --help

lgstab \
  --input assets/nadir.mp4 \
  --run-name intersection-01 \
  --devices cuda:0
```

For CPU inference:

```bash
lgstab \
  --input assets/nadir.mp4 \
  --run-name intersection-01 \
  --devices cpu \
  --no-amp
```

By default, output is written under `runs/<run-name>/`:

- `stabilized.mp4`
- `motion_inliers_outliers.mp4`
- `stabilization.npz`
- `stabilization.yaml`
- `tracks.mp4` when `--tracks` is enabled

The original script invocation remains available:

```bash
python scripts/superpoint/stabilize.py --help
```

Run `lgstab --help` for the complete set of stabilization, model, rendering,
and encoding options.
