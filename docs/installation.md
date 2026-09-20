# Installation

[← Home](index.md) · [CLI reference →](cli.md)

## Requirements

- Python 3.10 or newer.
- Git, because the pinned LightGlue dependency is installed from GitHub.
- FFmpeg available as `ffmpeg` on `PATH`, or supplied with `--ffmpeg`.
- PyTorch. CUDA is strongly recommended for practical video processing, but
  CPU execution is supported.

The Python package dependencies are declared in `pyproject.toml`: LightGlue,
NumPy, OpenCV, PyYAML, and PyTorch. LightGlue pulls its own supporting Python
packages.

## Editable development install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm that the entry point is installed:

```bash
lgstab --help
```

## Documentation install

Install the optional documentation dependency and serve the linked pages
locally:

```bash
python -m pip install -e '.[docs]'
mkdocs serve
```

Use `mkdocs build --strict` to build the static site and fail on invalid links
or other documentation warnings.

## CUDA and CPU selection

When no device option is provided, lgstab requests `cuda:0 cuda:1`. Override
that default on single-GPU systems:

```bash
lgstab -i input.mp4 -r single-gpu --devices cuda:0
```

For CPU inference, disable CUDA-oriented mixed precision and pinned memory:

```bash
lgstab -i input.mp4 -r cpu-run \
  --devices cpu \
  --no-amp \
  --no-pin-memory
```

CPU inference is substantially slower for typical video workloads.

## FFmpeg encoders

The defaults use `hevc_nvenc` for both the stabilized and visualization
videos. That requires an FFmpeg build with NVIDIA NVENC support. A portable
software-encoding configuration is:

```bash
lgstab -i input.mp4 -r software-encode \
  --devices cpu --no-amp --no-pin-memory \
  --codec libx264 --viz-codec libx264 \
  --crf 18 --diagnostic-crf 20
```

Check available encoders with `ffmpeg -encoders`.

## Model weights and offline execution

SuperPoint and LightGlue may need their pretrained weights available in the
local PyTorch cache. Perform an initial run while online before moving to an
offline machine, or provision the expected cache in advance.

## Legacy script entry point

The original source invocation remains valid from the repository root:

```bash
python scripts/superpoint/stabilize.py --help
```

The installed `lgstab` command is preferred because it works independently of
the current directory.

---

[← Home](index.md) · [Next: CLI reference →](cli.md)
