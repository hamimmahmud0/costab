# lgstab documentation

lgstab stabilizes stationary nadir video by matching sampled frames directly
to one fixed reference frame with SuperPoint and LightGlue. It rejects moving
objects through robust geometric estimation, interpolates absolute transforms
for frames between samples, and warps each source frame exactly once.

This direct-reference design avoids the drift that can accumulate when
frame-to-frame transforms are chained.

## Documentation map

- [Installation](installation.md) covers Python, FFmpeg, LightGlue, CUDA, and
  editable installs.
- [CLI reference](cli.md) documents every command-line parameter and default.
- [Python API](python-api.md) shows programmatic use and documents every class
  and function in `lgstab.stabilize`.
- [Outputs and coordinates](outputs.md) explains every deliverable, NPZ key,
  and transform direction.
- [Examples](examples.md) provides complete CLI and Python workflows.

## Pipeline

1. Decode a fixed reference frame.
2. Extract its SuperPoint features once per persistent inference worker.
3. Sample the input every `--step` frames and match each sample directly to
   the reference with LightGlue.
4. Estimate a homography, affine transform, or similarity transform with
   RANSAC and apply the configured quality gates.
5. Interpolate missing and intermediate **absolute** registrations by warped
   image corners, then optionally smooth those corner trajectories.
6. Compute a common crop or retain the full canvas with black borders.
7. Decode the source again, warp each frame once, and stream frames to FFmpeg.
8. Save videos, transform arrays, and YAML metadata.

## Quick start

```bash
python -m pip install -e .

lgstab \
  --input assets/nadir.mp4 \
  --run-name intersection-01 \
  --devices cuda:0
```

The default outputs are placed in `runs/intersection-01/`. Continue with the
[CLI reference](cli.md) for tuning or [Python examples](examples.md#python-examples)
for embedding the pipeline in another application.

---

[Next: Installation →](installation.md)
