#!/usr/bin/env python3
"""
stabilize_nadir.py

Fixed-reference stabilization for stationary nadir video using:
    SuperPoint + LightGlue + robust geometric registration.

Primary design invariant
------------------------
There is NO sequential transform chaining.

Every sampled frame is matched directly to one fixed reference frame.
Intermediate frames are obtained by interpolating ABSOLUTE reference
registrations. Each source frame is warped exactly once.

This is intended for stationary-drone traffic / vehicle trajectory video,
where the road/background should remain fixed and moving vehicles should be
rejected by RANSAC as geometric outliers.

Default deliverables
--------------------
<runs-dir>/<run-name>/stabilized.mp4
<runs-dir>/<run-name>/motion_inliers_outliers.mp4
<runs-dir>/<run-name>/stabilization.npz
<runs-dir>/<run-name>/stabilization.yaml

Optional:
<runs-dir>/<run-name>/tracks.mp4       (--tracks)

Important coordinate products
-----------------------------
stabilization.npz contains:
    transforms_current_to_reference : [T, 3, 3]
    reference_to_output             : [3, 3]
    output_to_reference             : [3, 3]
    raw_to_output                   : [T, 3, 3]

For a point p=(x,y) detected in raw frame t:
    p_output ~ raw_to_output[t] @ [x, y, 1]^T

This is useful for vehicle trajectory extraction.

Example
-------
python stabilize_nadir.py \
    -i assets/nadir.mp4 \
    -r intersection_01

python stabilize_nadir.py \
    -i assets/nadir.mp4 \
    -r intersection_01 \
    -s 4 \
    --crop \
    --crf 0 \
    --scale 0.5 \
    --smooth 0 \
    --devices cuda:0 cuda:1 \
    --tracks
"""

from __future__ import annotations

import argparse
import math
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


# ============================================================================
# Fresh Aqua visualization palette (OpenCV BGR)
# ============================================================================

AQUA_50 = (250, 253, 240)       # #F0FDFA
AQUA_100 = (241, 251, 204)      # #CCFBF1
AQUA_200 = (228, 246, 153)      # #99F6E4
AQUA_300 = (212, 234, 94)       # #5EEAD4
AQUA_400 = (191, 212, 45)       # #2DD4BF
AQUA_500 = (166, 184, 20)       # #14B8A6
AQUA_600 = (136, 148, 13)       # #0D9488
AQUA_700 = (110, 118, 15)       # #0F766E
AQUA_800 = (89, 94, 17)         # #115E59
AQUA_900 = (74, 78, 19)         # #134E4A
AQUA_950 = (46, 47, 4)          # #042F2E

ERROR_RED = (77, 72, 229)       # #E5484D
WARNING_AMBER = (11, 158, 245)  # #F59E0B
WHITE = (255, 255, 255)
MUTED_TEXT = (119, 122, 100)    # approximately #647A77 in BGR-ish presentation


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stabilize stationary nadir video with direct fixed-reference "
            "SuperPoint + LightGlue registration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----------------------------------------------------------------------
    # Required
    # ----------------------------------------------------------------------
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input video.",
    )
    parser.add_argument(
        "-r",
        "--run-name",
        required=True,
        help="Run name used for the output directory.",
    )

    # ----------------------------------------------------------------------
    # Core stabilization
    # ----------------------------------------------------------------------
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=4,
        help=(
            "Directly match every Nth frame to the fixed reference. "
            "Intermediate absolute transforms are interpolated."
        ),
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Zero-based fixed reference frame.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Inference scale only. Matching runs at this scale; final warps "
            "and output coordinates remain in original-resolution space."
        ),
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help=(
            "Temporal smoothing radius in frames applied to absolute warped "
            "corner trajectories. 0 = strict reference lock / no smoothing."
        ),
    )
    parser.add_argument(
        "--motion-model",
        choices=("homography", "affine", "similarity"),
        default="homography",
        help="Geometric registration model.",
    )

    border_group = parser.add_mutually_exclusive_group()
    border_group.add_argument(
        "--crop",
        dest="border_mode",
        action="store_const",
        const="crop",
        help="Crop to a common stable window with no black border.",
    )
    border_group.add_argument(
        "--black-border",
        dest="border_mode",
        action="store_const",
        const="black",
        help="Preserve full original canvas and allow black warp borders.",
    )
    parser.set_defaults(border_mode="crop")

    # ----------------------------------------------------------------------
    # Output paths
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Base run directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Stabilized video path. Default: <run-dir>/stabilized.mp4",
    )
    parser.add_argument(
        "--transforms-file",
        type=Path,
        default=None,
        help="NPZ diagnostics path. Default: <run-dir>/stabilization.npz",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="YAML metadata path. Default: <run-dir>/stabilization.yaml",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=None,
        help=(
            "Motion/inlier/outlier video. "
            "Default: <run-dir>/motion_inliers_outliers.mp4"
        ),
    )
    parser.add_argument(
        "--tracks-output",
        type=Path,
        default=None,
        help="Optional track video. Default: <run-dir>/tracks.mp4",
    )

    # ----------------------------------------------------------------------
    # SuperPoint
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=4096,
        help="Maximum SuperPoint keypoints per sampled frame.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.0005,
        help="SuperPoint detection threshold.",
    )
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=4,
        help="SuperPoint NMS radius.",
    )
    parser.add_argument(
        "--remove-borders",
        type=int,
        default=4,
        help="SuperPoint feature exclusion width at image borders.",
    )

    # ----------------------------------------------------------------------
    # LightGlue
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--lightglue-filter-threshold",
        type=float,
        default=0.1,
        help="LightGlue accepted-match confidence threshold.",
    )
    parser.add_argument(
        "--lightglue-depth-confidence",
        type=float,
        default=0.95,
        help=(
            "LightGlue adaptive early-stop confidence. "
            "Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--lightglue-width-confidence",
        type=float,
        default=0.99,
        help=(
            "LightGlue adaptive point-pruning confidence. "
            "Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--compile-lightglue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile LightGlue with torch.compile.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="torch.compile mode.",
    )

    # ----------------------------------------------------------------------
    # Robust estimation / quality gates
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=2.0,
        help=(
            "RANSAC reprojection threshold in ORIGINAL-resolution pixels. "
            "Automatically converted to inference resolution."
        ),
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=0.999,
        help="RANSAC confidence.",
    )
    parser.add_argument(
        "--ransac-max-iters",
        type=int,
        default=10000,
        help="RANSAC maximum iterations.",
    )
    parser.add_argument(
        "--ransac-refine-iters",
        type=int,
        default=10,
        help="Affine/similarity RANSAC refinement iterations.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=30,
        help="Minimum raw LightGlue matches for an accepted sample.",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=20,
        help="Minimum RANSAC inliers for an accepted sample.",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.25,
        help="Minimum accepted RANSAC inlier ratio.",
    )
    parser.add_argument(
        "--max-reprojection-rms",
        type=float,
        default=5.0,
        help=(
            "Maximum inlier RMS reprojection error in original pixels. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-corner-displacement",
        type=float,
        default=0.35,
        help=(
            "Reject if any warped image corner moves farther than this "
            "fraction of the image diagonal. 0 disables."
        ),
    )
    parser.add_argument(
        "--max-projective",
        type=float,
        default=0.003,
        help=(
            "Homography sanity limit for abs(h20) or abs(h21), after "
            "normalization. 0 disables."
        ),
    )
    parser.add_argument(
        "--max-interpolation-gap",
        type=int,
        default=0,
        help=(
            "Maximum frame distance allowed between valid direct-reference "
            "anchors. 0 disables this safety gate."
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Abort if any sampled direct-reference registration fails. "
            "Otherwise failed samples are filled only from neighboring "
            "absolute registrations."
        ),
    )

    # ----------------------------------------------------------------------
    # Crop
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--crop-mask-scale",
        type=float,
        default=0.25,
        help="Raster scale used for maximum common-rectangle calculation.",
    )
    parser.add_argument(
        "--crop-safety",
        type=int,
        default=4,
        help="Shrink common crop by this many original pixels per side.",
    )
    parser.add_argument(
        "--crop-resize-original",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resize the stable crop back to original input width/height. "
            "Default keeps the actual crop dimensions."
        ),
    )

    # ----------------------------------------------------------------------
    # Device / numerical
    # ----------------------------------------------------------------------
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help=(
            "Inference devices. Example: --devices cuda:0 cuda:1. "
            "Defaults to cuda:0 cuda:1."
        ),
    )
    device_group.add_argument(
        "--device",
        dest="legacy_devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help=(
            "Backward-compatible alias for --devices. It also accepts "
            "multiple devices, e.g. --device cuda:0 cuda:1, and a quoted "
            "value such as --device 'cuda:0 cuda:1'."
        ),
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        help=(
            "Persistent inference workers per device. Each worker owns its "
            "own SuperPoint + LightGlue model pair."
        ),
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=4,
        help=(
            "Maximum queued sampled frames per inference worker. "
            "This bounds CPU RAM while keeping both GPUs fed."
        ),
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pin CPU image tensors before CUDA transfer and use "
            "non-blocking host-to-device copies."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LightGlue mixed precision on CUDA.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="torch float32 matmul precision.",
    )
    parser.add_argument(
        "--torch-seed",
        type=int,
        default=0,
        help="Torch RNG seed.",
    )
    parser.add_argument(
        "--opencv-seed",
        type=int,
        default=0,
        help="OpenCV RNG seed used by robust estimation.",
    )

    # ----------------------------------------------------------------------
    # Main stabilized encoder
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable/path.",
    )
    parser.add_argument(
        "--codec",
        default="hevc_nvenc",
        help="Stabilized video codec.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help="Stabilized video CRF.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="Stabilized video encoder preset.",
    )
    parser.add_argument(
        "--pixel-format",
        default="yuv420p",
        help="Stabilized video pixel format.",
    )
    parser.add_argument(
        "--audio-codec",
        default="copy",
        help="Audio codec; use 'none' to omit audio.",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="192k",
        help="Audio bitrate when re-encoding audio.",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        choices=(
            "quiet",
            "panic",
            "fatal",
            "error",
            "warning",
            "info",
            "verbose",
            "debug",
        ),
        default="warning",
        help="FFmpeg log level.",
    )

    # ----------------------------------------------------------------------
    # Required motion + inlier/outlier visualization
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--diagnostic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate motion + inlier/outlier diagnostic video.",
    )
    parser.add_argument(
        "--diagnostic-crf",
        type=int,
        default=18,
        help="Diagnostic video CRF.",
    )
    parser.add_argument(
        "--viz-scale",
        type=float,
        default=0.40,
        help="Spatial scale of diagnostic/track visualizations.",
    )
    parser.add_argument(
        "--viz-header",
        type=int,
        default=88,
        help="Diagnostic header height in output pixels.",
    )
    parser.add_argument(
        "--viz-max-matches",
        type=int,
        default=800,
        help="Maximum match lines drawn per diagnostic frame. 0 = all.",
    )
    parser.add_argument(
        "--viz-point-radius",
        type=int,
        default=2,
        help="Diagnostic keypoint radius.",
    )
    parser.add_argument(
        "--viz-line-thickness",
        type=int,
        default=1,
        help="Diagnostic match line thickness.",
    )
    parser.add_argument(
        "--viz-codec",
        default="hevc_nvenc",
        help="Diagnostic/track visualization video codec.",
    )
    parser.add_argument(
        "--viz-preset",
        default="medium",
        help="Diagnostic/track visualization encoder preset.",
    )
    parser.add_argument(
        "--viz-pixel-format",
        default="yuv420p",
        help="Diagnostic/track visualization pixel format.",
    )

    # ----------------------------------------------------------------------
    # Optional background feature tracks visualization
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate sampled background feature-track visualization.",
    )
    parser.add_argument(
        "--tracks-crf",
        type=int,
        default=18,
        help="Track visualization CRF.",
    )
    parser.add_argument(
        "--track-length",
        type=int,
        default=24,
        help="Feature trail length in sampled observations.",
    )
    parser.add_argument(
        "--tracks-max-points",
        type=int,
        default=1200,
        help="Maximum persistent reference feature IDs drawn. 0 = unlimited.",
    )
    parser.add_argument(
        "--tracks-point-radius",
        type=int,
        default=2,
        help="Current track-point radius.",
    )
    parser.add_argument(
        "--tracks-line-thickness",
        type=int,
        default=1,
        help="Track trail thickness.",
    )

    # ----------------------------------------------------------------------
    # Runtime
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preview stabilized output while rendering.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Preview-only scale.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N frames. 0 disables.",
    )

    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    raw_devices = (
        args.devices
        if args.devices is not None
        else args.legacy_devices
    )

    if raw_devices is None:
        raw_devices = [
            "cuda:0",
            "cuda:1",
        ]

    normalized_devices = []

    for raw_device in raw_devices:
        # Accept all of:
        #   --devices cuda:0 cuda:1
        #   --device cuda:0 cuda:1
        #   --device "cuda:0 cuda:1"
        #   --devices "cuda:0,cuda:1"
        pieces = str(
            raw_device
        ).replace(
            ",",
            " ",
        ).split()

        normalized_devices.extend(
            pieces
        )

    if not normalized_devices:
        parser.error(
            "At least one device must be specified."
        )

    args.devices = normalized_devices

    if args.gpu_workers < 1:
        parser.error(
            "--gpu-workers must be >= 1."
        )

    if args.prefetch < 1:
        parser.error(
            "--prefetch must be >= 1."
        )

    if args.step < 1:
        parser.error("--step must be >= 1.")
    if args.reference_frame < 0:
        parser.error("--reference-frame must be >= 0.")
    if not (0.0 < args.scale <= 1.0):
        parser.error("--scale must satisfy 0 < value <= 1.")
    if args.smooth < 0:
        parser.error("--smooth must be >= 0.")
    if args.max_keypoints < 1:
        parser.error("--max-keypoints must be >= 1.")
    if args.detection_threshold < 0:
        parser.error("--detection-threshold must be >= 0.")
    if args.nms_radius < 0:
        parser.error("--nms-radius must be >= 0.")
    if args.remove_borders < 0:
        parser.error("--remove-borders must be >= 0.")
    for name in (
        "lightglue_filter_threshold",
        "min_inlier_ratio",
    ):
        value = getattr(args, name)
        if not (0.0 <= value <= 1.0):
            parser.error(f"--{name.replace('_', '-')} must be in [0,1].")
    for name in (
        "lightglue_depth_confidence",
        "lightglue_width_confidence",
    ):
        value = getattr(args, name)
        if not (value == -1 or 0.0 <= value <= 1.0):
            parser.error(
                f"--{name.replace('_', '-')} must be -1 or in [0,1]."
            )
    if args.ransac_threshold <= 0:
        parser.error("--ransac-threshold must be > 0.")
    if not (0.0 < args.ransac_confidence < 1.0):
        parser.error("--ransac-confidence must be in (0,1).")
    if args.ransac_max_iters < 1:
        parser.error("--ransac-max-iters must be >= 1.")
    if args.ransac_refine_iters < 0:
        parser.error("--ransac-refine-iters must be >= 0.")
    minimum_model_points = 4 if args.motion_model == "homography" else 3
    if args.min_matches < minimum_model_points:
        parser.error(
            f"--min-matches must be >= {minimum_model_points} "
            f"for {args.motion_model}."
        )
    if args.min_inliers < minimum_model_points:
        parser.error(
            f"--min-inliers must be >= {minimum_model_points} "
            f"for {args.motion_model}."
        )
    if args.max_reprojection_rms < 0:
        parser.error("--max-reprojection-rms must be >= 0.")
    if args.max_corner_displacement < 0:
        parser.error("--max-corner-displacement must be >= 0.")
    if args.max_projective < 0:
        parser.error("--max-projective must be >= 0.")
    if args.max_interpolation_gap < 0:
        parser.error("--max-interpolation-gap must be >= 0.")
    if not (0.0 < args.crop_mask_scale <= 1.0):
        parser.error("--crop-mask-scale must satisfy 0 < value <= 1.")
    if args.crop_safety < 0:
        parser.error("--crop-safety must be >= 0.")
    if args.crf < 0 or args.diagnostic_crf < 0 or args.tracks_crf < 0:
        parser.error("CRF values must be >= 0.")
    if args.viz_scale <= 0:
        parser.error("--viz-scale must be > 0.")
    if args.viz_header < 0:
        parser.error("--viz-header must be >= 0.")
    if args.viz_max_matches < 0:
        parser.error("--viz-max-matches must be >= 0.")
    if args.viz_point_radius < 1:
        parser.error("--viz-point-radius must be >= 1.")
    if args.viz_line_thickness < 1:
        parser.error("--viz-line-thickness must be >= 1.")
    if args.track_length < 2:
        parser.error("--track-length must be >= 2.")
    if args.tracks_max_points < 0:
        parser.error("--tracks-max-points must be >= 0.")
    if args.tracks_point_radius < 1:
        parser.error("--tracks-point-radius must be >= 1.")
    if args.tracks_line_thickness < 1:
        parser.error("--tracks-line-thickness must be >= 1.")
    if args.preview_scale <= 0:
        parser.error("--preview-scale must be > 0.")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0.")


# ============================================================================
# FFmpeg raw-video writer
# ============================================================================

class FFmpegRawWriter:
    def __init__(
        self,
        ffmpeg: str,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str,
        crf: int,
        preset: str,
        pixel_format: str,
        loglevel: str,
        source_audio: Path | None = None,
        audio_codec: str = "none",
        audio_bitrate: str = "192k",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.width = int(width)
        self.height = int(height)
        self.closed = False

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            loglevel,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
        ]

        if source_audio is not None and audio_codec != "none":
            command += [
                "-i",
                str(source_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
            ]
        else:
            command += [
                "-map",
                "0:v:0",
            ]

        command += [
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            pixel_format,
        ]

        if source_audio is not None and audio_codec != "none":
            if audio_codec == "copy":
                command += [
                    "-c:a",
                    "copy",
                ]
            else:
                command += [
                    "-c:a",
                    audio_codec,
                    "-b:a",
                    audio_bitrate,
                ]
            command += [
                "-shortest",
            ]
        else:
            command += [
                "-an",
            ]

        command += [
            str(self.output_path),
        ]

        self.command = command
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
        )

        if self.process.stdin is None:
            raise RuntimeError(
                "Could not open FFmpeg stdin."
            )

    def write(self, frame: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("Writer is already closed.")

        if frame.shape[:2] != (
            self.height,
            self.width,
        ):
            raise ValueError(
                "FFmpeg frame size mismatch: "
                f"expected {self.width}x{self.height}, "
                f"got {frame.shape[1]}x{frame.shape[0]}."
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                "FFmpeg writer expects uint8 BGR frames."
            )

        try:
            self.process.stdin.write(
                np.ascontiguousarray(frame).tobytes()
            )
        except BrokenPipeError as exc:
            return_code = self.process.poll()
            raise RuntimeError(
                "FFmpeg terminated while receiving frames. "
                f"Return code: {return_code}"
            ) from exc

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True

        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass

        return_code = self.process.wait()

        if return_code != 0:
            raise RuntimeError(
                "FFmpeg failed for "
                f"{self.output_path} with exit code {return_code}."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            if not self.closed:
                self.closed = True
                if self.process.stdin is not None:
                    try:
                        self.process.stdin.close()
                    except Exception:
                        pass
                self.process.kill()
                self.process.wait()
        return False


# ============================================================================
# Geometry helpers
# ============================================================================

def normalize_homography(
    H: np.ndarray,
) -> np.ndarray:
    H = np.asarray(
        H,
        dtype=np.float64,
    )

    if H.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 transform, got {H.shape}."
        )

    if abs(float(H[2, 2])) > 1e-12:
        H = H / H[2, 2]

    return H


def to_homography(
    affine_2x3: np.ndarray,
) -> np.ndarray:
    H = np.eye(
        3,
        dtype=np.float64,
    )
    H[:2, :] = affine_2x3
    return H


def scale_matrix(
    sx: float,
    sy: float,
) -> np.ndarray:
    return np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def lift_scaled_homography(
    H_scaled: np.ndarray,
    sx: float,
    sy: float,
) -> np.ndarray:
    """
    If scaled_coord = S @ original_coord and H_scaled maps
    current_scaled -> reference_scaled:

        H_original = S^-1 @ H_scaled @ S
    """
    S = scale_matrix(
        sx,
        sy,
    )
    H_original = (
        np.linalg.inv(S)
        @ H_scaled
        @ S
    )
    return normalize_homography(
        H_original
    )


def image_corners(
    width: int,
    height: int,
) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )


def warp_points(
    points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    points = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
        2,
    )

    warped = cv2.perspectiveTransform(
        points,
        np.asarray(
            H,
            dtype=np.float64,
        ),
    )

    return warped.reshape(
        -1,
        2,
    )


def corners_to_transform(
    source_corners: np.ndarray,
    target_corners: np.ndarray,
    motion_model: str,
) -> np.ndarray:
    source_corners = np.asarray(
        source_corners,
        dtype=np.float32,
    )
    target_corners = np.asarray(
        target_corners,
        dtype=np.float32,
    )

    if motion_model == "homography":
        H = cv2.getPerspectiveTransform(
            source_corners,
            target_corners,
        )
        return normalize_homography(
            H
        )

    if motion_model == "affine":
        A = cv2.getAffineTransform(
            source_corners[:3],
            target_corners[:3],
        )
        return normalize_homography(
            to_homography(A)
        )

    if motion_model == "similarity":
        A, _ = cv2.estimateAffinePartial2D(
            source_corners,
            target_corners,
            method=cv2.LMEDS,
        )
        if A is None:
            raise RuntimeError(
                "Could not construct similarity transform "
                "from interpolated corners."
            )
        return normalize_homography(
            to_homography(A)
        )

    raise ValueError(
        f"Unknown motion model: {motion_model}"
    )


def transform_motion_stats(
    H: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """
    Approximate local correction at the image center:
        dx, dy, rotation degrees, local isotropic-ish x scale.
    """
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    baseline = max(
        10.0,
        min(width, height) * 0.05,
    )

    probe = np.array(
        [
            [cx, cy],
            [cx + baseline, cy],
        ],
        dtype=np.float32,
    )
    warped = warp_points(
        probe,
        H,
    )

    dx = float(
        warped[0, 0] - cx
    )
    dy = float(
        warped[0, 1] - cy
    )

    vector = (
        warped[1]
        - warped[0]
    )
    rotation = math.degrees(
        math.atan2(
            float(vector[1]),
            float(vector[0]),
        )
    )
    scale = float(
        np.linalg.norm(vector)
        / baseline
    )

    return (
        dx,
        dy,
        rotation,
        scale,
    )


def transform_is_reasonable(
    H: np.ndarray,
    width: int,
    height: int,
    max_corner_displacement: float,
    max_projective: float,
) -> tuple[bool, str]:
    if H is None:
        return False, "transform estimation returned None"

    if not np.all(
        np.isfinite(H)
    ):
        return False, "non-finite transform"

    H = normalize_homography(
        H
    )

    if max_projective > 0:
        if (
            abs(float(H[2, 0])) > max_projective
            or abs(float(H[2, 1])) > max_projective
        ):
            return (
                False,
                "projective term exceeded limit",
            )

    if max_corner_displacement > 0:
        corners = image_corners(
            width,
            height,
        )
        moved = warp_points(
            corners,
            H,
        )
        distance = np.linalg.norm(
            moved - corners,
            axis=1,
        )
        diagonal = math.hypot(
            width,
            height,
        )

        if float(
            distance.max()
        ) > (
            max_corner_displacement
            * diagonal
        ):
            return (
                False,
                "corner displacement exceeded limit",
            )

    return True, ""


# ============================================================================
# Video / model helpers
# ============================================================================

def resolve_ffmpeg(
    executable: str,
) -> str:
    path = shutil.which(
        executable
    )

    if path is None:
        candidate = Path(
            executable
        )
        if candidate.is_file():
            return str(candidate)

        raise FileNotFoundError(
            f"FFmpeg not found: {executable}"
        )

    return path


def resolve_device(
    requested: str,
) -> torch.device:
    device = torch.device(
        requested
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested ({requested}) but CUDA is unavailable."
            )

        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )

        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device {device_index}, "
                f"but only {torch.cuda.device_count()} CUDA device(s) exist."
            )

    return device


def resolve_devices(
    requested_devices: list[str],
) -> list[torch.device]:
    devices = []

    for requested in requested_devices:
        device = resolve_device(
            requested
        )
        devices.append(
            device
        )

    canonical = [
        str(
            device
        )
        for device in devices
    ]

    if len(
        canonical
    ) != len(
        set(
            canonical
        )
    ):
        raise ValueError(
            "Duplicate entries in --devices are not useful. "
            "Use --gpu-workers to create multiple workers per device."
        )

    return devices


def get_video_metadata(
    video_path: Path,
) -> dict:
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )
    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )
    reported_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Could not determine video resolution."
        )

    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(
            "Could not determine a valid video FPS."
        )

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
    }


def read_reference_frame(
    video_path: Path,
    reference_index: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    index = 0
    reference = None

    while index <= reference_index:
        ok, frame = cap.read()

        if not ok:
            break

        if index == reference_index:
            reference = frame.copy()
            break

        index += 1

    cap.release()

    if reference is None:
        raise IndexError(
            f"Could not decode reference frame {reference_index}."
        )

    return reference


def resize_for_inference(
    frame: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, float, float]:
    height, width = frame.shape[:2]

    if scale == 1.0:
        return frame, 1.0, 1.0

    target_width = max(
        1,
        int(round(width * scale)),
    )
    target_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        frame,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    sx = target_width / width
    sy = target_height / height

    return (
        resized,
        sx,
        sy,
    )


def bgr_to_tensor(
    frame_bgr: np.ndarray,
    device: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor_cpu = (
        torch.from_numpy(frame_rgb)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div_(255.0)
    )

    use_pinned = (
        pin_memory
        and device.type == "cuda"
    )

    if use_pinned:
        tensor_cpu = tensor_cpu.pin_memory()

    tensor = tensor_cpu.to(
        device,
        non_blocking=use_pinned,
    )

    return tensor


def build_models(
    args: argparse.Namespace,
    device: torch.device,
):
    extractor = (
        SuperPoint(
            max_num_keypoints=args.max_keypoints,
            detection_threshold=args.detection_threshold,
            nms_radius=args.nms_radius,
            remove_borders=args.remove_borders,
        )
        .eval()
        .to(device)
    )

    matcher = (
        LightGlue(
            features="superpoint",
            depth_confidence=args.lightglue_depth_confidence,
            width_confidence=args.lightglue_width_confidence,
            filter_threshold=args.lightglue_filter_threshold,
            mp=(
                args.amp
                and device.type == "cuda"
            ),
        )
        .eval()
        .to(device)
    )

    if args.compile_lightglue:
        if not hasattr(
            matcher,
            "compile",
        ):
            raise RuntimeError(
                "This PyTorch version/model does not expose Module.compile()."
            )

        matcher.compile(
            mode=args.compile_mode
        )

    return (
        extractor,
        matcher,
    )


def extract_features(
    extractor,
    frame_scaled: np.ndarray,
    device: torch.device,
    pin_memory: bool,
):
    image = bgr_to_tensor(
        frame_scaled,
        device,
        pin_memory,
    )

    with torch.inference_mode():
        features = extractor.extract(
            image,
            resize=None,
        )

    return features


# ============================================================================
# Pair registration
# ============================================================================

def estimate_geometry(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    args: argparse.Namespace,
    inference_scale: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    n = len(
        current_points
    )

    if n == 0:
        return (
            None,
            np.zeros(
                0,
                dtype=bool,
            ),
        )

    ransac_threshold = (
        args.ransac_threshold
        * inference_scale
    )

    if args.motion_model == "homography":
        H, mask = cv2.findHomography(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=args.ransac_max_iters,
            confidence=args.ransac_confidence,
        )

    elif args.motion_model == "affine":
        A, mask = cv2.estimateAffine2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=args.ransac_max_iters,
            confidence=args.ransac_confidence,
            refineIters=args.ransac_refine_iters,
        )
        H = (
            None
            if A is None
            else to_homography(A)
        )

    elif args.motion_model == "similarity":
        A, mask = cv2.estimateAffinePartial2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=args.ransac_max_iters,
            confidence=args.ransac_confidence,
            refineIters=args.ransac_refine_iters,
        )
        H = (
            None
            if A is None
            else to_homography(A)
        )

    else:
        raise ValueError(
            f"Unknown motion model: {args.motion_model}"
        )

    if mask is None:
        inliers = np.zeros(
            n,
            dtype=bool,
        )
    else:
        inliers = (
            mask.reshape(-1)
            .astype(bool)
        )

    if H is not None:
        H = normalize_homography(
            H
        )

    return (
        H,
        inliers,
    )


def match_sample_to_reference(
    frame: np.ndarray,
    reference_features,
    extractor,
    matcher,
    device: torch.device,
    args: argparse.Namespace,
    width: int,
    height: int,
) -> dict:
    frame_scaled, sx, sy = resize_for_inference(
        frame,
        args.scale,
    )

    current_features = extract_features(
        extractor,
        frame_scaled,
        device,
        args.pin_memory,
    )

    with torch.inference_mode():
        matches01 = matcher(
            {
                "image0": reference_features,
                "image1": current_features,
            }
        )

    reference_unbatched = rbd(
        reference_features
    )
    current_unbatched = rbd(
        current_features
    )
    matches_unbatched = rbd(
        matches01
    )

    matches = matches_unbatched[
        "matches"
    ]

    num_matches = int(
        matches.shape[0]
    )

    result = {
        "valid": False,
        "reason": "",
        "H": None,
        "matches": num_matches,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "rms": float("nan"),
        "reference_indices": np.empty(
            0,
            dtype=np.int32,
        ),
        "reference_points_original": np.empty(
            (0, 2),
            dtype=np.float32,
        ),
        "current_points_original": np.empty(
            (0, 2),
            dtype=np.float32,
        ),
        "inlier_mask": np.empty(
            0,
            dtype=bool,
        ),
        "scores": np.empty(
            0,
            dtype=np.float32,
        ),
        "sx": sx,
        "sy": sy,
    }

    if num_matches < args.min_matches:
        result["reason"] = (
            f"too few matches ({num_matches} < {args.min_matches})"
        )
        return result

    ref_indices_t = matches[
        ...,
        0,
    ]
    cur_indices_t = matches[
        ...,
        1,
    ]

    reference_points_scaled_t = (
        reference_unbatched[
            "keypoints"
        ][
            ref_indices_t
        ]
    )

    current_points_scaled_t = (
        current_unbatched[
            "keypoints"
        ][
            cur_indices_t
        ]
    )

    reference_points_scaled = (
        reference_points_scaled_t
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    current_points_scaled = (
        current_points_scaled_t
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    H_scaled, inlier_mask = estimate_geometry(
        current_points_scaled,
        reference_points_scaled,
        args,
        inference_scale=(
            (sx + sy)
            * 0.5
        ),
    )

    scores_t = matches_unbatched.get(
        "scores"
    )

    if scores_t is None:
        scores = np.ones(
            num_matches,
            dtype=np.float32,
        )
    else:
        scores = (
            scores_t
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    reference_indices = (
        ref_indices_t
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.int32,
            copy=False,
        )
    )

    reference_points_original = (
        reference_points_scaled.copy()
    )
    current_points_original = (
        current_points_scaled.copy()
    )

    reference_points_original[
        :,
        0,
    ] /= sx
    reference_points_original[
        :,
        1,
    ] /= sy

    current_points_original[
        :,
        0,
    ] /= sx
    current_points_original[
        :,
        1,
    ] /= sy

    result.update(
        {
            "reference_indices": reference_indices,
            "reference_points_original": reference_points_original,
            "current_points_original": current_points_original,
            "inlier_mask": inlier_mask,
            "scores": scores,
        }
    )

    if H_scaled is None:
        result["reason"] = "geometric estimation failed"
        return result

    H_original = lift_scaled_homography(
        H_scaled,
        sx,
        sy,
    )

    num_inliers = int(
        inlier_mask.sum()
    )
    inlier_ratio = (
        num_inliers
        / max(
            1,
            num_matches,
        )
    )

    result[
        "inliers"
    ] = num_inliers
    result[
        "inlier_ratio"
    ] = float(
        inlier_ratio
    )
    result[
        "H"
    ] = H_original

    if num_inliers < args.min_inliers:
        result["reason"] = (
            f"too few inliers ({num_inliers} < {args.min_inliers})"
        )
        return result

    if inlier_ratio < args.min_inlier_ratio:
        result["reason"] = (
            f"low inlier ratio "
            f"({inlier_ratio:.3f} < {args.min_inlier_ratio:.3f})"
        )
        return result

    current_inliers = (
        current_points_original[
            inlier_mask
        ]
    )
    reference_inliers = (
        reference_points_original[
            inlier_mask
        ]
    )

    projected = warp_points(
        current_inliers,
        H_original,
    )

    error = np.linalg.norm(
        projected
        - reference_inliers,
        axis=1,
    )

    rms = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    result["rms"] = rms

    if (
        args.max_reprojection_rms > 0
        and rms > args.max_reprojection_rms
    ):
        result["reason"] = (
            f"high RMS reprojection error "
            f"({rms:.3f} > {args.max_reprojection_rms:.3f})"
        )
        return result

    reasonable, reason = transform_is_reasonable(
        H_original,
        width,
        height,
        args.max_corner_displacement,
        args.max_projective,
    )

    if not reasonable:
        result["reason"] = reason
        return result

    result["valid"] = True
    result["reason"] = "ok"

    return result



# ============================================================================
# Persistent multi-GPU registration workers
# ============================================================================

def registration_result_to_record(
    frame_index: int,
    result: dict | None,
    reference_keypoint_count: int,
    worker_id: int,
    worker_device: str,
) -> dict:
    if result is None:
        return {
            "frame_index": int(
                frame_index
            ),
            "valid": True,
            "reason": "reference identity",
            "H": np.eye(
                3,
                dtype=np.float64,
            ),
            "matches": int(
                reference_keypoint_count
            ),
            "inliers": int(
                reference_keypoint_count
            ),
            "inlier_ratio": 1.0,
            "rms": 0.0,
            "worker_id": int(
                worker_id
            ),
            "worker_device": str(
                worker_device
            ),
        }

    return {
        "frame_index": int(
            frame_index
        ),
        "valid": bool(
            result[
                "valid"
            ]
        ),
        "reason": str(
            result[
                "reason"
            ]
        ),
        "H": (
            result[
                "H"
            ]
            if result[
                "H"
            ] is not None
            else np.full(
                (
                    3,
                    3,
                ),
                np.nan,
                dtype=np.float64,
            )
        ),
        "matches": int(
            result[
                "matches"
            ]
        ),
        "inliers": int(
            result[
                "inliers"
            ]
        ),
        "inlier_ratio": float(
            result[
                "inlier_ratio"
            ]
        ),
        "rms": float(
            result[
                "rms"
            ]
        ),
        "worker_id": int(
            worker_id
        ),
        "worker_device": str(
            worker_device
        ),
    }


class RegistrationWorker(
    threading.Thread,
):
    """
    One persistent SuperPoint + LightGlue model pair.

    Workers share one bounded task queue. Therefore, whichever GPU becomes
    free first takes the next sampled frame. Each worker independently
    extracts the fixed reference features on its own device.
    """

    def __init__(
        self,
        worker_id: int,
        device: torch.device,
        task_queue: queue.Queue,
        result_queue: queue.Queue,
        status_queue: queue.Queue,
        stop_event: threading.Event,
        reference_scaled: np.ndarray,
        args: argparse.Namespace,
        width: int,
        height: int,
    ):
        super().__init__(
            name=(
                f"registration-worker-"
                f"{worker_id}-"
                f"{str(device).replace(':', '_')}"
            ),
            daemon=True,
        )

        self.worker_id = int(
            worker_id
        )
        self.device = device
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.status_queue = status_queue
        self.stop_event = stop_event
        self.reference_scaled = reference_scaled
        self.args = args
        self.width = int(
            width
        )
        self.height = int(
            height
        )

    def run(
        self,
    ) -> None:
        device_string = str(
            self.device
        )

        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(
                    self.device
                )

            torch.manual_seed(
                self.args.torch_seed
                + self.worker_id
            )

            extractor, matcher = build_models(
                self.args,
                self.device,
            )

            reference_features = extract_features(
                extractor,
                self.reference_scaled,
                self.device,
                self.args.pin_memory,
            )

            reference_unbatched = rbd(
                reference_features
            )

            reference_keypoint_count = int(
                reference_unbatched[
                    "keypoints"
                ].shape[
                    0
                ]
            )

            self.status_queue.put(
                {
                    "kind": "ready",
                    "worker_id": self.worker_id,
                    "device": device_string,
                    "reference_keypoints": reference_keypoint_count,
                }
            )

            while not self.stop_event.is_set():
                try:
                    task = self.task_queue.get(
                        timeout=0.25
                    )
                except queue.Empty:
                    continue

                try:
                    if task is None:
                        return

                    frame_index, frame = task

                    result = match_sample_to_reference(
                        frame=frame,
                        reference_features=reference_features,
                        extractor=extractor,
                        matcher=matcher,
                        device=self.device,
                        args=self.args,
                        width=self.width,
                        height=self.height,
                    )

                    self.result_queue.put(
                        {
                            "kind": "result",
                            "worker_id": self.worker_id,
                            "device": device_string,
                            "frame_index": int(
                                frame_index
                            ),
                            "frame": frame,
                            "result": result,
                        }
                    )

                except Exception as exc:
                    self.result_queue.put(
                        {
                            "kind": "error",
                            "worker_id": self.worker_id,
                            "device": device_string,
                            "frame_index": (
                                int(
                                    task[
                                        0
                                    ]
                                )
                                if task is not None
                                else -1
                            ),
                            "error": repr(
                                exc
                            ),
                            "traceback": traceback.format_exc(),
                        }
                    )

                finally:
                    self.task_queue.task_done()

        except Exception as exc:
            self.status_queue.put(
                {
                    "kind": "error",
                    "worker_id": self.worker_id,
                    "device": device_string,
                    "error": repr(
                        exc
                    ),
                    "traceback": traceback.format_exc(),
                }
            )


def start_registration_workers(
    devices: list[torch.device],
    args: argparse.Namespace,
    reference_scaled: np.ndarray,
    width: int,
    height: int,
):
    total_workers = (
        len(
            devices
        )
        * args.gpu_workers
    )

    task_queue = queue.Queue(
        maxsize=max(
            1,
            total_workers
            * args.prefetch,
        )
    )
    result_queue = queue.Queue()
    status_queue = queue.Queue()
    stop_event = threading.Event()

    workers = []
    worker_id = 0

    for device in devices:
        for _ in range(
            args.gpu_workers
        ):
            worker = RegistrationWorker(
                worker_id=worker_id,
                device=device,
                task_queue=task_queue,
                result_queue=result_queue,
                status_queue=status_queue,
                stop_event=stop_event,
                reference_scaled=reference_scaled,
                args=args,
                width=width,
                height=height,
            )

            worker.start()

            workers.append(
                worker
            )
            worker_id += 1

    ready = {}
    reference_keypoint_counts = []

    while len(
        ready
    ) < total_workers:
        message = status_queue.get()

        if message[
            "kind"
        ] == "error":
            stop_event.set()

            raise RuntimeError(
                "Inference worker failed during initialization:\n"
                f"worker={message['worker_id']} "
                f"device={message['device']}\n"
                f"{message['error']}\n"
                f"{message['traceback']}"
            )

        ready[
            int(
                message[
                    "worker_id"
                ]
            )
        ] = message

        reference_keypoint_counts.append(
            int(
                message[
                    "reference_keypoints"
                ]
            )
        )

    reference_keypoint_count = int(
        reference_keypoint_counts[
            0
        ]
    )

    if any(
        count
        != reference_keypoint_count
        for count in reference_keypoint_counts
    ):
        print(
            "WARNING: SuperPoint returned different reference keypoint "
            "counts across workers:",
            reference_keypoint_counts,
        )

    return (
        workers,
        task_queue,
        result_queue,
        stop_event,
        ready,
        reference_keypoint_count,
    )


def stop_registration_workers(
    workers: list[RegistrationWorker],
    task_queue: queue.Queue,
    stop_event: threading.Event,
    graceful: bool,
) -> None:
    if graceful:
        for _ in workers:
            task_queue.put(
                None
            )
    else:
        stop_event.set()

        for _ in workers:
            try:
                task_queue.put_nowait(
                    None
                )
            except queue.Full:
                break

    for worker in workers:
        worker.join(
            timeout=10.0
        )

    still_alive = [
        worker.name
        for worker in workers
        if worker.is_alive()
    ]

    if still_alive and graceful:
        stop_event.set()
        raise RuntimeError(
            "Inference worker(s) did not shut down cleanly: "
            + ", ".join(
                still_alive
            )
        )


# ============================================================================
# Visualization
# ============================================================================

def even_dimension(
    value: int,
    minimum: int = 2,
) -> int:
    value = max(
        minimum,
        int(value),
    )
    if value % 2:
        value += 1
    return value


def visualization_size(
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int]:
    return (
        even_dimension(
            round(
                width * scale
            )
        ),
        even_dimension(
            round(
                height * scale
            )
        ),
    )


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color=WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def select_visualization_matches(
    inlier_mask: np.ndarray,
    scores: np.ndarray,
    maximum: int,
) -> np.ndarray:
    n = len(
        inlier_mask
    )

    if maximum == 0 or n <= maximum:
        return np.arange(
            n,
            dtype=np.int64,
        )

    inlier_indices = np.flatnonzero(
        inlier_mask
    )
    outlier_indices = np.flatnonzero(
        ~inlier_mask
    )

    outlier_budget = min(
        len(outlier_indices),
        max(
            1,
            maximum // 3,
        ),
    )
    inlier_budget = (
        maximum
        - outlier_budget
    )

    def best(
        indices: np.ndarray,
        count: int,
    ) -> np.ndarray:
        if count <= 0 or len(indices) == 0:
            return np.empty(
                0,
                dtype=np.int64,
            )

        if len(indices) <= count:
            return indices

        ranked = indices[
            np.argsort(
                scores[
                    indices
                ]
            )[::-1]
        ]
        return ranked[
            :count
        ]

    chosen = np.concatenate(
        [
            best(
                inlier_indices,
                inlier_budget,
            ),
            best(
                outlier_indices,
                outlier_budget,
            ),
        ]
    )

    return chosen


def render_registration_diagnostic(
    reference_frame: np.ndarray,
    current_frame: np.ndarray,
    frame_index: int,
    reference_index: int,
    result: dict | None,
    width: int,
    height: int,
    viz_width: int,
    viz_height: int,
    header_height: int,
    args: argparse.Namespace,
) -> np.ndarray:
    reference_small = cv2.resize(
        reference_frame,
        (
            viz_width,
            viz_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    current_small = cv2.resize(
        current_frame,
        (
            viz_width,
            viz_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (
            header_height
            + viz_height,
            viz_width * 2,
            3,
        ),
        AQUA_950,
        dtype=np.uint8,
    )

    canvas[
        header_height:,
        :viz_width,
    ] = reference_small
    canvas[
        header_height:,
        viz_width:,
    ] = current_small

    sx_viz = (
        viz_width
        / width
    )
    sy_viz = (
        viz_height
        / height
    )

    status = "REFERENCE"
    status_color = AQUA_300

    if result is not None:
        status = (
            "VALID"
            if result["valid"]
            else "REJECTED"
        )
        status_color = (
            AQUA_300
            if result["valid"]
            else ERROR_RED
        )

        points_ref = result[
            "reference_points_original"
        ]
        points_cur = result[
            "current_points_original"
        ]
        inliers = result[
            "inlier_mask"
        ]
        scores = result[
            "scores"
        ]

        chosen = select_visualization_matches(
            inliers,
            scores,
            args.viz_max_matches,
        )

        overlay = canvas.copy()

        for index in chosen:
            ref_x, ref_y = points_ref[
                index
            ]
            cur_x, cur_y = points_cur[
                index
            ]

            p_ref = (
                int(
                    round(
                        ref_x
                        * sx_viz
                    )
                ),
                int(
                    round(
                        header_height
                        + ref_y
                        * sy_viz
                    )
                ),
            )
            p_cur = (
                int(
                    round(
                        viz_width
                        + cur_x
                        * sx_viz
                    )
                ),
                int(
                    round(
                        header_height
                        + cur_y
                        * sy_viz
                    )
                ),
            )

            color = (
                AQUA_300
                if bool(
                    inliers[
                        index
                    ]
                )
                else ERROR_RED
            )

            cv2.line(
                overlay,
                p_ref,
                p_cur,
                color,
                args.viz_line_thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                p_ref,
                args.viz_point_radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                p_cur,
                args.viz_point_radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

        canvas = cv2.addWeighted(
            overlay,
            0.88,
            canvas,
            0.12,
            0.0,
        )

        if result["H"] is not None:
            corners = image_corners(
                width,
                height,
            )
            projected = warp_points(
                corners,
                result["H"],
            )

            polygon = np.round(
                np.column_stack(
                    [
                        projected[
                            :,
                            0,
                        ]
                        * sx_viz,
                        header_height
                        + projected[
                            :,
                            1,
                        ]
                        * sy_viz,
                    ]
                )
            ).astype(
                np.int32
            )

            cv2.polylines(
                canvas,
                [
                    polygon.reshape(
                        -1,
                        1,
                        2,
                    )
                ],
                isClosed=True,
                color=AQUA_500,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            dx, dy, rotation, local_scale = transform_motion_stats(
                result["H"],
                width,
                height,
            )

            center_reference = np.array(
                [
                    [
                        (width - 1) * 0.5,
                        (height - 1) * 0.5,
                    ]
                ],
                dtype=np.float32,
            )
            center_warped = warp_points(
                center_reference,
                result["H"],
            )[0]

            arrow_start = (
                int(
                    round(
                        center_reference[
                            0,
                            0,
                        ]
                        * sx_viz
                    )
                ),
                int(
                    round(
                        header_height
                        + center_reference[
                            0,
                            1,
                        ]
                        * sy_viz
                    )
                ),
            )
            arrow_end = (
                int(
                    round(
                        center_warped[
                            0
                        ]
                        * sx_viz
                    )
                ),
                int(
                    round(
                        header_height
                        + center_warped[
                            1
                        ]
                        * sy_viz
                    )
                ),
            )

            cv2.arrowedLine(
                canvas,
                arrow_start,
                arrow_end,
                WARNING_AMBER,
                2,
                line_type=cv2.LINE_AA,
                tipLength=0.20,
            )

            motion_text = (
                f"correction dx={dx:+.2f}px  dy={dy:+.2f}px  "
                f"rot={rotation:+.3f}deg  local-scale={local_scale:.6f}"
            )
        else:
            motion_text = "correction unavailable"

        quality_text = (
            f"matches={result['matches']}  "
            f"inliers={result['inliers']}  "
            f"ratio={result['inlier_ratio']:.3f}  "
            f"rms={result['rms']:.3f}px"
            if math.isfinite(
                result["rms"]
            )
            else (
                f"matches={result['matches']}  "
                f"inliers={result['inliers']}  "
                f"ratio={result['inlier_ratio']:.3f}  rms=n/a"
            )
        )

        reason_text = result[
            "reason"
        ]
    else:
        motion_text = (
            "identity reference frame"
        )
        quality_text = (
            "direct fixed reference"
        )
        reason_text = "ok"

    put_text(
        canvas,
        (
            f"Fixed-reference stabilization | "
            f"reference={reference_index} | frame={frame_index} | {status}"
        ),
        (
            16,
            26,
        ),
        scale=0.58,
        color=status_color,
        thickness=2,
    )

    put_text(
        canvas,
        quality_text,
        (
            16,
            50,
        ),
        scale=0.50,
        color=WHITE,
    )

    put_text(
        canvas,
        motion_text,
        (
            16,
            72,
        ),
        scale=0.47,
        color=AQUA_100,
    )

    if header_height >= 104:
        put_text(
            canvas,
            f"reason: {reason_text}",
            (
                16,
                94,
            ),
            scale=0.45,
            color=(
                AQUA_200
                if status != "REJECTED"
                else ERROR_RED
            ),
        )

    put_text(
        canvas,
        f"Reference {reference_index}",
        (
            12,
            header_height + 25,
        ),
        scale=0.55,
        color=WHITE,
        thickness=2,
    )

    put_text(
        canvas,
        f"Current {frame_index}",
        (
            viz_width + 12,
            header_height + 25,
        ),
        scale=0.55,
        color=WHITE,
        thickness=2,
    )

    cv2.line(
        canvas,
        (
            viz_width,
            header_height,
        ),
        (
            viz_width,
            header_height + viz_height - 1,
        ),
        AQUA_300,
        2,
        lineType=cv2.LINE_AA,
    )

    return canvas


def update_track_histories(
    histories: dict[int, deque],
    allowed_ids: set[int],
    result: dict,
    maximum_ids: int,
    track_length: int,
) -> None:
    if not result["valid"]:
        return

    ids = result[
        "reference_indices"
    ]
    points = result[
        "current_points_original"
    ]
    inliers = result[
        "inlier_mask"
    ]

    for ref_id, point, inlier in zip(
        ids,
        points,
        inliers,
    ):
        if not bool(
            inlier
        ):
            continue

        key = int(
            ref_id
        )

        if key not in allowed_ids:
            if (
                maximum_ids != 0
                and len(
                    allowed_ids
                ) >= maximum_ids
            ):
                continue

            allowed_ids.add(
                key
            )

        history = histories.get(
            key
        )

        if history is None:
            history = deque(
                maxlen=track_length
            )
            histories[
                key
            ] = history

        history.append(
            (
                float(
                    point[0]
                ),
                float(
                    point[1]
                ),
            )
        )


def render_tracks(
    frame: np.ndarray,
    histories: dict[int, deque],
    frame_index: int,
    reference_index: int,
    width: int,
    height: int,
    viz_width: int,
    viz_height: int,
    args: argparse.Namespace,
) -> np.ndarray:
    canvas = cv2.resize(
        frame,
        (
            viz_width,
            viz_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    overlay = canvas.copy()

    sx = (
        viz_width
        / width
    )
    sy = (
        viz_height
        / height
    )

    palette = (
        AQUA_300,
        AQUA_400,
        AQUA_500,
        AQUA_200,
        AQUA_600,
    )

    active = 0

    for ref_id, history in histories.items():
        if not history:
            continue

        points = np.array(
            history,
            dtype=np.float32,
        )

        points[
            :,
            0,
        ] *= sx
        points[
            :,
            1,
        ] *= sy

        points_int = np.round(
            points
        ).astype(
            np.int32
        )

        color = palette[
            ref_id
            % len(
                palette
            )
        ]

        if len(
            points_int
        ) >= 2:
            cv2.polylines(
                overlay,
                [
                    points_int.reshape(
                        -1,
                        1,
                        2,
                    )
                ],
                isClosed=False,
                color=color,
                thickness=args.tracks_line_thickness,
                lineType=cv2.LINE_AA,
            )

        p = tuple(
            points_int[
                -1
            ]
        )

        cv2.circle(
            overlay,
            p,
            args.tracks_point_radius,
            AQUA_100,
            -1,
            lineType=cv2.LINE_AA,
        )

        active += 1

    canvas = cv2.addWeighted(
        overlay,
        0.86,
        canvas,
        0.14,
        0.0,
    )

    panel_height = 54
    panel = canvas.copy()
    cv2.rectangle(
        panel,
        (
            0,
            0,
        ),
        (
            viz_width,
            panel_height,
        ),
        AQUA_950,
        -1,
    )
    canvas = cv2.addWeighted(
        panel,
        0.82,
        canvas,
        0.18,
        0.0,
    )

    put_text(
        canvas,
        (
            f"Background SuperPoint tracks | frame={frame_index} | "
            f"reference={reference_index}"
        ),
        (
            14,
            23,
        ),
        scale=0.53,
        color=WHITE,
        thickness=2,
    )

    put_text(
        canvas,
        (
            f"active reference IDs={active} | "
            f"history={args.track_length} sampled observations"
        ),
        (
            14,
            45,
        ),
        scale=0.45,
        color=AQUA_100,
    )

    return canvas


# ============================================================================
# Absolute transform interpolation / smoothing
# ============================================================================

def moving_average_1d(
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return values.copy()

    n = len(
        values
    )

    if n == 0:
        return values.copy()

    padded = np.pad(
        values,
        (
            radius,
            radius,
        ),
        mode="edge",
    )

    kernel = np.ones(
        2 * radius + 1,
        dtype=np.float64,
    )
    kernel /= kernel.sum()

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid",
    )

    return smoothed[
        :n
    ]


def build_full_absolute_transforms(
    num_frames: int,
    sample_indices: np.ndarray,
    sample_transforms: np.ndarray,
    sample_valid: np.ndarray,
    width: int,
    height: int,
    motion_model: str,
    smooth_radius: int,
    reference_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_indices = sample_indices[
        sample_valid
    ]

    valid_transforms = sample_transforms[
        sample_valid
    ]

    if len(
        valid_indices
    ) == 0:
        raise RuntimeError(
            "No valid absolute reference registrations exist."
        )

    source_corners = image_corners(
        width,
        height,
    )

    valid_corner_positions = np.stack(
        [
            warp_points(
                source_corners,
                H,
            )
            for H in valid_transforms
        ],
        axis=0,
    )

    all_frames = np.arange(
        num_frames,
        dtype=np.float64,
    )

    full_corners = np.empty(
        (
            num_frames,
            4,
            2,
        ),
        dtype=np.float64,
    )

    # Absolute interpolation only: each coordinate is interpolated directly
    # between valid reference registrations. No incremental transforms.
    for corner_index in range(
        4
    ):
        for coordinate in range(
            2
        ):
            full_corners[
                :,
                corner_index,
                coordinate,
            ] = np.interp(
                all_frames,
                valid_indices.astype(
                    np.float64
                ),
                valid_corner_positions[
                    :,
                    corner_index,
                    coordinate,
                ],
            )

    if smooth_radius > 0:
        smoothed = np.empty_like(
            full_corners
        )

        for corner_index in range(
            4
        ):
            for coordinate in range(
                2
            ):
                smoothed[
                    :,
                    corner_index,
                    coordinate,
                ] = moving_average_1d(
                    full_corners[
                        :,
                        corner_index,
                        coordinate,
                    ],
                    smooth_radius,
                )

        full_corners = smoothed

        # Keep the selected reference frame exactly fixed.
        if 0 <= reference_frame < num_frames:
            full_corners[
                reference_frame
            ] = source_corners

    transforms = np.empty(
        (
            num_frames,
            3,
            3,
        ),
        dtype=np.float64,
    )

    for frame_index in range(
        num_frames
    ):
        transforms[
            frame_index
        ] = corners_to_transform(
            source_corners,
            full_corners[
                frame_index
            ].astype(
                np.float32
            ),
            motion_model,
        )

    return (
        transforms,
        full_corners,
    )


def maximum_anchor_gap(
    sample_indices: np.ndarray,
    sample_valid: np.ndarray,
) -> int:
    valid_indices = sample_indices[
        sample_valid
    ]

    if len(
        valid_indices
    ) < 2:
        return 0

    return int(
        np.max(
            np.diff(
                valid_indices
            )
        )
    )


# ============================================================================
# Common stable crop
# ============================================================================

def convex_common_region(
    transforms: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    base = image_corners(
        width,
        height,
    )

    common = cv2.convexHull(
        base
    ).reshape(
        -1,
        2,
    ).astype(
        np.float32
    )

    for H in transforms:
        warped = warp_points(
            base,
            H,
        ).astype(
            np.float32
        )

        warped_hull = cv2.convexHull(
            warped
        ).reshape(
            -1,
            2,
        ).astype(
            np.float32
        )

        area, intersection = cv2.intersectConvexConvex(
            common,
            warped_hull,
        )

        if (
            intersection is None
            or area <= 0
        ):
            raise RuntimeError(
                "No common valid stabilization region remains."
            )

        common = cv2.convexHull(
            intersection
        ).reshape(
            -1,
            2,
        ).astype(
            np.float32
        )

    return common


def largest_rectangle_in_binary(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Maximum axis-aligned rectangle of nonzero pixels.
    Returns x, y, width, height.
    """
    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    rows, cols = binary.shape
    heights = np.zeros(
        cols,
        dtype=np.int32,
    )

    best_area = 0
    best_rect = (
        0,
        0,
        0,
        0,
    )

    for row in range(
        rows
    ):
        heights = np.where(
            binary[
                row
            ] > 0,
            heights + 1,
            0,
        )

        stack = []

        for col in range(
            cols + 1
        ):
            current_height = (
                int(
                    heights[
                        col
                    ]
                )
                if col < cols
                else 0
            )

            start = col

            while (
                stack
                and stack[
                    -1
                ][
                    1
                ] > current_height
            ):
                start_index, popped_height = stack.pop()
                area = (
                    popped_height
                    * (
                        col
                        - start_index
                    )
                )

                if area > best_area:
                    best_area = area
                    x = start_index
                    y = (
                        row
                        - popped_height
                        + 1
                    )
                    w = (
                        col
                        - start_index
                    )
                    h = popped_height
                    best_rect = (
                        x,
                        y,
                        w,
                        h,
                    )

                start = start_index

            if (
                not stack
                or stack[
                    -1
                ][
                    1
                ] < current_height
            ):
                stack.append(
                    (
                        start,
                        current_height,
                    )
                )

    return best_rect


def compute_common_crop(
    transforms: np.ndarray,
    width: int,
    height: int,
    mask_scale: float,
    safety: int,
) -> tuple[int, int, int, int, np.ndarray]:
    common_polygon = convex_common_region(
        transforms,
        width,
        height,
    )

    mask_width = max(
        2,
        int(
            math.ceil(
                width
                * mask_scale
            )
        ),
    )
    mask_height = max(
        2,
        int(
            math.ceil(
                height
                * mask_scale
            )
        ),
    )

    sx = (
        mask_width
        / width
    )
    sy = (
        mask_height
        / height
    )

    polygon_small = common_polygon.copy()
    polygon_small[
        :,
        0,
    ] *= sx
    polygon_small[
        :,
        1,
    ] *= sy

    polygon_small = np.round(
        polygon_small
    ).astype(
        np.int32
    )

    mask = np.zeros(
        (
            mask_height,
            mask_width,
        ),
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        mask,
        polygon_small,
        255,
        lineType=cv2.LINE_8,
    )

    x_small, y_small, w_small, h_small = largest_rectangle_in_binary(
        mask
    )

    if w_small <= 0 or h_small <= 0:
        raise RuntimeError(
            "Could not determine a non-empty common stable crop."
        )

    # Move inward conservatively when mapping to full resolution.
    x0 = int(
        math.ceil(
            x_small
            / sx
        )
    )
    y0 = int(
        math.ceil(
            y_small
            / sy
        )
    )
    x1 = int(
        math.floor(
            (
                x_small
                + w_small
            )
            / sx
        )
    )
    y1 = int(
        math.floor(
            (
                y_small
                + h_small
            )
            / sy
        )
    )

    x0 += safety
    y0 += safety
    x1 -= safety
    y1 -= safety

    x0 = max(
        0,
        min(
            width - 1,
            x0,
        ),
    )
    y0 = max(
        0,
        min(
            height - 1,
            y0,
        ),
    )
    x1 = max(
        x0 + 1,
        min(
            width,
            x1,
        ),
    )
    y1 = max(
        y0 + 1,
        min(
            height,
            y1,
        ),
    )

    crop_width = (
        x1 - x0
    )
    crop_height = (
        y1 - y0
    )

    # yuv420p and many hardware encoders require even dimensions.
    if crop_width % 2:
        crop_width -= 1
    if crop_height % 2:
        crop_height -= 1

    if crop_width < 2 or crop_height < 2:
        raise RuntimeError(
            "Stable crop became too small after safety/even-size adjustment."
        )

    return (
        x0,
        y0,
        crop_width,
        crop_height,
        common_polygon,
    )


def output_coordinate_transforms(
    border_mode: str,
    crop_rect: tuple[int, int, int, int],
    width: int,
    height: int,
    crop_resize_original: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Return:
        reference_to_output
        output_to_reference
        output_width
        output_height
    """
    if border_mode == "black":
        reference_to_output = np.eye(
            3,
            dtype=np.float64,
        )
        output_to_reference = np.eye(
            3,
            dtype=np.float64,
        )
        return (
            reference_to_output,
            output_to_reference,
            width,
            height,
        )

    x, y, crop_width, crop_height = crop_rect

    crop_matrix = np.array(
        [
            [1.0, 0.0, -float(x)],
            [0.0, 1.0, -float(y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if not crop_resize_original:
        reference_to_output = crop_matrix
        output_to_reference = np.linalg.inv(
            reference_to_output
        )
        return (
            reference_to_output,
            output_to_reference,
            crop_width,
            crop_height,
        )

    resize_matrix = np.array(
        [
            [
                width
                / crop_width,
                0.0,
                0.0,
            ],
            [
                0.0,
                height
                / crop_height,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    reference_to_output = (
        resize_matrix
        @ crop_matrix
    )
    output_to_reference = np.linalg.inv(
        reference_to_output
    )

    return (
        reference_to_output,
        output_to_reference,
        width,
        height,
    )


# ============================================================================
# Main pipeline
# ============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(
        parser,
        args,
    )

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input video not found: {args.input}"
        )

    ffmpeg = resolve_ffmpeg(
        args.ffmpeg
    )

    run_dir = (
        args.runs_dir
        / args.run_name
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output
        if args.output is not None
        else run_dir / "stabilized.mp4"
    )
    transforms_path = (
        args.transforms_file
        if args.transforms_file is not None
        else run_dir / "stabilization.npz"
    )
    metadata_path = (
        args.metadata_file
        if args.metadata_file is not None
        else run_dir / "stabilization.yaml"
    )
    diagnostic_path = (
        args.diagnostic_output
        if args.diagnostic_output is not None
        else run_dir / "motion_inliers_outliers.mp4"
    )
    tracks_path = (
        args.tracks_output
        if args.tracks_output is not None
        else run_dir / "tracks.mp4"
    )

    for path in (
        output_path,
        transforms_path,
        metadata_path,
        diagnostic_path,
        tracks_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    torch.manual_seed(
        args.torch_seed
    )
    cv2.setRNGSeed(
        args.opencv_seed
    )
    torch.set_float32_matmul_precision(
        args.matmul_precision
    )

    devices = resolve_devices(
        args.devices
    )

    metadata = get_video_metadata(
        args.input
    )
    width = metadata[
        "width"
    ]
    height = metadata[
        "height"
    ]
    fps = metadata[
        "fps"
    ]
    reported_frames = metadata[
        "reported_frames"
    ]

    if (
        reported_frames > 0
        and args.reference_frame >= reported_frames
    ):
        raise IndexError(
            f"Reference frame {args.reference_frame} is outside "
            f"reported frame count {reported_frames}."
        )

    print("=" * 78)
    print("FIXED-REFERENCE NADIR VIDEO STABILIZATION")
    print("=" * 78)
    print("Input:", args.input)
    print("Run:", args.run_name)
    print("Resolution:", f"{width}x{height}")
    print("FPS:", f"{fps:.6f}")
    print("Reported frames:", reported_frames)
    print("Reference frame:", args.reference_frame)
    print("Match step:", args.step)
    print("Inference scale:", args.scale)
    print("Motion model:", args.motion_model)
    print("Smooth radius:", args.smooth)
    print("Border mode:", args.border_mode)
    print(
        "Devices:",
        " ".join(
            str(
                device
            )
            for device in devices
        ),
    )
    print(
        "Workers per device:",
        args.gpu_workers,
    )
    print(
        "Total inference workers:",
        len(
            devices
        )
        * args.gpu_workers,
    )
    print(
        "Prefetch per worker:",
        args.prefetch,
    )
    print(
        "Pinned host memory:",
        args.pin_memory,
    )
    print()

    # ----------------------------------------------------------------------
    # Load reference frame. Each persistent worker builds its own model pair
    # and extracts its own device-local copy of the fixed reference features.
    # ----------------------------------------------------------------------
    print("Loading fixed reference frame...")
    reference_frame = read_reference_frame(
        args.input,
        args.reference_frame,
    )

    reference_scaled, reference_sx, reference_sy = resize_for_inference(
        reference_frame,
        args.scale,
    )

    print(
        "Inference resolution:",
        f"{reference_scaled.shape[1]}x{reference_scaled.shape[0]}",
    )

    print(
        "Starting persistent SuperPoint + LightGlue workers..."
    )

    (
        workers,
        inference_task_queue,
        inference_result_queue,
        worker_stop_event,
        worker_ready,
        reference_keypoint_count,
    ) = start_registration_workers(
        devices=devices,
        args=args,
        reference_scaled=reference_scaled,
        width=width,
        height=height,
    )

    print(
        "Reference keypoints:",
        reference_keypoint_count,
    )

    for worker_id in sorted(
        worker_ready
    ):
        info = worker_ready[
            worker_id
        ]

        print(
            f"  worker {worker_id}: "
            f"{info['device']} "
            f"(reference keypoints="
            f"{info['reference_keypoints']})"
        )

    print()

    # ----------------------------------------------------------------------
    # Visualization writers for sampled registrations
    # ----------------------------------------------------------------------
    viz_width, viz_height = visualization_size(
        width,
        height,
        args.viz_scale,
    )

    viz_header = even_dimension(
        args.viz_header,
        minimum=0,
    )

    diagnostic_writer = None
    tracks_writer = None

    sample_video_fps = max(
        0.1,
        fps
        / args.step,
    )

    if args.diagnostic:
        diagnostic_writer = FFmpegRawWriter(
            ffmpeg=ffmpeg,
            output_path=diagnostic_path,
            width=viz_width * 2,
            height=viz_header + viz_height,
            fps=sample_video_fps,
            codec=args.viz_codec,
            crf=args.diagnostic_crf,
            preset=args.viz_preset,
            pixel_format=args.viz_pixel_format,
            loglevel=args.ffmpeg_loglevel,
        )

    if args.tracks:
        tracks_writer = FFmpegRawWriter(
            ffmpeg=ffmpeg,
            output_path=tracks_path,
            width=viz_width,
            height=viz_height,
            fps=sample_video_fps,
            codec=args.viz_codec,
            crf=args.tracks_crf,
            preset=args.viz_preset,
            pixel_format=args.viz_pixel_format,
            loglevel=args.ffmpeg_loglevel,
        )

    track_histories: dict[int, deque] = {}
    allowed_track_ids: set[int] = set()

    # ----------------------------------------------------------------------
    # First pass: directly match sampled frames to the fixed reference.
    #
    # The CPU decoder feeds one bounded shared queue. Persistent workers
    # dynamically take jobs, so both GPUs remain busy without assigning a
    # fixed frame sequence to either GPU. Results may finish out of order,
    # but they are committed strictly by frame index before diagnostics,
    # tracks, and transform interpolation.
    # ----------------------------------------------------------------------
    print("=" * 78)
    print("PASS 1/2: PARALLEL DIRECT REFERENCE REGISTRATION")
    print("=" * 78)

    cap = cv2.VideoCapture(
        str(args.input)
    )

    if not cap.isOpened():
        stop_registration_workers(
            workers,
            inference_task_queue,
            worker_stop_event,
            graceful=False,
        )
        raise RuntimeError(
            f"Could not open video: {args.input}"
        )

    sample_records = []
    sample_order = []
    pending_results = {}
    commit_position = 0
    submitted_jobs = 0
    received_jobs = 0

    frame_index = 0
    last_frame = None
    last_frame_index = -1
    sampled_indices_set = set()

    total_workers = len(
        workers
    )

    # Bound not only the task queue but also how far ordered commit may lag.
    # This prevents a single slow early frame from allowing later completed
    # full-resolution frames to accumulate without limit in RAM.
    max_uncommitted_samples = max(
        2,
        total_workers
        * (
            args.prefetch
            + 1
        ),
    )

    pass_completed = False

    def commit_ready_samples():
        nonlocal commit_position

        while commit_position < len(
            sample_order
        ):
            next_frame_index = sample_order[
                commit_position
            ]

            package = pending_results.get(
                next_frame_index
            )

            if package is None:
                break

            del pending_results[
                next_frame_index
            ]

            current_frame = package[
                "frame"
            ]
            visual_result = package[
                "result"
            ]
            worker_id = int(
                package[
                    "worker_id"
                ]
            )
            worker_device = str(
                package[
                    "device"
                ]
            )

            record = registration_result_to_record(
                frame_index=next_frame_index,
                result=visual_result,
                reference_keypoint_count=reference_keypoint_count,
                worker_id=worker_id,
                worker_device=worker_device,
            )

            sample_records.append(
                record
            )

            if diagnostic_writer is not None:
                diagnostic_frame = render_registration_diagnostic(
                    reference_frame=reference_frame,
                    current_frame=current_frame,
                    frame_index=next_frame_index,
                    reference_index=args.reference_frame,
                    result=visual_result,
                    width=width,
                    height=height,
                    viz_width=viz_width,
                    viz_height=viz_height,
                    header_height=viz_header,
                    args=args,
                )
                diagnostic_writer.write(
                    diagnostic_frame
                )

            if (
                tracks_writer is not None
                and visual_result is not None
            ):
                update_track_histories(
                    histories=track_histories,
                    allowed_ids=allowed_track_ids,
                    result=visual_result,
                    maximum_ids=args.tracks_max_points,
                    track_length=args.track_length,
                )

                tracks_frame = render_tracks(
                    frame=current_frame,
                    histories=track_histories,
                    frame_index=next_frame_index,
                    reference_index=args.reference_frame,
                    width=width,
                    height=height,
                    viz_width=viz_width,
                    viz_height=viz_height,
                    args=args,
                )
                tracks_writer.write(
                    tracks_frame
                )

            if (
                args.progress_every > 0
                and (
                    next_frame_index
                    % args.progress_every
                    == 0
                )
            ):
                print(
                    f"frame={next_frame_index:8d}  "
                    f"worker={worker_id:2d}  "
                    f"device={worker_device:8s}  "
                    f"valid={record['valid']}  "
                    f"matches={record['matches']:5d}  "
                    f"inliers={record['inliers']:5d}  "
                    f"ratio={record['inlier_ratio']:.3f}  "
                    f"rms={record['rms']:.3f}"
                )

            commit_position += 1

    def receive_one_result(
        block: bool,
    ) -> bool:
        nonlocal received_jobs

        try:
            if block:
                message = inference_result_queue.get()
            else:
                message = inference_result_queue.get_nowait()

        except queue.Empty:
            return False

        if message[
            "kind"
        ] == "error":
            raise RuntimeError(
                "Inference worker failed while processing a frame:\n"
                f"worker={message['worker_id']} "
                f"device={message['device']} "
                f"frame={message['frame_index']}\n"
                f"{message['error']}\n"
                f"{message['traceback']}"
            )

        current_frame_index = int(
            message[
                "frame_index"
            ]
        )

        if current_frame_index in pending_results:
            raise RuntimeError(
                f"Duplicate worker result for frame {current_frame_index}."
            )

        pending_results[
            current_frame_index
        ] = message

        received_jobs += 1

        commit_ready_samples()

        return True

    def drain_available_results():
        while receive_one_result(
            block=False
        ):
            pass

    def wait_until_capacity():
        while (
            len(
                sample_order
            )
            - commit_position
        ) >= max_uncommitted_samples:
            receive_one_result(
                block=True
            )

    def enqueue_sample(
        sample_frame_index: int,
        sample_frame: np.ndarray,
    ):
        nonlocal submitted_jobs

        wait_until_capacity()

        sample_order.append(
            int(
                sample_frame_index
            )
        )
        sampled_indices_set.add(
            int(
                sample_frame_index
            )
        )

        if (
            sample_frame_index
            == args.reference_frame
        ):
            pending_results[
                sample_frame_index
            ] = {
                "kind": "result",
                "worker_id": -1,
                "device": "reference",
                "frame_index": int(
                    sample_frame_index
                ),
                "frame": sample_frame,
                "result": None,
            }

            commit_ready_samples()

        else:
            inference_task_queue.put(
                (
                    int(
                        sample_frame_index
                    ),
                    sample_frame,
                )
            )

            submitted_jobs += 1

        drain_available_results()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            last_frame = frame.copy()
            last_frame_index = frame_index

            should_sample = (
                (
                    frame_index
                    - args.reference_frame
                )
                % args.step
                == 0
            ) or (
                frame_index
                == args.reference_frame
            )

            if should_sample:
                enqueue_sample(
                    frame_index,
                    frame,
                )

            frame_index += 1

        num_frames = frame_index

        if num_frames <= 0:
            raise RuntimeError(
                "Input video contains no decodable frames."
            )

        if args.reference_frame >= num_frames:
            raise IndexError(
                f"Reference frame {args.reference_frame} is outside "
                f"actual decoded frame count {num_frames}."
            )

        # Force a direct absolute reference registration on the final frame.
        if (
            last_frame_index >= 0
            and last_frame_index
            not in sampled_indices_set
        ):
            enqueue_sample(
                last_frame_index,
                last_frame,
            )

        # All jobs are submitted. Finish receiving and commit strictly in
        # temporal sample order, regardless of which GPU completed first.
        while commit_position < len(
            sample_order
        ):
            if not receive_one_result(
                block=False
            ):
                receive_one_result(
                    block=True
                )

        if received_jobs != submitted_jobs:
            raise RuntimeError(
                f"Received {received_jobs} inference results but "
                f"submitted {submitted_jobs} jobs."
            )

        pass_completed = True

    finally:
        cap.release()

        try:
            stop_registration_workers(
                workers,
                inference_task_queue,
                worker_stop_event,
                graceful=pass_completed,
            )
        finally:
            if diagnostic_writer is not None:
                diagnostic_writer.close()

            if tracks_writer is not None:
                tracks_writer.close()


    # Sort because reference-frame sampling or final insertion should never
    # be allowed to alter temporal anchor order.
    sample_records.sort(
        key=lambda item: item[
            "frame_index"
        ]
    )

    sample_indices = np.asarray(
        [
            item[
                "frame_index"
            ]
            for item in sample_records
        ],
        dtype=np.int32,
    )

    sample_valid = np.asarray(
        [
            item[
                "valid"
            ]
            for item in sample_records
        ],
        dtype=bool,
    )

    sample_transforms = np.stack(
        [
            item[
                "H"
            ]
            for item in sample_records
        ],
        axis=0,
    ).astype(
        np.float64,
        copy=False,
    )

    sample_matches = np.asarray(
        [
            item[
                "matches"
            ]
            for item in sample_records
        ],
        dtype=np.int32,
    )

    sample_inliers = np.asarray(
        [
            item[
                "inliers"
            ]
            for item in sample_records
        ],
        dtype=np.int32,
    )

    sample_inlier_ratio = np.asarray(
        [
            item[
                "inlier_ratio"
            ]
            for item in sample_records
        ],
        dtype=np.float32,
    )

    sample_rms = np.asarray(
        [
            item[
                "rms"
            ]
            for item in sample_records
        ],
        dtype=np.float32,
    )

    sample_reasons = np.asarray(
        [
            item[
                "reason"
            ]
            for item in sample_records
        ],
        dtype=np.str_,
    )

    sample_worker_ids = np.asarray(
        [
            item.get(
                "worker_id",
                -1,
            )
            for item in sample_records
        ],
        dtype=np.int32,
    )

    sample_worker_devices = np.asarray(
        [
            item.get(
                "worker_device",
                "unknown",
            )
            for item in sample_records
        ],
        dtype=np.str_,
    )

    failed_samples = int(
        (
            ~sample_valid
        ).sum()
    )

    valid_samples = int(
        sample_valid.sum()
    )

    print()
    print(
        "Decoded frames:",
        num_frames,
    )
    print(
        "Direct-reference samples:",
        len(
            sample_indices
        ),
    )
    print(
        "Valid samples:",
        valid_samples,
    )
    print(
        "Rejected samples:",
        failed_samples,
    )

    if args.strict and failed_samples:
        rejected = [
            (
                int(
                    sample_indices[
                        index
                    ]
                ),
                str(
                    sample_reasons[
                        index
                    ]
                ),
            )
            for index in np.flatnonzero(
                ~sample_valid
            )
        ]
        raise RuntimeError(
            "Strict mode: direct reference registration failed: "
            + "; ".join(
                f"frame {index}: {reason}"
                for index, reason in rejected
            )
        )

    gap = maximum_anchor_gap(
        sample_indices,
        sample_valid,
    )

    print(
        "Maximum valid-anchor gap:",
        gap,
        "frames",
    )

    if (
        args.max_interpolation_gap > 0
        and gap > args.max_interpolation_gap
    ):
        raise RuntimeError(
            f"Maximum valid-anchor gap is {gap} frames, exceeding "
            f"--max-interpolation-gap={args.max_interpolation_gap}."
        )

    # ----------------------------------------------------------------------
    # Expand absolute reference registration to every frame.
    # ----------------------------------------------------------------------
    print()
    print("Building per-frame ABSOLUTE transforms...")

    (
        transforms_current_to_reference,
        warped_corners,
    ) = build_full_absolute_transforms(
        num_frames=num_frames,
        sample_indices=sample_indices,
        sample_transforms=sample_transforms,
        sample_valid=sample_valid,
        width=width,
        height=height,
        motion_model=args.motion_model,
        smooth_radius=args.smooth,
        reference_frame=args.reference_frame,
    )

    # ----------------------------------------------------------------------
    # Crop / output coordinate system.
    # ----------------------------------------------------------------------
    if args.border_mode == "crop":
        print("Computing common stable crop...")

        (
            crop_x,
            crop_y,
            crop_width,
            crop_height,
            common_polygon,
        ) = compute_common_crop(
            transforms=transforms_current_to_reference,
            width=width,
            height=height,
            mask_scale=args.crop_mask_scale,
            safety=args.crop_safety,
        )

        crop_rect = (
            crop_x,
            crop_y,
            crop_width,
            crop_height,
        )

        retained_fraction = (
            crop_width
            * crop_height
            / (
                width
                * height
            )
        )

        print(
            "Stable crop:",
            crop_rect,
        )
        print(
            "Crop retained area:",
            f"{retained_fraction * 100:.2f}%",
        )

    else:
        crop_rect = (
            0,
            0,
            width,
            height,
        )
        common_polygon = image_corners(
            width,
            height,
        )
        retained_fraction = 1.0

    (
        reference_to_output,
        output_to_reference,
        output_width,
        output_height,
    ) = output_coordinate_transforms(
        border_mode=args.border_mode,
        crop_rect=crop_rect,
        width=width,
        height=height,
        crop_resize_original=args.crop_resize_original,
    )

    raw_to_output = np.einsum(
        "ij,tjk->tik",
        reference_to_output,
        transforms_current_to_reference,
    )

    print(
        "Output resolution:",
        f"{output_width}x{output_height}",
    )

    if (
        args.pixel_format
        in (
            "yuv420p",
            "yuv422p",
        )
        and (
            output_width % 2
            or output_height % 2
        )
    ):
        raise RuntimeError(
            f"{args.pixel_format} generally requires even dimensions, "
            f"but output is {output_width}x{output_height}. "
            "Use --pixel-format yuv444p, --crop, or an even source size."
        )

    # ----------------------------------------------------------------------
    # Second pass: original frame -> absolute warp -> fixed crop.
    # ----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("PASS 2/2: RENDER STABILIZED VIDEO")
    print("=" * 78)

    cap = cv2.VideoCapture(
        str(args.input)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not reopen video: {args.input}"
        )

    writer = FFmpegRawWriter(
        ffmpeg=ffmpeg,
        output_path=output_path,
        width=output_width,
        height=output_height,
        fps=fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        pixel_format=args.pixel_format,
        loglevel=args.ffmpeg_loglevel,
        source_audio=args.input,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
    )

    rendered = 0

    try:
        for frame_index in range(
            num_frames
        ):
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Video decode ended unexpectedly at frame {frame_index}; "
                    f"expected {num_frames} frames from pass 1."
                )

            H = transforms_current_to_reference[
                frame_index
            ]

            stabilized = cv2.warpPerspective(
                frame,
                H,
                (
                    width,
                    height,
                ),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(
                    0,
                    0,
                    0,
                ),
            )

            if args.border_mode == "crop":
                x, y, cw, ch = crop_rect

                stabilized = stabilized[
                    y:y + ch,
                    x:x + cw,
                ]

                if args.crop_resize_original:
                    stabilized = cv2.resize(
                        stabilized,
                        (
                            width,
                            height,
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )

            writer.write(
                stabilized
            )
            rendered += 1

            if args.preview:
                if args.preview_scale != 1.0:
                    preview_width = max(
                        1,
                        int(
                            round(
                                stabilized.shape[
                                    1
                                ]
                                * args.preview_scale
                            )
                        ),
                    )
                    preview_height = max(
                        1,
                        int(
                            round(
                                stabilized.shape[
                                    0
                                ]
                                * args.preview_scale
                            )
                        ),
                    )
                    preview = cv2.resize(
                        stabilized,
                        (
                            preview_width,
                            preview_height,
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    preview = stabilized

                cv2.imshow(
                    "Stabilized",
                    preview,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    27,
                ):
                    raise KeyboardInterrupt(
                        "Preview stopped by user."
                    )

            if (
                args.progress_every > 0
                and (
                    frame_index
                    % args.progress_every
                    == 0
                    or frame_index
                    == num_frames - 1
                )
            ):
                print(
                    f"Rendered {frame_index + 1}/{num_frames}"
                )

    finally:
        cap.release()

        if args.preview:
            cv2.destroyAllWindows()

        writer.close()

    if rendered != num_frames:
        raise RuntimeError(
            f"Rendered {rendered} frames but expected {num_frames}."
        )

    # ----------------------------------------------------------------------
    # Save transform data.
    # ----------------------------------------------------------------------
    transforms_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        transforms_path,
        frame_indices=np.arange(
            num_frames,
            dtype=np.int32,
        ),
        transforms_current_to_reference=(
            transforms_current_to_reference
            .astype(
                np.float64,
                copy=False,
            )
        ),
        warped_corners=(
            warped_corners
            .astype(
                np.float32,
                copy=False,
            )
        ),
        raw_to_output=(
            raw_to_output
            .astype(
                np.float64,
                copy=False,
            )
        ),
        reference_to_output=(
            reference_to_output
            .astype(
                np.float64,
                copy=False,
            )
        ),
        output_to_reference=(
            output_to_reference
            .astype(
                np.float64,
                copy=False,
            )
        ),
        crop_rect=np.asarray(
            crop_rect,
            dtype=np.int32,
        ),
        common_polygon=(
            common_polygon
            .astype(
                np.float32,
                copy=False,
            )
        ),
        sample_indices=sample_indices,
        sample_valid=sample_valid,
        sample_transforms_current_to_reference=sample_transforms,
        sample_matches=sample_matches,
        sample_inliers=sample_inliers,
        sample_inlier_ratio=sample_inlier_ratio,
        sample_reprojection_rms=sample_rms,
        sample_reasons=sample_reasons,
        sample_worker_ids=sample_worker_ids,
        sample_worker_devices=sample_worker_devices,
        width=np.int32(
            width
        ),
        height=np.int32(
            height
        ),
        output_width=np.int32(
            output_width
        ),
        output_height=np.int32(
            output_height
        ),
        fps=np.float64(
            fps
        ),
        frames=np.int32(
            num_frames
        ),
        reference_frame=np.int32(
            args.reference_frame
        ),
        step=np.int32(
            args.step
        ),
        inference_scale=np.float32(
            args.scale
        ),
        smooth_radius=np.int32(
            args.smooth
        ),
    )

    # ----------------------------------------------------------------------
    # YAML metadata.
    # ----------------------------------------------------------------------
    metadata_yaml = {
        "run_name": args.run_name,
        "input": {
            "file": str(
                args.input
            ),
            "width": int(
                width
            ),
            "height": int(
                height
            ),
            "fps": float(
                fps
            ),
            "frames": int(
                num_frames
            ),
        },
        "stabilization": {
            "reference_frame": int(
                args.reference_frame
            ),
            "step": int(
                args.step
            ),
            "motion_model": args.motion_model,
            "inference_scale": float(
                args.scale
            ),
            "smooth_radius": int(
                args.smooth
            ),
            "absolute_reference_matching": True,
            "sequential_transform_chaining": False,
            "intermediate_transform_method": (
                "linear interpolation of absolute warped image corners "
                "followed by projection back to the selected motion model"
            ),
        },
        "parallel_inference": {
            "devices": [
                str(
                    device
                )
                for device in devices
            ],
            "workers_per_device": int(
                args.gpu_workers
            ),
            "total_workers": int(
                len(
                    workers
                )
            ),
            "prefetch_per_worker": int(
                args.prefetch
            ),
            "pin_memory": bool(
                args.pin_memory
            ),
            "shared_dynamic_task_queue": True,
            "ordered_result_commit": True,
            "one_model_pair_per_worker": True,
            "one_reference_feature_copy_per_worker": True,
        },
        "superpoint": {
            "max_keypoints": int(
                args.max_keypoints
            ),
            "detection_threshold": float(
                args.detection_threshold
            ),
            "nms_radius": int(
                args.nms_radius
            ),
            "remove_borders": int(
                args.remove_borders
            ),
            "reference_keypoints": int(
                reference_keypoint_count
            ),
        },
        "lightglue": {
            "filter_threshold": float(
                args.lightglue_filter_threshold
            ),
            "depth_confidence": float(
                args.lightglue_depth_confidence
            ),
            "width_confidence": float(
                args.lightglue_width_confidence
            ),
            "mixed_precision": bool(
                args.amp
                and any(
                    device.type == "cuda"
                    for device in devices
                )
            ),
            "compiled": bool(
                args.compile_lightglue
            ),
        },
        "robust_estimation": {
            "ransac_threshold_original_px": float(
                args.ransac_threshold
            ),
            "ransac_confidence": float(
                args.ransac_confidence
            ),
            "ransac_max_iters": int(
                args.ransac_max_iters
            ),
            "min_matches": int(
                args.min_matches
            ),
            "min_inliers": int(
                args.min_inliers
            ),
            "min_inlier_ratio": float(
                args.min_inlier_ratio
            ),
            "max_reprojection_rms_px": float(
                args.max_reprojection_rms
            ),
            "valid_samples": int(
                valid_samples
            ),
            "rejected_samples": int(
                failed_samples
            ),
            "max_valid_anchor_gap_frames": int(
                gap
            ),
        },
        "output_geometry": {
            "border_mode": args.border_mode,
            "crop_rect_xywh": [
                int(
                    value
                )
                for value in crop_rect
            ],
            "crop_retained_area_fraction": float(
                retained_fraction
            ),
            "crop_resize_original": bool(
                args.crop_resize_original
            ),
            "width": int(
                output_width
            ),
            "height": int(
                output_height
            ),
            "reference_to_output": (
                reference_to_output
                .tolist()
            ),
            "output_to_reference": (
                output_to_reference
                .tolist()
            ),
        },
        "coordinate_usage": {
            "raw_to_output_key": "raw_to_output",
            "description": (
                "For raw-frame point [x,y,1] at frame t, multiply by "
                "raw_to_output[t] and homogeneous-normalize to obtain "
                "coordinates in the stabilized output video."
            ),
        },
        "encoding": {
            "codec": args.codec,
            "crf": int(
                args.crf
            ),
            "preset": args.preset,
            "pixel_format": args.pixel_format,
            "audio_codec": args.audio_codec,
        },
        "deliverables": {
            "stabilized_video": str(
                output_path
            ),
            "motion_inliers_outliers_video": (
                str(
                    diagnostic_path
                )
                if args.diagnostic
                else None
            ),
            "tracks_video": (
                str(
                    tracks_path
                )
                if args.tracks
                else None
            ),
            "transforms_npz": str(
                transforms_path
            ),
            "metadata_yaml": str(
                metadata_path
            ),
        },
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            metadata_yaml,
            handle,
            sort_keys=False,
        )

    print()
    print("=" * 78)
    print("STABILIZATION COMPLETE")
    print("=" * 78)
    print("Stabilized:", output_path)

    if args.diagnostic:
        print(
            "Motion / inliers / outliers:",
            diagnostic_path,
        )

    if args.tracks:
        print(
            "Track visualization:",
            tracks_path,
        )

    print(
        "Transforms:",
        transforms_path,
    )
    print(
        "Metadata:",
        metadata_path,
    )
    print()
    print(
        "Inference devices:",
        " ".join(
            str(
                device
            )
            for device in devices
        ),
    )
    print(
        "Inference workers:",
        len(
            workers
        ),
    )
    print()
    print(
        "No sequential motion transform was accumulated: "
        "all sampled transforms were estimated directly against "
        f"reference frame {args.reference_frame}."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()