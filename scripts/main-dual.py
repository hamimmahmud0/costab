import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque

import imageio.v3 as iio
import numpy as np
import torch
import yaml


# ============================================================
# Configuration / CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track a dense point grid with CoTracker3 Online across "
            "two CUDA devices and save the results as NPZ + YAML."
        ),
    )

    parser.add_argument(
        "--video-file",
        type=Path,
        default=Path("assets/clip01.mp4_1080p.mp4"),
        help="Input video path.",
    )
    parser.add_argument(
        "--devices",
        nargs=2,
        default=["cuda:0", "cuda:1"],
        metavar=("GPU0", "GPU1"),
        help="Exactly two torch CUDA device strings.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=96,
        help="Number of query points per grid dimension.",
    )
    parser.add_argument(
        "--query-frame",
        type=int,
        default=0,
        help=(
            "Video-frame index used as the query time. It must fall "
            "inside CoTracker's initial online step."
        ),
    )

    parser.add_argument(
        "--run-name",
        default="clip01",
        help="Run directory name created under --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--npz-name",
        default="track.npz",
        help="NPZ output filename inside the run directory.",
    )
    parser.add_argument(
        "--yaml-name",
        default="track.yaml",
        help="YAML metadata filename inside the run directory.",
    )

    parser.add_argument(
        "--hub-repo",
        default="facebookresearch/co-tracker",
        help="torch.hub repository containing the tracker model.",
    )
    parser.add_argument(
        "--model",
        default="cotracker3_online",
        help="torch.hub model entry point.",
    )
    parser.add_argument(
        "--video-plugin",
        default="pyav",
        help="ImageIO plugin/backend used to decode the video.",
    )

    args = parser.parse_args()

    if args.grid_size <= 0:
        parser.error("--grid-size must be greater than 0.")

    if args.query_frame < 0:
        parser.error("--query-frame must be >= 0.")

    if not args.run_name:
        parser.error("--run-name must not be empty.")

    if Path(args.npz_name).name != args.npz_name:
        parser.error("--npz-name must be a filename, not a path.")

    if Path(args.yaml_name).name != args.yaml_name:
        parser.error("--yaml-name must be a filename, not a path.")

    return args


args = parse_args()

VIDEO_FILE = args.video_file
DEVICES = args.devices
GRID_SIZE = args.grid_size
QUERY_FRAME = args.query_frame
RUN_NAME = args.run_name
HUB_REPO = args.hub_repo
MODEL_NAME = args.model
VIDEO_PLUGIN = args.video_plugin

RUN_DIR = args.runs_dir / RUN_NAME
NPZ_FILE = RUN_DIR / args.npz_name
YAML_FILE = RUN_DIR / args.yaml_name


# ============================================================
# Setup
# ============================================================

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not VIDEO_FILE.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_FILE}"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

if torch.cuda.device_count() < 2:
    raise RuntimeError(
        f"Need 2 CUDA GPUs, "
        f"but found {torch.cuda.device_count()}."
    )


# ============================================================
# Load CoTracker
# ============================================================

print("Loading CoTracker on GPU 0...")

tracker0 = (
    torch.hub.load(
        HUB_REPO,
        MODEL_NAME,
    )
    .to(DEVICES[0])
    .eval()
)


print("Loading CoTracker on GPU 1...")

tracker1 = (
    torch.hub.load(
        HUB_REPO,
        MODEL_NAME,
    )
    .to(DEVICES[1])
    .eval()
)


# Both trackers must use the same step.

if tracker0.step != tracker1.step:
    raise RuntimeError(
        "The two trackers have different step sizes."
    )


step = tracker0.step
window_size = step * 2


if QUERY_FRAME >= step:
    raise ValueError(
        f"--query-frame must be smaller than the tracker step "
        f"({step}); got {QUERY_FRAME}."
    )


print()
print("step:", step)
print("window:", window_size)


# One long-lived CPU worker per GPU. Reusing the pool avoids creating
# threads for every online chunk.
executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="cotracker-gpu",
)


# ============================================================
# Open video
# ============================================================

reader = iio.imiter(
    VIDEO_FILE,
    plugin=VIDEO_PLUGIN,
)


try:
    first_frame = next(reader)
except StopIteration:
    raise RuntimeError("Video contains no frames.")


H, W, C = first_frame.shape


print(
    "Resolution:",
    W,
    "x",
    H,
)


# ============================================================
# Create query grid
#
# queries:
#
#   [batch, point, (t, x, y)]
#
# With the default GRID_SIZE=96:
#
#   96 * 96 = 9216 points
# ============================================================

xs = torch.linspace(
    0,
    W - 1,
    GRID_SIZE,
    dtype=torch.float32,
)

ys = torch.linspace(
    0,
    H - 1,
    GRID_SIZE,
    dtype=torch.float32,
)


yy, xx = torch.meshgrid(
    ys,
    xs,
    indexing="ij",
)


xx = xx.reshape(-1)
yy = yy.reshape(-1)


# Every point is queried on QUERY_FRAME.

t = torch.full_like(
    xx,
    float(QUERY_FRAME),
)


queries = torch.stack(
    [
        t,
        xx,
        yy,
    ],
    dim=-1,
).unsqueeze(0)


num_points = queries.shape[1]


print(
    "Total queries:",
    queries.shape,
)

print(
    "Total points:",
    num_points,
)


# ============================================================
# Split points across GPUs
#
# Original point order is retained after concatenation:
#
#   GPU 0:
#       0 ... half-1
#
#   GPU 1:
#       half ... N-1
# ============================================================

half = num_points // 2


queries0 = queries[
    :,
    :half,
].to(DEVICES[0])


queries1 = queries[
    :,
    half:,
].to(DEVICES[1])


print(
    "GPU 0 points:",
    queries0.shape[1],
)

print(
    "GPU 1 points:",
    queries1.shape[1],
)


# ============================================================
# Convert frames to CoTracker input
#
# Input:
#
#   list of:
#       H W C
#
# Output:
#
#   1 T C H W
# ============================================================

def make_video_tensor(frames):

    frames_np = np.stack(
        frames,
        axis=0,
    )

    video_cpu = (
        torch.from_numpy(frames_np)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .float()
        .contiguous()
        .pin_memory()
    )

    return video_cpu


# ============================================================
# Initialize online trackers
#
# IMPORTANT:
#
# is_first_step=True returns:
#
#   (None, None)
#
# It only:
#
#   - resets online state
#   - registers queries
#   - prepares tracking
# ============================================================

def _initialize_one_tracker(tracker, video, query, device):
    with torch.cuda.device(device):
        return tracker(
            video_chunk=video,
            is_first_step=True,
            queries=query,
            grid_size=0,
        )


def _process_one_tracker(tracker, video, device):
    with torch.cuda.device(device):
        return tracker(
            video_chunk=video,
        )


def initialize_trackers(frames):

    video_cpu = make_video_tensor(frames)

    # Pinned host memory + non_blocking=True lets both H2D copies be
    # queued asynchronously. Each model call is then submitted from a
    # separate CPU worker so CUDA work can be launched on both devices
    # at the same time.
    video0 = video_cpu.to(
        DEVICES[0],
        non_blocking=True,
    )

    video1 = video_cpu.to(
        DEVICES[1],
        non_blocking=True,
    )

    future0 = executor.submit(
        _initialize_one_tracker,
        tracker0,
        video0,
        queries0,
        DEVICES[0],
    )

    future1 = executor.submit(
        _initialize_one_tracker,
        tracker1,
        video1,
        queries1,
        DEVICES[1],
    )

    result0 = future0.result()
    result1 = future1.result()


    # Current CoTracker online API should return
    # (None, None) during initialization.

    if result0 != (None, None):
        print(
            "Warning: GPU 0 initialization "
            "returned unexpected output:",
            result0,
        )

    if result1 != (None, None):
        print(
            "Warning: GPU 1 initialization "
            "returned unexpected output:",
            result1,
        )


    del video0
    del video1
    del video_cpu


# ============================================================
# Process one online tracking chunk
#
# Returns:
#
# tracks:
#     [1, T, 9216, 2]
#
# visibility:
#     [1, T, 9216]
#
# T is cumulative.
# ============================================================

def process_chunk(frames):

    video_cpu = make_video_tensor(frames)


    # --------------------------------------------------------
    # Same video goes to both GPUs
    # --------------------------------------------------------

    video0 = video_cpu.to(
        DEVICES[0],
        non_blocking=True,
    )

    video1 = video_cpu.to(
        DEVICES[1],
        non_blocking=True,
    )


    # --------------------------------------------------------
    # Track both point partitions concurrently.
    #
    # A separate CPU worker submits work to each CUDA device so one
    # model's Python-side forward pass does not delay launch of the
    # other model's kernels.
    # --------------------------------------------------------

    future0 = executor.submit(
        _process_one_tracker,
        tracker0,
        video0,
        DEVICES[0],
    )

    future1 = executor.submit(
        _process_one_tracker,
        tracker1,
        video1,
        DEVICES[1],
    )

    tracks0, visibility0 = future0.result()
    tracks1, visibility1 = future1.result()


    if tracks0 is None or visibility0 is None:
        raise RuntimeError(
            "GPU 0 tracker returned None during "
            "normal processing."
        )

    if tracks1 is None or visibility1 is None:
        raise RuntimeError(
            "GPU 1 tracker returned None during "
            "normal processing."
        )


    # --------------------------------------------------------
    # Move results back to CPU
    # --------------------------------------------------------

    tracks0 = tracks0.cpu()
    tracks1 = tracks1.cpu()

    visibility0 = visibility0.cpu()
    visibility1 = visibility1.cpu()


    # --------------------------------------------------------
    # Merge point dimension
    #
    # GPU 0:
    #
    #   [1, T, 4608, 2]
    #
    # GPU 1:
    #
    #   [1, T, 4608, 2]
    #
    # merged:
    #
    #   [1, T, 9216, 2]
    # --------------------------------------------------------

    tracks = torch.cat(
        [
            tracks0,
            tracks1,
        ],
        dim=2,
    )


    visibility = torch.cat(
        [
            visibility0,
            visibility1,
        ],
        dim=2,
    )


    del video0
    del video1
    del video_cpu

    del tracks0
    del tracks1

    del visibility0
    del visibility1


    return tracks, visibility


# ============================================================
# Online tracking
#
# CoTracker3 online:
#
#   step   = 8
#   window = 16
#
#
# The official cadence is roughly:
#
# frame index 8:
#
#   initialize using frames 0..7
#
# frame index 16:
#
#   process frames 0..15
#
# frame index 24:
#
#   process frames 8..23
#
# frame index 32:
#
#   process frames 16..31
#
# ...
#
#
# Important:
#
# Processing happens BEFORE current frame `i`
# is appended to the buffer.
# ============================================================

frames_buffer = deque(
    maxlen=window_size,
)


# Frame 0

frames_buffer.append(
    first_frame
)


num_video_frames = 1
last_frame_index = 0

initialized = False

tracks = None
visibility = None

num_tracking_calls = 0


print()
print("=" * 70)
print("Starting tracking")
print("=" * 70)


with torch.inference_mode():

    # --------------------------------------------------------
    # Remaining video frames
    # --------------------------------------------------------

    for frame_index, frame in enumerate(
        reader,
        start=1,
    ):

        # ----------------------------------------------------
        # Process BEFORE adding the current frame.
        #
        # At frame_index=8:
        #
        # buffer = frames 0..7
        #
        # At frame_index=16:
        #
        # buffer = frames 0..15
        # ----------------------------------------------------

        if (
            frame_index % step == 0
            and frame_index != 0
        ):

            chunk = list(frames_buffer)

            # Only use the most recent model window.
            chunk = chunk[-window_size:]


            # ------------------------------------------------
            # First call:
            # initialization only
            # ------------------------------------------------

            if not initialized:

                print(
                    f"Initializing at boundary "
                    f"{frame_index}: "
                    f"chunk={len(chunk)}"
                )

                initialize_trackers(
                    chunk
                )

                initialized = True


            # ------------------------------------------------
            # Subsequent calls:
            # actual tracking
            # ------------------------------------------------

            else:

                tracks, visibility = process_chunk(
                    chunk
                )

                num_tracking_calls += 1

                print(
                    f"Boundary {frame_index}:",
                    f"chunk={len(chunk)}",
                    "tracks=",
                    tuple(tracks.shape),
                    "visibility=",
                    tuple(visibility.shape),
                )


        # ----------------------------------------------------
        # Add current frame AFTER processing.
        # ----------------------------------------------------

        frames_buffer.append(
            frame
        )

        last_frame_index = frame_index
        num_video_frames += 1


    # ========================================================
    # Final processing call
    #
    # This is important even when the video frame count
    # happens to land exactly on a step boundary.
    #
    # This follows CoTracker's online_demo.py logic:
    #
    #   -(i % step) - step - 1
    #
    # Examples:
    #
    # Last frame index 88:
    #
    #   remainder = 0
    #   final chunk = 9 frames
    #
    # Last frame index 87:
    #
    #   remainder = 7
    #   final chunk = 16 frames
    #
    # This causes the final cumulative tensor's T
    # dimension to match the actual video length.
    # ========================================================

    if num_video_frames < 2:
        raise RuntimeError(
            "CoTracker requires at least two video frames."
        )


    remainder = last_frame_index % step


    final_chunk_size = (
        remainder
        + step
        + 1
    )


    final_chunk_size = min(
        final_chunk_size,
        window_size,
        len(frames_buffer),
    )


    final_chunk = list(
        frames_buffer
    )[-final_chunk_size:]


    # --------------------------------------------------------
    # Very short video:
    #
    # If no step boundary occurred, initialize now.
    # --------------------------------------------------------

    if not initialized:

        print(
            "Initializing tracker for short video:",
            f"chunk={len(final_chunk)}",
        )

        initialize_trackers(
            final_chunk
        )

        initialized = True


    # --------------------------------------------------------
    # Always make final tracking call.
    # --------------------------------------------------------

    tracks, visibility = process_chunk(
        final_chunk
    )

    num_tracking_calls += 1


    print(
        "Final:",
        f"video_frame={last_frame_index}",
        f"chunk={len(final_chunk)}",
        "tracks=",
        tuple(tracks.shape),
        "visibility=",
        tuple(visibility.shape),
    )


# ============================================================
# Validate final output
# ============================================================

if tracks is None:
    raise RuntimeError(
        "No track tensor was produced."
    )

if visibility is None:
    raise RuntimeError(
        "No visibility tensor was produced."
    )


expected_tracks_shape = (
    1,
    tracks.shape[1],
    num_points,
    2,
)


if tuple(tracks.shape) != expected_tracks_shape:
    raise RuntimeError(
        f"Unexpected tracks shape: "
        f"{tuple(tracks.shape)}"
    )


expected_visibility_shape = (
    1,
    tracks.shape[1],
    num_points,
)


if tuple(visibility.shape) != expected_visibility_shape:
    raise RuntimeError(
        f"Unexpected visibility shape: "
        f"{tuple(visibility.shape)}"
    )


# ============================================================
# Check temporal length
#
# After the final CoTracker call:
#
# tracks.shape[1]
#
# should normally equal:
#
# num_video_frames
# ============================================================

tracked_frames = tracks.shape[1]


print()
print("Video frames:", num_video_frames)
print("Tracked frames:", tracked_frames)


if tracked_frames != num_video_frames:

    print()
    print(
        "WARNING:"
    )

    print(
        f"Video contains {num_video_frames} frames "
        f"but CoTracker returned {tracked_frames} "
        f"temporal positions."
    )


# ============================================================
# Convert to NumPy
# ============================================================

tracks_np = (
    tracks
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float32,
        copy=False,
    )
)


visibility_np = (
    visibility
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.bool_,
        copy=False,
    )
)


queries_np = (
    queries
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float32,
        copy=False,
    )
)


# Track time index -> video frame index

frame_indices = np.arange(
    tracked_frames,
    dtype=np.int32,
)


# Point IDs

point_indices = np.arange(
    num_points,
    dtype=np.int32,
)


# ============================================================
# Save NPZ
# ============================================================

print()
print("Saving:", NPZ_FILE)


np.savez_compressed(
    NPZ_FILE,

    # --------------------------------------------------------
    # Main results
    # --------------------------------------------------------

    # [1, T, N, 2]
    tracks=tracks_np,

    # [1, T, N]
    visibility=visibility_np,

    # [1, N, 3]
    # query = [t, x, y]
    queries=queries_np,


    # --------------------------------------------------------
    # Index maps
    # --------------------------------------------------------

    # [T]
    frame_indices=frame_indices,

    # [N]
    point_indices=point_indices,


    # --------------------------------------------------------
    # Video metadata
    # --------------------------------------------------------

    width=np.int32(W),
    height=np.int32(H),
    channels=np.int32(C),

    video_frames=np.int32(
        num_video_frames
    ),

    tracked_frames=np.int32(
        tracked_frames
    ),


    # --------------------------------------------------------
    # Grid metadata
    # --------------------------------------------------------

    grid_size=np.int32(
        GRID_SIZE
    ),

    num_points=np.int32(
        num_points
    ),


    # --------------------------------------------------------
    # Tracker metadata
    # --------------------------------------------------------

    step=np.int32(
        step
    ),

    window_size=np.int32(
        window_size
    ),

    gpu0_points=np.int32(
        queries0.shape[1]
    ),

    gpu1_points=np.int32(
        queries1.shape[1]
    ),
)


# ============================================================
# Save YAML metadata
#
# YAML does not duplicate the huge track arrays.
# It describes the NPZ.
# ============================================================

metadata = {

    "run_name": RUN_NAME,

    "video": {

        "file": str(VIDEO_FILE),

        "width": int(W),

        "height": int(H),

        "channels": int(C),

        "frames": int(
            num_video_frames
        ),

        "last_frame_index": int(
            last_frame_index
        ),
    },


    "tracker": {

        "hub_repo": HUB_REPO,

        "model": MODEL_NAME,

        "devices": DEVICES,

        "video_plugin": VIDEO_PLUGIN,

        "step": int(
            step
        ),

        "window_size": int(
            window_size
        ),

        "tracking_calls": int(
            num_tracking_calls
        ),
    },


    "grid": {

        "grid_size": int(
            GRID_SIZE
        ),

        "num_points": int(
            num_points
        ),

        "query_frame": QUERY_FRAME,

        "query_format": [
            "t",
            "x",
            "y",
        ],

        "ordering": (
            "row-major: "
            "point_id = row * grid_size + column"
        ),

        "gpu0": {

            "point_start": 0,

            "point_end": int(
                half - 1
            ),

            "num_points": int(
                queries0.shape[1]
            ),
        },

        "gpu1": {

            "point_start": int(
                half
            ),

            "point_end": int(
                num_points - 1
            ),

            "num_points": int(
                queries1.shape[1]
            ),
        },
    },


    "output": {

        "npz_file": str(
            NPZ_FILE
        ),

        "tracks": {

            "key": "tracks",

            "shape": list(
                tracks_np.shape
            ),

            "dtype": str(
                tracks_np.dtype
            ),

            "layout": [
                "batch",
                "time",
                "point",
                "xy",
            ],

            "coordinate_order": [
                "x",
                "y",
            ],
        },


        "visibility": {

            "key": "visibility",

            "shape": list(
                visibility_np.shape
            ),

            "dtype": str(
                visibility_np.dtype
            ),

            "layout": [
                "batch",
                "time",
                "point",
            ],
        },


        "queries": {

            "key": "queries",

            "shape": list(
                queries_np.shape
            ),

            "dtype": str(
                queries_np.dtype
            ),

            "layout": [
                "batch",
                "point",
                "txy",
            ],
        },


        "frame_indices": {

            "key": "frame_indices",

            "shape": list(
                frame_indices.shape
            ),

            "description": (
                "Maps track time dimension "
                "to source video frame index."
            ),
        },
    },
}


print("Saving:", YAML_FILE)


with open(
    YAML_FILE,
    "w",
    encoding="utf-8",
) as f:

    yaml.safe_dump(
        metadata,
        f,
        sort_keys=False,
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("TRACKING COMPLETE")
print("=" * 70)

print(
    "Video:",
    VIDEO_FILE,
)

print(
    "Resolution:",
    f"{W}x{H}",
)

print(
    "Video frames:",
    num_video_frames,
)

print(
    "Tracked frames:",
    tracked_frames,
)

print(
    "Points:",
    num_points,
)

print(
    "Tracks:",
    tracks_np.shape,
)

print(
    "Visibility:",
    visibility_np.shape,
)

print(
    "Queries:",
    queries_np.shape,
)

print()

print(
    "NPZ:",
    NPZ_FILE,
)

print(
    "YAML:",
    YAML_FILE,
)

print("=" * 70)

# Release worker threads after all tracking/output work is complete.
executor.shutdown(wait=True)