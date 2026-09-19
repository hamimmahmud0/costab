import torch
import imageio.v3 as iio
import numpy as np
from collections import deque


VIDEO_FILE = "assets/clip01.mp4_1080p.mp4"
DEVICES = ["cuda:0", "cuda:1"]
GRID_SIZE = 96

RUN_NAME = "clip01"

# --------------------------------------------------
# Load one CoTracker instance per GPU
# --------------------------------------------------

print("Loading CoTracker on GPU 0...")
tracker0 = torch.hub.load(
    "facebookresearch/co-tracker",
    "cotracker3_online"
).to(DEVICES[0]).eval()

print("Loading CoTracker on GPU 1...")
tracker1 = torch.hub.load(
    "facebookresearch/co-tracker",
    "cotracker3_online"
).to(DEVICES[1]).eval()


step = tracker0.step
window_size = step * 2

print("step:", step)
print("window:", window_size)


# --------------------------------------------------
# Read first frame so we know H/W
# --------------------------------------------------

reader = iio.imiter(VIDEO_FILE, plugin="pyav")

first_frame = next(reader)

H, W, _ = first_frame.shape

print("Resolution:", W, "x", H)


# --------------------------------------------------
# Create GRID_SIZE x GRID_SIZE query points
#
# Query format:
#     [frame_index, x, y]
# --------------------------------------------------

xs = torch.linspace(0, W - 1, GRID_SIZE)
ys = torch.linspace(0, H - 1, GRID_SIZE)

yy, xx = torch.meshgrid(
    ys,
    xs,
    indexing="ij"
)

xx = xx.reshape(-1)
yy = yy.reshape(-1)

t = torch.zeros_like(xx)

queries = torch.stack(
    [t, xx, yy],
    dim=-1
).unsqueeze(0)

# shape:
# (1, 16384, 3)

print("Total queries:", queries.shape)


# --------------------------------------------------
# Split points between GPUs
# --------------------------------------------------

half = queries.shape[1] // 2

queries0 = queries[:, :half].to(DEVICES[0])
queries1 = queries[:, half:].to(DEVICES[1])

print("GPU 0 points:", queries0.shape[1])
print("GPU 1 points:", queries1.shape[1])


# --------------------------------------------------
# Frame buffer
# --------------------------------------------------

frames_buffer = deque(maxlen=window_size)

frames_buffer.append(first_frame)


initialized = False


all_tracks = []
all_visibility = []


with torch.inference_mode():

    for frame_index, frame in enumerate(reader, start=1):

        frames_buffer.append(frame)

        # Need full window first
        if len(frames_buffer) < window_size:
            continue

        # CoTracker advances by `step`
        if frame_index % step != 0:
            continue


        frames = np.stack(frames_buffer)

        # CPU:
        # T H W C -> 1 T C H W

        video_cpu = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
        )


        # Copy same video chunk to each GPU

        video0 = video_cpu.to(
            DEVICES[0],
            non_blocking=True
        )

        video1 = video_cpu.to(
            DEVICES[1],
            non_blocking=True
        )


        # ------------------------------------------
        # Initialize both online trackers
        # ------------------------------------------

        if not initialized:

            print("Initializing trackers...")

            tracker0(
                video_chunk=video0,
                is_first_step=True,
                queries=queries0
            )

            tracker1(
                video_chunk=video1,
                is_first_step=True,
                queries=queries1
            )

            initialized = True

            continue


        # ------------------------------------------
        # Run GPU 0
        # ------------------------------------------

        tracks0, visibility0 = tracker0(
            video_chunk=video0
        )


        # ------------------------------------------
        # Run GPU 1
        # ------------------------------------------

        tracks1, visibility1 = tracker1(
            video_chunk=video1
        )


        # Move output to CPU

        tracks0 = tracks0.cpu()
        tracks1 = tracks1.cpu()

        visibility0 = visibility0.cpu()
        visibility1 = visibility1.cpu()


        # ------------------------------------------
        # Merge point dimension
        #
        # B T N 2
        #      ^
        #      concatenate here
        # ------------------------------------------

        tracks = torch.cat(
            [tracks0, tracks1],
            dim=2
        )

        visibility = torch.cat(
            [visibility0, visibility1],
            dim=2
        )


        print(
            f"Frame {frame_index}:",
            "tracks =", tracks.shape,
            "visibility =", visibility.shape
        )


        all_tracks.append(tracks)
        all_visibility.append(visibility)

        # save the all_tracks and all_visibility combinedly as npz and yaml in ./runs/<RUN_NAME>/track.npz and ./runs/<RUN_NAME>/track.yaml 


        del video0
        del video1
        del video_cpu