import torch
import imageio.v3 as iio
from tqdm import tqdm
import numpy as np

VIDEO_FILE = 'assets/clip01.mp4'


#################################
######## Config #################
#################################

BATCH_SIZE = 50 
DEVICE = 'cpu'
GRID_SIZE = 128
STEP=1

batch = []

for frame in iio.imiter(VIDEO_FILE, plugin="pyav"):
    batch.append(frame)

    if len(batch) == BATCH_SIZE:
        batch = np.stack(batch)

        print(batch.shape)
        # (16, height, width, channels)

        # process batch here


                
        frames = batch

        device= DEVICE
        grid_size= GRID_SIZE

        print("Loading frames to memory")
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

        print('Fetching cotracker')
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

        # Run Online CoTracker, the same model with a different API:
        # Initialize online processing

        print("Initializing cotracker")
        cotracker.step = STEP
        cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

        # Process the video
        print("Processing video")
        for ind in tqdm(range(0, video.shape[1] - cotracker.step, cotracker.step)):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # B T N 2,  B T N 1

        batch = []

# process leftover frames
if batch:
    batch = np.stack(batch)
    print(batch.shape)
