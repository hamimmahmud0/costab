#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import (
    LightGlue,
    SuperPoint,
    DISK,
    SIFT,
    ALIKED,
    DoGHardNet,
)
from lightglue.utils import rbd


EXTRACTORS = {
    "superpoint": SuperPoint,
    "disk": DISK,
    "sift": SIFT,
    "aliked": ALIKED,
    "doghardnet": DoGHardNet,
}

# Fresh Aqua palette in OpenCV BGR order.
AQUA_500 = (166, 184, 20)       # #14B8A6
AQUA_300 = (212, 234, 94)       # #5EEAD4
AQUA_100 = (241, 251, 204)      # #CCFBF1
AQUA_950 = (46, 47, 4)          # #042F2E
TEXT_LIGHT = (250, 253, 240)    # #F0FDFA
WHITE = (255, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Match local features between two frames of a video using "
            "LightGlue and save a side-by-side visualization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input video file.",
    )

    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        nargs=2,
        metavar=("FRAME0", "FRAME1"),
        required=True,
        help="Two zero-based frame indices to match, e.g. --frames 0 9.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output visualization image. If omitted, a name is generated "
            "next to the input video."
        ),
    )

    parser.add_argument(
        "--extractor",
        choices=tuple(EXTRACTORS.keys()),
        default="superpoint",
        help="Local feature extractor.",
    )

    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Maximum number of keypoints extracted from each frame.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help=(
            "Resize the longest image side before feature extraction. "
            "Use the original video resolution when omitted."
        ),
    )

    parser.add_argument(
        "--max-draw-matches",
        type=int,
        default=500,
        help=(
            "Maximum number of matches drawn in the visualization. "
            "Use 0 to draw all matches."
        ),
    )

    parser.add_argument(
        "--point-radius",
        type=int,
        default=3,
        help="Radius of matched keypoint circles.",
    )

    parser.add_argument(
        "--line-thickness",
        type=int,
        default=1,
        help="Thickness of match lines.",
    )

    parser.add_argument(
        "--panel-height",
        type=int,
        default=86,
        help="Height of the Fresh Aqua information panel.",
    )

    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ output containing keypoints, matches, scores, "
            "and selected frame indices."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visualization after saving it.",
    )

    args = parser.parse_args()

    if args.frames[0] < 0 or args.frames[1] < 0:
        parser.error("Frame indices must be >= 0.")

    if args.frames[0] == args.frames[1]:
        parser.error("The two frame indices must be different.")

    if args.max_keypoints <= 0:
        parser.error("--max-keypoints must be > 0.")

    if args.max_draw_matches < 0:
        parser.error("--max-draw-matches must be >= 0.")

    if args.point_radius < 1:
        parser.error("--point-radius must be >= 1.")

    if args.line_thickness < 1:
        parser.error("--line-thickness must be >= 1.")

    if args.panel_height < 40:
        parser.error("--panel-height must be >= 40.")

    return args


def resolve_device(device_string):
    device = torch.device(device_string)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device_string}) but CUDA is unavailable."
        )

    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device {device_index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )

    return device


def generated_output_path(video_path, frame0, frame1):
    return video_path.with_name(
        f"{video_path.stem}_frames_{frame0}_{frame1}_matches.png"
    )


def read_exact_frames(video_path, requested_indices):
    """
    Decode sequentially and return the requested zero-based frames.

    Sequential decoding avoids relying on codec-dependent random seek behavior.
    """
    requested_indices = tuple(int(x) for x in requested_indices)
    targets = set(requested_indices)
    last_needed = max(targets)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if frame_count > 0:
        for index in targets:
            if index >= frame_count:
                capture.release()
                raise IndexError(
                    f"Requested frame {index}, but video contains "
                    f"{frame_count} frames."
                )

    found = {}
    frame_index = 0

    while frame_index <= last_needed:
        ok, frame = capture.read()

        if not ok:
            break

        if frame_index in targets:
            found[frame_index] = frame.copy()

            if len(found) == len(targets):
                break

        frame_index += 1

    capture.release()

    missing = [index for index in requested_indices if index not in found]

    if missing:
        raise RuntimeError(
            "Could not decode requested frame(s): "
            + ", ".join(str(index) for index in missing)
        )

    return (
        found[requested_indices[0]],
        found[requested_indices[1]],
        fps,
        frame_count,
    )


def bgr_to_lightglue_tensor(frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(frame_rgb)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div_(255.0)
        .to(device)
    )

    return tensor


def build_models(extractor_name, max_keypoints, device):
    extractor_class = EXTRACTORS[extractor_name]

    extractor = (
        extractor_class(
            max_num_keypoints=max_keypoints,
        )
        .eval()
        .to(device)
    )

    matcher = (
        LightGlue(
            features=extractor_name,
        )
        .eval()
        .to(device)
    )

    return extractor, matcher


def extract_and_match(
    extractor,
    matcher,
    image0,
    image1,
    resize,
):
    with torch.inference_mode():
        feats0 = extractor.extract(
            image0,
            resize=resize,
        )

        feats1 = extractor.extract(
            image1,
            resize=resize,
        )

        matches01 = matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
        )

    feats0, feats1, matches01 = [
        rbd(x)
        for x in (
            feats0,
            feats1,
            matches01,
        )
    ]

    matches = matches01["matches"]

    points0 = feats0["keypoints"][
        matches[..., 0]
    ]

    points1 = feats1["keypoints"][
        matches[..., 1]
    ]

    scores = matches01.get("scores")

    return (
        feats0,
        feats1,
        matches01,
        matches,
        points0,
        points1,
        scores,
    )


def choose_draw_indices(scores, num_matches, max_draw_matches):
    if max_draw_matches == 0 or num_matches <= max_draw_matches:
        return np.arange(
            num_matches,
            dtype=np.int64,
        )

    if scores is not None:
        scores_np = (
            scores.detach()
            .float()
            .cpu()
            .numpy()
        )

        order = np.argsort(
            scores_np
        )[::-1]

        return order[
            :max_draw_matches
        ]

    return np.linspace(
        0,
        num_matches - 1,
        max_draw_matches,
        dtype=np.int64,
    )


def draw_visualization(
    frame0,
    frame1,
    points0,
    points1,
    scores,
    frame_indices,
    extractor_name,
    num_keypoints0,
    num_keypoints1,
    max_draw_matches,
    point_radius,
    line_thickness,
    panel_height,
):
    height0, width0 = frame0.shape[:2]
    height1, width1 = frame1.shape[:2]

    content_height = max(
        height0,
        height1,
    )

    content_width = (
        width0
        + width1
    )

    canvas = np.full(
        (
            panel_height + content_height,
            content_width,
            3,
        ),
        AQUA_950,
        dtype=np.uint8,
    )

    y0 = panel_height

    canvas[
        y0:y0 + height0,
        0:width0,
    ] = frame0

    canvas[
        y0:y0 + height1,
        width0:width0 + width1,
    ] = frame1

    points0_np = (
        points0.detach()
        .float()
        .cpu()
        .numpy()
    )

    points1_np = (
        points1.detach()
        .float()
        .cpu()
        .numpy()
    )

    num_matches = len(
        points0_np
    )

    draw_indices = choose_draw_indices(
        scores=scores,
        num_matches=num_matches,
        max_draw_matches=max_draw_matches,
    )

    overlay = canvas.copy()

    for match_index in draw_indices:
        x0, y_point0 = points0_np[
            match_index
        ]

        x1, y_point1 = points1_np[
            match_index
        ]

        p0 = (
            int(round(x0)),
            int(round(y0 + y_point0)),
        )

        p1 = (
            int(round(width0 + x1)),
            int(round(y0 + y_point1)),
        )

        cv2.line(
            overlay,
            p0,
            p1,
            AQUA_500,
            line_thickness,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            overlay,
            p0,
            point_radius,
            AQUA_300,
            -1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            overlay,
            p1,
            point_radius,
            AQUA_100,
            -1,
            lineType=cv2.LINE_AA,
        )

    # Blend lines/points slightly so the source frames remain legible.
    canvas = cv2.addWeighted(
        overlay,
        0.88,
        canvas,
        0.12,
        0.0,
    )

    divider_x = width0

    cv2.line(
        canvas,
        (
            divider_x,
            panel_height,
        ),
        (
            divider_x,
            panel_height + content_height - 1,
        ),
        AQUA_300,
        2,
        lineType=cv2.LINE_AA,
    )

    frame0_index, frame1_index = frame_indices

    title = (
        f"LightGlue matches | {extractor_name.upper()} "
        f"| frames {frame0_index} -> {frame1_index}"
    )

    stats = (
        f"keypoints: {num_keypoints0} / {num_keypoints1}   "
        f"matches: {num_matches}   "
        f"drawn: {len(draw_indices)}"
    )

    cv2.putText(
        canvas,
        title,
        (
            20,
            32,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        WHITE,
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        stats,
        (
            20,
            64,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TEXT_LIGHT,
        1,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Frame {frame0_index}",
        (
            16,
            panel_height + 30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        WHITE,
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Frame {frame1_index}",
        (
            width0 + 16,
            panel_height + 30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        WHITE,
        2,
        lineType=cv2.LINE_AA,
    )

    return canvas


def save_npz(
    output_path,
    frame_indices,
    feats0,
    feats1,
    matches,
    points0,
    points1,
    scores,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "frame_indices": np.asarray(
            frame_indices,
            dtype=np.int32,
        ),
        "keypoints0": (
            feats0["keypoints"]
            .detach()
            .float()
            .cpu()
            .numpy()
        ),
        "keypoints1": (
            feats1["keypoints"]
            .detach()
            .float()
            .cpu()
            .numpy()
        ),
        "matches": (
            matches
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int32,
                copy=False,
            )
        ),
        "points0": (
            points0
            .detach()
            .float()
            .cpu()
            .numpy()
        ),
        "points1": (
            points1
            .detach()
            .float()
            .cpu()
            .numpy()
        ),
    }

    if scores is not None:
        payload["scores"] = (
            scores.detach()
            .float()
            .cpu()
            .numpy()
        )

    np.savez_compressed(
        output_path,
        **payload,
    )


def main():
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input video not found: {args.input}"
        )

    output_path = (
        args.output
        if args.output is not None
        else generated_output_path(
            args.input,
            args.frames[0],
            args.frames[1],
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    print("=" * 72)
    print("LIGHTGLUE VIDEO FRAME MATCHER")
    print("=" * 72)
    print("Video:", args.input)
    print("Frames:", args.frames)
    print("Extractor:", args.extractor)
    print("Max keypoints:", args.max_keypoints)
    print("Device:", device)
    print("Resize:", args.resize)
    print()

    print("Reading video frames...")

    frame0, frame1, fps, frame_count = read_exact_frames(
        args.input,
        args.frames,
    )

    print(
        "Frame resolution:",
        f"{frame0.shape[1]}x{frame0.shape[0]}",
    )

    if fps > 0:
        print(
            "FPS:",
            f"{fps:.3f}",
        )

    if frame_count > 0:
        print(
            "Video frames:",
            frame_count,
        )

    print()
    print("Loading extractor and LightGlue...")

    extractor, matcher = build_models(
        extractor_name=args.extractor,
        max_keypoints=args.max_keypoints,
        device=device,
    )

    image0 = bgr_to_lightglue_tensor(
        frame0,
        device,
    )

    image1 = bgr_to_lightglue_tensor(
        frame1,
        device,
    )

    print("Extracting and matching...")

    (
        feats0,
        feats1,
        matches01,
        matches,
        points0,
        points1,
        scores,
    ) = extract_and_match(
        extractor=extractor,
        matcher=matcher,
        image0=image0,
        image1=image1,
        resize=args.resize,
    )

    num_keypoints0 = int(
        feats0["keypoints"].shape[0]
    )

    num_keypoints1 = int(
        feats1["keypoints"].shape[0]
    )

    num_matches = int(
        matches.shape[0]
    )

    print()
    print(
        "Frame",
        args.frames[0],
        "keypoints:",
        num_keypoints0,
    )

    print(
        "Frame",
        args.frames[1],
        "keypoints:",
        num_keypoints1,
    )

    print(
        "Matches:",
        num_matches,
    )

    print()
    print("Rendering visualization...")

    visualization = draw_visualization(
        frame0=frame0,
        frame1=frame1,
        points0=points0,
        points1=points1,
        scores=scores,
        frame_indices=args.frames,
        extractor_name=args.extractor,
        num_keypoints0=num_keypoints0,
        num_keypoints1=num_keypoints1,
        max_draw_matches=args.max_draw_matches,
        point_radius=args.point_radius,
        line_thickness=args.line_thickness,
        panel_height=args.panel_height,
    )

    saved = cv2.imwrite(
        str(output_path),
        visualization,
    )

    if not saved:
        raise RuntimeError(
            f"Failed to save visualization: {output_path}"
        )

    print(
        "Visualization:",
        output_path,
    )

    if args.save_npz is not None:
        save_npz(
            output_path=args.save_npz,
            frame_indices=args.frames,
            feats0=feats0,
            feats1=feats1,
            matches=matches,
            points0=points0,
            points1=points1,
            scores=scores,
        )

        print(
            "Match data:",
            args.save_npz,
        )

    if args.show:
        cv2.imshow(
            "LightGlue matches",
            visualization,
        )

        print(
            "Press any key in the visualization window to close."
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("=" * 72)


if __name__ == "__main__":
    main()
