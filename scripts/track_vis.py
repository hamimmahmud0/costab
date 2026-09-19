import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# ============================================================
# Fresh Aqua theme
# ============================================================

THEME = {
    # Aqua core
    "aqua_50": "#F0FDFA",
    "aqua_100": "#CCFBF1",
    "aqua_200": "#99F6E4",
    "aqua_300": "#5EEAD4",
    "aqua_400": "#2DD4BF",
    "aqua_500": "#14B8A6",
    "aqua_600": "#0D9488",
    "aqua_700": "#0F766E",
    "aqua_800": "#115E59",
    "aqua_900": "#134E4A",
    "aqua_950": "#042F2E",

    # Light surfaces / text
    "white": "#FFFFFF",
    "warm_white": "#FCFFFE",
    "aqua_white": "#F8FFFD",
    "page_bg": "#F7FBFA",
    "surface_alt": "#F0FDFA",
    "border_light": "#DCEAE8",
    "border": "#CBDDD9",
    "muted_text": "#647A77",
    "secondary_text": "#435C59",
    "body_text": "#294B47",
    "heading": "#102E2B",

    # Status / chart palette
    "success": "#16A085",
    "info": "#0891B2",
    "warning": "#F59E0B",
    "error": "#E5484D",
    "series_1": "#14B8A6",
    "series_2": "#0891B2",
    "series_3": "#5EEAD4",
    "series_4": "#0F766E",
    "series_5": "#7DD3FC",
    "series_6": "#83C5BE",
    "series_7": "#F59E0B",
    "series_8": "#64748B",

    # Dark mode surfaces
    "dark_bg": "#071F1D",
    "dark_surface": "#103632",
    "dark_elevated": "#13413C",
    "dark_border": "#20554F",
    "dark_secondary_text": "#B6D8D3",
}

CHART_KEYS = [
    "series_1",
    "series_2",
    "series_3",
    "series_4",
    "series_5",
    "series_6",
    "series_7",
    "series_8",
]


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {value!r}")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return b, g, r


BGR = {name: hex_to_bgr(value) for name, value in THEME.items()}
CHART_COLORS = [BGR[key] for key in CHART_KEYS]


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize CoTracker NPZ/YAML outputs using a Fresh Aqua "
            "video-overlay theme."
        ),
    )

    parser.add_argument(
        "--yaml-file",
        type=Path,
        default=Path("runs/clip01/track.yaml"),
        help="Tracker YAML metadata file.",
    )
    parser.add_argument(
        "--npz-file",
        type=Path,
        default=None,
        help=(
            "Tracker NPZ file. If omitted, resolve output.npz_file from "
            "the YAML, then fall back to track.npz beside the YAML."
        ),
    )
    parser.add_argument(
        "--video-file",
        type=Path,
        default=None,
        help=(
            "Source video override. If omitted, resolve video.file from "
            "the YAML."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Rendered video path. Default: track_visualization.mp4 beside "
            "the YAML."
        ),
    )

    parser.add_argument(
        "--background",
        choices=("video", "light", "dark"),
        default="video",
        help=(
            "Visualization background. 'video' overlays tracks on the "
            "source video; 'light' and 'dark' render on themed canvases."
        ),
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="Output FPS. Use 0 to inherit the source video FPS.",
    )
    parser.add_argument(
        "--canvas-fps",
        type=float,
        default=30.0,
        help="FPS used for light/dark canvas mode when --output-fps is 0.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC output codec, e.g. mp4v, avc1, XVID.",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First source frame to render.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Last source frame to render, inclusive. -1 means all.",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=1600,
        help=(
            "Maximum number of tracks to draw. Use 0 to draw all selected "
            "points."
        ),
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=1,
        help=(
            "Spatial grid stride before applying --max-points. 2 draws "
            "approximately every second grid row/column."
        ),
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=16,
        help="Number of temporal positions shown in each track trail.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=2,
        help="Current track-point radius in pixels.",
    )
    parser.add_argument(
        "--trail-thickness",
        type=int,
        default=1,
        help="Track trail line thickness in pixels.",
    )
    parser.add_argument(
        "--track-alpha",
        type=float,
        default=0.90,
        help="Opacity of track graphics in the range 0..1.",
    )
    parser.add_argument(
        "--hidden-alpha",
        type=float,
        default=0.30,
        help="Relative opacity used for hidden points when enabled.",
    )
    parser.add_argument(
        "--draw-hidden",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw currently invisible points using the muted theme color.",
    )
    parser.add_argument(
        "--draw-trails",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw temporal trails behind current track positions.",
    )
    parser.add_argument(
        "--draw-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw current point markers.",
    )
    parser.add_argument(
        "--draw-query-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw original query locations as subtle hollow circles.",
    )

    parser.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the Fresh Aqua metadata/dashboard overlay.",
    )
    parser.add_argument(
        "--dashboard-alpha",
        type=float,
        default=0.82,
        help="Dashboard panel opacity in the range 0..1.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the bottom Fresh Aqua progress bar.",
    )
    parser.add_argument(
        "--title",
        default="CoTracker · Fresh Aqua",
        help="Dashboard title.",
    )

    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preview frames in an OpenCV window while rendering.",
    )
    parser.add_argument(
        "--window-name",
        default="CoTracker · Fresh Aqua",
        help="OpenCV preview window title.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Preview-window scale factor. Does not change saved output.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print rendering progress every N output frames. 0 disables.",
    )

    args = parser.parse_args()

    if args.start_frame < 0:
        parser.error("--start-frame must be >= 0.")
    if args.end_frame < -1:
        parser.error("--end-frame must be -1 or >= 0.")
    if args.end_frame != -1 and args.end_frame < args.start_frame:
        parser.error("--end-frame must be >= --start-frame.")
    if args.max_points < 0:
        parser.error("--max-points must be >= 0.")
    if args.point_stride <= 0:
        parser.error("--point-stride must be > 0.")
    if args.trail_length <= 0:
        parser.error("--trail-length must be > 0.")
    if args.point_radius < 0:
        parser.error("--point-radius must be >= 0.")
    if args.trail_thickness <= 0:
        parser.error("--trail-thickness must be > 0.")
    if not 0.0 <= args.track_alpha <= 1.0:
        parser.error("--track-alpha must be in the range 0..1.")
    if not 0.0 <= args.hidden_alpha <= 1.0:
        parser.error("--hidden-alpha must be in the range 0..1.")
    if not 0.0 <= args.dashboard_alpha <= 1.0:
        parser.error("--dashboard-alpha must be in the range 0..1.")
    if args.output_fps < 0:
        parser.error("--output-fps must be >= 0.")
    if args.canvas_fps <= 0:
        parser.error("--canvas-fps must be > 0.")
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters.")
    if args.preview_scale <= 0:
        parser.error("--preview-scale must be > 0.")
    if args.print_every < 0:
        parser.error("--print-every must be >= 0.")

    return args


# ============================================================
# Path / metadata loading
# ============================================================


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")

    return data


def resolve_recorded_path(raw_path: str | Path, yaml_file: Path) -> Path:
    """Resolve paths recorded when the tracker ran from another CWD."""
    raw = Path(raw_path).expanduser()

    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        # 1) Relative to current working directory (same as tracker run).
        candidates.append(raw)

        # 2) Relative to YAML directory.
        candidates.append(yaml_file.parent / raw)

        # 3) Just the basename beside the YAML. This is useful when YAML
        #    stores e.g. "runs/clip01/track.npz" and has since been moved.
        candidates.append(yaml_file.parent / raw.name)

        # 4) Try ancestors of the YAML directory as roots for recorded paths.
        for parent in yaml_file.parents:
            candidates.append(parent / raw)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    # Return the most natural unresolved candidate for a helpful error.
    if raw.is_absolute():
        return raw
    return (yaml_file.parent / raw.name).resolve(strict=False)


def resolve_npz_file(
    args: argparse.Namespace,
    metadata: dict,
    yaml_file: Path,
) -> Path:
    if args.npz_file is not None:
        path = args.npz_file.expanduser().resolve(strict=False)
    else:
        raw = metadata.get("output", {}).get("npz_file")
        if raw:
            path = resolve_recorded_path(raw, yaml_file)
        else:
            path = (yaml_file.parent / "track.npz").resolve(strict=False)

    if not path.is_file():
        raise FileNotFoundError(
            "NPZ file not found. Resolved path: "
            f"{path}. Pass --npz-file explicitly if needed."
        )

    return path


def resolve_video_file(
    args: argparse.Namespace,
    metadata: dict,
    yaml_file: Path,
) -> Optional[Path]:
    if args.video_file is not None:
        path = args.video_file.expanduser().resolve(strict=False)
    else:
        raw = metadata.get("video", {}).get("file")
        if raw:
            path = resolve_recorded_path(raw, yaml_file)
        else:
            path = None

    if args.background == "video":
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "Source video is required for --background video but could "
                f"not be resolved. Resolved path: {path}. Pass --video-file."
            )

    if path is not None and not path.is_file():
        return None

    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("tracks", "visibility", "queries")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                f"NPZ is missing required arrays: {', '.join(missing)}"
            )

        arrays = {key: data[key] for key in data.files}

    return arrays


# ============================================================
# Validation / selection
# ============================================================


def normalize_arrays(arrays: dict[str, np.ndarray]) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tracks = np.asarray(arrays["tracks"])
    visibility = np.asarray(arrays["visibility"])
    queries = np.asarray(arrays["queries"])

    if tracks.ndim != 4 or tracks.shape[0] != 1 or tracks.shape[-1] != 2:
        raise ValueError(
            "Expected tracks shape [1, T, N, 2], got "
            f"{tracks.shape}."
        )

    if visibility.ndim != 3 or visibility.shape[0] != 1:
        raise ValueError(
            "Expected visibility shape [1, T, N], got "
            f"{visibility.shape}."
        )

    if queries.ndim != 3 or queries.shape[0] != 1 or queries.shape[-1] != 3:
        raise ValueError(
            "Expected queries shape [1, N, 3], got "
            f"{queries.shape}."
        )

    tracks = tracks[0]
    visibility = visibility[0].astype(bool, copy=False)
    queries = queries[0]

    temporal_length, num_points, _ = tracks.shape

    if visibility.shape != (temporal_length, num_points):
        raise ValueError(
            "tracks/visibility shape mismatch: "
            f"tracks={tracks.shape}, visibility={visibility.shape}."
        )

    if queries.shape[0] != num_points:
        raise ValueError(
            "tracks/queries point-count mismatch: "
            f"tracks N={num_points}, queries N={queries.shape[0]}."
        )

    if "frame_indices" in arrays:
        frame_indices = np.asarray(arrays["frame_indices"]).reshape(-1)
        if frame_indices.shape[0] != temporal_length:
            raise ValueError(
                "frame_indices length does not match tracks temporal length: "
                f"{frame_indices.shape[0]} vs {temporal_length}."
            )
        frame_indices = frame_indices.astype(np.int64, copy=False)
    else:
        frame_indices = np.arange(temporal_length, dtype=np.int64)

    return tracks, visibility, queries, frame_indices


def infer_grid_size(metadata: dict, num_points: int) -> Optional[int]:
    value = metadata.get("grid", {}).get("grid_size")
    if value is not None:
        try:
            value = int(value)
            if value > 0 and value * value == num_points:
                return value
        except (TypeError, ValueError):
            pass

    root = int(round(np.sqrt(num_points)))
    if root * root == num_points:
        return root

    return None


def select_point_ids(
    num_points: int,
    grid_size: Optional[int],
    point_stride: int,
    max_points: int,
) -> np.ndarray:
    if grid_size is not None and grid_size * grid_size == num_points:
        rows = np.arange(0, grid_size, point_stride, dtype=np.int64)
        cols = np.arange(0, grid_size, point_stride, dtype=np.int64)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        selected = (rr * grid_size + cc).reshape(-1)
    else:
        selected = np.arange(0, num_points, point_stride, dtype=np.int64)

    if max_points > 0 and selected.size > max_points:
        positions = np.linspace(
            0,
            selected.size - 1,
            max_points,
            dtype=np.int64,
        )
        selected = selected[positions]

    return selected


# ============================================================
# Drawing helpers
# ============================================================


def blend_color(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(
        int(round((1.0 - t) * a + t * b))
        for a, b in zip(color_a, color_b)
    )


def point_color(point_id: int) -> tuple[int, int, int]:
    # Deterministic multiplicative hash distributes adjacent point IDs across
    # the Fresh Aqua chart palette without storing a large color table.
    index = ((int(point_id) * 2654435761) & 0xFFFFFFFF) % len(CHART_COLORS)
    return CHART_COLORS[index]


def as_int_point(x: float, y: float) -> tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def inside_frame(x: float, y: float, width: int, height: int) -> bool:
    return (
        np.isfinite(x)
        and np.isfinite(y)
        and -1.0 <= x <= width
        and -1.0 <= y <= height
    )


def alpha_rect(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
    border_color: Optional[tuple[int, int, int]] = None,
    border_thickness: int = 1,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    h, w = frame.shape[:2]

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    if border_color is not None and border_thickness > 0:
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2 - 1, y2 - 1),
            border_color,
            border_thickness,
            lineType=cv2.LINE_AA,
        )


def fit_text_scale(width: int, height: int) -> float:
    return max(0.42, min(0.82, min(width / 1600.0, height / 900.0) * 0.72))


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_dashboard(
    frame: np.ndarray,
    args: argparse.Namespace,
    metadata: dict,
    source_frame_index: int,
    track_time_index: Optional[int],
    total_track_frames: int,
    visible_selected: int,
    selected_count: int,
    grid_size: Optional[int],
    output_fps: float,
) -> None:
    h, w = frame.shape[:2]
    scale = fit_text_scale(w, h)
    pad = max(10, int(round(18 * scale / 0.72)))
    line_h = max(18, int(round(27 * scale / 0.72)))

    panel_w = min(w - 2 * pad, max(360, int(w * 0.33)))
    panel_h = 4 * line_h + 2 * pad
    x1 = pad
    y1 = pad
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    alpha_rect(
        frame,
        (x1, y1),
        (x2, y2),
        BGR["aqua_950"],
        args.dashboard_alpha,
        border_color=BGR["aqua_500"],
        border_thickness=1,
    )

    # Brand stripe.
    stripe_w = max(4, int(round(6 * scale / 0.72)))
    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + stripe_w, y2 - 1),
        BGR["aqua_500"],
        -1,
    )

    text_x = x1 + pad
    baseline = y1 + pad + line_h - 6

    draw_text(
        frame,
        args.title,
        (text_x, baseline),
        scale * 1.05,
        BGR["white"],
        max(1, int(round(scale * 2.0))),
    )

    run_name = str(metadata.get("run_name", "run"))
    tracker_name = str(metadata.get("tracker", {}).get("model", "CoTracker"))
    draw_text(
        frame,
        f"{run_name}  |  {tracker_name}",
        (text_x, baseline + line_h),
        scale * 0.80,
        BGR["aqua_200"],
        1,
    )

    if track_time_index is None:
        frame_label = f"Frame {source_frame_index}  |  no track sample"
    else:
        frame_label = (
            f"Frame {source_frame_index}  |  track {track_time_index + 1}/"
            f"{total_track_frames}"
        )

    draw_text(
        frame,
        frame_label,
        (text_x, baseline + 2 * line_h),
        scale * 0.78,
        BGR["dark_secondary_text"],
        1,
    )

    grid_label = f"grid {grid_size}x{grid_size}" if grid_size else "irregular grid"
    draw_text(
        frame,
        (
            f"Visible {visible_selected:,}/{selected_count:,}  |  "
            f"{grid_label}  |  {output_fps:.2f} FPS"
        ),
        (text_x, baseline + 3 * line_h),
        scale * 0.78,
        BGR["dark_secondary_text"],
        1,
    )

    # Compact visibility badge in the top-right corner.
    badge_text = (
        f"{(100.0 * visible_selected / selected_count):.1f}% visible"
        if selected_count > 0
        else "0.0% visible"
    )
    (tw, th), _ = cv2.getTextSize(
        badge_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.75,
        1,
    )
    badge_pad_x = max(8, int(10 * scale / 0.72))
    badge_pad_y = max(6, int(8 * scale / 0.72))
    bx2 = w - pad
    bx1 = bx2 - tw - 2 * badge_pad_x
    by1 = pad
    by2 = by1 + th + 2 * badge_pad_y

    alpha_rect(
        frame,
        (bx1, by1),
        (bx2, by2),
        BGR["aqua_700"],
        args.dashboard_alpha,
        border_color=BGR["aqua_300"],
        border_thickness=1,
    )
    draw_text(
        frame,
        badge_text,
        (bx1 + badge_pad_x, by2 - badge_pad_y),
        scale * 0.75,
        BGR["white"],
        1,
    )


def draw_progress_bar(
    frame: np.ndarray,
    progress: float,
) -> None:
    h, w = frame.shape[:2]
    bar_h = max(4, int(round(h * 0.006)))
    progress = float(np.clip(progress, 0.0, 1.0))

    cv2.rectangle(
        frame,
        (0, h - bar_h),
        (w, h),
        BGR["aqua_900"],
        -1,
    )
    cv2.rectangle(
        frame,
        (0, h - bar_h),
        (int(round(w * progress)), h),
        BGR["aqua_500"],
        -1,
    )


def draw_query_grid(
    frame: np.ndarray,
    queries: np.ndarray,
    selected_ids: np.ndarray,
    x_scale: float,
    y_scale: float,
) -> None:
    h, w = frame.shape[:2]
    color = BGR["aqua_200"]

    for point_id in selected_ids:
        _, x, y = queries[point_id]
        x *= x_scale
        y *= y_scale
        if not inside_frame(x, y, w, h):
            continue
        cv2.circle(
            frame,
            as_int_point(x, y),
            2,
            color,
            1,
            lineType=cv2.LINE_AA,
        )


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    queries: np.ndarray,
    track_time_index: int,
    selected_ids: np.ndarray,
    args: argparse.Namespace,
    x_scale: float,
    y_scale: float,
) -> int:
    h, w = frame.shape[:2]
    layer = frame.copy()

    if args.draw_query_grid:
        draw_query_grid(
            layer,
            queries,
            selected_ids,
            x_scale,
            y_scale,
        )

    trail_start = max(0, track_time_index - args.trail_length + 1)

    visible_now = visibility[track_time_index, selected_ids]
    visible_count = int(np.count_nonzero(visible_now))

    for point_id in selected_ids:
        base_color = point_color(int(point_id))
        is_visible_now = bool(visibility[track_time_index, point_id])

        if args.draw_trails:
            for t in range(trail_start + 1, track_time_index + 1):
                if not (
                    visibility[t - 1, point_id]
                    and visibility[t, point_id]
                ):
                    continue

                x0, y0 = tracks[t - 1, point_id]
                x1, y1 = tracks[t, point_id]
                x0 *= x_scale
                y0 *= y_scale
                x1 *= x_scale
                y1 *= y_scale

                if not (
                    inside_frame(x0, y0, w, h)
                    and inside_frame(x1, y1, w, h)
                ):
                    continue

                if track_time_index == trail_start:
                    age_fraction = 1.0
                else:
                    age_fraction = (
                        (t - trail_start)
                        / max(1, track_time_index - trail_start)
                    )

                trail_color = blend_color(
                    BGR["aqua_900"],
                    base_color,
                    0.35 + 0.65 * age_fraction,
                )

                cv2.line(
                    layer,
                    as_int_point(x0, y0),
                    as_int_point(x1, y1),
                    trail_color,
                    args.trail_thickness,
                    lineType=cv2.LINE_AA,
                )

        if not args.draw_points:
            continue

        x, y = tracks[track_time_index, point_id]
        x *= x_scale
        y *= y_scale

        if not inside_frame(x, y, w, h):
            continue

        point = as_int_point(x, y)

        if is_visible_now:
            if args.point_radius > 0:
                cv2.circle(
                    layer,
                    point,
                    args.point_radius + 1,
                    BGR["aqua_950"],
                    -1,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    layer,
                    point,
                    args.point_radius,
                    base_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
        elif args.draw_hidden and args.point_radius > 0:
            hidden_color = blend_color(
                BGR["muted_text"],
                BGR["aqua_200"],
                args.hidden_alpha,
            )
            cv2.circle(
                layer,
                point,
                args.point_radius,
                hidden_color,
                1,
                lineType=cv2.LINE_AA,
            )

    cv2.addWeighted(
        layer,
        args.track_alpha,
        frame,
        1.0 - args.track_alpha,
        0.0,
        dst=frame,
    )

    return visible_count


# ============================================================
# Rendering
# ============================================================


def get_metadata_dimensions(
    metadata: dict,
    arrays: dict[str, np.ndarray],
) -> tuple[int, int]:
    width = metadata.get("video", {}).get("width")
    height = metadata.get("video", {}).get("height")

    if width is None and "width" in arrays:
        width = int(np.asarray(arrays["width"]).item())
    if height is None and "height" in arrays:
        height = int(np.asarray(arrays["height"]).item())

    if width is None or height is None:
        raise RuntimeError(
            "Could not determine source width/height from YAML or NPZ."
        )

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid metadata resolution: {width}x{height}."
        )

    return width, height


def make_canvas(width: int, height: int, background: str) -> np.ndarray:
    if background == "light":
        color = BGR["page_bg"]
    elif background == "dark":
        color = BGR["dark_bg"]
    else:
        raise ValueError(f"Canvas requested for unsupported mode: {background}")

    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :] = color

    # Subtle Fresh Aqua grid makes motion easier to judge on blank canvases.
    spacing = max(48, int(round(min(width, height) / 12)))
    grid_color = (
        BGR["border_light"]
        if background == "light"
        else BGR["dark_border"]
    )

    for x in range(0, width, spacing):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, spacing):
        cv2.line(frame, (0, y), (width, y), grid_color, 1)

    return frame


def render(args: argparse.Namespace) -> None:
    yaml_file = args.yaml_file.expanduser().resolve(strict=False)
    metadata = load_yaml(yaml_file)

    npz_file = resolve_npz_file(args, metadata, yaml_file)
    video_file = resolve_video_file(args, metadata, yaml_file)

    if args.output_file is None:
        output_file = yaml_file.parent / "track_visualization.mp4"
    else:
        output_file = args.output_file.expanduser()
        if not output_file.is_absolute():
            output_file = output_file.resolve(strict=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    arrays = load_npz(npz_file)
    tracks, visibility, queries, frame_indices = normalize_arrays(arrays)

    temporal_length, num_points, _ = tracks.shape
    grid_size = infer_grid_size(metadata, num_points)
    selected_ids = select_point_ids(
        num_points=num_points,
        grid_size=grid_size,
        point_stride=args.point_stride,
        max_points=args.max_points,
    )

    metadata_w, metadata_h = get_metadata_dimensions(metadata, arrays)

    frame_to_track: dict[int, int] = {
        int(source_index): int(track_index)
        for track_index, source_index in enumerate(frame_indices)
    }

    capture: Optional[cv2.VideoCapture] = None

    if args.background == "video":
        assert video_file is not None
        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source video: {video_file}")

        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError(
                f"Could not read video resolution from: {video_file}"
            )

        if source_fps <= 0 or not np.isfinite(source_fps):
            source_fps = args.canvas_fps

        if source_frame_count <= 0:
            source_frame_count = int(frame_indices.max()) + 1
    else:
        width = metadata_w
        height = metadata_h
        source_fps = args.canvas_fps
        source_frame_count = int(frame_indices.max()) + 1

    output_fps = args.output_fps if args.output_fps > 0 else source_fps

    start_frame = args.start_frame
    natural_end = source_frame_count - 1
    end_frame = natural_end if args.end_frame == -1 else min(args.end_frame, natural_end)

    if start_frame > natural_end:
        if capture is not None:
            capture.release()
        raise ValueError(
            f"--start-frame {start_frame} exceeds last available frame "
            f"{natural_end}."
        )

    if end_frame < start_frame:
        if capture is not None:
            capture.release()
        raise ValueError("No frames selected for rendering.")

    output_frame_count = end_frame - start_frame + 1

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        if capture is not None:
            capture.release()
        raise RuntimeError(
            f"Could not open output video writer: {output_file} "
            f"(codec={args.codec!r})."
        )

    x_scale = width / float(metadata_w)
    y_scale = height / float(metadata_h)

    print("=" * 72)
    print("CoTracker visualization · Fresh Aqua")
    print("=" * 72)
    print("YAML:       ", yaml_file)
    print("NPZ:        ", npz_file)
    print("Video:      ", video_file if video_file is not None else "themed canvas")
    print("Output:     ", output_file)
    print("Resolution: ", f"{width}x{height}")
    print("FPS:        ", f"{output_fps:.3f}")
    print("Tracks:     ", f"T={temporal_length}, N={num_points}")
    print("Selected:   ", f"{selected_ids.size} points")
    print("Frames:     ", f"{start_frame}..{end_frame}")
    print("=" * 72)

    # Seek only once before sequential decoding.
    if capture is not None and start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rendered = 0

    try:
        for source_frame_index in range(start_frame, end_frame + 1):
            if capture is not None:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print(
                        f"Warning: video decode ended at source frame "
                        f"{source_frame_index}."
                    )
                    break
            else:
                frame = make_canvas(width, height, args.background)

            track_time_index = frame_to_track.get(source_frame_index)

            if track_time_index is not None:
                visible_selected = draw_tracks(
                    frame=frame,
                    tracks=tracks,
                    visibility=visibility,
                    queries=queries,
                    track_time_index=track_time_index,
                    selected_ids=selected_ids,
                    args=args,
                    x_scale=x_scale,
                    y_scale=y_scale,
                )
            else:
                visible_selected = 0
                if args.draw_query_grid:
                    draw_query_grid(
                        frame,
                        queries,
                        selected_ids,
                        x_scale,
                        y_scale,
                    )

            if args.dashboard:
                draw_dashboard(
                    frame=frame,
                    args=args,
                    metadata=metadata,
                    source_frame_index=source_frame_index,
                    track_time_index=track_time_index,
                    total_track_frames=temporal_length,
                    visible_selected=visible_selected,
                    selected_count=int(selected_ids.size),
                    grid_size=grid_size,
                    output_fps=output_fps,
                )

            if args.progress_bar:
                progress = (
                    (source_frame_index - start_frame + 1)
                    / float(output_frame_count)
                )
                draw_progress_bar(frame, progress)

            writer.write(frame)
            rendered += 1

            if args.show:
                preview = frame
                if args.preview_scale != 1.0:
                    preview = cv2.resize(
                        frame,
                        None,
                        fx=args.preview_scale,
                        fy=args.preview_scale,
                        interpolation=(
                            cv2.INTER_AREA
                            if args.preview_scale < 1.0
                            else cv2.INTER_LINEAR
                        ),
                    )

                cv2.imshow(args.window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Preview stopped by user.")
                    break

            if args.print_every > 0 and (
                rendered == 1
                or rendered % args.print_every == 0
                or source_frame_index == end_frame
            ):
                percent = 100.0 * rendered / output_frame_count
                print(
                    f"Rendered {rendered:>6}/{output_frame_count} "
                    f"({percent:6.2f}%) | source frame {source_frame_index}"
                )

    finally:
        writer.release()
        if capture is not None:
            capture.release()
        if args.show:
            cv2.destroyAllWindows()

    if rendered == 0:
        raise RuntimeError("No frames were rendered.")

    print()
    print("Visualization complete.")
    print("Saved:", output_file)


# ============================================================
# Entry point
# ============================================================


def main() -> None:
    args = parse_args()
    render(args)


if __name__ == "__main__":
    main()