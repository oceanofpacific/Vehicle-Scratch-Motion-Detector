from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


@dataclass
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0 or self.frame_count <= 0:
            return 0.0
        return self.frame_count / self.fps


@dataclass
class RawEvent:
    video_path: Path
    video_index: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    max_motion_area: float
    max_edge_score: float
    max_damage_score: float
    zone_counts: dict[str, int]
    damage_zone_counts: dict[str, int]
    best_frame: bytes
    best_boxes: list[tuple[int, int, int, int]]
    best_time: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class FinalEvent:
    event_id: int
    video_path: Path
    video_index: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration_seconds: float
    max_motion_area: float
    confidence_score: float
    max_damage_score: float
    damage_zones: str
    screenshot_path: Path
    clip_path: Path | None
    best_frame: bytes
    best_boxes: list[tuple[int, int, int, int]]
    best_time: float


@dataclass
class MotionDetection:
    box: tuple[int, int, int, int]
    area: float
    centroid: tuple[float, float]
    near_edge: bool
    edge_score: float
    damage_zone_names: list[str]
    damage_score: float


@dataclass
class MotionTrack:
    track_id: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    missed: int = 0
    active_start: float | None = None
    active_end: float = 0.0
    active_start_frame: int | None = None
    active_end_frame: int = 0
    max_area: float = 0.0
    max_edge_score: float = 0.0
    max_damage_score: float = 0.0
    zone_counts: dict[str, int] | None = None
    damage_zone_counts: dict[str, int] | None = None
    best_frame: np.ndarray | None = None
    best_boxes: list[tuple[int, int, int, int]] | None = None
    best_time: float = 0.0


@dataclass
class BehaviorObservation:
    timestamp: float
    frame_index: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    area: float
    near_vehicle: bool
    sample_seconds: float
    damage_zone_names: list[str]
    damage_score: float


@dataclass
class BehaviorTrack:
    track_id: int
    video_path: Path
    video_index: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    missed: int = 0
    observations: list[BehaviorObservation] | None = None
    max_area: float = 0.0
    best_near_area: float = 0.0
    max_damage_score: float = 0.0
    damage_zone_counts: dict[str, int] | None = None
    best_frame: bytes | None = None
    best_box: tuple[int, int, int, int] | None = None
    best_time: float = 0.0
    best_frame_index: int = 0


@dataclass
class BehaviorCandidate:
    event_id: int
    track_id: int
    video_path: Path
    video_index: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration_seconds: float
    near_vehicle_seconds: float
    elsewhere_seconds: float
    avg_speed_near_px_s: float | None
    avg_speed_elsewhere_px_s: float | None
    speed_ratio_near_to_elsewhere: float | None
    dwell_percentile: float
    anomaly_score: float
    max_motion_area: float
    max_damage_score: float
    damage_zones: str
    reason: str
    screenshot_path: Path | None
    best_frame: bytes
    best_box: tuple[int, int, int, int]
    path_points: list[tuple[int, int]]


@dataclass
class BehaviorDetection:
    box: tuple[int, int, int, int]
    area: float
    centroid: tuple[float, float]
    near_vehicle: bool
    near_score: float
    damage_zone_names: list[str]
    damage_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find suspicious motion around a parked vehicle ROI in fixed-camera "
            "surveillance videos. Results are candidates for review, not conclusions."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input video files. Globs such as videos/*.mp4 are supported.",
    )
    parser.add_argument("--output", default="output", help="Output directory.")
    parser.add_argument("--min-area", type=float, default=800.0, help="Minimum motion area in pixels.")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Minimum continuous motion duration in seconds before recording an event.",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=10.0,
        help="Merge events in the same video when the gap is this many seconds or less.",
    )
    parser.add_argument("--pre-roll", type=float, default=15.0, help="Clip seconds before event start.")
    parser.add_argument("--post-roll", type=float, default=15.0, help="Clip seconds after event end.")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=2,
        help="Analyze every Nth frame to speed up processing.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show an interactive ROI/mask/motion-box preview for tuning parameters.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Show random sampled detection frames to help tune min-area and min-duration.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=12,
        help="Number of random samples to show in calibration mode.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=7,
        help="Random seed used by calibration mode.",
    )
    parser.add_argument(
        "--calibration-warmup-seconds",
        type=float,
        default=2.0,
        help="Seconds before each calibration sample used to warm up the background model.",
    )
    parser.add_argument(
        "--annotate-vehicle-edge",
        action="store_true",
        help="Open an interactive window to click vehicle body edge points and save them.",
    )
    parser.add_argument(
        "--vehicle-edge-file",
        default=None,
        help="Path to vehicle edge annotation JSON. Defaults to output/vehicle_edges.json.",
    )
    parser.add_argument(
        "--trajectory-scan",
        action="store_true",
        help="Use saved vehicle edge annotation and motion tracking to find edge-proximity events.",
    )
    parser.add_argument(
        "--annotate-damage-zones",
        action="store_true",
        help="Interactively draw key damage-prone zones such as left rear and right side.",
    )
    parser.add_argument(
        "--damage-zones-file",
        default=None,
        help="Path to damage zones JSON. Defaults to output/damage_zones.json.",
    )
    parser.add_argument(
        "--damage-zone-names",
        default="left_rear,right_side",
        help="Comma-separated labels assigned to selected damage zones in order.",
    )
    parser.add_argument(
        "--damage-distance",
        type=int,
        default=80,
        help="Pixel distance around each damage zone treated as near-damage-zone.",
    )
    parser.add_argument(
        "--require-damage-zone",
        action="store_true",
        help="In trajectory scan, only keep events that touch or approach a marked damage zone.",
    )
    parser.add_argument(
        "--edge-distance",
        type=int,
        default=60,
        help="Pixel distance around the annotated vehicle edge treated as near-contact zone.",
    )
    parser.add_argument(
        "--track-max-distance",
        type=float,
        default=140.0,
        help="Maximum centroid distance in pixels for associating a moving target across sampled frames.",
    )
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=4,
        help="Sampled frames a tracked target may disappear before its proximity event is closed.",
    )
    parser.add_argument(
        "--behavior-scan",
        action="store_true",
        help="Track moving people/objects across the full frame and rank long-stay or slow-near-vehicle behavior.",
    )
    parser.add_argument(
        "--behavior-top",
        type=int,
        default=80,
        help="Maximum behavior candidates to keep in the behavior report.",
    )
    parser.add_argument(
        "--behavior-min-track-duration",
        type=float,
        default=2.0,
        help="Minimum tracked target duration in seconds for behavior analysis.",
    )
    parser.add_argument(
        "--behavior-min-near-duration",
        type=float,
        default=2.0,
        help="Minimum time a track must spend inside the vehicle ROI to appear in behavior analysis.",
    )
    parser.add_argument(
        "--behavior-min-elsewhere-duration",
        type=float,
        default=1.5,
        help="Minimum non-vehicle time needed before comparing near-vehicle speed to elsewhere speed.",
    )
    parser.add_argument(
        "--behavior-near-padding",
        type=int,
        default=0,
        help="Pixels to expand the vehicle ROI when deciding whether a target is near the vehicle.",
    )
    parser.add_argument(
        "--behavior-analysis-padding",
        type=int,
        default=220,
        help="Pixels around the vehicle ROI to include for behavior speed comparison.",
    )
    parser.add_argument(
        "--behavior-scale",
        type=float,
        default=0.5,
        help="Resize factor for behavior analysis frames. Lower values are faster but less precise.",
    )
    parser.add_argument(
        "--person-min-height",
        type=int,
        default=60,
        help="Minimum motion-box height in pixels for behavior-track candidates.",
    )
    parser.add_argument(
        "--person-min-height-width-ratio",
        type=float,
        default=1.0,
        help="Minimum height/width ratio for behavior-track candidates.",
    )
    parser.add_argument(
        "--person-max-area-ratio",
        type=float,
        default=0.12,
        help="Ignore motion boxes larger than this fraction of the full frame during behavior analysis.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI as x,y,w,h. If omitted, the first frame is shown for mouse selection.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to use for background model warmup at the start of each video.",
    )
    parser.add_argument(
        "--no-debug-video",
        action="store_true",
        help="Skip writing debug_video.mp4.",
    )
    return parser.parse_args()


def expand_inputs(inputs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = glob.glob(item)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(item))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            raise FileNotFoundError(f"Input video does not exist: {path}")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in VIDEO_EXTENSIONS:
                    child = child.resolve()
                    if child not in seen:
                        seen.add(child)
                        resolved.append(child)
            continue
        resolved.append(path)

    if not resolved:
        raise FileNotFoundError("No input videos found.")
    return resolved


def get_video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video dimensions: {path}")

    return VideoInfo(path=path, fps=fps, width=width, height=height, frame_count=frame_count)


def collect_video_infos(paths: Sequence[Path]) -> list[VideoInfo]:
    infos: list[VideoInfo] = []
    for path in paths:
        try:
            infos.append(get_video_info(path))
        except Exception as exc:
            print(f"Warning: skipping unreadable video: {path} ({exc})", file=sys.stderr)
    if not infos:
        raise RuntimeError("No readable videos found.")
    return infos


def read_first_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from: {path}")
    return frame


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be formatted as x,y,w,h")
    x, y, w, h = (int(float(part)) for part in parts)
    if w <= 0 or h <= 0:
        raise ValueError("--roi width and height must be positive")
    return x, y, w, h


def select_roi(first_frame: np.ndarray) -> tuple[int, int, int, int]:
    window_name = "Select vehicle ROI, then press ENTER or SPACE"
    print("Select the vehicle ROI and surrounding 1-2 meters, then press ENTER or SPACE.")
    roi = cv2.selectROI(window_name, first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection was cancelled or empty.")
    return x, y, w, h


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def resolve_vehicle_edge_file(output_dir: Path, value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return (output_dir / "vehicle_edges.json").resolve()


def resolve_damage_zones_file(output_dir: Path, value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return (output_dir / "damage_zones.json").resolve()


def parse_damage_zone_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    return names or ["zone_1", "zone_2"]


def draw_vehicle_edge(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    points: Sequence[Sequence[int]] | None,
    closed: bool = True,
    color: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    if not points:
        return frame
    x, y, _, _ = roi
    pts = np.array([[int(px) + x, int(py) + y] for px, py in points], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts], bool(closed and len(pts) >= 3), color, thickness, cv2.LINE_AA)
    for idx, point in enumerate(pts, start=1):
        cv2.circle(frame, tuple(point), 4, color, -1)
        cv2.putText(
            frame,
            str(idx),
            (int(point[0]) + 5, int(point[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_damage_zones(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    zones: Sequence[dict[str, object]] | None,
    color: tuple[int, int, int] = (0, 128, 255),
    thickness: int = 2,
) -> np.ndarray:
    if not zones:
        return frame
    rx, ry, _, _ = roi
    for zone in zones:
        x = int(zone.get("x", 0))
        y = int(zone.get("y", 0))
        w = int(zone.get("w", 0))
        h = int(zone.get("h", 0))
        name = str(zone.get("name", "zone"))
        if w <= 0 or h <= 0:
            continue
        p1 = (rx + x, ry + y)
        p2 = (rx + x + w, ry + y + h)
        cv2.rectangle(frame, p1, p2, color, thickness)
        cv2.putText(frame, name, (p1[0] + 4, max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def annotate_vehicle_edge(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    output_path: Path,
    source_video: Path,
) -> None:
    x, y, w, h = roi
    crop = frame[y : y + h, x : x + w].copy()
    points: list[tuple[int, int]] = []
    state = {"closed": True, "saved": False, "cancelled": False}
    window_name = "Vehicle edge annotation"

    def redraw() -> None:
        canvas = crop.copy()
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], bool(state["closed"] and len(points) >= 3), (255, 0, 255), 2, cv2.LINE_AA)
        for idx, point in enumerate(points, start=1):
            cv2.circle(canvas, point, 5, (0, 255, 255), -1)
            cv2.putText(canvas, str(idx), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        mode = "closed" if state["closed"] else "open"
        lines = [
            "Left click: add point | Backspace/U: undo | R: reset",
            "P: toggle open/closed | Enter/Space: save | Q/Esc: cancel",
            f"points={len(points)} mode={mode}",
        ]
        for i, line in enumerate(lines):
            y0 = 24 + i * 24
            cv2.rectangle(canvas, (8, y0 - 18), (min(w - 1, 8 + len(line) * 9), y0 + 6), (0, 0, 0), -1)
            cv2.putText(canvas, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)

    def on_mouse(event: int, mx: int, my: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((max(0, min(mx, w - 1)), max(0, min(my, h - 1))))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    print("Vehicle edge annotation:")
    print("  Left click points along the visible vehicle body edge.")
    print("  Press P to toggle open/closed path, Enter or Space to save, Q/Esc to cancel.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10, 32):
            if len(points) < 2:
                print("Need at least 2 edge points before saving.")
                continue
            state["saved"] = True
            break
        if key in (27, ord("q"), ord("Q")):
            state["cancelled"] = True
            break
        if key in (8, 127, ord("u"), ord("U")) and points:
            points.pop()
            redraw()
        if key in (ord("r"), ord("R")):
            points.clear()
            redraw()
        if key in (ord("p"), ord("P")):
            state["closed"] = not state["closed"]
            redraw()

    cv2.destroyWindow(window_name)
    if state["cancelled"] or not state["saved"]:
        raise RuntimeError("Vehicle edge annotation was cancelled.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_video": str(source_video),
        "roi": [int(v) for v in roi],
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "points": [[int(px), int(py)] for px, py in points],
        "closed": bool(state["closed"] and len(points) >= 3),
        "coordinate_space": "roi",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Vehicle edge annotation saved: {output_path}")


def annotate_damage_zones(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    output_path: Path,
    source_video: Path,
    zone_names: Sequence[str],
) -> None:
    x, y, w, h = roi
    crop = frame[y : y + h, x : x + w].copy()
    zones: list[dict[str, object]] = []
    state: dict[str, object] = {"dragging": False, "start": None, "current": None, "cancelled": False}
    window_name = "Damage zones annotation"

    def redraw() -> None:
        canvas = crop.copy()
        draw_damage_zones(canvas, (0, 0, w, h), zones)
        if state["dragging"] and state["start"] and state["current"]:
            sx, sy = state["start"]  # type: ignore[misc]
            cx, cy = state["current"]  # type: ignore[misc]
            cv2.rectangle(canvas, (sx, sy), (cx, cy), (0, 255, 255), 2)

        next_name = zone_names[len(zones)] if len(zones) < len(zone_names) else f"zone_{len(zones) + 1}"
        lines = [
            f"Drag box for: {next_name}",
            "Left drag: add box | Backspace/U: undo | R: reset",
            "Enter/Space/S: save | Q/Esc: cancel",
        ]
        for i, line in enumerate(lines):
            y0 = 24 + i * 24
            cv2.rectangle(canvas, (8, y0 - 18), (min(w - 1, 8 + len(line) * 9), y0 + 6), (0, 0, 0), -1)
            cv2.putText(canvas, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)

    def add_zone(p1: tuple[int, int], p2: tuple[int, int]) -> None:
        x1 = max(0, min(p1[0], p2[0]))
        y1 = max(0, min(p1[1], p2[1]))
        x2 = min(w - 1, max(p1[0], p2[0]))
        y2 = min(h - 1, max(p1[1], p2[1]))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return
        name = zone_names[len(zones)] if len(zones) < len(zone_names) else f"zone_{len(zones) + 1}"
        zones.append({"name": name, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})

    def on_mouse(event: int, mx: int, my: int, _flags: int, _userdata: object) -> None:
        point = (max(0, min(mx, w - 1)), max(0, min(my, h - 1)))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["start"] = point
            state["current"] = point
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["current"] = point
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            start = state["start"]
            if start:
                add_zone(start, point)  # type: ignore[arg-type]
            state["dragging"] = False
            state["start"] = None
            state["current"] = None
            redraw()

    print("Damage zone annotation:")
    print("  Drag boxes for key damage areas, for example left rear then right side.")
    print("  Press Enter/Space/S to save. Press Q/Esc to cancel.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10, 32, ord("s"), ord("S")):
            if not zones:
                print("Need at least one damage zone before saving.")
                continue
            break
        if key in (27, ord("q"), ord("Q")):
            state["cancelled"] = True
            break
        if key in (8, 127, ord("u"), ord("U")) and zones:
            zones.pop()
            redraw()
        if key in (ord("r"), ord("R")):
            zones.clear()
            redraw()

    cv2.destroyWindow(window_name)
    if state["cancelled"]:
        raise RuntimeError("Damage zone annotation was cancelled.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_video": str(source_video),
        "roi": [int(v) for v in roi],
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "zones": zones,
        "coordinate_space": "roi",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Damage zones saved: {output_path}")


def load_vehicle_edge_annotation(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Vehicle edge annotation not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(f"Vehicle edge annotation needs at least 2 points: {path}")
    return payload


def load_damage_zones_annotation(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Damage zones annotation not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    zones = payload.get("zones")
    if not isinstance(zones, list) or not zones:
        raise ValueError(f"Damage zones annotation has no zones: {path}")
    return payload


def damage_zones_from_annotation(annotation: dict[str, object]) -> list[dict[str, object]]:
    zones = annotation.get("zones", [])
    result: list[dict[str, object]] = []
    for idx, zone in enumerate(zones):  # type: ignore[assignment]
        if not isinstance(zone, dict):
            continue
        result.append(
            {
                "name": str(zone.get("name", f"zone_{idx + 1}")),
                "x": int(zone.get("x", 0)),
                "y": int(zone.get("y", 0)),
                "w": int(zone.get("w", 0)),
                "h": int(zone.get("h", 0)),
            }
        )
    return result


def edge_points_from_annotation(annotation: dict[str, object]) -> list[tuple[int, int]]:
    points = annotation.get("points", [])
    return [(int(point[0]), int(point[1])) for point in points]  # type: ignore[index]


def edge_closed_from_annotation(annotation: dict[str, object]) -> bool:
    return bool(annotation.get("closed", True))


def build_edge_zone_mask(
    roi: tuple[int, int, int, int],
    points: Sequence[tuple[int, int]],
    closed: bool,
    edge_distance: int,
) -> np.ndarray:
    _, _, w, h = roi
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 2:
        return mask
    pts = np.array(points, dtype=np.int32)
    thickness = max(3, int(edge_distance) * 2 + 1)
    cv2.polylines(mask, [pts], bool(closed and len(points) >= 3), 255, thickness, cv2.LINE_AA)
    return mask


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def safe_name(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return clean or "video"


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    screenshots_dir = output_dir / "screenshots"
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    return screenshots_dir, clips_dir


def write_image(path: Path, image: np.ndarray) -> None:
    encoded = encode_image_bytes(image, path.suffix or ".jpg")
    path.write_bytes(encoded)


def encode_image_bytes(image: np.ndarray, extension: str = ".jpg") -> bytes:
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError("Could not encode image")
    return encoded.tobytes()


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode stored event image")
    return image


def create_debug_writer(
    output_dir: Path,
    first_info: VideoInfo,
    sample_every: int,
    enabled: bool,
) -> tuple[cv2.VideoWriter | None, tuple[int, int] | None]:
    if not enabled:
        return None, None

    debug_path = output_dir / "debug_video.mp4"
    output_fps = max(1.0, first_info.fps / max(1, sample_every))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (first_info.width, first_info.height)
    writer = cv2.VideoWriter(str(debug_path), fourcc, output_fps, size)
    if not writer.isOpened():
        print(f"Warning: could not create debug video at {debug_path}", file=sys.stderr)
        return None, None
    return writer, size


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def find_motion_boxes(mask: np.ndarray, min_area: float) -> tuple[float, list[tuple[int, int, int, int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0.0
    boxes: list[tuple[int, int, int, int]] = []
    box_area_threshold = max(40.0, min_area * 0.08)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        total_area += area
        if area >= box_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    return total_area, boxes


def find_motion_detections(mask: np.ndarray, min_area: float) -> list[tuple[tuple[int, int, int, int], float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[tuple[tuple[int, int, int, int], float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        detections.append((cv2.boundingRect(contour), area))
    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def bbox_centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def bbox_edge_proximity(
    box: tuple[int, int, int, int],
    edge_zone_mask: np.ndarray,
) -> tuple[bool, float]:
    x, y, w, h = box
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(edge_zone_mask.shape[1], x + w)
    y2 = min(edge_zone_mask.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return False, 0.0
    zone_pixels = int(cv2.countNonZero(edge_zone_mask[y1:y2, x1:x2]))
    if zone_pixels <= 0:
        return False, 0.0
    box_area = max(1, (x2 - x1) * (y2 - y1))
    score = clamp01(zone_pixels / max(1.0, box_area * 0.35))
    return True, score


def bbox_damage_zone_proximity(
    box: tuple[int, int, int, int],
    damage_zones: Sequence[dict[str, object]] | None,
    damage_distance: int,
) -> tuple[list[str], float]:
    if not damage_zones:
        return [], 0.0
    bx, by, bw, bh = box
    motion_rect = (float(bx), float(by), float(bx + bw), float(by + bh))
    names: list[str] = []
    best_score = 0.0
    threshold = max(1.0, float(damage_distance))
    for zone in damage_zones:
        zx = float(zone.get("x", 0))
        zy = float(zone.get("y", 0))
        zw = float(zone.get("w", 0))
        zh = float(zone.get("h", 0))
        if zw <= 0 or zh <= 0:
            continue
        zone_rect = (zx, zy, zx + zw, zy + zh)
        distance = rect_distance(motion_rect, zone_rect)
        if distance <= threshold:
            names.append(str(zone.get("name", "zone")))
            best_score = max(best_score, clamp01(1.0 - distance / threshold))
    return names, best_score


def build_motion_detection(
    box: tuple[int, int, int, int],
    area: float,
    edge_zone_mask: np.ndarray,
    damage_zones: Sequence[dict[str, object]] | None = None,
    damage_distance: int = 80,
) -> MotionDetection:
    near_edge, edge_score = bbox_edge_proximity(box, edge_zone_mask)
    damage_zone_names, damage_score = bbox_damage_zone_proximity(box, damage_zones, damage_distance)
    return MotionDetection(
        box=box,
        area=area,
        centroid=bbox_centroid(box),
        near_edge=near_edge,
        edge_score=edge_score,
        damage_zone_names=damage_zone_names,
        damage_score=damage_score,
    )


def bbox_mask_proximity(box: tuple[int, int, int, int], mask: np.ndarray) -> tuple[bool, float]:
    x, y, w, h = box
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(mask.shape[1], x + w)
    y2 = min(mask.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return False, 0.0
    overlap = int(cv2.countNonZero(mask[y1:y2, x1:x2]))
    if overlap <= 0:
        return False, 0.0
    area = max(1, (x2 - x1) * (y2 - y1))
    return True, clamp01(overlap / max(1.0, area * 0.25))


def expand_rect(
    rect: tuple[int, int, int, int],
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    pad = max(0, int(padding))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def build_behavior_near_mask(
    frame_width: int,
    frame_height: int,
    roi: tuple[int, int, int, int],
    near_padding: int,
) -> np.ndarray:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    x, y, w, h = expand_rect(roi, near_padding, frame_width, frame_height)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def full_frame_damage_zone_proximity(
    box: tuple[int, int, int, int],
    roi: tuple[int, int, int, int],
    damage_zones: Sequence[dict[str, object]] | None,
    damage_distance: int,
) -> tuple[list[str], float]:
    if not damage_zones:
        return [], 0.0
    roi_x, roi_y, _, _ = roi
    bx, by, bw, bh = box
    roi_box = (bx - roi_x, by - roi_y, bw, bh)
    return bbox_damage_zone_proximity(roi_box, damage_zones, damage_distance)


def is_behavior_detection_candidate(
    box: tuple[int, int, int, int],
    area: float,
    frame_width: int,
    frame_height: int,
    min_height: int,
    min_height_width_ratio: float,
    max_area_ratio: float,
) -> bool:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    if h < max(1, min_height):
        return False
    if h / max(1.0, float(w)) < min_height_width_ratio:
        return False
    frame_area = max(1, frame_width * frame_height)
    if area > frame_area * max(0.001, max_area_ratio):
        return False
    if w > frame_width * 0.65 or h > frame_height * 0.95:
        return False
    return True


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def rect_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def motion_edge_score(boxes: Iterable[tuple[int, int, int, int]], roi: tuple[int, int, int, int]) -> float:
    _, _, roi_w, roi_h = roi
    if roi_w <= 0 or roi_h <= 0:
        return 0.0

    # The ROI includes the vehicle plus nearby space. Treat the central inset
    # rectangle as a simple vehicle-body proxy for scoring, not as evidence.
    inset_x = roi_w * 0.18
    inset_y = roi_h * 0.18
    vehicle_proxy = (inset_x, inset_y, roi_w - inset_x, roi_h - inset_y)
    near_range = max(20.0, min(roi_w, roi_h) * 0.25)

    best = 0.0
    for bx, by, bw, bh in boxes:
        motion_rect = (float(bx), float(by), float(bx + bw), float(by + bh))
        distance = rect_distance(motion_rect, vehicle_proxy)
        best = max(best, clamp01(1.0 - distance / near_range))
    return best


def motion_zone_counts(
    boxes: Iterable[tuple[int, int, int, int]],
    roi: tuple[int, int, int, int],
    cols: int = 4,
    rows: int = 4,
) -> dict[str, int]:
    _, _, roi_w, roi_h = roi
    if roi_w <= 0 or roi_h <= 0:
        return {}

    touched: set[str] = set()
    for bx, by, bw, bh in boxes:
        center_x = clamp01((bx + bw / 2.0) / roi_w)
        center_y = clamp01((by + bh / 2.0) / roi_h)
        col = min(cols - 1, int(center_x * cols))
        row = min(rows - 1, int(center_y * rows))
        touched.add(f"{row}:{col}")
    return {key: 1 for key in touched}


def merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    merged = dict(a)
    for key, value in b.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def confidence_score(event: RawEvent, min_area: float) -> float:
    duration_part = clamp01(event.duration_seconds / 8.0) * 30.0
    area_part = clamp01(event.max_motion_area / max(min_area * 5.0, 1.0)) * 20.0
    edge_part = clamp01(event.max_edge_score) * 25.0
    repeated_zone_hits = max(event.zone_counts.values(), default=0)
    repeat_part = clamp01(repeated_zone_hits / 8.0) * 10.0
    damage_part = clamp01(event.max_damage_score) * 15.0
    return round(min(100.0, duration_part + area_part + edge_part + repeat_part + damage_part), 1)


def annotate_frame(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    boxes: Iterable[tuple[int, int, int, int]],
    timestamp_seconds: float,
    video_path: Path,
    motion_area: float | None = None,
    active: bool = False,
    edge_points: Sequence[Sequence[int]] | None = None,
    edge_closed: bool = True,
    damage_zones: Sequence[dict[str, object]] | None = None,
    label_suffix: str | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    x, y, w, h = roi
    roi_color = (0, 255, 255)
    box_color = (0, 0, 255)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), roi_color, 2)
    draw_vehicle_edge(annotated, roi, edge_points, closed=edge_closed, color=(255, 0, 255), thickness=2)
    draw_damage_zones(annotated, roi, damage_zones)

    for bx, by, bw, bh in boxes:
        cv2.rectangle(
            annotated,
            (x + bx, y + by),
            (x + bx + bw, y + by + bh),
            box_color,
            2,
        )

    label_parts = [video_path.name, format_time(timestamp_seconds)]
    if motion_area is not None:
        label_parts.append(f"area={motion_area:.0f}")
    if active:
        label_parts.append("candidate")
    if label_suffix:
        label_parts.append(label_suffix)
    label = " | ".join(label_parts)

    cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 1, 8 + len(label) * 10), 40), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        label,
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def finalize_raw_event(
    raw_events: list[RawEvent],
    video_path: Path,
    video_index: int,
    start_time: float | None,
    end_time: float,
    start_frame: int | None,
    end_frame: int,
    max_motion_area: float,
    max_edge_score: float,
    max_damage_score: float,
    zone_counts: dict[str, int],
    damage_zone_counts: dict[str, int] | None,
    best_frame: np.ndarray | None,
    best_boxes: list[tuple[int, int, int, int]],
    best_time: float,
    min_duration: float,
) -> None:
    if start_time is None or start_frame is None or best_frame is None:
        return
    duration = max(0.0, end_time - start_time)
    if duration < min_duration:
        return
    raw_events.append(
        RawEvent(
            video_path=video_path,
            video_index=video_index,
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            max_motion_area=max_motion_area,
            max_edge_score=max_edge_score,
            max_damage_score=max_damage_score,
            zone_counts=dict(zone_counts),
            damage_zone_counts=dict(damage_zone_counts or {}),
            best_frame=encode_image_bytes(best_frame),
            best_boxes=list(best_boxes),
            best_time=best_time,
        )
    )


def process_video(
    info: VideoInfo,
    video_index: int,
    roi: tuple[int, int, int, int],
    min_area: float,
    min_duration: float,
    sample_every: int,
    warmup_frames: int,
    debug_writer: cv2.VideoWriter | None,
    debug_size: tuple[int, int] | None,
) -> list[RawEvent]:
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {info.path}")

    local_roi = clamp_roi(roi, info.width, info.height)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    raw_events: list[RawEvent] = []
    sample_every = max(1, sample_every)
    sample_step_seconds = sample_every / info.fps

    active_start: float | None = None
    active_end = 0.0
    active_start_frame: int | None = None
    active_end_frame = 0
    active_max_area = 0.0
    active_max_edge_score = 0.0
    active_zone_counts: dict[str, int] = {}
    active_best_frame: np.ndarray | None = None
    active_best_boxes: list[tuple[int, int, int, int]] = []
    active_best_time = 0.0

    def close_active_event() -> None:
        nonlocal active_start
        nonlocal active_end
        nonlocal active_start_frame
        nonlocal active_end_frame
        nonlocal active_max_area
        nonlocal active_max_edge_score
        nonlocal active_zone_counts
        nonlocal active_best_frame
        nonlocal active_best_boxes
        nonlocal active_best_time

        if active_start is not None:
            finalize_raw_event(
                raw_events=raw_events,
                video_path=info.path,
                video_index=video_index,
                start_time=active_start,
                end_time=active_end,
                start_frame=active_start_frame,
                end_frame=active_end_frame,
                max_motion_area=active_max_area,
                max_edge_score=active_max_edge_score,
                max_damage_score=0.0,
                zone_counts=active_zone_counts,
                damage_zone_counts={},
                best_frame=active_best_frame,
                best_boxes=active_best_boxes,
                best_time=active_best_time,
                min_duration=min_duration,
            )

        active_start = None
        active_end = 0.0
        active_start_frame = None
        active_end_frame = 0
        active_max_area = 0.0
        active_max_edge_score = 0.0
        active_zone_counts = {}
        active_best_frame = None
        active_best_boxes = []
        active_best_time = 0.0

    def recover_decode_gap() -> bool:
        nonlocal frame_index

        if info.frame_count <= 0:
            return False
        next_frame = frame_index + max(1, sample_every)
        if next_frame >= info.frame_count:
            return False
        close_active_event()
        frame_index = next_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return True

    frame_index = 0
    processed = 0

    while True:
        if frame_index % sample_every != 0:
            ok = cap.grab()
            if not ok:
                if recover_decode_gap():
                    continue
                break
            frame_index += 1
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            if recover_decode_gap():
                continue
            break

        timestamp = frame_index / info.fps
        x, y, w, h = local_roi
        roi_frame = frame[y : y + h, x : x + w]

        fgmask = subtractor.apply(roi_frame)
        mask = preprocess_mask(fgmask)
        motion_area, boxes = find_motion_boxes(mask, min_area)
        edge_score = motion_edge_score(boxes, local_roi)
        zone_counts = motion_zone_counts(boxes, local_roi)

        warming_up = frame_index < warmup_frames
        motion_detected = (motion_area >= min_area) and not warming_up

        if motion_detected:
            if active_start is None:
                active_start = timestamp
                active_start_frame = frame_index
                active_max_area = 0.0
                active_max_edge_score = 0.0
                active_zone_counts = {}
            active_end = min(info.duration_seconds or (timestamp + sample_step_seconds), timestamp + sample_step_seconds)
            active_end_frame = min(
                max(info.frame_count - 1, frame_index),
                frame_index + sample_every - 1,
            )
            active_max_edge_score = max(active_max_edge_score, edge_score)
            active_zone_counts = merge_counts(active_zone_counts, zone_counts)
            if motion_area >= active_max_area:
                active_max_area = motion_area
                active_best_frame = frame.copy()
                active_best_boxes = list(boxes)
                active_best_time = timestamp
        elif active_start is not None:
            close_active_event()

        if debug_writer is not None:
            annotated = annotate_frame(
                frame=frame,
                roi=local_roi,
                boxes=boxes,
                timestamp_seconds=timestamp,
                video_path=info.path,
                motion_area=motion_area,
                active=motion_detected,
            )
            if debug_size is not None and (annotated.shape[1], annotated.shape[0]) != debug_size:
                annotated = cv2.resize(annotated, debug_size)
            debug_writer.write(annotated)

        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} sampled frames from {info.path.name}...", flush=True)

        frame_index += 1

    if active_start is not None:
        close_active_event()

    cap.release()
    return raw_events


def process_video_trajectory(
    info: VideoInfo,
    video_index: int,
    roi: tuple[int, int, int, int],
    vehicle_edge_points: Sequence[tuple[int, int]],
    vehicle_edge_closed: bool,
    damage_zones: Sequence[dict[str, object]] | None,
    edge_distance: int,
    damage_distance: int,
    require_damage_zone: bool,
    min_area: float,
    min_duration: float,
    sample_every: int,
    warmup_frames: int,
    track_max_distance: float,
    track_max_missed: int,
    debug_writer: cv2.VideoWriter | None,
    debug_size: tuple[int, int] | None,
) -> list[RawEvent]:
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {info.path}")

    local_roi = clamp_roi(roi, info.width, info.height)
    edge_zone_mask = build_edge_zone_mask(local_roi, vehicle_edge_points, vehicle_edge_closed, edge_distance)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    raw_events: list[RawEvent] = []
    tracks: dict[int, MotionTrack] = {}
    next_track_id = 1
    sample_every = max(1, sample_every)
    sample_step_seconds = sample_every / info.fps
    track_max_missed = max(0, track_max_missed)

    def close_track_event(track: MotionTrack) -> None:
        if track.active_start is None:
            return
        finalize_raw_event(
            raw_events=raw_events,
            video_path=info.path,
            video_index=video_index,
            start_time=track.active_start,
            end_time=track.active_end,
            start_frame=track.active_start_frame,
            end_frame=track.active_end_frame,
            max_motion_area=track.max_area,
            max_edge_score=track.max_edge_score,
            max_damage_score=track.max_damage_score,
            zone_counts=track.zone_counts or {},
            damage_zone_counts=track.damage_zone_counts or {},
            best_frame=track.best_frame,
            best_boxes=track.best_boxes or [track.box],
            best_time=track.best_time,
            min_duration=min_duration,
        )
        track.active_start = None
        track.active_end = 0.0
        track.active_start_frame = None
        track.active_end_frame = 0
        track.max_area = 0.0
        track.max_edge_score = 0.0
        track.max_damage_score = 0.0
        track.zone_counts = {}
        track.damage_zone_counts = {}
        track.best_frame = None
        track.best_boxes = []
        track.best_time = 0.0

    def update_track_event(track: MotionTrack, detection: MotionDetection, frame: np.ndarray, timestamp: float, frame_index: int) -> None:
        if track.active_start is None:
            track.active_start = timestamp
            track.active_start_frame = frame_index
            track.max_area = 0.0
            track.max_edge_score = 0.0
            track.max_damage_score = 0.0
            track.zone_counts = {}
            track.damage_zone_counts = {}
        track.active_end = min(info.duration_seconds or (timestamp + sample_step_seconds), timestamp + sample_step_seconds)
        track.active_end_frame = min(max(info.frame_count - 1, frame_index), frame_index + sample_every - 1)
        track.max_edge_score = max(track.max_edge_score, detection.edge_score)
        track.max_damage_score = max(track.max_damage_score, detection.damage_score)
        track.zone_counts = merge_counts(track.zone_counts or {}, motion_zone_counts([detection.box], local_roi))
        damage_counts = track.damage_zone_counts or {}
        for name in detection.damage_zone_names:
            damage_counts[name] = damage_counts.get(name, 0) + 1
        track.damage_zone_counts = damage_counts
        if detection.area >= track.max_area:
            track.max_area = detection.area
            track.best_frame = frame.copy()
            track.best_boxes = [detection.box]
            track.best_time = timestamp

    def recover_decode_gap() -> bool:
        nonlocal frame_index

        if info.frame_count <= 0:
            return False
        next_frame = frame_index + max(1, sample_every)
        if next_frame >= info.frame_count:
            return False
        for track in tracks.values():
            close_track_event(track)
        tracks.clear()
        frame_index = next_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return True

    frame_index = 0
    processed = 0

    while True:
        if frame_index % sample_every != 0:
            ok = cap.grab()
            if not ok:
                if recover_decode_gap():
                    continue
                break
            frame_index += 1
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            if recover_decode_gap():
                continue
            break

        timestamp = frame_index / info.fps
        x, y, w, h = local_roi
        roi_frame = frame[y : y + h, x : x + w]
        fgmask = subtractor.apply(roi_frame)
        mask = preprocess_mask(fgmask)
        warming_up = frame_index < warmup_frames

        detections: list[MotionDetection] = []
        if not warming_up:
            detections = [
                build_motion_detection(box, area, edge_zone_mask, damage_zones, damage_distance)
                for box, area in find_motion_detections(mask, min_area)
            ]

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for det_index, detection in enumerate(detections):
            best_track_id: int | None = None
            best_distance = float("inf")
            for track_id, track in tracks.items():
                if track_id in matched_tracks:
                    continue
                distance = math.hypot(detection.centroid[0] - track.centroid[0], detection.centroid[1] - track.centroid[1])
                if distance < best_distance and distance <= track_max_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                track = MotionTrack(
                    track_id=next_track_id,
                    centroid=detection.centroid,
                    box=detection.box,
                    zone_counts={},
                    damage_zone_counts={},
                    best_boxes=[],
                )
                tracks[next_track_id] = track
                best_track_id = next_track_id
                next_track_id += 1
            else:
                track = tracks[best_track_id]
                track.centroid = detection.centroid
                track.box = detection.box
                track.missed = 0

            matched_tracks.add(best_track_id)
            matched_detections.add(det_index)
            qualifies = detection.near_edge and (not require_damage_zone or bool(detection.damage_zone_names))
            if qualifies:
                update_track_event(track, detection, frame, timestamp, frame_index)
            else:
                close_track_event(track)

        for track_id in list(tracks):
            if track_id in matched_tracks:
                continue
            track = tracks[track_id]
            track.missed += 1
            if track.missed > track_max_missed:
                close_track_event(track)
                del tracks[track_id]

        if debug_writer is not None:
            all_boxes = [detection.box for detection in detections]
            near_boxes = [detection.box for detection in detections if detection.near_edge]
            damage_boxes = [detection.box for detection in detections if detection.damage_zone_names]
            annotated = annotate_frame(
                frame=frame,
                roi=local_roi,
                boxes=all_boxes,
                timestamp_seconds=timestamp,
                video_path=info.path,
                motion_area=sum(detection.area for detection in detections),
                active=bool(near_boxes),
                edge_points=vehicle_edge_points,
                edge_closed=vehicle_edge_closed,
                damage_zones=damage_zones,
                label_suffix="trajectory-edge",
            )
            for bx, by, bw, bh in near_boxes:
                cv2.rectangle(annotated, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 128, 255), 3)
            for bx, by, bw, bh in damage_boxes:
                cv2.rectangle(annotated, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 255, 0), 3)
            if debug_size is not None and (annotated.shape[1], annotated.shape[0]) != debug_size:
                annotated = cv2.resize(annotated, debug_size)
            debug_writer.write(annotated)

        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} sampled frames from {info.path.name}...", flush=True)

        frame_index += 1

    for track in list(tracks.values()):
        close_track_event(track)
    cap.release()
    return raw_events


def process_video_behavior(
    info: VideoInfo,
    video_index: int,
    roi: tuple[int, int, int, int],
    damage_zones: Sequence[dict[str, object]] | None,
    damage_distance: int,
    min_area: float,
    sample_every: int,
    warmup_frames: int,
    track_max_distance: float,
    track_max_missed: int,
    near_padding: int,
    analysis_padding: int,
    analysis_scale: float,
    person_min_height: int,
    person_min_height_width_ratio: float,
    person_max_area_ratio: float,
    debug_writer: cv2.VideoWriter | None,
    debug_size: tuple[int, int] | None,
) -> list[BehaviorTrack]:
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {info.path}")

    local_roi = clamp_roi(roi, info.width, info.height)
    analysis_roi = expand_rect(local_roi, analysis_padding, info.width, info.height)
    analysis_x, analysis_y, analysis_w, analysis_h = analysis_roi
    scale = max(0.05, min(1.0, float(analysis_scale)))
    scaled_min_area = max(20.0, min_area * scale * scale)
    scaled_min_height = max(1, int(round(person_min_height * scale)))
    near_vehicle_mask = build_behavior_near_mask(info.width, info.height, local_roi, near_padding)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    live_tracks: dict[int, BehaviorTrack] = {}
    finished_tracks: list[BehaviorTrack] = []
    next_track_id = 1
    sample_every = max(1, sample_every)
    sample_step_seconds = sample_every / info.fps
    track_max_missed = max(0, track_max_missed)

    def close_track(track: BehaviorTrack) -> None:
        if track.observations:
            finished_tracks.append(track)

    def add_observation(
        track: BehaviorTrack,
        detection: BehaviorDetection,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> None:
        track.centroid = detection.centroid
        track.box = detection.box
        track.missed = 0
        sample_seconds = sample_step_seconds
        if info.duration_seconds > 0:
            sample_seconds = min(sample_seconds, max(0.0, info.duration_seconds - timestamp))
        observation = BehaviorObservation(
            timestamp=timestamp,
            frame_index=frame_index,
            centroid=detection.centroid,
            box=detection.box,
            area=detection.area,
            near_vehicle=detection.near_vehicle,
            sample_seconds=sample_seconds,
            damage_zone_names=detection.damage_zone_names,
            damage_score=detection.damage_score,
        )
        observations = track.observations or []
        observations.append(observation)
        track.observations = observations
        track.max_area = max(track.max_area, detection.area)
        track.max_damage_score = max(track.max_damage_score, detection.damage_score)
        damage_counts = track.damage_zone_counts or {}
        for name in detection.damage_zone_names:
            damage_counts[name] = damage_counts.get(name, 0) + 1
        track.damage_zone_counts = damage_counts

        should_use_frame = track.best_frame is None
        if detection.near_vehicle and detection.area >= track.best_near_area:
            track.best_near_area = detection.area
            should_use_frame = True
        if should_use_frame:
            track.best_frame = encode_image_bytes(frame)
            track.best_box = detection.box
            track.best_time = timestamp
            track.best_frame_index = frame_index

    def recover_decode_gap() -> bool:
        nonlocal frame_index

        if info.frame_count <= 0:
            return False
        next_frame = frame_index + max(1, sample_every)
        if next_frame >= info.frame_count:
            return False
        for track in live_tracks.values():
            close_track(track)
        live_tracks.clear()
        frame_index = next_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return True

    frame_index = 0
    processed = 0

    while True:
        if frame_index % sample_every != 0:
            ok = cap.grab()
            if not ok:
                if recover_decode_gap():
                    continue
                break
            frame_index += 1
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            if recover_decode_gap():
                continue
            break

        timestamp = frame_index / info.fps
        analysis_frame = frame[analysis_y : analysis_y + analysis_h, analysis_x : analysis_x + analysis_w]
        if scale < 0.999:
            analysis_input = cv2.resize(
                analysis_frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            analysis_input = analysis_frame
        fgmask = subtractor.apply(analysis_input)
        mask = preprocess_mask(fgmask)
        warming_up = frame_index < warmup_frames

        detections: list[BehaviorDetection] = []
        if not warming_up:
            for scaled_box, scaled_area in find_motion_detections(mask, scaled_min_area):
                sx, sy, sw, sh = scaled_box
                box = (
                    analysis_x + int(round(sx / scale)),
                    analysis_y + int(round(sy / scale)),
                    max(1, int(round(sw / scale))),
                    max(1, int(round(sh / scale))),
                )
                area = scaled_area / max(1e-6, scale * scale)
                if not is_behavior_detection_candidate(
                    box=box,
                    area=area,
                    frame_width=info.width,
                    frame_height=info.height,
                    min_height=scaled_min_height if scale >= 0.999 else person_min_height,
                    min_height_width_ratio=person_min_height_width_ratio,
                    max_area_ratio=person_max_area_ratio,
                ):
                    continue
                near_vehicle, near_score = bbox_mask_proximity(box, near_vehicle_mask)
                damage_zone_names, damage_score = full_frame_damage_zone_proximity(
                    box,
                    local_roi,
                    damage_zones,
                    damage_distance,
                )
                detections.append(
                    BehaviorDetection(
                        box=box,
                        area=area,
                        centroid=bbox_centroid(box),
                        near_vehicle=near_vehicle,
                        near_score=near_score,
                        damage_zone_names=damage_zone_names,
                        damage_score=damage_score,
                    )
                )

        matched_tracks: set[int] = set()
        for detection in detections:
            best_track_id: int | None = None
            best_distance = float("inf")
            for track_id, track in live_tracks.items():
                if track_id in matched_tracks:
                    continue
                distance = math.hypot(detection.centroid[0] - track.centroid[0], detection.centroid[1] - track.centroid[1])
                if distance < best_distance and distance <= track_max_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                track = BehaviorTrack(
                    track_id=next_track_id,
                    video_path=info.path,
                    video_index=video_index,
                    centroid=detection.centroid,
                    box=detection.box,
                    observations=[],
                    damage_zone_counts={},
                )
                live_tracks[next_track_id] = track
                best_track_id = next_track_id
                next_track_id += 1
            else:
                track = live_tracks[best_track_id]

            matched_tracks.add(best_track_id)
            add_observation(track, detection, frame, timestamp, frame_index)

        for track_id in list(live_tracks):
            if track_id in matched_tracks:
                continue
            track = live_tracks[track_id]
            track.missed += 1
            if track.missed > track_max_missed:
                close_track(track)
                del live_tracks[track_id]

        if debug_writer is not None:
            boxes = [detection.box for detection in detections]
            annotated = annotate_frame(
                frame=frame,
                roi=local_roi,
                boxes=[],
                timestamp_seconds=timestamp,
                video_path=info.path,
                motion_area=sum(detection.area for detection in detections),
                active=any(detection.near_vehicle for detection in detections),
                label_suffix="behavior",
            )
            x, y, _, _ = local_roi
            for detection in detections:
                bx, by, bw, bh = detection.box
                color = (0, 255, 0) if detection.near_vehicle else (255, 128, 0)
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(
                annotated,
                "green=inside vehicle ROI, blue=elsewhere",
                (max(10, x), max(30, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if debug_size is not None and (annotated.shape[1], annotated.shape[0]) != debug_size:
                annotated = cv2.resize(annotated, debug_size)
            debug_writer.write(annotated)

        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} behavior sampled frames from {info.path.name}...", flush=True)

        frame_index += 1

    for track in list(live_tracks.values()):
        close_track(track)
    cap.release()
    return finished_tracks


def merge_raw_events(raw_events: list[RawEvent], merge_gap: float) -> list[RawEvent]:
    if not raw_events:
        return []

    events = sorted(raw_events, key=lambda e: (e.video_index, e.start_time))
    merged: list[RawEvent] = []

    current = events[0]
    for event in events[1:]:
        same_video = event.video_index == current.video_index and event.video_path == current.video_path
        gap = event.start_time - current.end_time
        if same_video and gap <= merge_gap:
            best_frame = current.best_frame
            best_boxes = current.best_boxes
            best_time = current.best_time
            max_area = current.max_motion_area
            if event.max_motion_area > max_area:
                best_frame = event.best_frame
                best_boxes = event.best_boxes
                best_time = event.best_time
                max_area = event.max_motion_area
            current = RawEvent(
                video_path=current.video_path,
                video_index=current.video_index,
                start_time=min(current.start_time, event.start_time),
                end_time=max(current.end_time, event.end_time),
                start_frame=min(current.start_frame, event.start_frame),
                end_frame=max(current.end_frame, event.end_frame),
                max_motion_area=max_area,
                max_edge_score=max(current.max_edge_score, event.max_edge_score),
                max_damage_score=max(current.max_damage_score, event.max_damage_score),
                zone_counts=merge_counts(current.zone_counts, event.zone_counts),
                damage_zone_counts=merge_counts(current.damage_zone_counts, event.damage_zone_counts),
                best_frame=best_frame,
                best_boxes=list(best_boxes),
                best_time=best_time,
            )
        else:
            merged.append(current)
            current = event
    merged.append(current)
    return merged


def write_event_screenshot(
    event: RawEvent,
    event_id: int,
    screenshots_dir: Path,
    roi: tuple[int, int, int, int],
    edge_points: Sequence[Sequence[int]] | None = None,
    edge_closed: bool = True,
    damage_zones: Sequence[dict[str, object]] | None = None,
) -> Path:
    filename = f"event_{event_id:04d}_{safe_name(event.video_path)}_{format_time(event.start_time).replace(':', '-')}.jpg"
    path = screenshots_dir / filename
    best_frame = decode_image_bytes(event.best_frame)
    annotated = annotate_frame(
        frame=best_frame,
        roi=roi,
        boxes=event.best_boxes,
        timestamp_seconds=event.best_time,
        video_path=event.video_path,
        motion_area=event.max_motion_area,
        active=True,
        edge_points=edge_points,
        edge_closed=edge_closed,
        damage_zones=damage_zones,
    )
    write_image(path, annotated)
    return path


def export_clip(
    ffmpeg_path: str | None,
    event: RawEvent,
    event_id: int,
    clips_dir: Path,
    video_info: VideoInfo,
    pre_roll: float,
    post_roll: float,
) -> Path | None:
    if ffmpeg_path is None:
        return None

    start = max(0.0, event.start_time - pre_roll)
    end = event.end_time + post_roll
    if video_info.duration_seconds > 0:
        end = min(video_info.duration_seconds, end)
    duration = max(0.1, end - start)

    output_path = clips_dir / f"event_{event_id:04d}_{safe_name(event.video_path)}.mp4"
    copy_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(event.video_path),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(copy_cmd, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    encode_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(event.video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(encode_cmd, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    print(f"Warning: ffmpeg failed to export clip for event {event_id}.", file=sys.stderr)
    return None


def build_final_events(
    merged_events: list[RawEvent],
    video_infos: dict[Path, VideoInfo],
    screenshots_dir: Path,
    clips_dir: Path,
    roi: tuple[int, int, int, int],
    min_area: float,
    pre_roll: float,
    post_roll: float,
    edge_points: Sequence[Sequence[int]] | None = None,
    edge_closed: bool = True,
    damage_zones: Sequence[dict[str, object]] | None = None,
) -> list[FinalEvent]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("ffmpeg was not found. Clip export will be skipped.")
    else:
        print(f"Using ffmpeg: {ffmpeg_path}")

    final_events: list[FinalEvent] = []
    for event_id, event in enumerate(merged_events, start=1):
        screenshot_path = write_event_screenshot(event, event_id, screenshots_dir, roi, edge_points, edge_closed, damage_zones)
        damage_zone_names = ",".join(sorted(event.damage_zone_counts))
        clip_path = export_clip(
            ffmpeg_path=ffmpeg_path,
            event=event,
            event_id=event_id,
            clips_dir=clips_dir,
            video_info=video_infos[event.video_path],
            pre_roll=pre_roll,
            post_roll=post_roll,
        )
        final_events.append(
            FinalEvent(
                event_id=event_id,
                video_path=event.video_path,
                video_index=event.video_index,
                start_time=event.start_time,
                end_time=event.end_time,
                start_frame=event.start_frame,
                end_frame=event.end_frame,
                duration_seconds=event.duration_seconds,
                max_motion_area=event.max_motion_area,
                confidence_score=confidence_score(event, min_area),
                max_damage_score=event.max_damage_score,
                damage_zones=damage_zone_names,
                screenshot_path=screenshot_path,
                clip_path=clip_path,
                best_frame=event.best_frame,
                best_boxes=event.best_boxes,
                best_time=event.best_time,
            )
        )
    return final_events


def write_events_csv(events: list[FinalEvent], csv_path: Path) -> None:
    fieldnames = [
        "event_id",
        "start_time",
        "end_time",
        "start_timestamp_hhmmss",
        "end_timestamp_hhmmss",
        "frame_start",
        "frame_end",
        "duration_seconds",
        "video_filename",
        "max_motion_area",
        "confidence_score",
        "max_damage_score",
        "damage_zones",
        "screenshot_path",
        "clip_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "start_time": format_time(event.start_time),
                    "end_time": format_time(event.end_time),
                    "start_timestamp_hhmmss": format_time(event.start_time),
                    "end_timestamp_hhmmss": format_time(event.end_time),
                    "frame_start": event.start_frame,
                    "frame_end": event.end_frame,
                    "duration_seconds": f"{event.duration_seconds:.3f}",
                    "video_filename": event.video_path.name,
                    "max_motion_area": f"{event.max_motion_area:.1f}",
                    "confidence_score": f"{event.confidence_score:.1f}",
                    "max_damage_score": f"{event.max_damage_score:.3f}",
                    "damage_zones": event.damage_zones,
                    "screenshot_path": str(event.screenshot_path),
                    "clip_path": str(event.clip_path) if event.clip_path is not None else "",
                }
            )


def html_link_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().as_uri()
        except ValueError:
            return str(path)


def write_html_report(events: list[FinalEvent], report_path: Path, output_dir: Path) -> None:
    rows: list[str] = []
    for event in events:
        screenshot_src = html.escape(html_link_path(event.screenshot_path, output_dir))
        damage_zones = event.damage_zones or "None"
        clip_html = "No clip"
        if event.clip_path is not None:
            clip_href = html.escape(html_link_path(event.clip_path, output_dir))
            clip_html = f'<a href="{clip_href}">Open clip</a>'

        rows.append(
            f"""
            <article class="event">
              <div class="event-head">
                <div>
                  <h2>Event {event.event_id}</h2>
                  <p>{html.escape(event.video_path.name)}</p>
                </div>
                <div class="score">{event.confidence_score:.1f}</div>
              </div>
              <dl>
                <div><dt>Start</dt><dd>{html.escape(format_time(event.start_time))}</dd></div>
                <div><dt>End</dt><dd>{html.escape(format_time(event.end_time))}</dd></div>
                <div><dt>Frames</dt><dd>{event.start_frame} - {event.end_frame}</dd></div>
                <div><dt>Duration</dt><dd>{event.duration_seconds:.2f}s</dd></div>
                <div><dt>Max area</dt><dd>{event.max_motion_area:.0f}</dd></div>
                <div><dt>Damage zones</dt><dd>{html.escape(damage_zones)}</dd></div>
                <div><dt>Damage score</dt><dd>{event.max_damage_score:.2f}</dd></div>
                <div><dt>Clip</dt><dd>{clip_html}</dd></div>
              </dl>
              <a href="{screenshot_src}"><img src="{screenshot_src}" alt="Event {event.event_id} screenshot"></a>
            </article>
            """
        )

    content = "\n".join(rows) if rows else '<p class="empty">No suspicious candidate events found.</p>'
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vehicle Motion Candidate Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    header {{
      padding: 24px 32px;
      background: #17212b;
      color: white;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    header p {{
      margin: 0;
      color: #cbd5df;
    }}
    main {{
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    .event {{
      margin-bottom: 20px;
      padding: 18px;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      background: white;
    }}
    .event-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .event-head p {{
      margin: 4px 0 0;
      color: #5b6875;
    }}
    .score {{
      min-width: 74px;
      padding: 10px 12px;
      border-radius: 8px;
      background: #0f766e;
      color: white;
      font-weight: 700;
      text-align: center;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    dt {{
      font-size: 12px;
      color: #687586;
      text-transform: uppercase;
    }}
    dd {{
      margin: 3px 0 0;
      font-weight: 600;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 680px;
      object-fit: contain;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: #111827;
    }}
    a {{
      color: #0f766e;
    }}
    .empty {{
      padding: 24px;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      background: white;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Vehicle Motion Candidate Report</h1>
    <p>Candidate clips for manual review only. This report does not determine intent, responsibility, or crime.</p>
  </header>
  <main>
    {content}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def optional_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def percentile_rank(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.0
    below_or_equal = sum(1 for item in values if item <= value)
    return 100.0 * below_or_equal / len(values)


def percentile_value(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = clamp01(percentile / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def thin_path_points(observations: Sequence[BehaviorObservation], max_points: int = 180) -> list[tuple[int, int]]:
    if not observations:
        return []
    step = max(1, math.ceil(len(observations) / max(1, max_points)))
    points = [
        (int(round(obs.centroid[0])), int(round(obs.centroid[1])))
        for idx, obs in enumerate(observations)
        if idx % step == 0
    ]
    last = observations[-1]
    last_point = (int(round(last.centroid[0])), int(round(last.centroid[1])))
    if not points or points[-1] != last_point:
        points.append(last_point)
    return points


def behavior_track_stats(track: BehaviorTrack) -> dict[str, object] | None:
    observations = sorted(track.observations or [], key=lambda obs: obs.timestamp)
    if not observations or track.best_frame is None or track.best_box is None:
        return None

    start = observations[0]
    end = observations[-1]
    duration = max(0.0, end.timestamp - start.timestamp + end.sample_seconds)
    near_seconds = sum(obs.sample_seconds for obs in observations if obs.near_vehicle)
    elsewhere_seconds = sum(obs.sample_seconds for obs in observations if not obs.near_vehicle)

    near_distance = 0.0
    near_speed_seconds = 0.0
    elsewhere_distance = 0.0
    elsewhere_speed_seconds = 0.0
    for prev, current in zip(observations, observations[1:]):
        dt = current.timestamp - prev.timestamp
        max_gap = max(prev.sample_seconds, current.sample_seconds) * 6.0
        if dt <= 0 or dt > max_gap:
            continue
        distance = math.hypot(current.centroid[0] - prev.centroid[0], current.centroid[1] - prev.centroid[1])
        if prev.near_vehicle and current.near_vehicle:
            near_distance += distance
            near_speed_seconds += dt
        elif not prev.near_vehicle and not current.near_vehicle:
            elsewhere_distance += distance
            elsewhere_speed_seconds += dt

    avg_speed_near = near_distance / near_speed_seconds if near_speed_seconds > 0 else None
    avg_speed_elsewhere = elsewhere_distance / elsewhere_speed_seconds if elsewhere_speed_seconds > 0 else None
    speed_ratio = None
    if avg_speed_near is not None and avg_speed_elsewhere is not None and avg_speed_elsewhere > 1e-6:
        speed_ratio = avg_speed_near / avg_speed_elsewhere

    return {
        "track": track,
        "observations": observations,
        "start_time": start.timestamp,
        "end_time": end.timestamp + end.sample_seconds,
        "start_frame": start.frame_index,
        "end_frame": end.frame_index,
        "duration_seconds": duration,
        "near_vehicle_seconds": near_seconds,
        "elsewhere_seconds": elsewhere_seconds,
        "avg_speed_near_px_s": avg_speed_near,
        "avg_speed_elsewhere_px_s": avg_speed_elsewhere,
        "speed_ratio_near_to_elsewhere": speed_ratio,
        "path_points": thin_path_points(observations),
    }


def behavior_reason(
    dwell_percentile: float,
    near_seconds: float,
    speed_ratio: float | None,
    avg_speed_near: float | None,
    median_near_speed: float,
    max_damage_score: float,
) -> str:
    reasons: list[str] = []
    if dwell_percentile >= 85.0:
        reasons.append("long_near_vehicle_dwell")
    if near_seconds >= 20.0:
        reasons.append("absolute_long_near_vehicle_dwell")
    if speed_ratio is not None and speed_ratio <= 0.65:
        reasons.append("slower_near_vehicle_than_elsewhere")
    elif avg_speed_near is not None and median_near_speed > 0 and avg_speed_near <= median_near_speed * 0.45:
        reasons.append("slow_vs_other_near_vehicle_tracks")
    if max_damage_score > 0:
        reasons.append("near_marked_damage_zone")
    return ",".join(reasons or ["near_vehicle_track"])


def build_behavior_candidates(
    tracks: list[BehaviorTrack],
    min_track_duration: float,
    min_near_duration: float,
    min_elsewhere_duration: float,
    top_n: int,
) -> list[BehaviorCandidate]:
    stats = [item for item in (behavior_track_stats(track) for track in tracks) if item is not None]
    stats = [
        item
        for item in stats
        if float(item["duration_seconds"]) >= min_track_duration
        and float(item["near_vehicle_seconds"]) >= min_near_duration
    ]
    if not stats:
        return []

    near_by_video: dict[Path, list[float]] = {}
    near_speeds: list[float] = []
    for item in stats:
        track = item["track"]
        assert isinstance(track, BehaviorTrack)
        near_by_video.setdefault(track.video_path, []).append(float(item["near_vehicle_seconds"]))
        speed = item["avg_speed_near_px_s"]
        if isinstance(speed, (int, float)) and math.isfinite(float(speed)):
            near_speeds.append(float(speed))

    median_near_speed = percentile_value(near_speeds, 50.0)
    scored: list[tuple[float, dict[str, object], float, str]] = []
    for item in stats:
        track = item["track"]
        assert isinstance(track, BehaviorTrack)
        near_seconds = float(item["near_vehicle_seconds"])
        duration = max(0.001, float(item["duration_seconds"]))
        video_near_values = near_by_video.get(track.video_path, [])
        dwell_percentile = percentile_rank(video_near_values, near_seconds)
        median_near = percentile_value(video_near_values, 50.0)
        p90_near = percentile_value(video_near_values, 90.0)
        dwell_relative = clamp01((near_seconds - median_near) / max(1.0, p90_near - median_near))
        dwell_percentile_part = dwell_percentile / 100.0
        speed_ratio = item["speed_ratio_near_to_elsewhere"]
        avg_speed_near = item["avg_speed_near_px_s"]
        elsewhere_seconds = float(item["elsewhere_seconds"])
        slow_part = 0.0
        if isinstance(speed_ratio, (int, float)) and elsewhere_seconds >= min_elsewhere_duration:
            slow_part = clamp01((0.75 - float(speed_ratio)) / 0.75)
        elif isinstance(avg_speed_near, (int, float)) and median_near_speed > 0:
            slow_part = clamp01((median_near_speed - float(avg_speed_near)) / median_near_speed)
        near_share = clamp01(near_seconds / duration)
        damage_part = clamp01(track.max_damage_score)
        score = round(
            min(
                100.0,
                dwell_relative * 35.0
                + dwell_percentile_part * 25.0
                + slow_part * 30.0
                + near_share * 5.0
                + damage_part * 5.0,
            ),
            1,
        )
        reason = behavior_reason(
            dwell_percentile=dwell_percentile,
            near_seconds=near_seconds,
            speed_ratio=float(speed_ratio) if isinstance(speed_ratio, (int, float)) else None,
            avg_speed_near=float(avg_speed_near) if isinstance(avg_speed_near, (int, float)) else None,
            median_near_speed=median_near_speed,
            max_damage_score=track.max_damage_score,
        )
        scored.append((score, item, dwell_percentile, reason))

    scored.sort(
        key=lambda row: (
            row[0],
            float(row[1]["near_vehicle_seconds"]),
            float(row[1]["duration_seconds"]),
        ),
        reverse=True,
    )
    if top_n > 0:
        scored = scored[:top_n]

    candidates: list[BehaviorCandidate] = []
    for event_id, (score, item, dwell_percentile, reason) in enumerate(scored, start=1):
        track = item["track"]
        assert isinstance(track, BehaviorTrack)
        damage_zone_names = ",".join(sorted(track.damage_zone_counts or {}))
        candidates.append(
            BehaviorCandidate(
                event_id=event_id,
                track_id=track.track_id,
                video_path=track.video_path,
                video_index=track.video_index,
                start_time=float(item["start_time"]),
                end_time=float(item["end_time"]),
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                duration_seconds=float(item["duration_seconds"]),
                near_vehicle_seconds=float(item["near_vehicle_seconds"]),
                elsewhere_seconds=float(item["elsewhere_seconds"]),
                avg_speed_near_px_s=(
                    float(item["avg_speed_near_px_s"]) if isinstance(item["avg_speed_near_px_s"], (int, float)) else None
                ),
                avg_speed_elsewhere_px_s=(
                    float(item["avg_speed_elsewhere_px_s"])
                    if isinstance(item["avg_speed_elsewhere_px_s"], (int, float))
                    else None
                ),
                speed_ratio_near_to_elsewhere=(
                    float(item["speed_ratio_near_to_elsewhere"])
                    if isinstance(item["speed_ratio_near_to_elsewhere"], (int, float))
                    else None
                ),
                dwell_percentile=round(dwell_percentile, 1),
                anomaly_score=score,
                max_motion_area=track.max_area,
                max_damage_score=track.max_damage_score,
                damage_zones=damage_zone_names,
                reason=reason,
                screenshot_path=None,
                best_frame=track.best_frame or b"",
                best_box=track.best_box or track.box,
                path_points=item["path_points"],  # type: ignore[arg-type]
            )
        )
    return candidates


def annotate_behavior_frame(
    frame: np.ndarray,
    candidate: BehaviorCandidate,
    roi: tuple[int, int, int, int],
    edge_points: Sequence[Sequence[int]] | None,
    edge_closed: bool,
    damage_zones: Sequence[dict[str, object]] | None,
) -> np.ndarray:
    annotated = frame.copy()
    x, y, w, h = roi
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
    draw_vehicle_edge(annotated, roi, edge_points, closed=edge_closed, color=(255, 0, 255), thickness=2)
    draw_damage_zones(annotated, roi, damage_zones)

    if len(candidate.path_points) >= 2:
        pts = np.array(candidate.path_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], False, (255, 128, 0), 2, cv2.LINE_AA)
        cv2.circle(annotated, candidate.path_points[0], 5, (255, 255, 0), -1)
        cv2.circle(annotated, candidate.path_points[-1], 5, (0, 255, 255), -1)

    bx, by, bw, bh = candidate.best_box
    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
    lines = [
        f"Behavior {candidate.event_id} score={candidate.anomaly_score:.1f}",
        f"near={candidate.near_vehicle_seconds:.1f}s speed_near={optional_float(candidate.avg_speed_near_px_s, 1)} px/s",
        f"elsewhere={candidate.elsewhere_seconds:.1f}s ratio={optional_float(candidate.speed_ratio_near_to_elsewhere, 2)}",
        candidate.reason,
    ]
    top = 32
    for idx, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (16, top + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (16, top + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def write_behavior_screenshots(
    candidates: list[BehaviorCandidate],
    screenshots_dir: Path,
    roi: tuple[int, int, int, int],
    edge_points: Sequence[Sequence[int]] | None,
    edge_closed: bool,
    damage_zones: Sequence[dict[str, object]] | None,
) -> None:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        frame = decode_image_bytes(candidate.best_frame)
        annotated = annotate_behavior_frame(frame, candidate, roi, edge_points, edge_closed, damage_zones)
        filename = f"behavior_{candidate.event_id:04d}_{safe_name(candidate.video_path)}_{format_time(candidate.start_time).replace(':', '-')}.jpg"
        screenshot_path = screenshots_dir / filename
        write_image(screenshot_path, annotated)
        candidate.screenshot_path = screenshot_path


def write_behavior_csv(candidates: list[BehaviorCandidate], csv_path: Path) -> None:
    fieldnames = [
        "event_id",
        "track_id",
        "video_filename",
        "start_time",
        "end_time",
        "start_timestamp_hhmmss",
        "end_timestamp_hhmmss",
        "frame_start",
        "frame_end",
        "duration_seconds",
        "near_vehicle_seconds",
        "elsewhere_seconds",
        "avg_speed_near_px_s",
        "avg_speed_elsewhere_px_s",
        "speed_ratio_near_to_elsewhere",
        "dwell_percentile",
        "anomaly_score",
        "max_motion_area",
        "max_damage_score",
        "damage_zones",
        "reason",
        "screenshot_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "event_id": candidate.event_id,
                    "track_id": candidate.track_id,
                    "video_filename": candidate.video_path.name,
                    "start_time": format_time(candidate.start_time),
                    "end_time": format_time(candidate.end_time),
                    "start_timestamp_hhmmss": format_time(candidate.start_time),
                    "end_timestamp_hhmmss": format_time(candidate.end_time),
                    "frame_start": candidate.start_frame,
                    "frame_end": candidate.end_frame,
                    "duration_seconds": f"{candidate.duration_seconds:.3f}",
                    "near_vehicle_seconds": f"{candidate.near_vehicle_seconds:.3f}",
                    "elsewhere_seconds": f"{candidate.elsewhere_seconds:.3f}",
                    "avg_speed_near_px_s": optional_float(candidate.avg_speed_near_px_s, 3),
                    "avg_speed_elsewhere_px_s": optional_float(candidate.avg_speed_elsewhere_px_s, 3),
                    "speed_ratio_near_to_elsewhere": optional_float(candidate.speed_ratio_near_to_elsewhere, 3),
                    "dwell_percentile": f"{candidate.dwell_percentile:.1f}",
                    "anomaly_score": f"{candidate.anomaly_score:.1f}",
                    "max_motion_area": f"{candidate.max_motion_area:.1f}",
                    "max_damage_score": f"{candidate.max_damage_score:.3f}",
                    "damage_zones": candidate.damage_zones,
                    "reason": candidate.reason,
                    "screenshot_path": str(candidate.screenshot_path or ""),
                }
            )


def write_behavior_html_report(
    candidates: list[BehaviorCandidate],
    report_path: Path,
    output_dir: Path,
) -> None:
    rows: list[str] = []
    for candidate in candidates:
        screenshot_src = ""
        if candidate.screenshot_path is not None:
            screenshot_src = html.escape(html_link_path(candidate.screenshot_path, output_dir))
        damage_zones = candidate.damage_zones or "None"
        speed_ratio = optional_float(candidate.speed_ratio_near_to_elsewhere, 2) or "N/A"
        rows.append(
            f"""
            <article class="event">
              <div class="event-head">
                <div>
                  <h2>Behavior {candidate.event_id}</h2>
                  <p>{html.escape(candidate.video_path.name)} | track {candidate.track_id}</p>
                </div>
                <div class="score">{candidate.anomaly_score:.1f}</div>
              </div>
              <dl>
                <div><dt>Start</dt><dd>{html.escape(format_time(candidate.start_time))}</dd></div>
                <div><dt>End</dt><dd>{html.escape(format_time(candidate.end_time))}</dd></div>
                <div><dt>Duration</dt><dd>{candidate.duration_seconds:.2f}s</dd></div>
                <div><dt>Near vehicle</dt><dd>{candidate.near_vehicle_seconds:.2f}s</dd></div>
                <div><dt>Elsewhere</dt><dd>{candidate.elsewhere_seconds:.2f}s</dd></div>
                <div><dt>Near speed</dt><dd>{optional_float(candidate.avg_speed_near_px_s, 1) or "N/A"} px/s</dd></div>
                <div><dt>Elsewhere speed</dt><dd>{optional_float(candidate.avg_speed_elsewhere_px_s, 1) or "N/A"} px/s</dd></div>
                <div><dt>Speed ratio</dt><dd>{speed_ratio}</dd></div>
                <div><dt>Dwell percentile</dt><dd>{candidate.dwell_percentile:.1f}</dd></div>
                <div><dt>Damage zones</dt><dd>{html.escape(damage_zones)}</dd></div>
                <div><dt>Reason</dt><dd>{html.escape(candidate.reason)}</dd></div>
              </dl>
              <a href="{screenshot_src}"><img src="{screenshot_src}" alt="Behavior {candidate.event_id} screenshot"></a>
            </article>
            """
        )

    content = "\n".join(rows) if rows else '<p class="empty">No long-dwell or slow-near-vehicle behavior candidates found.</p>'
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vehicle-Area Behavior Candidate Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    header {{
      padding: 24px 32px;
      background: #17212b;
      color: white;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    header p {{
      margin: 0;
      color: #cbd5df;
    }}
    main {{
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    .event {{
      margin-bottom: 20px;
      padding: 18px;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      background: white;
    }}
    .event-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .event-head p {{
      margin: 4px 0 0;
      color: #5b6875;
    }}
    .score {{
      min-width: 74px;
      padding: 10px 12px;
      border-radius: 8px;
      background: #7c2d12;
      color: white;
      font-weight: 700;
      text-align: center;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    dt {{
      font-size: 12px;
      color: #687586;
      text-transform: uppercase;
    }}
    dd {{
      margin: 3px 0 0;
      font-weight: 600;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 680px;
      object-fit: contain;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: #111827;
    }}
    a {{
      color: #0f766e;
    }}
    .empty {{
      padding: 24px;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      background: white;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Vehicle-Area Behavior Candidate Report</h1>
    <p>Ranks tracks that stay near the vehicle longer than peers or move slower near the vehicle than elsewhere. Candidates for manual review only.</p>
  </header>
  <main>
    {content}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def make_preview_canvas(
    frame: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
    timestamp_seconds: float,
    video_path: Path,
    motion_area: float,
    active: bool,
) -> np.ndarray:
    annotated = annotate_frame(
        frame=frame,
        roi=roi,
        boxes=boxes,
        timestamp_seconds=timestamp_seconds,
        video_path=video_path,
        motion_area=motion_area,
        active=active,
    )
    mask_panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for bx, by, bw, bh in boxes:
        cv2.rectangle(mask_panel, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
    cv2.putText(
        mask_panel,
        f"mask | area={motion_area:.0f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    mask_panel = cv2.resize(mask_panel, (annotated.shape[1], annotated.shape[0]))
    return np.hstack([annotated, mask_panel])


def run_preview(
    info: VideoInfo,
    roi: tuple[int, int, int, int],
    min_area: float,
    sample_every: int,
    warmup_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {info.path}")

    local_roi = clamp_roi(roi, info.width, info.height)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    sample_every = max(1, sample_every)
    frame_index = 0
    window_name = "Preview: annotated frame + ROI mask"
    print("Preview mode: press Space to pause/resume, Q or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            timestamp = frame_index / info.fps
            x, y, w, h = local_roi
            roi_frame = frame[y : y + h, x : x + w]
            fgmask = subtractor.apply(roi_frame)
            mask = preprocess_mask(fgmask)
            motion_area, boxes = find_motion_boxes(mask, min_area)
            active = motion_area >= min_area and frame_index >= warmup_frames
            canvas = make_preview_canvas(frame, mask, local_roi, boxes, timestamp, info.path, motion_area, active)

            try:
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(max(1, int(1000 * sample_every / info.fps))) & 0xFF
            except cv2.error as exc:
                print(f"OpenCV window error: {exc}", file=sys.stderr)
                break

            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                while True:
                    key = cv2.waitKey(50) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        return
                    if key == 32:
                        break

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def analyze_frame_at(
    info: VideoInfo,
    roi: tuple[int, int, int, int],
    target_frame: int,
    min_area: float,
    sample_every: int,
    warmup_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], float]:
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {info.path}")

    local_roi = clamp_roi(roi, info.width, info.height)
    start_frame = max(0, target_frame - max(1, warmup_frames))
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    last_frame: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_boxes: list[tuple[int, int, int, int]] = []
    last_area = 0.0
    current_frame = start_frame
    sample_every = max(1, sample_every)

    try:
        while current_frame <= target_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            should_process = current_frame % sample_every == 0 or current_frame == target_frame
            if should_process:
                x, y, w, h = local_roi
                roi_frame = frame[y : y + h, x : x + w]
                fgmask = subtractor.apply(roi_frame)
                mask = preprocess_mask(fgmask)
                area, boxes = find_motion_boxes(mask, min_area)
                last_frame = frame.copy()
                last_mask = mask
                last_boxes = boxes
                last_area = area

            current_frame += 1
    finally:
        cap.release()

    if last_frame is None or last_mask is None:
        raise RuntimeError(f"Could not read calibration frame {target_frame} from {info.path}")
    return last_frame, last_mask, last_boxes, last_area


def run_calibration(
    infos: list[VideoInfo],
    roi: tuple[int, int, int, int],
    output_dir: Path,
    min_area: float,
    sample_every: int,
    warmup_frames: int,
    warmup_seconds: float,
    sample_count: int,
    seed: int,
) -> None:
    calibration_dir = output_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    usable_infos = [info for info in infos if info.frame_count > 1]
    if not usable_infos:
        usable_infos = infos

    sample_count = max(1, sample_count)
    window_name = "Calibration samples"
    print(f"Calibration mode: saving samples to {calibration_dir}")
    print("Press any key for next sample, Q or Esc to quit.")

    for sample_id in range(1, sample_count + 1):
        info = rng.choice(usable_infos)
        max_frame = max(0, info.frame_count - 1)
        target_frame = rng.randint(0, max_frame) if max_frame > 0 else 0
        timestamp = target_frame / info.fps
        local_roi = clamp_roi(roi, info.width, info.height)
        sample_warmup_frames = max(warmup_frames, int(max(0.0, warmup_seconds) * info.fps))

        frame, mask, boxes, motion_area = analyze_frame_at(
            info=info,
            roi=local_roi,
            target_frame=target_frame,
            min_area=min_area,
            sample_every=sample_every,
            warmup_frames=sample_warmup_frames,
        )
        active = motion_area >= min_area
        canvas = make_preview_canvas(frame, mask, local_roi, boxes, timestamp, info.path, motion_area, active)
        out_path = calibration_dir / f"sample_{sample_id:03d}_{safe_name(info.path)}_{target_frame}.jpg"
        write_image(out_path, canvas)
        print(
            f"Sample {sample_id}/{sample_count}: {info.path.name} "
            f"at {format_time(timestamp)}, frame {target_frame}, area={motion_area:.0f}"
        )

        try:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(0) & 0xFF
        except cv2.error as exc:
            print(f"OpenCV window error: {exc}", file=sys.stderr)
            break

        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()

    if args.sample_every < 1:
        print("--sample-every must be >= 1", file=sys.stderr)
        return 2
    if args.min_area <= 0:
        print("--min-area must be > 0", file=sys.stderr)
        return 2
    if args.min_duration < 0:
        print("--min-duration must be >= 0", file=sys.stderr)
        return 2
    if args.calibration_samples < 1:
        print("--calibration-samples must be >= 1", file=sys.stderr)
        return 2
    if args.edge_distance < 1:
        print("--edge-distance must be >= 1", file=sys.stderr)
        return 2
    if args.damage_distance < 1:
        print("--damage-distance must be >= 1", file=sys.stderr)
        return 2
    if args.track_max_distance <= 0:
        print("--track-max-distance must be > 0", file=sys.stderr)
        return 2
    if args.track_max_missed < 0:
        print("--track-max-missed must be >= 0", file=sys.stderr)
        return 2
    if args.behavior_top < 0:
        print("--behavior-top must be >= 0", file=sys.stderr)
        return 2
    if args.behavior_min_track_duration < 0:
        print("--behavior-min-track-duration must be >= 0", file=sys.stderr)
        return 2
    if args.behavior_min_near_duration < 0:
        print("--behavior-min-near-duration must be >= 0", file=sys.stderr)
        return 2
    if args.behavior_min_elsewhere_duration < 0:
        print("--behavior-min-elsewhere-duration must be >= 0", file=sys.stderr)
        return 2
    if args.behavior_near_padding < 0:
        print("--behavior-near-padding must be >= 0", file=sys.stderr)
        return 2
    if args.behavior_analysis_padding < 0:
        print("--behavior-analysis-padding must be >= 0", file=sys.stderr)
        return 2
    if not (0 < args.behavior_scale <= 1.0):
        print("--behavior-scale must be > 0 and <= 1", file=sys.stderr)
        return 2
    if args.person_min_height < 1:
        print("--person-min-height must be >= 1", file=sys.stderr)
        return 2
    if args.person_min_height_width_ratio <= 0:
        print("--person-min-height-width-ratio must be > 0", file=sys.stderr)
        return 2
    if args.person_max_area_ratio <= 0:
        print("--person-max-area-ratio must be > 0", file=sys.stderr)
        return 2

    try:
        input_paths = expand_inputs(args.input)
    except Exception as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output).expanduser().resolve()
    vehicle_edge_file = resolve_vehicle_edge_file(output_dir, args.vehicle_edge_file)
    damage_zones_file = resolve_damage_zones_file(output_dir, args.damage_zones_file)

    first_frame = read_first_frame(input_paths[0])
    if args.roi:
        roi = parse_roi(args.roi)
    else:
        roi = select_roi(first_frame)
    roi = clamp_roi(roi, first_frame.shape[1], first_frame.shape[0])
    print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    if args.annotate_vehicle_edge:
        try:
            annotate_vehicle_edge(first_frame, roi, vehicle_edge_file, input_paths[0])
        except Exception as exc:
            print(f"Annotation error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.annotate_damage_zones:
        try:
            annotate_damage_zones(
                first_frame,
                roi,
                damage_zones_file,
                input_paths[0],
                parse_damage_zone_names(args.damage_zone_names),
            )
        except Exception as exc:
            print(f"Damage zone annotation error: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        infos = collect_video_infos(input_paths)
    except Exception as exc:
        print(f"Video error: {exc}", file=sys.stderr)
        return 1
    info_by_path = {info.path: info for info in infos}

    if args.preview:
        run_preview(
            info=infos[0],
            roi=roi,
            min_area=args.min_area,
            sample_every=args.sample_every,
            warmup_frames=max(0, args.warmup_frames),
        )
        return 0

    if args.calibration:
        run_calibration(
            infos=infos,
            roi=roi,
            output_dir=output_dir,
            min_area=args.min_area,
            sample_every=args.sample_every,
            warmup_frames=max(0, args.warmup_frames),
            warmup_seconds=args.calibration_warmup_seconds,
            sample_count=args.calibration_samples,
            seed=args.calibration_seed,
        )
        return 0

    screenshots_dir, clips_dir = ensure_dirs(output_dir)

    edge_points: list[tuple[int, int]] | None = None
    edge_closed = True
    damage_zones: list[dict[str, object]] | None = None
    if args.trajectory_scan or args.behavior_scan:
        annotation: dict[str, object] | None = None
        if args.trajectory_scan or vehicle_edge_file.exists():
            try:
                annotation = load_vehicle_edge_annotation(vehicle_edge_file)
                edge_points = edge_points_from_annotation(annotation)
                edge_closed = edge_closed_from_annotation(annotation)
                annotated_roi = annotation.get("roi")
                if annotated_roi and [int(v) for v in annotated_roi] != [int(v) for v in roi]:  # type: ignore[union-attr]
                    print(
                        f"Warning: annotation ROI {annotated_roi} differs from current ROI {list(roi)}. "
                        "Using annotation points relative to current ROI.",
                        file=sys.stderr,
                    )
                print(f"Using vehicle edge annotation: {vehicle_edge_file}")
            except Exception as exc:
                if args.trajectory_scan:
                    print(f"Vehicle edge error: {exc}", file=sys.stderr)
                    print("Run --annotate-vehicle-edge first, or pass --vehicle-edge-file.", file=sys.stderr)
                    return 1
                print(f"Warning: could not load vehicle edge annotation: {exc}", file=sys.stderr)

        if damage_zones_file.exists():
            try:
                damage_annotation = load_damage_zones_annotation(damage_zones_file)
                damage_zones = damage_zones_from_annotation(damage_annotation)
                print(f"Using damage zones annotation: {damage_zones_file}")
            except Exception as exc:
                print(f"Warning: could not load damage zones: {exc}", file=sys.stderr)
        elif args.require_damage_zone:
            print(f"Damage zones file is required with --require-damage-zone: {damage_zones_file}", file=sys.stderr)
            return 1

    debug_writer, debug_size = create_debug_writer(
        output_dir=output_dir,
        first_info=infos[0],
        sample_every=args.sample_every,
        enabled=not args.no_debug_video,
    )

    if args.behavior_scan:
        all_behavior_tracks: list[BehaviorTrack] = []
        try:
            for index, info in enumerate(infos):
                print(f"Analyzing behavior in {info.path} ({index + 1}/{len(infos)})")
                tracks = process_video_behavior(
                    info=info,
                    video_index=index,
                    roi=roi,
                    damage_zones=damage_zones,
                    damage_distance=args.damage_distance,
                    min_area=args.min_area,
                    sample_every=args.sample_every,
                    warmup_frames=max(0, args.warmup_frames),
                    track_max_distance=args.track_max_distance,
                    track_max_missed=args.track_max_missed,
                    near_padding=args.behavior_near_padding,
                    analysis_padding=args.behavior_analysis_padding,
                    analysis_scale=args.behavior_scale,
                    person_min_height=args.person_min_height,
                    person_min_height_width_ratio=args.person_min_height_width_ratio,
                    person_max_area_ratio=args.person_max_area_ratio,
                    debug_writer=debug_writer,
                    debug_size=debug_size,
                )
                print(f"Found {len(tracks)} behavior track(s) in {info.path.name}.")
                all_behavior_tracks.extend(tracks)
        finally:
            if debug_writer is not None:
                debug_writer.release()

        behavior_candidates = build_behavior_candidates(
            tracks=all_behavior_tracks,
            min_track_duration=args.behavior_min_track_duration,
            min_near_duration=args.behavior_min_near_duration,
            min_elsewhere_duration=args.behavior_min_elsewhere_duration,
            top_n=args.behavior_top,
        )
        behavior_screenshots_dir = output_dir / "behavior_screenshots"
        write_behavior_screenshots(
            behavior_candidates,
            behavior_screenshots_dir,
            roi,
            edge_points,
            edge_closed,
            damage_zones,
        )
        behavior_csv_path = output_dir / "behavior_events.csv"
        write_behavior_csv(behavior_candidates, behavior_csv_path)
        behavior_report_path = output_dir / "behavior_report.html"
        write_behavior_html_report(behavior_candidates, behavior_report_path, output_dir)

        print("")
        print("Done.")
        print(f"Behavior CSV: {behavior_csv_path}")
        print(f"Behavior HTML report: {behavior_report_path}")
        print(f"Behavior screenshots: {behavior_screenshots_dir}")
        if not args.no_debug_video:
            print(f"Debug video: {output_dir / 'debug_video.mp4'}")
        print(f"Behavior tracks scanned: {len(all_behavior_tracks)}")
        print(f"Final behavior candidates: {len(behavior_candidates)}")
        print("Reminder: these are only behavior candidates for review, not conclusions about intent, responsibility, or crime.")
        return 0

    all_raw_events: list[RawEvent] = []
    try:
        for index, info in enumerate(infos):
            print(f"Analyzing {info.path} ({index + 1}/{len(infos)})")
            if args.trajectory_scan:
                assert edge_points is not None
                raw_events = process_video_trajectory(
                    info=info,
                    video_index=index,
                    roi=roi,
                    vehicle_edge_points=edge_points,
                    vehicle_edge_closed=edge_closed,
                    damage_zones=damage_zones,
                    edge_distance=args.edge_distance,
                    damage_distance=args.damage_distance,
                    require_damage_zone=args.require_damage_zone,
                    min_area=args.min_area,
                    min_duration=args.min_duration,
                    sample_every=args.sample_every,
                    warmup_frames=max(0, args.warmup_frames),
                    track_max_distance=args.track_max_distance,
                    track_max_missed=args.track_max_missed,
                    debug_writer=debug_writer,
                    debug_size=debug_size,
                )
            else:
                raw_events = process_video(
                    info=info,
                    video_index=index,
                    roi=roi,
                    min_area=args.min_area,
                    min_duration=args.min_duration,
                    sample_every=args.sample_every,
                    warmup_frames=max(0, args.warmup_frames),
                    debug_writer=debug_writer,
                    debug_size=debug_size,
                )
            print(f"Found {len(raw_events)} raw suspicious candidate event(s) in {info.path.name}.")
            all_raw_events.extend(raw_events)
    finally:
        if debug_writer is not None:
            debug_writer.release()

    merged_events = merge_raw_events(all_raw_events, merge_gap=args.merge_gap)
    final_events = build_final_events(
        merged_events=merged_events,
        video_infos=info_by_path,
        screenshots_dir=screenshots_dir,
        clips_dir=clips_dir,
        roi=roi,
        min_area=args.min_area,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        edge_points=edge_points,
        edge_closed=edge_closed,
        damage_zones=damage_zones,
    )

    csv_path = output_dir / "events.csv"
    write_events_csv(final_events, csv_path)
    report_path = output_dir / "report.html"
    write_html_report(final_events, report_path, output_dir)

    print("")
    print("Done.")
    print(f"Events CSV: {csv_path}")
    print(f"HTML report: {report_path}")
    print(f"Screenshots: {screenshots_dir}")
    if shutil.which("ffmpeg") is not None:
        print(f"Clips: {clips_dir}")
    if not args.no_debug_video:
        print(f"Debug video: {output_dir / 'debug_video.mp4'}")
    print(f"Final suspicious candidate events: {len(final_events)}")
    print("Reminder: these are only candidate clips for review, not conclusions about intent or crime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
