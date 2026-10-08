"""
Utilities for Configuration, Video Capture, and OpenCV Visual Overlay
Includes HUD rendering, virtual line visualization, track annotations,
and resilient video streaming with reconnection support.
"""

from __future__ import annotations
import os
import time
import yaml
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from .line_counter import LineCounter
from .detector import TrackedPerson


DEFAULT_CONFIG: Dict[str, Any] = {
    "source": "video.mp4/video.mp4.mp4",
    "model": "yolov8n.pt",
    "confidence": 0.40,
    "iou": 0.50,
    "tracker": "bytetrack.yaml",
    "line": {
        "point1": [100, 360],
        "point2": [540, 360],
        "buffer_distance": 20.0,
        "in_direction": "A_to_B",
        "check_segment_bounds": False,
    },
    "performance": {
        "frame_skip": 1,
        "resize_width": 640,
        "resize_height": 480,
        "display_fps": True,
    },
    "logging": {
        "db_path": "people_counter.db",
        "csv_path": "outputs/crossings_log.csv",
        "export_csv_on_exit": True,
    },
    "video_output": {
        "save_video": False,
        "output_path": "outputs/annotated_output.mp4",
        "fps": 30,
    },
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file and merge with default values.

    Args:
        config_path: Filepath to YAML configuration.

    Returns:
        Dictionary with merged configuration settings.
    """
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            # Deep update top-level keys
            for k, v in user_config.items():
                if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v
        except Exception as e:
            print(f"[WARN] Error reading config file '{config_path}': {e}. Using defaults.")
    return config


def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> None:
    """
    Save configuration dictionary back to YAML file.

    Args:
        config: Configuration dictionary to serialize.
        config_path: Output filepath.
    """
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"[WARN] Could not save config to '{config_path}': {e}")


class VideoStream:
    """
    Robust video capture supporting webcams, video files, and RTSP streams.
    Includes validation, connection retry, and reconnection on stream drops.
    """

    def __init__(self, source: str | int, max_reconnect_attempts: int = 5, reconnect_delay: float = 2.0) -> None:
        self.raw_source = source
        self.is_rtsp = isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("rtsps://") or source.startswith("http://") or source.startswith("https://"))
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self._parsed_source = self._parse_source(source)
        self._open()

    def _parse_source(self, src: str | int) -> str | int:
        """Parse numeric string to int for webcam, or resolve path."""
        if isinstance(src, int):
            return src
        if isinstance(src, str):
            if src.isdigit():
                return int(src)
            # Check if file exists if it's a local path
            if not self.is_rtsp:
                if not os.path.exists(src):
                    # Try resolving relative to workspace
                    raise FileNotFoundError(f"Video file source not found: '{src}'")
        return src

    def _open(self) -> bool:
        """Open VideoCapture device or stream."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self._parsed_source)
        if not self.cap.isOpened():
            return False
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame. If RTSP stream dropped, attempt reconnect.
        """
        if self.cap is None or not self.cap.isOpened():
            if self.is_rtsp:
                if not self._reconnect():
                    return False, None
            else:
                return False, None

        ret, frame = self.cap.read()
        if not ret:
            if self.is_rtsp:
                print("[WARN] RTSP stream dropped. Attempting reconnect...")
                if self._reconnect():
                    ret, frame = self.cap.read()
                    return ret, frame
            return False, None

        return True, frame

    def _reconnect(self) -> bool:
        """Attempt to reconnect to stream with retry attempts."""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(f"[INFO] Reconnection attempt {attempt}/{self.max_reconnect_attempts}...")
            time.sleep(self.reconnect_delay)
            if self._open():
                print("[INFO] Successfully reconnected to video stream.")
                return True
        print("[ERROR] Failed to reconnect after maximum attempts.")
        return False

    @property
    def fps(self) -> float:
        """Get stream FPS (defaults to 30.0 if not available)."""
        if self.cap and self.cap.isOpened():
            val = self.cap.get(cv2.CAP_PROP_FPS)
            if val and val > 0:
                return val
        return 30.0

    @property
    def width(self) -> int:
        """Get native frame width."""
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 640

    @property
    def height(self) -> int:
        """Get native frame height."""
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 480

    def release(self) -> None:
        """Release capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def draw_hud(
    frame: np.ndarray,
    in_count: int,
    out_count: int,
    occupancy: int,
    fps: float,
    is_calibrating: bool = False,
    calib_step: int = 0,
) -> np.ndarray:
    """
    Renders a modern, semi-transparent HUD banner with live metrics and status indicators.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw dark top banner
    banner_height = 55
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 20, 25), -1)
    # Bottom subtle border
    cv2.line(overlay, (0, banner_height), (w, banner_height), (50, 60, 75), 1)

    # Blend banner overlay (alpha=0.85)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Metrics layout
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness = 1

    # Badges: IN (Green), OUT (Red), OCCUPANCY (Cyan), FPS (White)
    # IN Badge
    cv2.circle(frame, (25, 28), 6, (46, 204, 113), -1)
    cv2.putText(frame, f"IN: {in_count}", (38, 34), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

    # OUT Badge
    cv2.circle(frame, (135, 28), 6, (60, 60, 235), -1)
    cv2.putText(frame, f"OUT: {out_count}", (148, 34), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

    # OCCUPANCY Badge
    occ_color = (255, 180, 0) if occupancy > 0 else (180, 180, 180)
    cv2.circle(frame, (260, 28), 6, occ_color, -1)
    cv2.putText(frame, f"OCCUPANCY: {occupancy}", (275, 34), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # FPS Indicator
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
    cv2.putText(frame, fps_text, (w - tw - 20, 34), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    # Calibration Banner when active
    if is_calibrating:
        calib_overlay = frame.copy()
        cv2.rectangle(calib_overlay, (0, h - 40), (w, h), (0, 120, 220), -1)
        cv2.addWeighted(calib_overlay, 0.85, frame, 0.15, 0, frame)
        prompt = "CALIBRATION: Click Point 1 (Start)" if calib_step == 0 else "CALIBRATION: Click Point 2 (End)"
        cv2.putText(frame, f"{prompt}  |  Press ESC or 'L' to cancel", (20, h - 14), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Subtle key shortcut footer
        hint_text = "[L] Redefine Line  |  [R] Reset  |  [Q] Quit"
        cv2.putText(frame, hint_text, (15, h - 12), font, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def draw_line_and_buffer(
    frame: np.ndarray,
    line_counter: LineCounter,
    is_calibrating: bool = False,
    temp_point: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Renders the virtual counting line, buffer boundaries, endpoints, and direction labels.
    """
    p1 = line_counter.point1
    p2 = line_counter.point2

    # Draw buffer boundary lines (dashed effect / translucent)
    line_a, line_b = line_counter.get_buffer_boundary_lines()
    overlay = frame.copy()

    # Draw buffer zone as light shaded strip
    poly_pts = np.array([line_a[0], line_a[1], line_b[1], line_b[0]], dtype=np.int32)
    cv2.fillPoly(overlay, [poly_pts], (40, 180, 220) if is_calibrating else (40, 40, 40))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Draw boundary limit lines
    cv2.line(frame, line_a[0], line_a[1], (100, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, line_b[0], line_b[1], (100, 200, 255), 1, cv2.LINE_AA)

    # Draw main virtual counting line
    main_color = (0, 255, 255) if is_calibrating else (0, 230, 115)
    cv2.line(frame, p1, p2, main_color, 2, cv2.LINE_AA)

    # Draw interactive anchor endpoints
    for p, label in [(p1, "P1"), (p2, "P2")]:
        cv2.circle(frame, p, 7, (20, 20, 20), -1)
        cv2.circle(frame, p, 5, main_color, -1)
        cv2.putText(frame, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # If currently picking point 2, preview line to temp point
    if is_calibrating and temp_point is not None:
        cv2.line(frame, p1, temp_point, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, temp_point, 5, (0, 165, 255), -1)

    return frame


def draw_tracks(
    frame: np.ndarray,
    tracked_people: List[TrackedPerson],
    line_counter: LineCounter,
) -> np.ndarray:
    """
    Renders person bounding boxes, persistent IDs, confidence scores,
    and centroid motion trails.
    """
    for person in tracked_people:
        x1, y1, x2, y2 = person.bbox
        tid = person.track_id
        conf = person.confidence
        cx, cy = person.centroid

        # Determine color by side/state
        side, _ = line_counter.determine_side((cx, cy))
        if side == "BUFFER":
            box_color = (0, 220, 255)  # Yellow/Orange in buffer
        elif side == "A":
            box_color = (46, 204, 113)  # Green for Side A
        else:
            box_color = (60, 120, 240)  # Red/Blue for Side B

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Label tag
        label = f"#{tid} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y1 = max(y1 - lh - 6, 0)
        cv2.rectangle(frame, (x1, label_y1), (x1 + lw + 6, label_y1 + lh + 6), box_color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y1 + lh + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

        # Draw centroid
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # Draw centroid trajectory trail
        history = line_counter.get_track_history(tid)
        if len(history) > 1:
            pts = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 220, 255), 1, cv2.LINE_AA)

    return frame
