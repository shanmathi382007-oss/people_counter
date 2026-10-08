"""
Real-Time People Counter with YOLOv8, ByteTrack, Virtual Line Crossing, and SQLite/CSV Logging
Entry point for CLI execution.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

# Add local package directory to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from counter.detector import PersonDetector
from counter.line_counter import LineCounter, CrossingEvent
from counter.logger import CrossingLogger
from counter.utils import (
    load_config,
    save_config,
    draw_hud,
    draw_line_and_buffer,
    draw_tracks,
    VideoStream,
)


class CalibrationState:
    """Stores mouse calibration state for interactive line placement."""
    def __init__(self) -> None:
        self.active: bool = False
        self.step: int = 0  # 0: waiting for point 1, 1: waiting for point 2
        self.point1: Optional[Tuple[int, int]] = None
        self.temp_cursor: Optional[Tuple[int, int]] = None
        self.new_line_ready: bool = False
        self.new_point1: Optional[Tuple[int, int]] = None
        self.new_point2: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self.active = False
        self.step = 0
        self.point1 = None
        self.temp_cursor = None
        self.new_line_ready = False
        self.new_point1 = None
        self.new_point2 = None


def mouse_callback(event: int, x: int, y: int, flags: int, param: CalibrationState) -> None:
    """OpenCV mouse callback for clicking line endpoints."""
    calib = param
    if not calib.active:
        return

    if event == cv2.EVENT_MOUSEMOVE:
        calib.temp_cursor = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if calib.step == 0:
            calib.point1 = (x, y)
            calib.step = 1
            print(f"[CALIB] Point 1 set to: ({x}, {y}). Now click Point 2.")
        elif calib.step == 1:
            calib.new_point1 = calib.point1
            calib.new_point2 = (x, y)
            calib.new_line_ready = True
            calib.active = False
            calib.step = 0
            print(f"[CALIB] Point 2 set to: ({x}, {y}). Line calibration complete!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time People Counter with YOLOv8 and Virtual Line Crossing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input video source: webcam index (0), video file path, or RTSP stream URL.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO model weights file (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated video output (e.g. outputs/output.mp4).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=None,
        help="Flag to enable video output recording.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window (for server environments, unit testing, or background tasks).",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=None,
        help="Override buffer distance in pixels.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit after processing this many frames (useful for testing).",
    )
    return parser.parse_args()


def run_counter(
    source_override: Optional[str] = None,
    config_path: str = "config.yaml",
    model_override: Optional[str] = None,
    output_override: Optional[str] = None,
    save_video_override: Optional[bool] = None,
    headless: bool = False,
    buffer_override: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Main execution loop for People Counter.

    Returns:
        Tuple of (total_in, total_out, final_occupancy)
    """
    # 1. Load Configuration
    config = load_config(config_path)

    # Apply overrides
    source = source_override if source_override is not None else config.get("source", 0)
    model_name = model_override if model_override is not None else config.get("model", "yolov8n.pt")
    confidence = config.get("confidence", 0.40)
    iou = config.get("iou", 0.50)
    tracker = config.get("tracker", "bytetrack.yaml")
    device = config.get("device", None)

    line_cfg = config.get("line", {})
    p1 = tuple(line_cfg.get("point1", [100, 360]))
    p2 = tuple(line_cfg.get("point2", [540, 360]))
    buffer_dist = buffer_override if buffer_override is not None else line_cfg.get("buffer_distance", 20.0)
    in_dir = line_cfg.get("in_direction", "A_to_B")
    check_bounds = line_cfg.get("check_segment_bounds", False)

    perf_cfg = config.get("performance", {})
    frame_skip = max(1, perf_cfg.get("frame_skip", 1))
    resize_w = perf_cfg.get("resize_width", 640)
    resize_h = perf_cfg.get("resize_height", 480)

    log_cfg = config.get("logging", {})
    db_path = log_cfg.get("db_path", "people_counter.db")
    csv_path = log_cfg.get("csv_path", "outputs/crossings_log.csv")
    export_csv = log_cfg.get("export_csv_on_exit", True)

    vid_cfg = config.get("video_output", {})
    save_video = save_video_override if save_video_override is not None else vid_cfg.get("save_video", False)
    out_video_path = output_override if output_override is not None else vid_cfg.get("output_path", "outputs/annotated_output.mp4")

    print("=" * 60)
    print(" YOLOv8 Real-Time People Counter")
    print(f" Source:      {source}")
    print(f" Model:       {model_name}")
    print(f" Line:        {p1} -> {p2} (Buffer: ±{buffer_dist}px, Dir: {in_dir})")
    print(f" Database:    {db_path}")
    print(f" Save Video:  {save_video} ({out_video_path})")
    print(f" Headless:    {headless}")
    print("=" * 60)

    # 2. Initialize Components
    logger = CrossingLogger(db_path=db_path, default_csv_path=csv_path)
    line_counter = LineCounter(
        point1=p1,
        point2=p2,
        buffer_distance=buffer_dist,
        in_direction=in_dir,
        check_segment_bounds=check_bounds,
    )

    try:
        stream = VideoStream(source)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 0, 0, 0
    except Exception as e:
        print(f"[ERROR] Failed to initialize video stream from '{source}': {e}")
        return 0, 0, 0

    print("[INFO] Initializing YOLOv8 model (downloading weights if necessary)...")
    detector = PersonDetector(
        model_name=model_name,
        confidence=confidence,
        iou=iou,
        tracker=tracker,
        device=device,
    )
    print("[INFO] YOLOv8 model initialized successfully.")

    # 3. Video Writer Setup
    video_writer = None
    if save_video:
        os.makedirs(os.path.dirname(os.path.abspath(out_video_path)), exist_ok=True)
        target_w = resize_w if resize_w else stream.width
        target_h = resize_h if resize_h else stream.height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_video_path, fourcc, float(stream.fps), (target_w, target_h))

    # 4. GUI & Calibration Setup
    window_name = "YOLOv8 People Counter"
    calib = CalibrationState()
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback, calib)

    # 5. Main Processing Loop
    frame_count = 0
    fps_val = 0.0
    fps_start_time = time.time()
    fps_frame_count = 0
    last_tracked_people = []

    print("[INFO] Processing stream. Key commands: [L] Redefine Line  [R] Reset  [Q] Quit")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("[INFO] End of video stream or stream disconnected.")
                break

            frame_count += 1

            if max_frames and frame_count > max_frames:
                print(f"[INFO] Reached max frame limit ({max_frames}). Stopping.")
                break

            # Resize if configured
            if resize_w and resize_h:
                frame = cv2.resize(frame, (resize_w, resize_h))

            # Handle new line calibration from mouse clicks
            if calib.new_line_ready and calib.new_point1 and calib.new_point2:
                line_counter.set_line(calib.new_point1, calib.new_point2)
                # Persist to config dict and file
                config["line"]["point1"] = list(calib.new_point1)
                config["line"]["point2"] = list(calib.new_point2)
                save_config(config, config_path)
                print(f"[INFO] Virtual line updated: {calib.new_point1} -> {calib.new_point2} and saved to config.")
                calib.reset()

            # Detection and tracking (with frame skip support)
            if frame_count % frame_skip == 0:
                tracked_people = detector.track(frame)
                last_tracked_people = tracked_people
            else:
                tracked_people = last_tracked_people

            # Extract track centroids and update line counter
            active_tracks = [(p.track_id, p.centroid) for p in tracked_people]
            crossing_events = line_counter.update(active_tracks)

            # Log events
            for event in crossing_events:
                logger.log_event(event)
                arrow = "==> [IN]" if event.direction == "IN" else "<== [OUT]"
                print(f"[EVENT] {arrow} Track #{event.track_id} | Total IN: {event.in_total} | Total OUT: {event.out_total} | Occupancy: {event.occupancy}")

            # Compute smoothed FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 0.5:
                fps_val = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Render annotations
            annotated_frame = frame.copy()
            draw_line_and_buffer(
                annotated_frame,
                line_counter,
                is_calibrating=calib.active,
                temp_point=calib.temp_cursor if (calib.active and calib.step == 1) else None,
            )
            draw_tracks(annotated_frame, tracked_people, line_counter)
            draw_hud(
                annotated_frame,
                in_count=line_counter.total_in,
                out_count=line_counter.total_out,
                occupancy=line_counter.occupancy,
                fps=fps_val,
                is_calibrating=calib.active,
                calib_step=calib.step,
            )

            # Write to output video file if requested
            if video_writer is not None:
                video_writer.write(annotated_frame)

            # Display GUI window if not in headless mode
            if not headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    print("[INFO] Quit requested by user.")
                    break
                elif key in (ord("r"), ord("R")):
                    line_counter.reset()
                    print("[INFO] Counters and tracks reset to 0.")
                elif key in (ord("l"), ord("L")):
                    calib.active = not calib.active
                    calib.step = 0
                    calib.point1 = None
                    if calib.active:
                        print("[INFO] Calibration Mode ON: Click Point 1 on the video window.")
                    else:
                        print("[INFO] Calibration Mode cancelled.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # 6. Cleanup & Export
        stream.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Annotated video saved to: {os.path.abspath(out_video_path)}")

        if not headless:
            cv2.destroyAllWindows()

        if export_csv:
            csv_file = logger.export_csv(csv_path)
            print(f"[INFO] Crossing event log exported to: {csv_file}")

        print("=" * 60)
        print(" People Counter Session Summary")
        print(f" Total IN:         {line_counter.total_in}")
        print(f" Total OUT:        {line_counter.total_out}")
        print(f" Final Occupancy:  {line_counter.occupancy}")
        print("=" * 60)

    return line_counter.total_in, line_counter.total_out, line_counter.occupancy


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_counter(
        source_override=args.source,
        config_path=args.config,
        model_override=args.model,
        output_override=args.output,
        save_video_override=args.save_video,
        headless=args.headless,
        buffer_override=args.buffer,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
