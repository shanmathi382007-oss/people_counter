"""
YOLOv8 Person Detection and ByteTrack Multi-Object Tracking
Supports both Ultralytics YOLO (PyTorch) and high-performance ONNX Runtime / OpenCV DNN
with built-in ByteTrack multi-object association.
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


@dataclass
class TrackedPerson:
    """Represents a person detected and tracked in a video frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    centroid: Tuple[int, int]         # (cx, cy)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanBoxTracker:
    """
    Lightweight Kalman filter for tracking 2D bounding boxes [x1, y1, x2, y2]
    with constant velocity model.
    """
    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int], score: float) -> None:
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.score = score
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0

        self.state = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next bounding box based on velocity."""
        # Simple constant velocity step: x += vx
        self.state[0] += self.state[4]
        self.state[1] += self.state[5]
        self.state[2] += self.state[6]
        self.state[3] += self.state[7]
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox: Tuple[int, int, int, int], score: float) -> None:
        """Update tracker state with newly observed bounding box."""
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0

        # Velocity smoothing
        alpha = 0.6
        self.state[4] = alpha * (cx - self.state[0]) + (1 - alpha) * self.state[4]
        self.state[5] = alpha * (cy - self.state[1]) + (1 - alpha) * self.state[5]
        self.state[6] = alpha * (w - self.state[2]) + (1 - alpha) * self.state[6]
        self.state[7] = alpha * (h - self.state[3]) + (1 - alpha) * self.state[7]

        self.state[0] = cx
        self.state[1] = cy
        self.state[2] = w
        self.state[3] = h

        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get current bounding box coordinates [x1, y1, x2, y2]."""
        cx, cy, w, h = self.state[:4]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return (x1, y1, x2, y2)


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by associating high-confidence and low-confidence
    detection boxes in two hierarchical matching stages.
    """

    def __init__(self, track_thresh: float = 0.35, match_thresh: float = 0.25, max_age: int = 30) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[TrackedPerson]:
        """
        Update tracks with current frame detections.

        Args:
            detections: List of ((x1, y1, x2, y2), confidence) for person class.

        Returns:
            List of TrackedPerson objects with active track IDs.
        """
        # Step 1: Predict positions for all existing trackers
        for t in self.trackers:
            t.predict()

        # Step 2: Separate high and low confidence detections
        dets_high = [d for d in detections if d[1] >= self.track_thresh]
        dets_low = [d for d in detections if d[1] < self.track_thresh]

        # Step 3: First association with high score detections
        matched_high, unmatched_trks, unmatched_dets_high = self._associate(
            self.trackers, dets_high, self.match_thresh
        )

        for trk_idx, det_idx in matched_high:
            box, score = dets_high[det_idx]
            self.trackers[trk_idx].update(box, score)

        # Step 4: Second association with low score detections for remaining tracks
        remaining_trks = [self.trackers[i] for i in unmatched_trks]
        matched_low, unmatched_trks_low_idx, _ = self._associate(
            remaining_trks, dets_low, self.match_thresh
        )

        for rem_idx, det_idx in matched_low:
            box, score = dets_low[det_idx]
            remaining_trks[rem_idx].update(box, score)

        # Step 5: Initialize new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets_high:
            box, score = dets_high[det_idx]
            self.trackers.append(KalmanBoxTracker(box, score))

        # Step 6: Filter active tracks and prune lost tracks
        active_tracked: List[TrackedPerson] = []
        kept_trackers: List[KalmanBoxTracker] = []

        for t in self.trackers:
            if t.time_since_update == 0 and (t.hit_streak >= 1 or t.hits >= 2):
                bbox = t.get_bbox()
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                active_tracked.append(
                    TrackedPerson(
                        track_id=t.id,
                        bbox=bbox,
                        confidence=t.score,
                        centroid=(cx, cy),
                    )
                )

            if t.time_since_update <= self.max_age:
                kept_trackers.append(t)

        self.trackers = kept_trackers
        return active_tracked

    def _associate(
        self,
        trackers: List[KalmanBoxTracker],
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU bipartite matching between trackers and detections."""
        if len(trackers) == 0 or len(detections) == 0:
            return [], list(range(len(trackers))), list(range(len(detections)))

        iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for t_idx, trk in enumerate(trackers):
            tb = trk.get_bbox()
            for d_idx, (db, _) in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(tb, db)

        matched: List[Tuple[int, int]] = []
        matched_trks = set()
        matched_dets = set()

        # Sort candidate matches by highest IoU first
        flat_indices = np.argsort(-iou_matrix.ravel())
        for idx in flat_indices:
            t_idx = idx // len(detections)
            d_idx = idx % len(detections)

            if iou_matrix[t_idx, d_idx] < iou_thresh:
                break
            if t_idx in matched_trks or d_idx in matched_dets:
                continue

            matched_trks.add(t_idx)
            matched_dets.add(d_idx)
            matched.append((t_idx, d_idx))

        unmatched_trks = [i for i in range(len(trackers)) if i not in matched_trks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
        return matched, unmatched_trks, unmatched_dets


class PersonDetector:
    """
    Unified YOLOv8 person detector supporting:
    1. Ultralytics YOLO (PyTorch) if available
    2. ONNX Runtime / OpenCV DNN with ByteTrack for maximum speed and compatibility
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.50,
        tracker: str = "bytetrack.yaml",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.tracker_name = tracker
        self.device = device

        self.engine_type = "ultralytics"
        self._ultralytics_model = None
        self._ort_session = None
        self._cv2_net = None
        self._bytetracker = ByteTracker(track_thresh=self.confidence, match_thresh=self.iou)

        self._init_engine()

    def _init_engine(self) -> None:
        """Attempt to load Ultralytics YOLO; fall back to ONNX Runtime or OpenCV DNN."""
        # Check if ultralytics can be imported without Windows SAC / torch policy errors
        can_use_ultralytics = False
        try:
            import ultralytics
            from ultralytics import YOLO
            # Check if model ends with .pt and exists or downloads
            self._ultralytics_model = YOLO(self.model_name)
            self.engine_type = "ultralytics"
            can_use_ultralytics = True
            print(f"[INFO] Initialized PersonDetector with Ultralytics YOLO ({self.model_name})")
        except Exception as e:
            # Ultralytics or Torch blocked / unavailable
            can_use_ultralytics = False

        if not can_use_ultralytics:
            # Fall back to ONNX Runtime / OpenCV DNN
            onnx_path = self._find_onnx_model()
            if onnx_path is None:
                raise FileNotFoundError(
                    f"Neither PyTorch/Ultralytics nor ONNX model found. "
                    f"Please provide '{self.model_name}' as an ONNX model or install torch."
                )

            # Try ONNX Runtime first (fastest, signed by Microsoft)
            try:
                import onnxruntime as ort
                # Suppress verbose warnings
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                self._ort_session = ort.InferenceSession(onnx_path, sess_options=opts)
                self.engine_type = "onnxruntime"
                print(f"[INFO] Initialized PersonDetector with ONNX Runtime using '{onnx_path}'")
            except Exception as e:
                # Try OpenCV DNN fallback
                self._cv2_net = cv2.dnn.readNetFromONNX(onnx_path)
                self.engine_type = "cv2_dnn"
                print(f"[INFO] Initialized PersonDetector with OpenCV DNN using '{onnx_path}'")

    def _find_onnx_model(self) -> Optional[str]:
        """Locate YOLOv8 ONNX model in common directories."""
        candidates = [
            self.model_name,
            os.path.splitext(self.model_name)[0] + ".onnx",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov8n.onnx"),
            "yolov8n.onnx",
            os.path.join("people_counter", "yolov8n.onnx"),
        ]
        for c in candidates:
            if os.path.exists(c) and c.endswith(".onnx"):
                return os.path.abspath(c)
        return None

    def track(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Run person detection and tracking on a video frame.

        Args:
            frame: BGR numpy image array.

        Returns:
            List of TrackedPerson objects with persistent track IDs.
        """
        if frame is None or frame.size == 0:
            return []

        if self.engine_type == "ultralytics" and self._ultralytics_model is not None:
            return self._track_ultralytics(frame)
        else:
            return self._track_onnx(frame)

    def _track_ultralytics(self, frame: np.ndarray) -> List[TrackedPerson]:
        """Tracking with Ultralytics built-in ByteTrack."""
        results = self._ultralytics_model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_name,
            classes=[0],  # Person class
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        tracked: List[TrackedPerson] = []
        if not results or len(results) == 0:
            return tracked

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return tracked

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(track_ids)):
            x1, y1, x2, y2 = map(int, coords[i])
            tracked.append(
                TrackedPerson(
                    track_id=int(track_ids[i]),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confs[i]),
                    centroid=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return tracked

    def _track_onnx(self, frame: np.ndarray) -> List[TrackedPerson]:
        """YOLOv8 ONNX inference followed by ByteTrack association."""
        img_h, img_w = frame.shape[:2]

        # Preprocess frame for YOLOv8 (640x640 letterbox/resize, RGB, /255.0)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(640, 640),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        # Inference
        if self.engine_type == "onnxruntime" and self._ort_session is not None:
            input_name = self._ort_session.get_inputs()[0].name
            preds = self._ort_session.run(None, {input_name: blob})[0]  # [1, 84, 8400]
        else:
            self._cv2_net.setInput(blob)
            preds = self._cv2_net.forward()

        # Output shape: (1, 84, 8400) -> transpose to (8400, 84)
        output = preds[0].transpose(1, 0)

        # Extract boxes and person scores (class index 0 is person, offset by 4 for box coords)
        person_scores = output[:, 4]  # Column 4 corresponds to class 0 ('person')
        mask = person_scores > 0.15   # Filter low confidence detections
        output = output[mask]
        person_scores = person_scores[mask]

        if len(output) == 0:
            return self._bytetracker.update([])

        # Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to original frame dimensions
        x_factor = img_w / 640.0
        y_factor = img_h / 640.0

        cx = output[:, 0] * x_factor
        cy = output[:, 1] * y_factor
        w = output[:, 2] * x_factor
        h = output[:, 3] * y_factor

        x1 = np.maximum(0, cx - w / 2.0)
        y1 = np.maximum(0, cy - h / 2.0)
        x2 = np.minimum(img_w, cx + w / 2.0)
        y2 = np.minimum(img_h, cy + h / 2.0)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            person_scores.tolist(),
            score_threshold=0.15,
            nms_threshold=self.iou,
        )

        detections: List[Tuple[Tuple[int, int, int, int], float]] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx = tuple(map(int, boxes_xyxy[idx]))
                score = float(person_scores[idx])
                detections.append((bx, score))

        # Update ByteTrack tracker
        return self._bytetracker.update(detections)
