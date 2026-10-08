"""
Line Crossing Detection and Occupancy Logic
Handles virtual line definition, 2D vector signed distance calculations,
anti-jitter buffer zones, and direction classification (IN / OUT).
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class CrossingEvent:
    """Represents a single verified line crossing event."""
    track_id: int
    direction: str  # "IN" or "OUT"
    timestamp: datetime
    in_total: int
    out_total: int
    occupancy: int
    centroid: Tuple[int, int]


@dataclass
class TrackState:
    """Maintains tracking and crossing state for an individual track ID."""
    track_id: int
    last_confirmed_side: Optional[str] = None  # 'A', 'B', or None
    is_in_buffer: bool = False
    last_crossed_direction: Optional[str] = None
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    cross_count: int = 0


class LineCounter:
    """
    Virtual line crossing counter using 2D vector geometry and a buffer zone
    to prevent double counting from tracking jitter.
    """

    def __init__(
        self,
        point1: Tuple[int, int] = (100, 360),
        point2: Tuple[int, int] = (540, 360),
        buffer_distance: float = 20.0,
        in_direction: str = "A_to_B",
        check_segment_bounds: bool = False,
        bounds_margin: float = 0.25,
    ) -> None:
        """
        Initialize the LineCounter.

        Args:
            point1: Starting coordinate (x1, y1) of the virtual line.
            point2: Ending coordinate (x2, y2) of the virtual line.
            buffer_distance: Half-width of the buffer zone in pixels (±buffer_distance).
            in_direction: Direction that constitutes an "IN" count:
                          "A_to_B" means crossing from Side A to Side B is IN.
                          "B_to_A" means crossing from Side B to Side A is IN.
            check_segment_bounds: If True, only counts crossings that occur within
                                  the line segment bounds (plus margin).
            bounds_margin: Fractional margin beyond segment endpoints (default 0.25).
        """
        self.point1 = point1
        self.point2 = point2
        self.buffer_distance = max(1.0, float(buffer_distance))
        self.in_direction = in_direction
        self.check_segment_bounds = check_segment_bounds
        self.bounds_margin = bounds_margin

        self.total_in = 0
        self.total_out = 0
        self.tracks: Dict[int, TrackState] = {}
        self._precompute_line_geometry()

    def _precompute_line_geometry(self) -> None:
        """Precomputes vector components and length of the virtual line."""
        x1, y1 = self.point1
        x2, y2 = self.point2
        self.dx = float(x2 - x1)
        self.dy = float(y2 - y1)
        self.length = math.hypot(self.dx, self.dy)
        if self.length < 1e-6:
            # Fallback for degenerate line
            self.length = 1.0
            self.dx = 1.0
            self.dy = 0.0

    def set_line(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> None:
        """
        Update the virtual line coordinates at runtime.

        Args:
            point1: New start coordinate (x1, y1).
            point2: New end coordinate (x2, y2).
        """
        self.point1 = (int(point1[0]), int(point1[1]))
        self.point2 = (int(point2[0]), int(point2[1]))
        self._precompute_line_geometry()
        # Reset track states since reference line moved
        self.tracks.clear()

    def set_buffer_distance(self, distance: float) -> None:
        """Update buffer distance threshold."""
        self.buffer_distance = max(1.0, float(distance))

    def reset(self) -> None:
        """Reset all counters and active track states."""
        self.total_in = 0
        self.total_out = 0
        self.tracks.clear()

    @property
    def occupancy(self) -> int:
        """
        Current occupancy: IN - OUT, guaranteed never to fall below 0.
        """
        return max(0, self.total_in - self.total_out)

    def compute_signed_distance(self, point: Tuple[float, float]) -> float:
        """
        Compute perpendicular signed distance from a point to the directed line.

        Positive distance represents Side A, negative represents Side B.
        Zero represents lying exactly on the line.

        Distance formula:
            cross_product = dx * (y - y1) - dy * (x - x1)
            signed_dist = cross_product / length
        """
        x, y = point
        x1, y1 = self.point1
        cross = self.dx * (y - y1) - self.dy * (x - x1)
        return cross / self.length

    def compute_segment_projection(self, point: Tuple[float, float]) -> float:
        """
        Compute parametric projection parameter t along the line segment.
        t = 0 corresponds to point1, t = 1 corresponds to point2.
        """
        x, y = point
        x1, y1 = self.point1
        dot = (x - x1) * self.dx + (y - y1) * self.dy
        return dot / (self.length ** 2)

    def determine_side(self, point: Tuple[float, float]) -> Tuple[str, float]:
        """
        Determine which side of the line a point falls on with respect to the buffer.

        Returns:
            Tuple of (side, signed_distance) where side is 'A', 'B', or 'BUFFER'.
        """
        dist = self.compute_signed_distance(point)
        if dist > self.buffer_distance:
            return "A", dist
        elif dist < -self.buffer_distance:
            return "B", dist
        else:
            return "BUFFER", dist

    def update(
        self,
        active_tracks: List[Tuple[int, Tuple[int, int]]],
        current_time: Optional[datetime] = None,
    ) -> List[CrossingEvent]:
        """
        Process current frame detections and detect line crossings.

        Args:
            active_tracks: List of tuples (track_id, (cx, cy))
            current_time: Optional timestamp (defaults to datetime.now())

        Returns:
            List of CrossingEvent objects generated in this frame.
        """
        if current_time is None:
            current_time = datetime.now()

        events: List[CrossingEvent] = []
        active_ids = set()

        for track_id, centroid in active_tracks:
            active_ids.add(track_id)

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackState(track_id=track_id)

            state = self.tracks[track_id]
            state.positions.append(centroid)

            # Check segment bounds if enabled
            if self.check_segment_bounds:
                t = self.compute_segment_projection(centroid)
                if t < -self.bounds_margin or t > (1.0 + self.bounds_margin):
                    # Outside line segment span, skip crossing check
                    continue

            current_side, _ = self.determine_side(centroid)

            # State transition logic with buffer zone anti-jitter
            if current_side == "BUFFER":
                # Currently within the neutral buffer zone
                state.is_in_buffer = True

            elif current_side == "A":
                # Centroid is confirmed on Side A
                if state.last_confirmed_side == "B":
                    # Crossed from B to A!
                    direction = "OUT" if self.in_direction == "A_to_B" else "IN"
                    event = self._record_crossing(track_id, direction, centroid, current_time)
                    events.append(event)
                    state.last_crossed_direction = direction
                    state.cross_count += 1

                # Update confirmed side to A
                state.last_confirmed_side = "A"
                state.is_in_buffer = False

            elif current_side == "B":
                # Centroid is confirmed on Side B
                if state.last_confirmed_side == "A":
                    # Crossed from A to B!
                    direction = "IN" if self.in_direction == "A_to_B" else "OUT"
                    event = self._record_crossing(track_id, direction, centroid, current_time)
                    events.append(event)
                    state.last_crossed_direction = direction
                    state.cross_count += 1

                # Update confirmed side to B
                state.last_confirmed_side = "B"
                state.is_in_buffer = False

        # Prune very old tracks not seen recently (optional garbage collection)
        if len(self.tracks) > 500:
            stale_ids = [tid for tid in self.tracks if tid not in active_ids]
            if len(stale_ids) > 200:
                for tid in stale_ids[:100]:
                    del self.tracks[tid]

        return events

    def _record_crossing(
        self,
        track_id: int,
        direction: str,
        centroid: Tuple[int, int],
        timestamp: datetime,
    ) -> CrossingEvent:
        """Internal helper to increment counters and create event."""
        if direction == "IN":
            self.total_in += 1
        else:
            self.total_out += 1

        return CrossingEvent(
            track_id=track_id,
            direction=direction,
            timestamp=timestamp,
            in_total=self.total_in,
            out_total=self.total_out,
            occupancy=self.occupancy,
            centroid=centroid,
        )

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Return recorded centroid trajectory for a given track ID."""
        if track_id in self.tracks:
            return list(self.tracks[track_id].positions)
        return []

    def get_buffer_boundary_lines(self) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calculate offset line coordinates representing the buffer boundaries
        for visualization.
        """
        # Normal vector perpendicular to (dx, dy): (-dy / length, dx / length)
        nx = -self.dy / self.length
        ny = self.dx / self.length

        ox = nx * self.buffer_distance
        oy = ny * self.buffer_distance

        x1, y1 = self.point1
        x2, y2 = self.point2

        line_a = ((int(round(x1 + ox)), int(round(y1 + oy))), (int(round(x2 + ox)), int(round(y2 + oy))))
        line_b = ((int(round(x1 - ox)), int(round(y1 - oy))), (int(round(x2 - ox)), int(round(y2 - oy))))
        return line_a, line_b
