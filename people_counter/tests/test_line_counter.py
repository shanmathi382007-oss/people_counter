"""
Unit Tests for Line Crossing Logic (LineCounter)
Tests covering both directions (IN/OUT), tracking jitter rejection,
touching the line without crossing, vertical/diagonal line orientations,
and occupancy boundary guarantees.
"""

from __future__ import annotations
import pytest
from datetime import datetime
from counter.line_counter import LineCounter


@pytest.fixture
def horizontal_counter():
    """
    Standard horizontal line at y=300 from x=100 to x=500.
    With buffer_distance=20.0:
    - Side A: y > 320 (cross_product > 0)
    - Buffer: 280 <= y <= 320
    - Side B: y < 280 (cross_product < 0)
    Default in_direction='A_to_B', so crossing from y > 320 to y < 280 is IN.
    """
    return LineCounter(
        point1=(100, 300),
        point2=(500, 300),
        buffer_distance=20.0,
        in_direction="A_to_B",
        check_segment_bounds=False,
    )


def test_crossing_in_direction(horizontal_counter):
    """Verify that moving completely from Side A to Side B triggers exactly one IN count."""
    counter = horizontal_counter
    t0 = datetime.now()

    # Frame 1: Person 10 starts well within Side A (y=360)
    events = counter.update([(10, (300, 360))], current_time=t0)
    assert len(events) == 0
    assert counter.total_in == 0
    assert counter.total_out == 0

    # Frame 2: Moves into buffer zone (y=305)
    events = counter.update([(10, (300, 305))], current_time=t0)
    assert len(events) == 0
    assert counter.total_in == 0

    # Frame 3: Crosses completely into Side B (y=260)
    events = counter.update([(10, (300, 260))], current_time=t0)
    assert len(events) == 1
    assert events[0].direction == "IN"
    assert events[0].track_id == 10
    assert counter.total_in == 1
    assert counter.total_out == 0
    assert counter.occupancy == 1

    # Frame 4: Continues walking further into Side B (y=220) - should NOT trigger again
    events = counter.update([(10, (300, 220))], current_time=t0)
    assert len(events) == 0
    assert counter.total_in == 1


def test_crossing_out_direction(horizontal_counter):
    """Verify that moving completely from Side B to Side A triggers exactly one OUT count."""
    counter = horizontal_counter
    t0 = datetime.now()

    # Frame 1: Person 20 starts on Side B (y=250)
    events = counter.update([(20, (300, 250))], current_time=t0)
    assert len(events) == 0

    # Frame 2: Person enters buffer zone (y=300)
    events = counter.update([(20, (300, 300))], current_time=t0)
    assert len(events) == 0

    # Frame 3: Person reaches Side A (y=350)
    events = counter.update([(20, (300, 350))], current_time=t0)
    assert len(events) == 1
    assert events[0].direction == "OUT"
    assert events[0].track_id == 20
    assert counter.total_in == 0
    assert counter.total_out == 1
    # Occupancy cannot fall below 0
    assert counter.occupancy == 0


def test_roundtrip_crossings(horizontal_counter):
    """Verify entering (IN) and subsequently exiting (OUT) updates occupancy properly."""
    counter = horizontal_counter
    t0 = datetime.now()

    # Enter: A -> B
    counter.update([(1, (300, 360))], current_time=t0)
    events = counter.update([(1, (300, 250))], current_time=t0)
    assert len(events) == 1
    assert events[0].direction == "IN"
    assert counter.total_in == 1
    assert counter.total_out == 0
    assert counter.occupancy == 1

    # Exit: B -> A
    events = counter.update([(1, (300, 350))], current_time=t0)
    assert len(events) == 1
    assert events[0].direction == "OUT"
    assert counter.total_in == 1
    assert counter.total_out == 1
    assert counter.occupancy == 0


def test_jitter_within_buffer(horizontal_counter):
    """Verify that jittering/oscillating inside the buffer zone does not trigger double counts."""
    counter = horizontal_counter
    t0 = datetime.now()

    # Object starts on Side A
    counter.update([(5, (300, 350))], current_time=t0)

    # Object enters buffer zone and jitters around
    buffer_positions = [(300, 315), (300, 290), (300, 310), (300, 285), (300, 300)]
    for pos in buffer_positions:
        events = counter.update([(5, pos)], current_time=t0)
        assert len(events) == 0, f"Unexpected crossing at position {pos}"

    assert counter.total_in == 0
    assert counter.total_out == 0


def test_touch_line_and_revert(horizontal_counter):
    """Verify that a track that moves into buffer/touches line then returns to origin side is not counted."""
    counter = horizontal_counter
    t0 = datetime.now()

    # Starts on Side A (y=360)
    counter.update([(7, (300, 360))], current_time=t0)

    # Approaches and touches line (y=300, inside buffer zone [-20, +20])
    events = counter.update([(7, (300, 300))], current_time=t0)
    assert len(events) == 0

    # Backs off to buffer edge
    events = counter.update([(7, (300, 315))], current_time=t0)
    assert len(events) == 0

    # Returns completely to Side A (y=370)
    events = counter.update([(7, (300, 370))], current_time=t0)
    assert len(events) == 0
    assert counter.total_in == 0
    assert counter.total_out == 0


def test_vertical_line():
    """Verify crossing detection with a vertical dividing line."""
    # Line from (300, 100) to (300, 500)
    # dx = 0, dy = 400
    # signed_distance for (x, y): (-dy*(x - x1)) / length = -400*(x - 300) / 400 = -(x - 300) = 300 - x
    # If x < 280: dist = 300 - 280 = +20 -> Side A
    # If x > 320: dist = 300 - 320 = -20 -> Side B
    counter = LineCounter(
        point1=(300, 100),
        point2=(300, 500),
        buffer_distance=20.0,
        in_direction="A_to_B",
    )
    t0 = datetime.now()

    # Move from left (x=240, Side A) to right (x=350, Side B) -> should be IN
    counter.update([(1, (240, 250))], current_time=t0)
    events = counter.update([(1, (350, 250))], current_time=t0)
    assert len(events) == 1
    assert events[0].direction == "IN"
    assert counter.total_in == 1


def test_diagonal_line():
    """Verify crossing detection with an angled / diagonal virtual line."""
    # Line from (100, 100) to (500, 500) (45 degree line)
    counter = LineCounter(
        point1=(100, 100),
        point2=(500, 500),
        buffer_distance=15.0,
        in_direction="A_to_B",
    )
    t0 = datetime.now()

    # Normal is perpendicular to (1, 1). Point (350, 200) vs (200, 350)
    counter.update([(8, (350, 200))], current_time=t0)
    events = counter.update([(8, (200, 350))], current_time=t0)
    assert len(events) == 1
    assert counter.total_in == 1 or counter.total_out == 1


def test_segment_bounds_constraint():
    """Verify that when check_segment_bounds is True, crossings outside line endpoints are ignored."""
    counter = LineCounter(
        point1=(200, 300),
        point2=(400, 300),
        buffer_distance=20.0,
        check_segment_bounds=True,
        bounds_margin=0.1,  # x must roughly be within [180, 420]
    )
    t0 = datetime.now()

    # Crossing far outside the line segment at x = 700
    counter.update([(11, (700, 360))], current_time=t0)
    events = counter.update([(11, (700, 240))], current_time=t0)
    assert len(events) == 0
    assert counter.total_in == 0

    # Crossing within the line segment at x = 300
    counter.update([(12, (300, 360))], current_time=t0)
    events = counter.update([(12, (300, 240))], current_time=t0)
    assert len(events) == 1
    assert counter.total_in == 1


def test_occupancy_boundary_guarantee(horizontal_counter):
    """Occupancy should never fall below zero even if OUT counts exceed IN counts."""
    counter = horizontal_counter
    t0 = datetime.now()

    # Two people start on Side B and leave to Side A
    counter.update([(1, (300, 240)), (2, (320, 240))], current_time=t0)
    counter.update([(1, (300, 360)), (2, (320, 360))], current_time=t0)

    assert counter.total_in == 0
    assert counter.total_out == 2
    assert counter.occupancy == 0  # max(0, 0 - 2) == 0


def test_reset_functionality(horizontal_counter):
    """Verify reset clears counts and internal track state."""
    counter = horizontal_counter
    t0 = datetime.now()

    # Trigger a crossing
    counter.update([(1, (300, 360))], current_time=t0)
    counter.update([(1, (300, 250))], current_time=t0)
    assert counter.total_in == 1

    # Reset
    counter.reset()
    assert counter.total_in == 0
    assert counter.total_out == 0
    assert counter.occupancy == 0
    assert len(counter.tracks) == 0
