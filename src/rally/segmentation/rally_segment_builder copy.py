# file: src/rally/segmentation/rally_segment_builder.py

from typing import List, Dict


class RallySegment:
    """
    Container for a rally segment.
    Includes trajectory slice, speeds, positions, frames, and player_boxes.
    """

    def __init__(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame

        # Filled by builder
        self.frames: List[int] = []
        self.positions: List[tuple] = []
        self.speeds: List[float] = []
        self.directions: List[float] = []
        self.trajectory: List[Dict] = []

        # Player boxes (metadata engine expects this)
        self.player_boxes: List[Dict] = []

        # Filled later by metadata engine
        self.start_time = None
        self.end_time = None
        self.metadata = {}

    def __repr__(self):
        return f"RallySegment(start={self.start_frame}, end={self.end_frame})"


class RallySegmentBuilder:
    """
    Story 5.3 – Rally Segment Builder
    Pairs start/end events and attaches trajectory slices.
    """

    def __init__(self):
        self.start_events: List[Dict] = []
        self.end_events: List[Dict] = []

    def add_start_events(self, events: List[Dict]):
        self.start_events.extend(events)

    def add_end_events(self, events: List[Dict]):
        self.end_events.extend(events)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _slice_trajectory(self, trajectory: List[Dict], start: int, end: int):
        """Extract trajectory points between start_frame and end_frame."""
        return [p for p in trajectory if start <= p["frame"] <= end]

    def _compute_speeds(self, traj: List[Dict]) -> List[float]:
        speeds = []
        for i in range(1, len(traj)):
            x1, y1 = traj[i - 1]["x"], traj[i - 1]["y"]
            x2, y2 = traj[i]["x"], traj[i]["y"]
            dx = x2 - x1
            dy = y2 - y1
            speeds.append((dx * dx + dy * dy) ** 0.5)
        return speeds

    def _compute_directions(self, traj: List[Dict]) -> List[float]:
        dirs = []
        for i in range(1, len(traj)):
            dirs.append(traj[i].get("direction", 0.0))
        return dirs

    # ---------------------------------------------------------
    # Main builder
    # ---------------------------------------------------------

    def build_segments(self, trajectory: List[Dict]) -> List[RallySegment]:
        """
        Build rally segments by pairing start and end events
        and attaching trajectory slices.
        """

        segments: List[RallySegment] = []

        # Sort events by frame index
        starts = sorted(self.start_events, key=lambda e: e["frame"])
        ends = sorted(self.end_events, key=lambda e: e["frame"])

        end_idx = 0

        for start in starts:
            start_frame = start["frame"]

            # Find the first end event that occurs AFTER this start
            while end_idx < len(ends) and ends[end_idx]["frame"] <= start_frame:
                end_idx += 1

            if end_idx < len(ends):
                end_frame = ends[end_idx]["frame"]
                seg = RallySegment(start_frame, end_frame)

                # Slice trajectory
                traj_slice = self._slice_trajectory(trajectory, start_frame, end_frame)
                seg.trajectory = traj_slice

                # Extract frames & positions
                seg.frames = [p["frame"] for p in traj_slice]
                seg.positions = [(p["x"], p["y"]) for p in traj_slice]

                # Compute speeds & directions
                seg.speeds = self._compute_speeds(traj_slice)
                seg.directions = self._compute_directions(traj_slice)

                # Player boxes — currently unknown, leave empty
                seg.player_boxes = []

                segments.append(seg)
                end_idx += 1

        return segments
