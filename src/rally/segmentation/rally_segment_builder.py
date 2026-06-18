# file name : src/rally/segmentation/rally_segment_builder.py

# ============================================================
# SECTION: Imports & Data Models
# ============================================================

from typing import List, Dict, Any, Optional
import math


# ============================================================
# SECTION: RallySegment Data Container
# ============================================================

class RallySegment:
    def __init__(self, rally_id: int, start_frame: int, end_frame: int):
        self.rally_id = rally_id
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.frames: List[int] = []
        self.positions: List[tuple] = []
        self.speeds: List[float] = []
        self.directions: List[float] = []
        self.trajectory: List[Dict[str, Any]] = []

        self.fps: Optional[float] = None
        self.player_boxes: Dict[str, List[Any]] = {}
        self.last_hitter: Optional[str] = None
        self.player_positions: Dict[str, Any] = {}

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_s: Optional[float] = None

        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"RallySegment(id={self.rally_id}, start={self.start_frame}, end={self.end_frame})"


# ============================================================
# SECTION: Forgiving Rally Segment Builder
# ============================================================

class RallySegmentBuilder:
    """
    Forgiving Rally Segment Builder

    FEATURES:
    • Merges noisy start/end events
    • Fallback rally detection using trajectory
    • Rejects micro-rallies
    • Ensures trajectory continuity
    • Safe to reuse across multiple videos (resettable)
    """

    def __init__(
        self,
        min_duration_s: float = 0.5,
        fps: float = 30.0,
        merge_window_frames: int = 12,
        fallback_speed_threshold: float = 1.0,
        fallback_min_frames: int = 10,
    ):
        self.start_events: List[Dict] = []
        self.end_events: List[Dict] = []
        self.min_duration_s = min_duration_s
        self.fps = fps

        self.merge_window_frames = merge_window_frames
        self.fallback_speed_threshold = fallback_speed_threshold
        self.fallback_min_frames = fallback_min_frames

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def reset(self):
        """
        Clear any previously registered start/end events so the builder
        can be reused for a fresh video/pipeline run.
        """
        self.start_events = []
        self.end_events = []

    # --------------------------------------------------------
    # Event Registration
    # --------------------------------------------------------

    def add_start_events(self, events: List[Dict]):
        self.start_events.extend(events)

    def add_end_events(self, events: List[Dict]):
        self.end_events.extend(events)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _slice_trajectory(self, traj: List[Dict], start: int, end: int) -> List[Dict]:
        return [p for p in traj if start <= p["frame"] <= end]

    def _compute_speeds(self, traj: List[Dict]) -> List[float]:
        speeds: List[float] = []
        for i in range(1, len(traj)):
            dx = traj[i]["x"] - traj[i - 1]["x"]
            dy = traj[i]["y"] - traj[i - 1]["y"]
            speeds.append(math.sqrt(dx * dx + dy * dy))
        return speeds

    def _compute_directions(self, traj: List[Dict]) -> List[float]:
        return [p.get("direction", 0.0) for p in traj[1:]]

    # --------------------------------------------------------
    # Merge noisy events
    # --------------------------------------------------------

    def _merge_events(self, events: List[int]) -> List[int]:
        if not events:
            return []
        events = sorted(events)
        merged = [events[0]]
        for f in events[1:]:
            if f - merged[-1] <= self.merge_window_frames:
                # Treat events within the merge window as the same logical event
                continue
            merged.append(f)
        return merged

    # --------------------------------------------------------
    # Fallback rally detection
    # --------------------------------------------------------

    def _fallback_detect(self, trajectory: List[Dict]) -> List[tuple]:
        """
        Detect rallies based on continuous movement when start/end detectors fail
        or produce no usable events.
        """
        fallback_segments: List[tuple] = []
        active = False
        start_f: Optional[int] = None
        count = 0

        for i in range(1, len(trajectory)):
            dx = trajectory[i]["x"] - trajectory[i - 1]["x"]
            dy = trajectory[i]["y"] - trajectory[i - 1]["y"]
            speed = math.sqrt(dx * dx + dy * dy)

            if speed >= self.fallback_speed_threshold:
                if not active:
                    active = True
                    start_f = trajectory[i]["frame"]
                count += 1
            else:
                if active and count >= self.fallback_min_frames and start_f is not None:
                    end_f = trajectory[i]["frame"]
                    fallback_segments.append((start_f, end_f))
                active = False
                count = 0

        # Handle case where movement continues until the last frame
        if active and count >= self.fallback_min_frames and start_f is not None:
            end_f = trajectory[-1]["frame"]
            fallback_segments.append((start_f, end_f))

        return fallback_segments

    # --------------------------------------------------------
    # Core Builder Logic
    # --------------------------------------------------------

    def build_segments(self, trajectory: List[Dict]) -> List[RallySegment]:

        segments: List[RallySegment] = []

        if not trajectory:
            return segments

        # Sort trajectory by frame to ensure continuity
        traj = sorted(trajectory, key=lambda p: p["frame"])

        # Extract raw start/end frames from events
        start_frames = [e["frame"] for e in self.start_events if "frame" in e]
        end_frames = [e["frame"] for e in self.end_events if "frame" in e]

        # Merge noisy events
        start_frames = self._merge_events(start_frames)
        end_frames = self._merge_events(end_frames)

        # Fallback detection if no usable events
        if not start_frames or not end_frames:
            fallback = self._fallback_detect(traj)
            for s, e in fallback:
                start_frames.append(s)
                end_frames.append(e)

        # Sort again after fallback
        start_frames = sorted(start_frames)
        end_frames = sorted(end_frames)

        if not start_frames or not end_frames:
            # Still nothing usable – return empty list
            return segments

        # Pair start/end
        end_idx = 0
        rally_id = 1
        last_end = -9999

        for s in start_frames:
            # Skip starts that occur before the last accepted end
            if s <= last_end:
                continue

            # Find the next end frame strictly after this start
            while end_idx < len(end_frames) and end_frames[end_idx] <= s:
                end_idx += 1
            if end_idx >= len(end_frames):
                break

            e = end_frames[end_idx]

            # Duration check
            duration_s = (e - s) / self.fps
            if duration_s < self.min_duration_s:
                continue

            # Slice trajectory for this rally
            traj_slice = self._slice_trajectory(traj, s, e)
            if len(traj_slice) < 2:
                continue

            seg = RallySegment(rally_id, s, e)
            seg.fps = self.fps
            seg.duration_s = duration_s
            seg.start_time = s / self.fps
            seg.end_time = e / self.fps

            seg.trajectory = traj_slice
            seg.frames = [p["frame"] for p in traj_slice]
            seg.positions = [(p["x"], p["y"]) for p in traj_slice]
            seg.speeds = self._compute_speeds(traj_slice)
            seg.directions = self._compute_directions(traj_slice)

            segments.append(seg)
            rally_id += 1
            last_end = e

        return segments
