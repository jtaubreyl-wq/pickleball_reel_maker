import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class TrajectoryPoint:
    """
    Single ball position sample for a given frame.

    Attributes
    ----------
    frame : int
        Frame index in the source video.
    x : float
        X coordinate of the ball in image space.
    y : float
        Y coordinate of the ball in image space.
    velocity : float
        Per-frame speed (pixel distance between consecutive frames).
    direction : float
        Per-frame direction in degrees (atan2 of dy, dx).
    """
    frame: int
    x: float
    y: float
    velocity: float = 0.0
    direction: float = 0.0


class BallTrajectoryExtractor:
    """
    Extracts ball trajectory features and rally segments from a raw track.

    This class:
    - Converts raw tracking data into typed trajectory points.
    - Computes per-frame velocity and direction.
    - Detects direction changes and bounce candidates.
    - Segments the track into rallies based on bounces and direction changes.
    - Always returns at least one rally segment so the downstream
      highlight generator does not fall back to a 1-second clip.
    """

    def __init__(
        self,
        bounce_vel_drop_ratio: float = 0.55,
        bounce_vel_spike_ratio: float = 1.35,
        min_direction_change_deg: float = 25.0,
        min_rally_length_frames: int = 15,
    ):
        """
        Parameters
        ----------
        bounce_vel_drop_ratio : float
            Threshold ratio for velocity drop to consider a bounce candidate.
        bounce_vel_spike_ratio : float
            Threshold ratio for velocity spike after a drop to confirm a bounce.
        min_direction_change_deg : float
            Minimum direction change (in degrees) to be considered significant.
        min_rally_length_frames : int
            Minimum number of frames for a rally to be considered valid.
            If no rally passes this filter, the full track is used as a single rally.
        """
        self.bounce_vel_drop_ratio = bounce_vel_drop_ratio
        self.bounce_vel_spike_ratio = bounce_vel_spike_ratio
        self.min_direction_change_deg = min_direction_change_deg
        self.min_rally_length_frames = min_rally_length_frames

    # ---------------------------------------------------------
    # 1. Convert raw track into TrajectoryPoint objects
    # ---------------------------------------------------------
    def _to_points(self, track: List[Dict]) -> List[TrajectoryPoint]:
        """
        Convert a list of raw dicts into TrajectoryPoint objects.

        Expected input dict keys: "frame", "x", "y".
        """
        return [
            TrajectoryPoint(
                frame=int(t["frame"]),
                x=float(t["x"]),
                y=float(t["y"]),
            )
            for t in track
        ]

    # ---------------------------------------------------------
    # 2. Compute per-frame velocity
    # ---------------------------------------------------------
    def _compute_velocity(self, pts: List[TrajectoryPoint]) -> None:
        """
        Compute per-frame velocity (pixel distance between consecutive points)."""
        for i in range(1, len(pts)):
            dx = pts[i].x - pts[i - 1].x
            dy = pts[i].y - pts[i - 1].y
            pts[i].velocity = math.sqrt(dx * dx + dy * dy)

    # ---------------------------------------------------------
    # 3. Compute direction angle (degrees)
    # ---------------------------------------------------------
    def _compute_direction(self, pts: List[TrajectoryPoint]) -> None:
        """
        Compute per-frame direction angle in degrees using atan2(dy, dx)."""
        for i in range(1, len(pts)):
            dx = pts[i].x - pts[i - 1].x
            dy = pts[i].y - pts[i - 1].y
            pts[i].direction = math.degrees(math.atan2(dy, dx))

    # ---------------------------------------------------------
    # 4. Detect direction changes
    # ---------------------------------------------------------
    def _compute_direction_changes(self, pts: List[TrajectoryPoint]) -> List[int]:
        """
        Detect frames where the direction changes significantly.

        Returns
        -------
        List[int]
            List of frame indices where the direction change exceeds
            `min_direction_change_deg`.
        """
        changes: List[int] = []
        for i in range(2, len(pts)):
            prev = pts[i - 1].direction
            curr = pts[i].direction
            delta = abs(curr - prev)

            # Normalize to [0, 180]
            if delta > 180:
                delta = 360 - delta

            if delta >= self.min_direction_change_deg:
                changes.append(pts[i].frame)

        return changes

    # ---------------------------------------------------------
    # 5. Bounce detection
    # ---------------------------------------------------------
    def _detect_bounces(self, pts: List[TrajectoryPoint]) -> List[int]:
        """
        Detect bounce frames based on velocity drop followed by a spike.

        Returns
        -------
        List[int]
            List of frame indices considered as bounce events.
        """
        bounces: List[int] = []
        for i in range(2, len(pts)):
            v_prev = pts[i - 1].velocity
            v_curr = pts[i].velocity

            if v_prev > 0 and v_curr < v_prev * self.bounce_vel_drop_ratio:
                if i + 1 < len(pts):
                    v_next = pts[i + 1].velocity
                    if v_next > v_prev * self.bounce_vel_spike_ratio:
                        bounces.append(pts[i].frame)

        # Ensure unique and sorted
        bounces = sorted(set(bounces))
        return bounces

    # ---------------------------------------------------------
    # 6. Rally segmentation
    # ---------------------------------------------------------
    def _segment_rallies(
        self,
        pts: List[TrajectoryPoint],
        bounces: List[int],
        direction_changes: List[int],
    ) -> List[Tuple[int, int]]:
        """
        Segment the trajectory into rallies using bounces and direction changes.

        Logic
        -----
        - If we have multiple bounces, we split rallies between them when
          there is at least one direction change in between.
        - If there are no bounces at all, we treat the entire track as a
          single rally.
        - We filter out rallies that are shorter than `min_rally_length_frames`.
        - If after filtering there are no rallies, we fall back to a single
          rally spanning the full track.

        Returns
        -------
        List[Tuple[int, int]]
            List of (start_frame, end_frame) rally segments.
        """
        if not pts:
            return []

        # If no bounces detected, treat the entire track as one rally.
        if not bounces:
            full_segment = (pts[0].frame, pts[-1].frame)
            return [full_segment]

        rally_segments: List[Tuple[int, int]] = []
        current_start = bounces[0]

        for i in range(1, len(bounces)):
            prev = bounces[i - 1]
            curr = bounces[i]

            # If there is a direction change between two bounces,
            # we consider that the end of a rally.
            if any(prev < dc < curr for dc in direction_changes):
                rally_segments.append((current_start, prev))
                current_start = curr

        # Close the last rally at the last bounce
        rally_segments.append((current_start, bounces[-1]))

        # Filter out very short rallies
        filtered: List[Tuple[int, int]] = []
        for start, end in rally_segments:
            if end > start and (end - start + 1) >= self.min_rally_length_frames:
                filtered.append((start, end))

        # If everything was filtered out, fall back to a single rally
        # spanning the full track. This prevents the downstream pipeline
        # from producing a tiny 1-second "highlight".
        if not filtered:
            filtered = [(pts[0].frame, pts[-1].frame)]

        return filtered

    # ---------------------------------------------------------
    # 7. Main extraction function
    # ---------------------------------------------------------
    def extract(self, track: List[Dict]) -> List[Dict]:
        """
        Main entry point: compute trajectory features and rally segments.

        Parameters
        ----------
        track : List[Dict]
            Raw tracking data. Each dict must contain:
            - "frame": int
            - "x": float
            - "y": float

        Returns
        -------
        List[Dict]
            Per-frame trajectory data with:
            - frame, x, y
            - vx, vy (vx is velocity magnitude, vy is kept as direction for compatibility)
            - direction, velocity
            - is_bounce: bool
            - is_direction_change: bool
            - bounces: List[int] (same list on every frame for convenience)
            - direction_changes: List[int]
            - rallies: List[Tuple[int, int]] rally segments
        """
        pts = self._to_points(track)

        # Not enough points to compute anything meaningful
        if len(pts) < 2:
            return []

        # Compute basic kinematics
        self._compute_velocity(pts)
        self._compute_direction(pts)

        # Higher-level events
        direction_changes = self._compute_direction_changes(pts)
        bounces = self._detect_bounces(pts)
        rallies = self._segment_rallies(pts, bounces, direction_changes)

        # Precompute sets for quick per-frame flags
        bounce_set = set(bounces)
        direction_change_set = set(direction_changes)

        # ---------------------------------------------------------
        # Return list of dicts (what pipeline expects)
        # ---------------------------------------------------------
        trajectory: List[Dict] = []
        for p in pts:
            is_bounce = p.frame in bounce_set
            is_direction_change = p.frame in direction_change_set

            trajectory.append(
                {
                    # Core geometry
                    "frame": p.frame,
                    "x": p.x,
                    "y": p.y,
                    # Keep vx/vy naming for compatibility with existing pipeline.
                    # vx: magnitude of velocity, vy: we store direction (deg).
                    "vx": p.velocity,
                    "vy": p.direction,
                    # Explicit fields
                    "direction": p.direction,
                    "velocity": p.velocity,
                    # Event flags
                    "is_bounce": is_bounce,
                    "is_direction_change": is_direction_change,
                    # Global lists (same on every frame for convenience)
                    "bounces": bounces,
                    "direction_changes": direction_changes,
                    "rallies": rallies,
                }
            )

        return trajectory
