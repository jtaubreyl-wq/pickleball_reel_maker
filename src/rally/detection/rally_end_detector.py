# FILE: src/rally/rally_end_detector.py
"""
Dynamic Rally End Detector (ball + curvature + motion + player reset to baselines)

Key features:
• Uses ball motion, curvature, and missing frames to detect rally end
• Supports dynamic court orientation (vertical or horizontal)
• Baseline orientation is configured from RallyStartDetector based on first serve
• Detects rally end when players retreat back to their baselines facing opposite sides
"""

# ============================================================
# SECTION: Imports & Data Models
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


# ------------------------------------------------------------
# Ball & Player State Models
# ------------------------------------------------------------

@dataclass
class BallState:
    """
    Ball state at a given frame.

    frame_idx:
        Index of the frame in the video.

    x, y:
        Normalized ball position in [0, 1] coordinates.

    vx, vy:
        Ball velocity components in normalized units per frame.
    """
    frame_idx: int
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class PlayerState:
    """
    Player state for a single frame, built by PlayerFilter / PlayerStateBuilder.

    left_player_x, left_player_y:
        Normalized center position of the left player.

    right_player_x, right_player_y:
        Normalized center position of the right player.

    left_vx, left_vy, right_vx, right_vy:
        Approximate per-frame velocity of each player.

    left_player_in_court, right_player_in_court:
        Whether each player is inside the court bounds.

    left_player_active, right_player_active:
        Whether each player is considered "active" (moving).
    """
    left_player_x: float
    left_player_y: float
    right_player_x: float
    right_player_y: float

    left_vx: float
    left_vy: float
    right_vx: float
    right_vy: float

    left_player_in_court: bool
    right_player_in_court: bool
    left_player_active: bool
    right_player_active: bool

    @property
    def ready_state(self) -> bool:
        return (
            self.left_player_in_court
            and self.right_player_in_court
            and (self.left_player_active or self.right_player_active)
        )


@dataclass
class CourtBounds:
    """
    Normalized court bounds in [0, 1] coordinates.

    left, right:
        Horizontal bounds.

    top, bottom:
        Vertical bounds.
    """
    left: float
    right: float
    top: float
    bottom: float


# ============================================================
# SECTION: Utility — Shrink Bounds
# ============================================================

def shrink_bounds(bounds: CourtBounds, margin: float = 0.12) -> CourtBounds:
    """
    Shrink the court bounds by a margin to create an 'active zone' for the ball.

    margin:
        Fraction of width/height to shrink from each side.
    """
    width = bounds.right - bounds.left
    height = bounds.bottom - bounds.top
    dx = width * margin
    dy = height * margin
    return CourtBounds(
        left=bounds.left + dx,
        right=bounds.right - dx,
        top=bounds.top + dy,
        bottom=bounds.bottom - dy,
    )


# ============================================================
# SECTION: Dynamic Rally End Detector
# ============================================================

class RallyEndDetector:
    """
    Dynamic Rally End Detector (curvature + motion + debounce + player reset).

    Detection signals:
    • Ball leaving active zone
    • Ball disappearing / too slow for too long
    • Ball becoming stationary
    • Curvature collapse (flat trajectory)
    • Curvature spike (impact)
    • Players retreating to their baselines (orientation-aware)

    Dynamic orientation:
    • RallyStartDetector should call `configure_baselines_from_serve(...)`
      once per rally, based on the first serve direction.
    • This allows the same detector to work on any pickleball court orientation
      (vertical or horizontal, rotated, etc.).
    """

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def __init__(
        self,
        court_bounds: CourtBounds,
        fps: float,
        curvature_dead_threshold: float = 0.0005,
        curvature_spike_threshold: float = 0.25,
        missing_frames_limit: int = 10,
        stop_speed_threshold: float = 0.3,
        min_stop_frames: int = 6,
        min_rally_frames: int = 12,
        curvature_flat_frames: int = 6,
        player_reset_frames: int = 6,
        player_baseline_tolerance: float = 0.12,
    ):
        """
        Initialize the rally end detector.

        player_reset_frames:
            Number of consecutive frames where players must be near their baselines
            to trigger a player-based rally end.

        player_baseline_tolerance:
            Fraction of court dimension around the baseline coordinate considered
            "close enough" to count as being at the baseline.
        """
        self.court_bounds = court_bounds
        self.active_zone = shrink_bounds(court_bounds, margin=0.10)
        self.fps = fps

        # thresholds
        self.curvature_dead_threshold = curvature_dead_threshold
        self.curvature_spike_threshold = curvature_spike_threshold
        self.missing_frames_limit = missing_frames_limit
        self.stop_speed_threshold = stop_speed_threshold
        self.min_stop_frames = min_stop_frames
        self.min_rally_frames = min_rally_frames
        self.curvature_flat_frames = curvature_flat_frames
        self.player_reset_frames = player_reset_frames
        self.player_baseline_tolerance = player_baseline_tolerance

        # rally state
        self.in_rally = False
        self.start_frame: Optional[int] = None
        self.prev1: Optional[BallState] = None
        self.prev2: Optional[BallState] = None
        self.missing_frames = 0
        self.stationary_count = 0
        self.flat_count = 0
        self.player_reset_count = 0

        # dynamic orientation state
        # axis: "x" or "y" (which axis defines baselines)
        # left_baseline_coord / right_baseline_coord: coordinate along that axis
        self.baseline_axis: str = "y"
        self.left_baseline_coord: Optional[float] = None
        self.right_baseline_coord: Optional[float] = None

    # --------------------------------------------------------
    # Orientation Configuration (called by RallyStartDetector)
    # --------------------------------------------------------

    def configure_baselines_from_serve(
        self,
        serve_ball_before: BallState,
        serve_ball_after: BallState,
        left_player_pos: tuple[float, float],
        right_player_pos: tuple[float, float],
    ) -> None:
        """
        Configure baseline orientation dynamically from the first serve.

        This is intended to be called by RallyStartDetector once per rally.

        Logic:
        • Determine dominant serve direction (horizontal vs vertical) from ball motion.
        • Choose baseline axis accordingly ("x" or "y").
        • Assign left/right baselines based on which player is closer to the
          serve origin along that axis.

        Parameters:
            serve_ball_before:
                Ball state just before the serve motion.

            serve_ball_after:
                Ball state just after the serve motion.

            left_player_pos:
                (x, y) of the left player at serve time.

            right_player_pos:
                (x, y) of the right player at serve time.
        """
        dx = serve_ball_after.x - serve_ball_before.x
        dy = serve_ball_after.y - serve_ball_before.y

        # Decide axis: whichever has larger absolute motion
        if abs(dx) >= abs(dy):
            axis = "x"
        else:
            axis = "y"

        self.baseline_axis = axis

        # Project players onto that axis
        left_coord = left_player_pos[0] if axis == "x" else left_player_pos[1]
        right_coord = right_player_pos[0] if axis == "x" else right_player_pos[1]

        # Serve direction sign along axis
        serve_delta = dx if axis == "x" else dy

        # The serving player is the one closer to the serve origin along axis.
        # We don't strictly need to know which one served; we just need
        # consistent baselines for left/right players.
        # We define:
        # • left player's baseline at their serve-time coordinate
        # • right player's baseline at their serve-time coordinate
        self.left_baseline_coord = left_coord
        self.right_baseline_coord = right_coord

        # Optional: if you want to bias baselines slightly away from center
        # based on serve direction, you could adjust here using serve_delta.

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _speed(self, b: BallState) -> float:
        """Compute ball speed magnitude."""
        return math.sqrt(b.vx**2 + b.vy**2)

    def _in_active_zone(self, b: BallState) -> bool:
        """Check if ball is inside the active zone."""
        a = self.active_zone
        return a.left < b.x < a.right and a.top < b.y < a.bottom

    def _curvature(self, p2: BallState, p1: BallState, p0: BallState) -> float:
        """
        Compute curvature using cross product of velocity vectors.

        Uses three consecutive ball positions.
        """
        ax, ay = p1.x - p0.x, p1.y - p0.y
        bx, by = p2.x - p1.x, p2.y - p1.y
        cross = abs(ax * by - ay * bx)
        dist = math.sqrt((p2.x - p0.x) ** 2 + (p2.y - p0.y) ** 2)
        if dist < 1e-5:
            return 0.0
        return cross / (dist ** 3)

    # --------------------------------------------------------
    # Player Reset / Baseline Logic
    # --------------------------------------------------------

    def _players_reset_to_baselines(self, players: PlayerState) -> bool:
        """
        Check if both players are near their baselines and effectively "reset".

        Uses:
        • Dynamic baseline axis ("x" or "y")
        • Baseline coordinates configured from serve
        • Player positions and velocities
        """
        if self.left_baseline_coord is None or self.right_baseline_coord is None:
            return False

        axis = self.baseline_axis

        if axis == "x":
            left_pos = players.left_player_x
            right_pos = players.right_player_x
            left_vel = players.left_vx
            right_vel = players.right_vx
            court_min = self.court_bounds.left
            court_max = self.court_bounds.right
        else:
            left_pos = players.left_player_y
            right_pos = players.right_player_y
            left_vel = players.left_vy
            right_vel = players.right_vy
            court_min = self.court_bounds.top
            court_max = self.court_bounds.bottom

        # Tolerance window around baselines
        court_size = court_max - court_min
        tol = court_size * self.player_baseline_tolerance

        def near_baseline(pos: float, baseline: float) -> bool:
            return abs(pos - baseline) <= tol

        left_near = near_baseline(left_pos, self.left_baseline_coord)
        right_near = near_baseline(right_pos, self.right_baseline_coord)

        # Optional: require that players are not moving much (settled)
        left_slow = abs(left_vel) < 0.003
        right_slow = abs(right_vel) < 0.003

        # Require in-court as well
        if not (players.left_player_in_court and players.right_player_in_court):
            return False

        return left_near and right_near and left_slow and right_slow

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def start_rally(self, frame_idx: int):
        """
        Mark the start of a rally.

        RallyStartDetector should call this when a new rally begins.
        """
        self.in_rally = True
        self.start_frame = frame_idx
        self.prev1 = None
        self.prev2 = None
        self.missing_frames = 0
        self.stationary_count = 0
        self.flat_count = 0
        self.player_reset_count = 0

    def reset_rally_state(self):
        """Reset all internal rally state."""
        self.in_rally = False
        self.start_frame = None
        self.prev1 = None
        self.prev2 = None
        self.missing_frames = 0
        self.stationary_count = 0
        self.flat_count = 0
        self.player_reset_count = 0

    # --------------------------------------------------------
    # Detect
    # --------------------------------------------------------

    def detect(self, ball: BallState, players: PlayerState) -> Optional[Dict[str, Any]]:
        """
        Main detection entry point.

        Returns:
            A rally_end event dict or None if rally continues.
        """

        # Ignore until a rally has started
        if not self.in_rally or self.start_frame is None:
            self.prev2 = self.prev1
            self.prev1 = ball
            return None

        # Enforce minimum rally duration
        if ball.frame_idx - self.start_frame < self.min_rally_frames:
            self.prev2 = self.prev1
            self.prev1 = ball
            return None

        speed = self._speed(ball)

        # Track missing / very slow frames
        if speed < 0.1:
            self.missing_frames += 1
        else:
            self.missing_frames = 0

        if self.missing_frames >= self.missing_frames_limit:
            return self._end(ball, "ball_missing")

        # Ball leaves active zone
        if not self._in_active_zone(ball):
            return self._end(ball, "ball_left_zone")

        # Stationary debounce
        if speed < self.stop_speed_threshold:
            self.stationary_count += 1
            if self.stationary_count >= self.min_stop_frames:
                return self._end(ball, "ball_stationary")
        else:
            self.stationary_count = 0

        # Need 3 points for curvature
        if self.prev1 is None or self.prev2 is None:
            self.prev2 = self.prev1
            self.prev1 = ball
            return None

        curvature = self._curvature(ball, self.prev1, self.prev2)

        # Curvature spike (impact)
        if curvature >= self.curvature_spike_threshold:
            return self._end(ball, "curvature_spike")

        # Curvature flat debounce
        if curvature <= self.curvature_dead_threshold:
            self.flat_count += 1
            if self.flat_count >= self.curvature_flat_frames:
                return self._end(ball, "curvature_flat")
        else:
            self.flat_count = 0

        # ----------------------------------------------------
        # Player reset to baselines (orientation-aware)
        # ----------------------------------------------------
        if self._players_reset_to_baselines(players):
            self.player_reset_count += 1
            if self.player_reset_count >= self.player_reset_frames:
                return self._end(ball, "players_reset_baselines")
        else:
            self.player_reset_count = 0

        # Update history
        self.prev2 = self.prev1
        self.prev1 = ball
        return None

    # --------------------------------------------------------
    # Event builder
    # --------------------------------------------------------

    def _end(self, ball: BallState, reason: str) -> Dict[str, Any]:
        """
        Build a rally_end event and reset internal state.
        """
        timestamp = ball.frame_idx / self.fps
        event = {
            "type": "rally_end",
            "frame": ball.frame_idx,
            "time": timestamp,
            "ball_pos": (ball.x, ball.y),
            "reason": reason,
        }
        self.reset_rally_state()
        return event

    # --------------------------------------------------------
    # Debug overlay
    # --------------------------------------------------------

    def draw_debug_overlay(self, frame, event, color=(0, 0, 255)):
        """
        Draw debug overlay for a rally_end event on a video frame.
        """
        if event is None:
            return frame
        import cv2
        x, y = map(int, event["ball_pos"])
        text = f"RALLY END f={event['frame']} ({event['reason']})"
        cv2.circle(frame, (x, y), 10, color, 2)
        cv2.putText(
            frame,
            text,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        return frame
