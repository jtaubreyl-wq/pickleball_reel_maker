# FILE: src/rally/rally_start_detector.py
"""
Dynamic Rally Start Detector

This version adds:
• Serve-direction analysis
• Automatic court-orientation detection (vertical or horizontal)
• Automatic baseline coordinate detection for both players
• Integration hook for RallyEndDetector.configure_baselines_from_serve()

A rally starts when:
• Ball speed exceeds threshold, OR
• Ball accelerates sharply, OR
• Ball changes direction sharply, OR
• Ball reappears after missing
AND
• Players are in a ready state
AND
• Ball is inside the active zone
"""

# ============================================================
# SECTION: Imports & Data Models
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math


# ------------------------------------------------------------
# Ball & Player State Models
# ------------------------------------------------------------

@dataclass
class BallState:
    frame_idx: int
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class PlayerState:
    """
    Updated PlayerState from PlayerFilter/PlayerStateBuilder.
    """
    left_player_in_court: bool
    right_player_in_court: bool
    left_player_active: bool
    right_player_active: bool

    left_player_x: float
    left_player_y: float
    right_player_x: float
    right_player_y: float

    left_vx: float
    left_vy: float
    right_vx: float
    right_vy: float

    @property
    def ready_state(self) -> bool:
        return (
            self.left_player_in_court
            and self.right_player_in_court
            and (self.left_player_active or self.right_player_active)
        )


@dataclass
class CourtBounds:
    left: float
    right: float
    top: float
    bottom: float


# ============================================================
# SECTION: Utility — Shrink Court Bounds
# ============================================================

def shrink_bounds(bounds: CourtBounds, margin: float = 0.10) -> CourtBounds:
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
# SECTION: Dynamic Rally Start Detector
# ============================================================

class RallyStartDetector:
    """
    Dynamic Rally Start Detector.

    New:
    • Detects serve direction
    • Determines court orientation (vertical/horizontal)
    • Determines baseline coordinates for both players
    • Passes orientation to RallyEndDetector
    """

    def __init__(
        self,
        court_bounds: CourtBounds,
        fps: float,
        velocity_threshold: float = 1.0,
        accel_threshold: float = 1.5,
        direction_change_deg: float = 25.0,
        min_sustain_frames: int = 1,
        reappear_frames: int = 5,
    ):
        self.court_bounds = court_bounds
        self.active_zone = shrink_bounds(court_bounds, margin=0.10)
        self.fps = fps

        self.velocity_threshold = velocity_threshold
        self.accel_threshold = accel_threshold
        self.direction_change_deg = direction_change_deg
        self.sustain_frames = max(1, int(min_sustain_frames))
        self.reappear_frames = reappear_frames

        self.in_rally = False
        self.prev_ball: Optional[BallState] = None
        self.missing_frames = 0
        self.recent_fast_frames: List[int] = []

        # For serve-direction analysis
        self.serve_ball_before: Optional[BallState] = None
        self.serve_ball_after: Optional[BallState] = None
        self.serve_captured = False

        # RallyEndDetector reference (optional)
        self.rally_end_detector = None

    # --------------------------------------------------------
    # External hookup
    # --------------------------------------------------------

    def attach_rally_end_detector(self, detector):
        """
        Attach RallyEndDetector so we can configure baselines dynamically.
        """
        self.rally_end_detector = detector

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _speed(self, b: BallState) -> float:
        return math.sqrt(b.vx**2 + b.vy**2)

    def _direction_change(self, b1: BallState, b2: BallState) -> float:
        v1 = (b1.vx, b1.vy)
        v2 = (b2.vx, b2.vy)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 < 1e-3 or mag2 < 1e-3:
            return 0.0
        cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return abs(math.degrees(math.acos(cosang)))

    def _in_active_zone(self, b: BallState) -> bool:
        a = self.active_zone
        return a.left < b.x < a.right and a.top < b.y < a.bottom

    # --------------------------------------------------------
    # Serve Direction Analysis
    # --------------------------------------------------------

    def _capture_serve_motion(self, ball: BallState):
        """
        Capture the first two meaningful ball states to determine serve direction.
        """
        if self.serve_ball_before is None:
            self.serve_ball_before = ball
            return

        if self.serve_ball_after is None:
            # Only capture if ball actually moved
            if self._speed(ball) > 0.2:
                self.serve_ball_after = ball
                self.serve_captured = True

    def _configure_baselines(self, players: PlayerState):
        """
        Once serve direction is known, configure baselines in RallyEndDetector.
        """
        if not self.serve_captured:
            return

        if self.rally_end_detector is None:
            return

        self.rally_end_detector.configure_baselines_from_serve(
            serve_ball_before=self.serve_ball_before,
            serve_ball_after=self.serve_ball_after,
            left_player_pos=(players.left_player_x, players.left_player_y),
            right_player_pos=(players.right_player_x, players.right_player_y),
        )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset_rally_state(self):
        self.in_rally = False
        self.prev_ball = None
        self.missing_frames = 0
        self.recent_fast_frames.clear()

        # Reset serve capture
        self.serve_ball_before = None
        self.serve_ball_after = None
        self.serve_captured = False

    # --------------------------------------------------------
    # Detect
    # --------------------------------------------------------

    def detect(self, ball: BallState, players: PlayerState) -> Optional[Dict[str, Any]]:

        # Capture serve motion BEFORE rally starts
        if not self.in_rally:
            self._capture_serve_motion(ball)

        # Already in rally
        if self.in_rally:
            self.prev_ball = ball
            return None

        # Must be in active zone
        if not self._in_active_zone(ball):
            self.prev_ball = ball
            return None

        # Players must be ready
        if not players.ready_state:
            self.prev_ball = ball
            return None

        speed = self._speed(ball)

        # Missing frames logic
        if speed < 0.1:
            self.missing_frames += 1
        else:
            self.missing_frames = 0

        # Condition 1 — reappearance
        if self.missing_frames >= self.reappear_frames and speed > 0.1:
            self.in_rally = True
            self._configure_baselines(players)
            return self._make_event(ball, speed)

        # Condition 2 — sustained fast movement
        if speed >= self.velocity_threshold:
            self.recent_fast_frames.append(ball.frame_idx)
            if len(self.recent_fast_frames) >= self.sustain_frames:
                self.in_rally = True
                self._configure_baselines(players)
                return self._make_event(ball, speed)
        else:
            self.recent_fast_frames.clear()

        # Condition 3 — acceleration or direction change
        if self.prev_ball is not None:
            prev_speed = self._speed(self.prev_ball)
            accel = speed - prev_speed

            if accel >= self.accel_threshold:
                self.in_rally = True
                self._configure_baselines(players)
                return self._make_event(ball, speed)

            direction_change = self._direction_change(self.prev_ball, ball)
            if direction_change >= self.direction_change_deg:
                self.in_rally = True
                self._configure_baselines(players)
                return self._make_event(ball, speed)

        self.prev_ball = ball
        return None

    # --------------------------------------------------------
    # Event builder
    # --------------------------------------------------------

    def _make_event(self, ball: BallState, speed: float) -> Dict[str, Any]:
        timestamp = ball.frame_idx / self.fps
        return {
            "type": "rally_start",
            "frame": ball.frame_idx,
            "time": timestamp,
            "ball_pos": (ball.x, ball.y),
            "ball_speed": speed,
        }

    # --------------------------------------------------------
    # Debug overlay
    # --------------------------------------------------------

    def draw_debug_overlay(self, frame, event, color=(0, 255, 0)):
        if event is None:
            return frame
        import cv2
        x, y = map(int, event["ball_pos"])
        text = f"RALLY START f={event['frame']} v={event['ball_speed']:.1f}"
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
