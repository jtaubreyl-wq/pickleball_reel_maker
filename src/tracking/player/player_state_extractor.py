# ============================================================
# FILE: src/tracking/player/player_state_extractor.py
# ============================================================
"""
PlayerStateExtractor

Converts tracked player positions into a fully-populated PlayerState object.

Inputs:
    - tracked_players: output from PlayerTrackerSORT
    - ball_state: BallState object
    - court_bounds: CourtBounds object

Outputs:
    - PlayerState with:
        left_player_x, left_player_y
        right_player_x, right_player_y
        left_vx, left_vy
        right_vx, right_vy
        left_player_in_court
        right_player_in_court
        left_player_active
        right_player_active
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from src.rally.detection.rally_start_detector import PlayerState, CourtBounds
from src.rally.detection.rally_start_detector import BallState


class PlayerStateExtractor:
    """
    Converts tracked players into a PlayerState object.
    """

    def __init__(self, court_bounds: CourtBounds):
        self.court_bounds = court_bounds

        # Store previous positions for velocity estimation
        self.prev_positions = {
            "left": None,
            "right": None,
        }

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def build_player_state(
        self,
        tracked_players: List[Dict],
        ball_state: BallState,
    ) -> PlayerState:
        """
        Build a PlayerState object from tracked players + ball state.
        """

        # -----------------------------------------------------
        # 1. Identify left vs right player
        # -----------------------------------------------------
        left, right = self._assign_left_right(tracked_players)

        # -----------------------------------------------------
        # 2. Compute velocities
        # -----------------------------------------------------
        left_vx, left_vy = self._compute_velocity("left", left)
        right_vx, right_vy = self._compute_velocity("right", right)

        # -----------------------------------------------------
        # 3. Determine in-court flags
        # -----------------------------------------------------
        left_in = self._is_in_court(left)
        right_in = self._is_in_court(right)

        # -----------------------------------------------------
        # 4. Determine active player (closest to ball)
        # -----------------------------------------------------
        left_active, right_active = self._compute_active(left, right, ball_state)

        # -----------------------------------------------------
        # 5. Build PlayerState
        # -----------------------------------------------------
        return PlayerState(
            left_player_x=left["cx"],
            left_player_y=left["cy"],
            right_player_x=right["cx"],
            right_player_y=right["cy"],
            left_vx=left_vx,
            left_vy=left_vy,
            right_vx=right_vx,
            right_vy=right_vy,
            left_player_in_court=left_in,
            right_player_in_court=right_in,
            left_player_active=left_active,
            right_player_active=right_active,
        )

    # ---------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------
    def _assign_left_right(self, players: List[Dict]):
        """
        Assign players to left/right based on x-position.
        """
        if len(players) == 0:
            # No detections → return dummy players
            return (
                {"cx": 0.0, "cy": 0.0},
                {"cx": 0.0, "cy": 0.0},
            )

        if len(players) == 1:
            p = players[0]
            # Assume single player is on the left
            return (
                {"cx": p["cx"], "cy": p["cy"]},
                {"cx": p["cx"], "cy": p["cy"]},
            )

        # Two or more players → pick two closest to center
        players_sorted = sorted(players, key=lambda p: p["cx"])
        left = players_sorted[0]
        right = players_sorted[-1]

        return left, right

    def _compute_velocity(self, side: str, player: Dict):
        """
        Compute velocity using previous frame position.
        """
        prev = self.prev_positions.get(side)

        if prev is None:
            self.prev_positions[side] = (player["cx"], player["cy"])
            return 0.0, 0.0

        vx = player["cx"] - prev[0]
        vy = player["cy"] - prev[1]

        self.prev_positions[side] = (player["cx"], player["cy"])
        return vx, vy

    def _is_in_court(self, player: Dict) -> bool:
        """
        Check if player is inside court bounds.
        """
        x, y = player["cx"], player["cy"]
        cb = self.court_bounds

        return (
            cb.left <= x <= cb.right and
            cb.top <= y <= cb.bottom
        )

    def _compute_active(
        self,
        left: Dict,
        right: Dict,
        ball_state: BallState,
    ):
        """
        Determine which player is 'active' based on proximity to ball.
        """
        bx, by = ball_state.x, ball_state.y

        left_dist = np.hypot(left["cx"] - bx, left["cy"] - by)
        right_dist = np.hypot(right["cx"] - bx, right["cy"] - by)

        if left_dist < right_dist:
            return True, False
        else:
            return False, True
