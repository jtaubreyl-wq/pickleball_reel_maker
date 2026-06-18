# src/rally/detection/winner_forced_error_detector.py

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


@dataclass
class WinnerForcedErrorResult:
    winner: Optional[str]
    ending_event: Optional[str]   # "winner_shot", "forced_error", "unforced_error"


class WinnerForcedErrorDetector:
    """
    Story 6.2 — Winner / Forced Error Detection
    """

    def __init__(self, court_bounds, forced_error_distance_thresh: float = 1.0):
        """
        court_bounds: dict or object with xmin/xmax/ymin/ymax
        forced_error_distance_thresh: distance threshold for forced error logic
        """
        self.court = self._normalize_bounds(court_bounds)
        self.forced_thresh = forced_error_distance_thresh

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def analyze(
        self,
        trajectory: List[Any],
        last_hitter: Optional[str],
        player_positions: Dict[str, Any]
    ) -> WinnerForcedErrorResult:

        # Normalize player positions to (x, y)
        player_positions = self._normalize_player_positions(player_positions)

        # Safety: empty trajectory → cannot determine winner
        if not trajectory:
            return WinnerForcedErrorResult(
                winner=None,
                ending_event=None
            )

        # Final trajectory point may be dict or tuple
        final = trajectory[-1]
        x, y = self._extract_xy(final)
        ball_xy = np.array([x, y], dtype=float)

        # If last_hitter is missing, we cannot determine winner
        if not last_hitter:
            # But we can still detect OUT OF BOUNDS
            if self._is_out(ball_xy):
                return WinnerForcedErrorResult(
                    winner=None,
                    ending_event="unforced_error"
                )
            return WinnerForcedErrorResult(winner=None, ending_event=None)

        # 1. OUT OF BOUNDS → unforced error
        if self._is_out(ball_xy):
            winner = self._opposite_team(last_hitter)
            return WinnerForcedErrorResult(
                winner=winner,
                ending_event="unforced_error"
            )

        # 2. Defender reachability
        defender = self._opposite_team(last_hitter)

        # If defender position missing → cannot classify
        if defender not in player_positions:
            return WinnerForcedErrorResult(
                winner=None,
                ending_event=None
            )

        defender_xy = np.array(player_positions[defender], dtype=float)
        dist = np.linalg.norm(ball_xy - defender_xy)

        # Pickleball logic:
        # If defender CANNOT reach → winner shot
        # If defender CAN reach → forced error
        if dist > self.forced_thresh:
            ending_event = "winner_shot"
            winner = last_hitter
        else:
            ending_event = "forced_error"
            winner = last_hitter

        return WinnerForcedErrorResult(
            winner=winner,
            ending_event=ending_event
        )

    # ------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------
    def _normalize_bounds(self, b):
        """Accept dict or object with left/right/top/bottom."""
        if isinstance(b, dict):
            return {
                "xmin": b.get("xmin", b.get("left")),
                "xmax": b.get("xmax", b.get("right")),
                "ymin": b.get("ymin", b.get("top")),
                "ymax": b.get("ymax", b.get("bottom")),
            }
        return {
            "xmin": b.left,
            "xmax": b.right,
            "ymin": b.top,
            "ymax": b.bottom,
        }

    def _normalize_player_positions(
        self,
        positions: Dict[str, Any],
    ) -> Dict[str, Tuple[float, float]]:
        """Convert any format into {'Team A': (x,y), 'Team B': (x,y)}."""
        out = {}
        for k, v in positions.items():
            if isinstance(v, dict):
                out[k] = (float(v.get("x")), float(v.get("y")))
            else:
                x, y = v
                out[k] = (float(x), float(y))
        return out

    def _extract_xy(self, p: Any) -> Tuple[float, float]:
        """
        Extract (x, y) from a trajectory point that may be:
        - dict with "x", "y"
        - tuple/list (x, y)
        """
        if isinstance(p, dict):
            return float(p["x"]), float(p["y"])
        x, y = p
        return float(x), float(y)

    def _is_out(self, xy):
        x, y = xy
        return (
            x < self.court["xmin"] or
            x > self.court["xmax"] or
            y < self.court["ymin"] or
            y > self.court["ymax"]
        )

    def _opposite_team(self, team: Optional[str]) -> str:
        if not team:
            return "Unknown"
        t = team.lower()
        if t in ["team a", "a", "left", "p1"]:
            return "Team B"
        return "Team A"
