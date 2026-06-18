# FILE: src/rally/detection/winner_forced_error_detector.py

# ============================================================
# SECTION: Imports & Data Models
# ============================================================

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


# ============================================================
# SECTION: Output Container
# ============================================================

@dataclass
class WinnerForcedErrorResult:
    """
    Output container for Story 6.2.

    winner:
        "Team A", "Team B", or None

    ending_event:
        "winner_shot"     → attacker hit an unreturnable shot
        "forced_error"    → defender reached but failed
        "unforced_error"  → ball went out or into net
        "unknown"         → insufficient data
    """
    winner: Optional[str]
    ending_event: Optional[str]


# ============================================================
# SECTION: Winner / Forced Error Detector (Story 6.2)
# ============================================================

class WinnerForcedErrorDetector:
    """
    Story 6.2 — Winner / Forced Error Detection

    Purpose:
        Determine how the rally ended:
            • Winner shot
            • Forced error
            • Unforced error
            • Unknown (insufficient data)

    Inputs:
        • Final ball trajectory point
        • Last hitter ("Team A" or "Team B")
        • Player positions at rally end (from metadata engine)

    NEW (optional):
        • If players clearly reset to opposite baselines at rally end,
          ambiguous endings may be classified as "forced_error" instead of "unknown".
    """

    # --------------------------------------------------------
    # SECTION: Initialization
    # --------------------------------------------------------

    def __init__(self, court_bounds: Dict[str, float], forced_error_distance_thresh: float = 1.0):
        """
        Args:
            court_bounds:
                dict with xmin/xmax/ymin/ymax or object with left/right/top/bottom

            forced_error_distance_thresh:
                Maximum distance (pixels) the defender can be from the ball
                to be considered "reachable".
        """
        self.court = self._normalize_bounds(court_bounds)
        self.forced_thresh = forced_error_distance_thresh

    # --------------------------------------------------------
    # SECTION: Public API — Analyze Rally Outcome
    # --------------------------------------------------------

    def analyze(
        self,
        trajectory: List[Any],
        last_hitter: Optional[str],
        player_positions: Dict[str, Any],
    ) -> WinnerForcedErrorResult:
        """
        Determine rally outcome based on final ball position and defender reachability.

        Args:
            trajectory:
                List of dicts or tuples containing ball positions.

            last_hitter:
                "Team A" or "Team B"

            player_positions:
                {"Team A": (x,y), "Team B": (x,y)}
                OR
                {"Team A": {"cx":..., "cy":...}, ...}  (from metadata engine)

        Returns:
            WinnerForcedErrorResult
        """

        # ----------------------------------------------------
        # Safety Checks
        # ----------------------------------------------------

        if not trajectory:
            return WinnerForcedErrorResult(None, "unknown")

        if last_hitter is None:
            return WinnerForcedErrorResult(None, "unknown")

        # Normalize player positions
        player_positions = self._normalize_player_positions(player_positions)

        # ----------------------------------------------------
        # Extract final ball position
        # ----------------------------------------------------

        final = trajectory[-1]

        if isinstance(final, dict):
            x, y = final.get("x"), final.get("y")
        else:
            x, y = final

        if x is None or y is None:
            return WinnerForcedErrorResult(None, "unknown")

        ball_xy = np.array([x, y], dtype=float)

        # ----------------------------------------------------
        # Condition 1 — Ball Out of Bounds → Unforced Error
        # ----------------------------------------------------

        if self._is_out(ball_xy):
            winner = self._opposite_team(last_hitter)
            return WinnerForcedErrorResult(winner, "unforced_error")

        # ----------------------------------------------------
        # Condition 2 — Defender Reachability
        # ----------------------------------------------------

        defender = self._opposite_team(last_hitter)

        if defender not in player_positions:
            # Try baseline reset fallback
            if self._players_reset_to_baseline(player_positions):
                return WinnerForcedErrorResult(last_hitter, "forced_error")
            return WinnerForcedErrorResult(None, "unknown")

        defender_xy = np.array(player_positions[defender], dtype=float)

        # Distance between defender and final ball position
        dist = np.linalg.norm(ball_xy - defender_xy)

        # ----------------------------------------------------
        # Pickleball Logic:
        # ----------------------------------------------------
        # If defender CANNOT reach → winner shot
        # If defender CAN reach → forced error
        # ----------------------------------------------------

        if dist > self.forced_thresh:
            return WinnerForcedErrorResult(last_hitter, "winner_shot")
        else:
            return WinnerForcedErrorResult(last_hitter, "forced_error")

    # --------------------------------------------------------
    # SECTION: Internal Helpers
    # --------------------------------------------------------

    def _normalize_bounds(self, b: Any) -> Dict[str, float]:
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

    def _normalize_player_positions(self, positions: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Convert any format into:
            {"Team A": (x,y), "Team B": (x,y)}

        Supports:
            • {"Team A": {"cx":..., "cy":...}}
            • {"Team A": (x,y)}
        """
        out: Dict[str, Tuple[float, float]] = {}
        for k, v in positions.items():
            if isinstance(v, dict):
                # Metadata engine format
                out[k] = (v.get("cx"), v.get("cy"))
            else:
                out[k] = tuple(v)
        return out

    def _is_out(self, xy: np.ndarray) -> bool:
        """Check if ball is outside court bounds."""
        x, y = xy
        return (
            x < self.court["xmin"]
            or x > self.court["xmax"]
            or y < self.court["ymin"]
            or y > self.court["ymax"]
        )

    def _opposite_team(self, team: Optional[str]) -> Optional[str]:
        """Return the opposing team label."""
        if team is None:
            return None
        t = team.lower()
        if t in ["team a", "a", "left", "p1"]:
            return "Team B"
        return "Team A"

    # --------------------------------------------------------
    # SECTION: Baseline Reset Logic (Optional Enhancement)
    # --------------------------------------------------------

    def _players_reset_to_baseline(self, positions: Dict[str, Tuple[float, float]]) -> bool:
        """
        Determine whether players ended the rally on opposite sides of the court.

        This is a simple heuristic:
            • If we have at least 2 players with y-positions,
              and their vertical spread is large,
              we assume they reset to opposite baselines.

        This is used ONLY as a fallback when defender position is missing.
        """
        if not positions or len(positions) < 2:
            return False

        ys = [p[1] for p in positions.values() if p[1] is not None]
        if len(ys) < 2:
            return False

        spread = max(ys) - min(ys)

        # Threshold is heuristic; matches metadata engine
        return spread > 100.0
