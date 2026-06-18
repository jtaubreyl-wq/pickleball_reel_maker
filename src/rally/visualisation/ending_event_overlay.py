# FILE: src/rally/visualization/ending_event_overlay.py

# ============================================================
# SECTION: Imports
# ============================================================

import cv2
from typing import Dict, List, Optional, Tuple


# ============================================================
# SECTION: Ending Event Debug Overlay (Story 6.2)
# ============================================================

class EndingEventDebugOverlay:
    """
    Story 6.2 — Ending Event Debug Overlay

    Purpose:
        Visualize the rally-ending event on the frame:
            • Winner (Team A / Team B)
            • Ending event type:
                - winner_shot
                - forced_error
                - unforced_error
                - unknown
            • Final ball position
            • Optional trajectory path

    Why this matters:
        If the ending event is missing or incorrect, the highlight scoring
        engine cannot assign proper bonuses. This often results in
        highlight_score = 0 → highlight selector rejects the rally →
        reel assembler outputs a 1‑second fallback clip.

        This overlay helps you *see* whether the ending event was detected.
    """

    # --------------------------------------------------------
    # SECTION: Initialization
    # --------------------------------------------------------

    def __init__(self):
        # Colors for different ending events
        self.colors = {
            "winner_shot": (0, 255, 0),      # green
            "forced_error": (0, 165, 255),   # orange
            "unforced_error": (0, 0, 255),   # red
            "unknown": (255, 255, 255),      # white
        }

        self.text_color = (255, 255, 255)
        self.path_color = (255, 0, 0)        # blue for trajectory path
        self.warn_color = (0, 255, 255)      # yellow for warnings

    # --------------------------------------------------------
    # SECTION: Internal Helpers
    # --------------------------------------------------------

    def _draw_trajectory_path(self, frame, trajectory: List[Dict]):
        """
        Draw the full trajectory path for debugging.
        """
        pts = [(int(p["x"]), int(p["y"])) for p in trajectory if "x" in p and "y" in p]

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], self.path_color, 2)

    # --------------------------------------------------------
    # SECTION: Public API — Draw Overlay
    # --------------------------------------------------------

    def draw(
        self,
        frame,
        trajectory: List[Dict],
        result: Dict,
        rally_id: Optional[int] = None,
        draw_path: bool = True,
    ):
        """
        Draw ending event overlay.

        Parameters:
            frame: np.ndarray — video frame
            trajectory: list of dicts with keys {frame, x, y}
            result: dict with keys:
                {
                    "winner": str | None,
                    "ending_event": str | None
                }
            rally_id: optional rally ID for labeling
            draw_path: whether to draw the trajectory path
        """

        # ----------------------------------------------------
        # Validate trajectory
        # ----------------------------------------------------
        if not trajectory:
            cv2.putText(
                frame,
                "ENDING EVENT: NO TRAJECTORY",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.warn_color,
                2,
            )
            return frame

        # ----------------------------------------------------
        # Draw trajectory path (optional)
        # ----------------------------------------------------
        if draw_path:
            self._draw_trajectory_path(frame, trajectory)

        # ----------------------------------------------------
        # Final ball position
        # ----------------------------------------------------
        final_point = trajectory[-1]
        x, y = int(final_point["x"]), int(final_point["y"])

        # ----------------------------------------------------
        # Extract ending event info
        # ----------------------------------------------------
        event = result.get("ending_event", "unknown")
        winner = result.get("winner", "Unknown")

        color = self.colors.get(event, self.colors["unknown"])

        # ----------------------------------------------------
        # Draw final ball marker
        # ----------------------------------------------------
        cv2.circle(frame, (x, y), 14, color, -1)

        # ----------------------------------------------------
        # Draw label text
        # ----------------------------------------------------
        label = f"{winner} — {event}"
        cv2.putText(
            frame,
            label,
            (x + 20, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        # ----------------------------------------------------
        # Draw rally ID (if provided)
        # ----------------------------------------------------
        if rally_id is not None:
            cv2.putText(
                frame,
                f"Rally {rally_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.text_color,
                2,
            )

        return frame
