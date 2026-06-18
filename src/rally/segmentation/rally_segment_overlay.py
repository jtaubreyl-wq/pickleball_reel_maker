# FILE: src/rally/visualization/rally_segment_overlay.py

# ============================================================
# SECTION: Imports
# ============================================================

import cv2
from typing import Dict, List, Optional, Tuple


# ============================================================
# SECTION: Rally Segment Debug Overlay (Story 5.3)
# ============================================================

class RallySegmentDebugOverlay:
    """
    Story 5.3 — Rally Segment Debug Overlay

    Purpose:
        Draw visual markers on video frames to show:
            • Rally start location
            • Rally end location
            • Rally ID
            • Rally duration
            • Optional trajectory path
            • Optional ball speed visualization

    Why this matters:
        If rally segmentation is incorrect (e.g., start == end, missing trajectory),
        the highlight pipeline collapses into a 1‑second fallback clip.
        This overlay helps you *see* segmentation issues immediately.
    """

    # --------------------------------------------------------
    # SECTION: Initialization
    # --------------------------------------------------------

    def __init__(self):
        # Colors for drawing
        self.start_color = (0, 255, 0)     # Green
        self.end_color = (0, 0, 255)       # Red
        self.text_color = (255, 255, 255)  # White
        self.warn_color = (0, 255, 255)    # Yellow
        self.path_color = (255, 0, 0)      # Blue

    # --------------------------------------------------------
    # SECTION: Internal Helpers
    # --------------------------------------------------------

    def _find_point(self, trajectory: List[Dict], frame_idx: int) -> Optional[Tuple[int, int]]:
        """
        Find the (x, y) position of the ball at a specific frame.

        trajectory: list of dicts with keys:
            { "frame": int, "x": float, "y": float }

        Returns:
            (x, y) as ints, or None if not found.
        """
        for p in trajectory:
            if p.get("frame") == frame_idx:
                return int(p["x"]), int(p["y"])
        return None

    def _draw_trajectory_path(self, frame, trajectory: List[Dict], start_frame: int, end_frame: int):
        """
        Draw the ball trajectory path between start and end frames.
        """
        pts = [
            (int(p["x"]), int(p["y"]))
            for p in trajectory
            if start_frame <= p.get("frame", -1) <= end_frame
        ]

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], self.path_color, 2)

    # --------------------------------------------------------
    # SECTION: Public API — Draw Overlay
    # --------------------------------------------------------

    def draw(self, frame, segment: Dict, trajectory: List[Dict], draw_path: bool = True):
        """
        Draw rally start/end markers on a frame.

        segment dict contains:
            {
                "rally_id": int,
                "start_frame": int,
                "end_frame": int,
                "duration_s": float
            }

        trajectory is the full ball trajectory list.
        """

        rally_id = segment.get("rally_id", -1)
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]
        duration_s = segment.get("duration_s", 0.0)

        # --- Find ball positions ---
        start_point = self._find_point(trajectory, start_frame)
        end_point = self._find_point(trajectory, end_frame)

        # ----------------------------------------------------
        # Draw Trajectory Path (optional)
        # ----------------------------------------------------
        if draw_path:
            self._draw_trajectory_path(frame, trajectory, start_frame, end_frame)

        # ----------------------------------------------------
        # Draw Start Marker
        # ----------------------------------------------------
        if start_point:
            x, y = start_point
            cv2.circle(frame, (x, y), 10, self.start_color, -1)
            cv2.putText(
                frame,
                f"R{rally_id} START ({start_frame})",
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.start_color,
                2,
            )
        else:
            cv2.putText(
                frame,
                f"R{rally_id} START MISSING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.warn_color,
                2,
            )

        # ----------------------------------------------------
        # Draw End Marker
        # ----------------------------------------------------
        if end_point:
            x, y = end_point
            cv2.circle(frame, (x, y), 10, self.end_color, -1)
            cv2.putText(
                frame,
                f"R{rally_id} END ({end_frame})",
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.end_color,
                2,
            )
        else:
            cv2.putText(
                frame,
                f"R{rally_id} END MISSING",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.warn_color,
                2,
            )

        # ----------------------------------------------------
        # Draw Duration Label
        # ----------------------------------------------------
        label = f"Duration: {duration_s:.2f}s"
        cv2.putText(
            frame,
            label,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.text_color,
            2,
        )

        # ----------------------------------------------------
        # Draw Rally ID (top-left)
        # ----------------------------------------------------
        cv2.putText(
            frame,
            f"Rally {rally_id}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 255),
            2,
        )

        return frame
