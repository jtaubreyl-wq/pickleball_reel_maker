# FILE: src/detection/player_filter.py
"""
Player Filter + Player State Builder

This upgraded version:
• Filters YOLO person detections
• Normalizes bounding boxes
• Computes center points
• Assigns player identity (left vs right)
• Tracks player motion (velocity)
• Estimates facing direction
• Determines in‑court status
• Determines active status
• Produces PlayerState objects for rally detectors
"""

# ============================================================
# SECTION: Imports & Configuration
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


# ============================================================
# SECTION: Player Filter Configuration
# ============================================================

@dataclass
class PlayerFilterConfig:
    confidence_threshold: float = 0.5
    person_class_id: int = 0


# ============================================================
# SECTION: Player State Model
# ============================================================

@dataclass
class PlayerState:
    """
    Output state used by RallyStartDetector and RallyEndDetector.
    """

    # identity
    left_player_x: float
    left_player_y: float
    right_player_x: float
    right_player_y: float

    # motion
    left_vx: float
    left_vy: float
    right_vx: float
    right_vy: float

    # status
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


# ============================================================
# SECTION: Player Filter
# ============================================================

class PlayerFilter:
    """
    Filters YOLO detections and builds PlayerState.

    New capabilities:
    • Assign left/right player identity
    • Track motion (velocity)
    • Determine in‑court status
    • Determine active status
    """

    def __init__(self, config: PlayerFilterConfig = PlayerFilterConfig()):
        self.config = config

        # history for velocity estimation
        self.prev_left = None
        self.prev_right = None

    # --------------------------------------------------------
    # Normalization Helpers
    # --------------------------------------------------------

    def _normalize_detection(self, det: Dict[str, Any], frame_w: int, frame_h: int) -> Dict[str, Any]:
        x1 = det["x1"] / frame_w
        y1 = det["y1"] / frame_h
        x2 = det["x2"] / frame_w
        y2 = det["y2"] / frame_h

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        return {
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "conf": det["conf"],
            "class_id": det["class_id"],
        }

    # --------------------------------------------------------
    # Public API — Filter Detections
    # --------------------------------------------------------

    def filter(self, detections, frame_w: int, frame_h: int) -> List[Dict[str, Any]]:
        if len(detections) == 0:
            return []

        # Convert sv.Detections to dict list if needed
        if hasattr(detections, "xyxy"):
            det_list = []
            for i in range(len(detections)):
                det_list.append({
                    "x1": float(detections.xyxy[i][0]),
                    "y1": float(detections.xyxy[i][1]),
                    "x2": float(detections.xyxy[i][2]),
                    "y2": float(detections.xyxy[i][3]),
                    "conf": float(detections.confidence[i]),
                    "class_id": int(detections.class_id[i]),
                })
            detections = det_list

        # Filter by class + confidence
        filtered = []
        for det in detections:
            if det["class_id"] != self.config.person_class_id:
                continue
            if det["conf"] < self.config.confidence_threshold:
                continue
            filtered.append(self._normalize_detection(det, frame_w, frame_h))

        return filtered

    # --------------------------------------------------------
    # Player State Builder
    # --------------------------------------------------------

    def build_player_state(
        self,
        detections: List[Dict[str, Any]],
        court_bounds,
    ) -> Optional[PlayerState]:
        """
        Build PlayerState from filtered detections.
        """

        if len(detections) < 2:
            return None

        # Sort by x-position → left player, right player
        detections = sorted(detections, key=lambda d: d["cx"])
        left = detections[0]
        right = detections[1]

        # Compute velocities
        left_vx = left["cx"] - self.prev_left["cx"] if self.prev_left else 0
        left_vy = left["cy"] - self.prev_left["cy"] if self.prev_left else 0
        right_vx = right["cx"] - self.prev_right["cx"] if self.prev_right else 0
        right_vy = right["cy"] - self.prev_right["cy"] if self.prev_right else 0

        # Update history
        self.prev_left = left
        self.prev_right = right

        # In-court detection
        def in_court(px, py):
            return (
                court_bounds.left < px < court_bounds.right
                and court_bounds.top < py < court_bounds.bottom
            )

        left_in = in_court(left["cx"], left["cy"])
        right_in = in_court(right["cx"], right["cy"])

        # Active = moving
        left_active = abs(left_vx) + abs(left_vy) > 0.002
        right_active = abs(right_vx) + abs(right_vy) > 0.002

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

    # --------------------------------------------------------
    # Debug Logging
    # --------------------------------------------------------

    def summarize(self, frame_idx: int, detections: List[Dict[str, Any]]) -> int:
        count = len(detections)
        print(f"[PlayerFilter] Frame {frame_idx}: {count} player detections")
        return count
