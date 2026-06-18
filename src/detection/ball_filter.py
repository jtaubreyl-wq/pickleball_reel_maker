# FILE: src/detection/ball_filter.py

# ============================================================
# SECTION: Imports & Constants
# ============================================================

import numpy as np
from typing import Dict, List, Any, Optional


# ============================================================
# SECTION: Ball Filter (Story 4.1 Pre‑Processing)
# ============================================================

class BallFilter:
    """
    Purpose:
        Clean YOLO ball detections before they enter the tracking pipeline.

    Why this matters:
        Bad detections → broken trajectory → rally detectors fail →
        → highlight pipeline collapses → 1‑second fallback video.

    This upgraded version:
        • Normalizes bounding boxes
        • Rejects boxes that are too large
        • Rejects boxes with wrong aspect ratio
        • Rejects low‑confidence detections
        • Computes center point (x, y)
        • Ensures consistent output format
    """

    # --------------------------------------------------------
    # SECTION: Initialization
    # --------------------------------------------------------

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        max_box_fraction: float = 0.12,
        min_confidence: float = 0.25,
        max_aspect_ratio: float = 2.0,
    ):
        """
        frame_width, frame_height:
            Dimensions of the video frame.

        max_box_fraction:
            Maximum allowed box size relative to frame width.
            (Ball should be tiny — usually < 5% of frame width.)

        min_confidence:
            Minimum YOLO confidence to accept a detection.

        max_aspect_ratio:
            Reject boxes that are too elongated (ball should be round-ish).
        """
        self.w = frame_width
        self.h = frame_height
        self.max_box_fraction = max_box_fraction
        self.min_confidence = min_confidence
        self.max_aspect_ratio = max_aspect_ratio

    # --------------------------------------------------------
    # SECTION: Normalization Helpers
    # --------------------------------------------------------

    def _normalize_box(self, det: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert absolute pixel coordinates to normalized [0,1] values.
        Also compute center point (cx, cy).
        """
        x1 = det["x1"] / self.w
        y1 = det["y1"] / self.h
        x2 = det["x2"] / self.w
        y2 = det["y2"] / self.h

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "conf": det["conf"],
        }

    # --------------------------------------------------------
    # SECTION: Validation Checks
    # --------------------------------------------------------

    def _is_valid_box(self, det: Dict[str, Any]) -> bool:
        """
        Validate detection box using:
            • Confidence threshold
            • Maximum size threshold
            • Aspect ratio threshold
        """

        # Confidence check
        if det["conf"] < self.min_confidence:
            return False

        # Box size check
        box_w = det["x2"] - det["x1"]
        box_h = det["y2"] - det["y1"]

        if box_w <= 0 or box_h <= 0:
            return False

        if box_w > self.max_box_fraction or box_h > self.max_box_fraction:
            return False

        # Aspect ratio check
        aspect = max(box_w / box_h, box_h / box_w)
        if aspect > self.max_aspect_ratio:
            return False

        return True

    # --------------------------------------------------------
    # SECTION: Public API — Filter Detections
    # --------------------------------------------------------

    def filter(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter YOLO detections and return normalized, valid ball boxes.

        Input format:
            [
                {"x1": px, "y1": px, "x2": px, "y2": px, "conf": float},
                ...
            ]

        Output format:
            [
                {
                    "x1": float, "y1": float,
                    "x2": float, "y2": float,
                    "cx": float, "cy": float,
                    "conf": float
                },
                ...
            ]
        """

        if not detections:
            return []

        valid = []
        for det in detections:
            if self._is_valid_box(det):
                valid.append(self._normalize_box(det))

        return valid
