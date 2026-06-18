# ============================================================
# SECTION: Imports & Configuration
# ============================================================

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import numpy as np
from src.tracking.sort.sort import Sort


# ============================================================
# SECTION: Debug Helper
# ============================================================

def debug(msg: str):
    """
    Lightweight debug logger for ball tracking.
    """
    print(f"[DEBUG][BallTracker] {msg}", flush=True)


# ============================================================
# SECTION: Ball Tracking Configuration
# ============================================================

@dataclass
class BallTrackConfig:
    """
    Configuration for SORT-based ball tracking.
    """
    max_age: int = 10
    min_hits: int = 1
    iou_threshold: float = 0.1

    # YOLO filtering
    ball_class_id: int = 0
    min_confidence: float = 0.20

    # Optional smoothing
    smoothing_alpha: float = 0.25  # EMA smoothing for trajectory


# ============================================================
# SECTION: YOLO → SORT Conversion
# ============================================================

def dict_to_sort_format(detections: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Convert YOLO detection dicts into SORT's expected format:
        [x1, y1, x2, y2, confidence, class_id]
    """
    converted = []
    for d in detections:
        try:
            converted.append([
                float(d["x1"]),
                float(d["y1"]),
                float(d["x2"]),
                float(d["y2"]),
                float(d["conf"]),
                float(d["class_id"]),
            ])
        except Exception as e:
            debug(f"Malformed detection skipped: {d} ({e})")
    return converted


# ============================================================
# SECTION: Ball Tracker (SORT Wrapper)
# ============================================================

class BallTracker:
    """
    SORT-based ball tracker that accepts YOLO dict detections.

    Now supports:
        • dict input: {frame_idx: [detections]}
        • list input: [[detections], [detections], ...]
    """

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def __init__(self, config: Optional[BallTrackConfig] = None):
        self.config = config or BallTrackConfig()

        debug("Initializing SORT tracker...")
        self.tracker = Sort(
            max_age=self.config.max_age,
            min_hits=self.config.min_hits,
            iou_threshold=self.config.iou_threshold,
        )
        debug("SORT tracker initialized.")

        # For smoothing
        self.last_positions: Dict[int, tuple] = {}

    # --------------------------------------------------------
    # Internal Smoothing Helper
    # --------------------------------------------------------

    def _smooth(self, track_id: int, x: float, y: float) -> tuple:
        """
        Apply exponential moving average smoothing to reduce jitter.
        """
        if track_id not in self.last_positions:
            self.last_positions[track_id] = (x, y)
            return x, y

        prev_x, prev_y = self.last_positions[track_id]
        alpha = self.config.smoothing_alpha

        smoothed_x = prev_x * (1 - alpha) + x * alpha
        smoothed_y = prev_y * (1 - alpha) + y * alpha

        self.last_positions[track_id] = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    # --------------------------------------------------------
    # Update Tracker for a Single Frame
    # --------------------------------------------------------

    def update(self, detections_dict: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update SORT tracker with YOLO detections for a single frame.
        """

        # Convert YOLO → SORT
        dets = dict_to_sort_format(detections_dict)

        # Filter ball detections
        ball_dets = [
            d for d in dets
            if int(d[5]) == self.config.ball_class_id and d[4] >= self.config.min_confidence
        ]

        sort_input = [d[:4] for d in ball_dets]

        start_time = time.time()

        try:
            tracked = self.tracker.update(sort_input)
        except Exception as e:
            debug(f"ERROR inside SORT.update(): {e}")
            return []

        if time.time() - start_time > 1.0:
            debug("WARNING: SORT.update() took >1 second — possible hang detected")

        # Convert SORT output → dicts
        out = []
        for x1, y1, x2, y2, track_id in tracked:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Apply smoothing
            cx, cy = self._smooth(track_id, cx, cy)

            out.append({
                "frame": None,
                "x": float(cx),
                "y": float(cy),
                "conf": 1.0,
                "track_id": int(track_id),
            })

        return out

    # --------------------------------------------------------
    # Track Entire Sequence
    # --------------------------------------------------------

    def track_sequence(self, frames) -> List[Dict[str, Any]]:
        """
        Track ball across all frames.

        Accepts:
            • dict: {frame_idx: [detections]}
            • list: [[detections], [detections], ...]

        Returns:
            List of tracked points with frame indices.
        """

        # --- NEW: Accept list OR dict ---
        if isinstance(frames, list):
            debug("Input is list — converting to dict format")
            frames = {i: det for i, det in enumerate(frames)}

        debug(f"track_sequence() starting — total frames: {len(frames)}")

        timeline = []

        for frame_idx in sorted(frames.keys()):
            dets = frames.get(frame_idx, [])

            try:
                tracked = self.update(dets)
            except Exception as e:
                debug(f"ERROR updating frame {frame_idx}: {e}")
                continue

            for item in tracked:
                item["frame"] = frame_idx
                timeline.append(item)

        debug(f"track_sequence() complete — total tracked points: {len(timeline)}")
        return timeline
