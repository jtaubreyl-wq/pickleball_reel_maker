from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from src.tracking.sort.sort import Sort


# ------------------------------------------------------------
# Configuration dataclass
# ------------------------------------------------------------

@dataclass
class BallTrackConfig:
    max_age: int = 10
    min_hits: int = 1
    iou_threshold: float = 0.1
    ball_class_id: int = 0
    min_confidence: float = 0.2


# ------------------------------------------------------------
# Convert YOLO detections → internal ball detection format
# ------------------------------------------------------------

def yolo_to_ball_detections(
    yolo_dets: List[Dict[str, Any]],
    ball_class_id: int,
    min_confidence: float
) -> List[Dict[str, Any]]:
    output = []

    for det in yolo_dets:
        if det["class_id"] != ball_class_id:
            continue
        if det["confidence"] < min_confidence:
            continue

        x1, y1, x2, y2 = det["bbox"]

        output.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "class_id": det["class_id"],
            "confidence": det["confidence"],
        })

    return output


# ------------------------------------------------------------
# Select the single best ball detection
# ------------------------------------------------------------

def select_single_ball(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Select the single best ball detection from a list.
    Typically chooses the highest-confidence detection.
    Returns None if the list is empty.
    """
    if not detections:
        return None

    return max(detections, key=lambda d: d["confidence"])


# ------------------------------------------------------------
# Ball Tracker (wraps SORT)
# ------------------------------------------------------------

class BallTracker:
    def __init__(self, config: BallTrackConfig):
        self.config = config
        self.tracker = Sort(
            max_age=config.max_age,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold
        )

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Filter for ball detections only
        ball_dets = [
            d for d in detections
            if d["class_id"] == self.config.ball_class_id
            and d["confidence"] >= self.config.min_confidence
        ]

        # Convert to SORT format
        sort_input = [
            [d["x1"], d["y1"], d["x2"], d["y2"]]
            for d in ball_dets
        ]

        # Update SORT tracker
        tracked = self.tracker.update(sort_input)

        # Convert SORT output to timeline format
        output = []
        for x1, y1, x2, y2, track_id in tracked:
            output.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "track_id": int(track_id)
            })

        return output
