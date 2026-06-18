# ============================================================
# FILE: src/players/player_detector.py
# ============================================================
"""
PlayerDetector

Detects players in each video frame using YOLO.
Outputs clean bounding boxes and confidence scores.

This module is the foundation for:
    - Player tracking (SORT / ByteTrack)
    - PlayerState extraction
    - Rally detection (Stories 5.1 / 5.2)
    - Winner detection (Story 6.2)
    - Highlight scoring (Story 6.3)
"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


class PlayerDetector:
    """
    YOLO-based player detector.
    Detects 'person' class only.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.35):
        """
        Args:
            model_path: Path to YOLO model (.pt)
            conf_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in a single frame.

        Returns:
            List of dicts:
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "conf": 0.87,
                    "cx": 512,
                    "cy": 300,
                },
                ...
            ]
        """
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # YOLO class 0 = person
            if cls != 0:
                continue
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "cx": cx,
                "cy": cy,
            })

        return detections

    # ---------------------------------------------------------
    # BATCH API (optional)
    # ---------------------------------------------------------
    def detect_video(self, video_path: str) -> Dict[int, List[Dict]]:
        """
        Run detection on an entire video.

        Returns:
            { frame_idx: [player detections] }
        """
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        output = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            output[frame_idx] = self.detect_players(frame)

        cap.release()
        return output
