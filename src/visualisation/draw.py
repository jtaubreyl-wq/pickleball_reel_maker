import os
import cv2
import numpy as np
from supervision import Detections

DEBUG_DIR = "data/processed/debug/"

PLAYER_COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
]


class DetectionVisualizer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        os.makedirs(DEBUG_DIR, exist_ok=True)

    def draw(self, frame: np.ndarray, detections: Detections, frame_idx: int):
        """
        Draw bounding boxes + labels on a frame and save to debug directory.
        """
        if not self.enabled:
            return

        annotated = frame.copy()

        for i, (xyxy, conf, cls) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            x1, y1, x2, y2 = map(int, xyxy)

            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            label = f"Player {i} ({conf:.2f})"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - 20),
                (x1 + 150, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        out_path = os.path.join(DEBUG_DIR, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, annotated)
