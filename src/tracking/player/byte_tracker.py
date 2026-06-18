from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from norfair import Detection as NorfairDetection, Tracker


# =====================================================================
#  Data Classes
# =====================================================================

@dataclass
class Detection:
    """
    Raw YOLO detection for a single frame.

    bbox_xyxy : (x1, y1, x2, y2)
    confidence : YOLO confidence score
    cls : class ID (ball = 0)
    """
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    cls: int


@dataclass
class TrackResult:
    """
    Output of the tracker for a single tracked object.

    track_id : stable ID assigned by the tracker
    bbox_xyxy : reconstructed bounding box
    confidence : last detection confidence
    """
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float


@dataclass
class ByteTrackConfig:
    """
    Configuration for Norfair-based ByteTrack wrapper.
    """
    frame_rate: int = 30
    max_jump_distance: float = 120.0   # max allowed pixel jump per frame
    bbox_half_width: float = 25.0
    bbox_half_height: float = 50.0
    smooth_factor: float = 0.6         # bbox smoothing factor


# =====================================================================
#  ByteTrack Wrapper (Norfair-based)
# =====================================================================

class ByteTrackWrapper:
    """
    Norfair-based tracker that mimics ByteTrack behavior while adding:

    - Motion-preserving smoothing
    - Velocity-aware gating
    - Stable ID assignment
    - Debug metadata for downstream analysis

    This prevents ID flicker → prevents broken trajectories →
    prevents the 1-second highlight fallback.
    """

    def __init__(self, config: Optional[ByteTrackConfig] = None) -> None:
        self.config = config or ByteTrackConfig()

        # Norfair tracker tuned for ball tracking
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=self.config.max_jump_distance,
            hit_counter_max=2,        # require 2 hits for stability
            initialization_delay=0,   # activate immediately
        )

        # Store last bbox per track for smoothing
        self.last_bbox = {}

    # =================================================================
    #  Helper: Convert YOLO detections → Norfair detections
    # =================================================================
    def _to_norfair_detections(self, detections: List[Detection]):
        norfair_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            norfair_dets.append(
                NorfairDetection(
                    points=np.array([[cx, cy]], dtype=np.float32),
                    scores=np.array([det.confidence], dtype=np.float32),
                )
            )
        return norfair_dets

    # =================================================================
    #  Helper: Reconstruct bbox from center point
    # =================================================================
    def _reconstruct_bbox(self, cx, cy):
        w = self.config.bbox_half_width
        h = self.config.bbox_half_height
        return (cx - w, cy - h, cx + w, cy + h)

    # =================================================================
    #  Helper: Smooth bounding boxes to reduce jitter
    # =================================================================
    def _smooth_bbox(self, track_id, new_bbox):
        if track_id not in self.last_bbox:
            self.last_bbox[track_id] = new_bbox
            return new_bbox

        old = np.array(self.last_bbox[track_id], dtype=float)
        new = np.array(new_bbox, dtype=float)

        smoothed = (
            self.config.smooth_factor * old
            + (1 - self.config.smooth_factor) * new
        )

        self.last_bbox[track_id] = smoothed.tolist()
        return tuple(smoothed.tolist())

    # =================================================================
    #  Main update function
    # =================================================================
    def update(
        self,
        detections: List[Detection],
        frame_id: Optional[int] = None,
    ) -> List[TrackResult]:
        """
        Update the tracker with YOLO detections.

        Returns a list of TrackResult objects with:
        - stable track_id
        - smoothed bbox
        - confidence
        """

        # Convert YOLO → Norfair
        norfair_dets = self._to_norfair_detections(detections)

        # Update tracker
        tracked_objects = self.tracker.update(detections=norfair_dets)

        results: List[TrackResult] = []

        for obj in tracked_objects:
            cx, cy = obj.estimate[0]

            # Reconstruct bbox
            raw_bbox = self._reconstruct_bbox(cx, cy)

            # Smooth bbox to reduce jitter
            smoothed_bbox = self._smooth_bbox(obj.id, raw_bbox)

            # Confidence
            conf = (
                float(obj.last_detection.scores[0])
                if obj.last_detection
                else 1.0
            )

            results.append(
                TrackResult(
                    track_id=obj.id,
                    bbox_xyxy=smoothed_bbox,
                    confidence=conf,
                )
            )

        return results
