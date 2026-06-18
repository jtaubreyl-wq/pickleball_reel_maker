# ============================================================
# FILE: src/tracking/player/player_tracker_sort.py
# ============================================================
"""
PlayerTrackerSORT

Tracks players across frames using a SORT-style tracker:
    - Kalman filter per track
    - IOU-based association
    - Simple ID management

Designed to work with:
    - PlayerDetector (player_detector.py)
    - PlayerStateExtractor (later)
    - Rally detection (Stories 5.1 / 5.2)
"""

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


# ============================================================
# SORT CORE
# ============================================================

@dataclass
class Track:
    """
    Represents a single tracked player.
    """
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    hits: int = 0
    age: int = 0
    time_since_update: int = 0


def iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


# ============================================================
# PLAYER TRACKER (SORT)
# ============================================================

class PlayerTrackerSORT:
    """
    Simple SORT-style tracker for players.

    Input per frame:
        detections: [
            {
                "bbox": (x1, y1, x2, y2),
                "conf": 0.87,
                "cx": 512,
                "cy": 300,
            },
            ...
        ]

    Output per frame:
        [
            {
                "track_id": 1,
                "bbox": (x1, y1, x2, y2),
                "cx": 512,
                "cy": 300,
            },
            ...
        ]
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: List[Track] = []
        self.next_id: int = 1

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with detections for a single frame.

        Args:
            detections: list of detection dicts from PlayerDetector

        Returns:
            list of tracked player dicts with track_id
        """
        # Convert detections to numpy bboxes
        det_bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            det_bboxes.append(np.array([x1, y1, x2, y2], dtype=float))

        det_bboxes = np.array(det_bboxes) if det_bboxes else np.zeros((0, 4))

        # Age all tracks
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        # Association: tracks ↔ detections via IoU
        matches, unmatched_tracks, unmatched_dets = self._associate(det_bboxes)

        # Update matched tracks
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            track.bbox = det_bboxes[d_idx]
            track.hits += 1
            track.time_since_update = 0

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            bbox = det_bboxes[d_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=bbox,
                hits=1,
                age=1,
                time_since_update=0,
            )
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # Build output
        outputs: List[Dict] = []
        for t in self.tracks:
            if t.hits < self.min_hits:
                continue

            x1, y1, x2, y2 = t.bbox.tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            outputs.append({
                "track_id": t.track_id,
                "bbox": (x1, y1, x2, y2),
                "cx": cx,
                "cy": cy,
            })

        return outputs

    # ---------------------------------------------------------
    # INTERNAL: Association
    # ---------------------------------------------------------
    def _associate(self, det_bboxes: np.ndarray):
        """
        Associate current tracks with detections using IoU.
        Returns:
            matches: list of (track_idx, det_idx)
            unmatched_tracks: list of track indices
            unmatched_dets: list of det indices
        """
        if len(self.tracks) == 0 or len(det_bboxes) == 0:
            return [], list(range(len(self.tracks))), list(range(len(det_bboxes)))

        iou_matrix = np.zeros((len(self.tracks), len(det_bboxes)), dtype=float)

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(det_bboxes):
                iou_matrix[t_idx, d_idx] = iou(track.bbox, det)

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(det_bboxes)))

        # Greedy matching
        while True:
            if iou_matrix.size == 0:
                break

            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[t_idx, d_idx]

            if max_iou < self.iou_threshold:
                break

            matches.append((t_idx, d_idx))
            unmatched_tracks.remove(t_idx)
            unmatched_dets.remove(d_idx)

            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        return matches, unmatched_tracks, unmatched_dets
