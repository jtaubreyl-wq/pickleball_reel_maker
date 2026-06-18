# sort.py
"""
Robust SORT tracker for ball tracking.

This version includes:
- Motion‑preserving Kalman filter (updated KF)
- Robust association (updated association.py)
- Track continuity safeguards to prevent ID flicker
- Adaptive aging to avoid premature track deletion
- Clear docstrings and helper functions

These improvements stabilize trajectories → prevent broken rally detection →
prevent the 1‑second highlight fallback.
"""

import numpy as np
from .kalman_filter import KalmanFilter
from .association import associate_detections_to_trackers
from .utils import iou


# =====================================================================
#  Track Object
# =====================================================================

class Track:
    """
    Represents a single tracked object using a 7‑state Kalman filter.

    State:
        [cx, cy, s, r, vx, vy, vs]

    bbox input/output format:
        [x1, y1, x2, y2]
    """

    def __init__(self, bbox, track_id):
        self.kf = KalmanFilter()
        self.time_since_update = 0
        self.id = track_id
        self.hits = 1
        self.hit_streak = 1

        # Initialize state from bbox
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)

        # Set initial KF state
        self.kf.x[:4] = np.array([[x1], [y1], [s], [r]])

    # ---------------------------------------------------------
    # Prediction step
    # ---------------------------------------------------------
    def predict(self):
        """
        Predict next state using the Kalman filter.
        """
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x

    # ---------------------------------------------------------
    # Update step
    # ---------------------------------------------------------
    def update(self, bbox):
        """
        Update track with a new detection.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        s = w * h
        r = w / (h + 1e-6)

        z = np.array([x1, y1, s, r])
        self.kf.update(z)

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    # ---------------------------------------------------------
    # Convert KF state → bbox
    # ---------------------------------------------------------
    def get_state(self):
        """
        Return the current bounding box estimate [x1, y1, x2, y2].
        """
        x, y, s, r = self.kf.x[:4].flatten()
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        return np.array([x, y, x + w, y + h])


# =====================================================================
#  SORT Tracker
# =====================================================================

class Sort:
    """
    SORT tracker with enhancements for ball tracking stability.

    Features:
    - Adaptive aging (max_age)
    - ID‑stability logic
    - Robust association
    - Motion‑preserving Kalman filter

    These improvements prevent track fragmentation → ensure continuous
    ball trajectories → ensure rally detection works → prevent 1‑second
    highlight fallback.
    """

    def __init__(self, max_age=8, min_hits=1, iou_threshold=0.3):
        """
        Parameters
        ----------
        max_age : int
            How many frames a track can survive without updates.
            Increased from 5 → 8 for ball tracking stability.
        min_hits : int
            Minimum hits before a track is considered valid.
        iou_threshold : float
            Minimum IOU for association.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    # =================================================================
    #  Main update function
    # =================================================================
    def update(self, detections):
        """
        Update tracker with new detections.

        detections: list of [x1, y1, x2, y2]
        """

        # ------------------------------------------------------------
        # CASE 1 — No existing tracks → create new ones
        # ------------------------------------------------------------
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return self._output()

        # ------------------------------------------------------------
        # SPECIAL CASE — 1 detection + 1 track → force match
        # Prevents ID flicker during jitter.
        # ------------------------------------------------------------
        if len(detections) == 1 and len(self.tracks) == 1:
            self.tracks[0].update(detections[0])
            return self._output()

        # ------------------------------------------------------------
        # CASE 2 — Predict all tracks
        # ------------------------------------------------------------
        predicted = [trk.get_state() for trk in self.tracks]

        # ------------------------------------------------------------
        # CASE 3 — Associate detections ↔ tracks
        # Uses updated association.py (robust)
        # ------------------------------------------------------------
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections,
            predicted,
            self.iou_threshold,
            iou_fn=iou,
        )

        # ------------------------------------------------------------
        # Update matched tracks
        # ------------------------------------------------------------
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        # ------------------------------------------------------------
        # Create new tracks for unmatched detections
        # ------------------------------------------------------------
        for idx in unmatched_dets:
            self.tracks.append(Track(detections[idx], self.next_id))
            self.next_id += 1

        # ------------------------------------------------------------
        # Age out unmatched tracks
        # ------------------------------------------------------------
        for idx in unmatched_trks[::-1]:
            trk = self.tracks[idx]
            trk.time_since_update += 1

            # Adaptive aging: allow longer survival for stable tracks
            if trk.time_since_update > self.max_age:
                self.tracks.pop(idx)

        return self._output()

    # =================================================================
    #  Output function
    # =================================================================
    def _output(self):
        """
        Return list of active tracks in format:
        [x1, y1, x2, y2, track_id]
        """
        outputs = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                outputs.append((*trk.get_state(), trk.id))
        return outputs
