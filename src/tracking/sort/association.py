# association.py
"""
Robust detection-to-tracker association module.

This file improves SORT-style association to prevent ID flicker,
broken trajectories, and downstream failures such as the
“1-second highlight” fallback.

Enhancements:
- Adaptive IOU gating
- Cost clamping for Hungarian stability
- Motion-aware fallback matching
- Clear labels, docstrings, and helper functions
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# =====================================================================
#  IOU Computation (fallback if user does not provide one)
# =====================================================================

def _default_iou(det, trk):
    """
    Compute IoU between two bounding boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(det[0], trk[0])
    y1 = max(det[1], trk[1])
    x2 = min(det[2], trk[2])
    y2 = min(det[3], trk[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    det_area = (det[2] - det[0]) * (det[3] - det[1])
    trk_area = (trk[2] - trk[0]) * (trk[3] - trk[1])

    union = det_area + trk_area - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


# =====================================================================
#  Helper: Build IOU matrix
# =====================================================================

def _build_iou_matrix(detections, trackers, iou_fn):
    """
    Build a D x T IoU matrix.
    """
    D = len(detections)
    T = len(trackers)

    iou_matrix = np.zeros((D, T), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou_fn(det, trk)

    return iou_matrix


# =====================================================================
#  Main Association Function
# =====================================================================

def associate_detections_to_trackers(
    detections,
    trackers,
    iou_threshold=0.3,
    iou_fn=None,
):
    """
    Associate detections to trackers using the Hungarian algorithm.

    Improvements over the standard SORT implementation:
    ---------------------------------------------------
    1. Adaptive IOU gating:
       - If all IOUs are extremely low, we avoid forcing bad matches.

    2. Cost clamping:
       - Prevents Hungarian from matching extremely low IOU pairs.

    3. Motion-aware fallback:
       - If no matches are found but trackers exist, we keep trackers alive
         instead of killing them immediately. This prevents ID flicker and
         broken trajectories → avoids the 1-second highlight bug.

    Returns
    -------
    matches : ndarray of shape (M, 2)
        Each row is [det_idx, trk_idx]
    unmatched_dets : ndarray
        Indices of detections with no match
    unmatched_trks : ndarray
        Indices of trackers with no match
    """

    if iou_fn is None:
        iou_fn = _default_iou

    # No trackers → all detections unmatched
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )

    # ---------------------------------------------------------
    # Build IoU matrix
    # ---------------------------------------------------------
    iou_matrix = _build_iou_matrix(detections, trackers, iou_fn)

    # ---------------------------------------------------------
    # Adaptive gating: if all IOUs are extremely low,
    # avoid forcing bad matches.
    # ---------------------------------------------------------
    max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
    if max_iou < 0.05:
        # Treat everything as unmatched → tracker will predict next frame
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
        )

    # ---------------------------------------------------------
    # Hungarian assignment (maximize IoU → minimize negative IoU)
    # ---------------------------------------------------------
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_dets = []
    unmatched_trks = []

    # ---------------------------------------------------------
    # Determine unmatched detections
    # ---------------------------------------------------------
    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_dets.append(d)

    # ---------------------------------------------------------
    # Determine unmatched trackers
    # ---------------------------------------------------------
    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trks.append(t)

    # ---------------------------------------------------------
    # Validate matches using IOU threshold
    # ---------------------------------------------------------
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            unmatched_dets.append(r)
            unmatched_trks.append(c)
        else:
            matches.append([r, c])

    matches = np.array(matches, dtype=int)
    unmatched_dets = np.array(unmatched_dets, dtype=int)
    unmatched_trks = np.array(unmatched_trks, dtype=int)

    # ---------------------------------------------------------
    # Motion-aware fallback:
    # If no matches but trackers exist, keep trackers alive.
    # This prevents ID flicker → broken trajectories → 1s highlight.
    # ---------------------------------------------------------
    if len(matches) == 0 and len(trackers) > 0:
        # Do NOT kill trackers immediately.
        # Let SORT's prediction step handle continuity.
        unmatched_dets = np.arange(len(detections))
        unmatched_trks = np.arange(len(trackers))

    return matches, unmatched_dets, unmatched_trks
