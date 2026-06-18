# utils.py
"""
Utility functions for tracking and geometry.

This module provides:
- Numerically stable IoU computation
- Optional Generalized IoU (GIoU) for small-object stability
- Safe clamping to avoid NaN/inf issues during association

These improvements help maintain stable tracking → continuous ball
trajectories → prevent the 1‑second highlight fallback.
"""

import numpy as np


# =====================================================================
#  Standard IoU (with numerical stability)
# =====================================================================

def iou(bb_test, bb_gt):
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes.

    Parameters
    ----------
    bb_test : array-like
        [x1, y1, x2, y2]
    bb_gt : array-like
        [x1, y1, x2, y2]

    Returns
    -------
    float
        IoU value in [0, 1].

    Notes
    -----
    This version is numerically stable and clamps invalid values.
    It prevents NaN/inf propagation into the Hungarian association,
    which can break tracking → break trajectories → cause the
    1‑second highlight fallback.
    """

    # Convert to float arrays
    bb_test = np.asarray(bb_test, dtype=float)
    bb_gt = np.asarray(bb_gt, dtype=float)

    # Intersection box
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h

    # Areas
    area_test = max(1e-6, (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]))
    area_gt   = max(1e-6, (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]))

    union = area_test + area_gt - inter
    union = max(union, 1e-6)

    iou_val = inter / union

    # Clamp to valid range
    if not np.isfinite(iou_val):
        return 0.0

    return float(max(0.0, min(1.0, iou_val)))


# =====================================================================
#  Optional: Generalized IoU (GIoU)
# =====================================================================

def giou(bb_test, bb_gt):
    """
    Compute Generalized IoU (GIoU), which is more stable for small objects.

    GIoU is useful for pickleball tracking because the ball is tiny and
    bounding boxes often barely overlap, causing unstable IoU values.

    Returns a value in [-1, 1].
    """

    bb_test = np.asarray(bb_test, dtype=float)
    bb_gt = np.asarray(bb_gt, dtype=float)

    # Standard IoU
    iou_val = iou(bb_test, bb_gt)

    # Coordinates of smallest enclosing box
    x1 = min(bb_test[0], bb_gt[0])
    y1 = min(bb_test[1], bb_gt[1])
    x2 = max(bb_test[2], bb_gt[2])
    y2 = max(bb_test[3], bb_gt[3])

    enclosing_area = max(1e-6, (x2 - x1) * (y2 - y1))

    # Areas
    area_test = max(1e-6, (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]))
    area_gt   = max(1e-6, (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]))

    # Intersection
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h

    union = area_test + area_gt - inter
    union = max(union, 1e-6)

    # GIoU penalty
    giou_val = iou_val - (enclosing_area - union) / enclosing_area

    # Clamp
    if not np.isfinite(giou_val):
        return -1.0

    return float(max(-1.0, min(1.0, giou_val)))
