import numpy as np


# =====================================================================
#  Outlier Removal — Modified Z-Score
# =====================================================================

def remove_outliers_zscore(values, threshold=3.5):
    """
    Remove extreme spikes using a robust median-based Z-score.

    This prevents single-frame YOLO jitter from creating fake bounces
    or direction changes that break rally segmentation.
    """
    arr = np.array(values, dtype=float)

    if len(arr) < 3:
        return arr

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))

    if mad == 0:
        return arr

    modified_z = 0.6745 * (arr - median) / mad
    mask = np.abs(modified_z) < threshold

    cleaned = arr.copy()

    for i, valid in enumerate(mask):
        if not valid:
            left = next((j for j in range(i - 1, -1, -1) if mask[j]), None)
            right = next((j for j in range(i + 1, len(arr)) if mask[j]), None)

            if left is not None and right is not None:
                cleaned[i] = (
                    arr[left]
                    if abs(arr[left] - arr[i]) < abs(arr[right] - arr[i])
                    else arr[right]
                )
            elif left is not None:
                cleaned[i] = arr[left]
            elif right is not None:
                cleaned[i] = arr[right]

    return cleaned


# =====================================================================
#  Exponential Moving Average (EMA)
# =====================================================================

def smooth_ema(values, alpha=0.5):
    """
    Smooth values using an exponential moving average.

    EMA is gentle and preserves motion well, making it ideal for
    preventing the “flat trajectory → 1-second highlight” bug.
    """
    if len(values) == 0:
        return []

    smoothed = [float(values[0])]

    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

    return smoothed


# =====================================================================
#  Pure NumPy Savitzky–Golay Smoothing
# =====================================================================

def smooth_savgol(values, window=5, poly=3):
    """
    Pure NumPy Savitzky–Golay smoothing.
    Fully compatible with NumPy 2.x and SciPy-free.

    Adaptive windowing ensures we never oversmooth short sequences.
    """
    arr = np.array(values, dtype=float)
    n = len(arr)

    if n < 3:
        return arr

    # Ensure odd window
    if window % 2 == 0:
        window += 1

    # Window cannot exceed sequence length
    if window > n:
        window = n if n % 2 == 1 else n - 1

    # Poly must be < window
    if poly >= window:
        poly = window - 1

    if window < 3 or poly < 1:
        return arr

    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, poly + 1, increasing=True)

    ATA_inv = np.linalg.pinv(A.T @ A)
    coeffs = (ATA_inv @ A.T)[0]

    return np.convolve(arr, coeffs[::-1], mode="same")


# =====================================================================
#  Motion Preservation Layer
# =====================================================================

def preserve_motion(original, smoothed, min_motion=0.4):
    """
    Prevent smoothing from flattening the trajectory.

    If smoothed motion < threshold, blend with original motion.
    This is the KEY fix for the 1-second highlight bug.
    """
    corrected = []
    for i in range(len(original)):
        if i == 0:
            corrected.append(smoothed[i])
            continue

        orig_dx = original[i] - original[i - 1]
        smooth_dx = smoothed[i] - smoothed[i - 1]

        if abs(smooth_dx) < min_motion:
            corrected.append(0.7 * smoothed[i] + 0.3 * original[i])
        else:
            corrected.append(smoothed[i])

    return corrected


# =====================================================================
#  Timeline Smoothing (x, y per frame)
# =====================================================================

def smooth_timeline(
    timeline,
    method="ema",
    alpha=0.5,
    remove_outliers=False,
    min_motion_threshold=0.4,
    add_debug_labels=True,
):
    """
    Smooth a full timeline of ball positions grouped by track_id.

    This function:
    - Removes outliers
    - Applies EMA or Savitzky–Golay smoothing
    - Preserves motion to avoid flattening
    - Adds debug labels for inspection

    Returns a smoothed timeline sorted by frame.
    """
    if len(timeline) == 0:
        return timeline

    # Group by track_id
    tracks = {}
    for item in timeline:
        tid = item["track_id"]
        tracks.setdefault(tid, []).append(item)

    smoothed_output = []

    for tid, items in tracks.items():
        xs = [i["x"] for i in items]
        ys = [i["y"] for i in items]

        # -----------------------------
        # Step 1 — Outlier removal
        # -----------------------------
        if remove_outliers:
            xs = remove_outliers_zscore(xs)
            ys = remove_outliers_zscore(ys)

        # -----------------------------
        # Step 2 — Smoothing
        # -----------------------------
        if method == "ema":
            xs_s = smooth_ema(xs, alpha)
            ys_s = smooth_ema(ys, alpha)
        elif method == "savgol":
            xs_s = smooth_savgol(xs)
            ys_s = smooth_savgol(ys)
        else:
            xs_s, ys_s = xs, ys

        # -----------------------------
        # Step 3 — Motion preservation
        # -----------------------------
        xs_final = preserve_motion(xs, xs_s, min_motion=min_motion_threshold)
        ys_final = preserve_motion(ys, ys_s, min_motion=min_motion_threshold)

        # -----------------------------
        # Step 4 — Rebuild timeline
        # -----------------------------
        for i, item in enumerate(items):
            entry = {
                "frame": item["frame"],
                "x": float(xs_final[i]),
                "y": float(ys_final[i]),
                "conf": float(item["conf"]),
                "track_id": tid,
            }

            if add_debug_labels:
                entry["debug"] = {
                    "original_x": float(xs[i]),
                    "original_y": float(ys[i]),
                    "smoothed_x": float(xs_s[i]),
                    "smoothed_y": float(ys_s[i]),
                    "final_x": float(xs_final[i]),
                    "final_y": float(ys_final[i]),
                }

            smoothed_output.append(entry)

    smoothed_output.sort(key=lambda x: x["frame"])
    return smoothed_output
