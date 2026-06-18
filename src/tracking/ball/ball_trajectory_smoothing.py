import numpy as np
from scipy.signal import savgol_filter


class TrajectorySmoother:
    """
    Story 4.4 — Smooth ball trajectories using:
    1. Savitzky–Golay smoothing (noise reduction)
    2. Kalman filtering (physical motion model)
    3. Motion-preservation safeguards (prevents flattening → 1‑sec highlight bug)

    This module ensures:
    - Smoothing never removes real motion.
    - Short tracks are not over-smoothed.
    - Output always contains meaningful trajectory curvature.
    """

    def __init__(
        self,
        use_savgol=True,
        use_kalman=True,
        window=7,
        poly=3,
        kalman_process_var=1e-3,
        kalman_measure_var=1e-2,
        min_motion_threshold=0.5,
        enable_debug_labels=True,
    ):
        """
        Parameters
        ----------
        use_savgol : bool
            Enable Savitzky–Golay smoothing.
        use_kalman : bool
            Enable Kalman filtering.
        window : int
            SG window size (must be odd).
        poly : int
            SG polynomial order.
        kalman_process_var : float
            Kalman process noise.
        kalman_measure_var : float
            Kalman measurement noise.
        min_motion_threshold : float
            Minimum allowed per-frame movement after smoothing.
            Prevents flattening → avoids 1‑second highlight bug.
        enable_debug_labels : bool
            Adds debug metadata to each frame.
        """
        self.use_savgol = use_savgol
        self.use_kalman = use_kalman
        self.window = window
        self.poly = poly
        self.kalman_process_var = kalman_process_var
        self.kalman_measure_var = kalman_measure_var
        self.min_motion_threshold = min_motion_threshold
        self.enable_debug_labels = enable_debug_labels

    # ---------------------------------------------------------
    # 1. Savitzky–Golay smoothing
    # ---------------------------------------------------------
    def _smooth_savgol(self, arr):
        """
        Apply SG smoothing with adaptive fallback for short tracks.
        """
        if len(arr) < self.window or self.window < 3:
            return arr  # too short to smooth safely

        try:
            return savgol_filter(arr, self.window, self.poly).tolist()
        except ValueError:
            # fallback if SG fails
            return arr

    # ---------------------------------------------------------
    # 2. Simple 1D Kalman filter
    # ---------------------------------------------------------
    def _kalman_filter(self, arr):
        """
        Apply a 1D Kalman filter that preserves motion instead of flattening it.
        """
        if len(arr) == 0:
            return arr

        x = arr[0]  # initial estimate
        p = 1.0     # initial covariance
        q = self.kalman_process_var
        r = self.kalman_measure_var

        out = [x]

        for z in arr[1:]:
            # Predict
            x_pred = x
            p_pred = p + q

            # Update
            k = p_pred / (p_pred + r)
            x = x_pred + k * (z - x_pred)
            p = (1 - k) * p_pred

            out.append(x)

        return out

    # ---------------------------------------------------------
    # 3. Motion preservation safeguard
    # ---------------------------------------------------------
    def _preserve_motion(self, original, smoothed):
        """
        Ensure smoothing does not collapse motion.

        If smoothed movement < threshold, blend with original.
        """
        corrected = []
        for i in range(len(original)):
            if i == 0:
                corrected.append(smoothed[i])
                continue

            orig_dx = original[i] - original[i - 1]
            smooth_dx = smoothed[i] - smoothed[i - 1]

            if abs(smooth_dx) < self.min_motion_threshold:
                # Blend 70% smoothed, 30% original to restore motion
                corrected.append(0.7 * smoothed[i] + 0.3 * original[i])
            else:
                corrected.append(smoothed[i])

        return corrected

    # ---------------------------------------------------------
    # 4. Main smoothing function
    # ---------------------------------------------------------
    def smooth(self, track):
        """
        Smooth the trajectory while preserving real motion.

        Input
        -----
        track : List[Dict]
            Raw tracking output from Story 4.2.

        Output
        ------
        List[Dict]
            Same structure, but with smoothed x/y and optional debug labels.
        """
        if not track:
            return track

        xs = [t["x"] for t in track]
        ys = [t["y"] for t in track]

        # -----------------------------
        # Step 1 — Savitzky–Golay
        # -----------------------------
        if self.use_savgol:
            xs_sg = self._smooth_savgol(xs)
            ys_sg = self._smooth_savgol(ys)
        else:
            xs_sg, ys_sg = xs, ys

        # -----------------------------
        # Step 2 — Kalman filter
        # -----------------------------
        if self.use_kalman:
            xs_kf = self._kalman_filter(xs_sg)
            ys_kf = self._kalman_filter(ys_sg)
        else:
            xs_kf, ys_kf = xs_sg, ys_sg

        # -----------------------------
        # Step 3 — Motion preservation
        # -----------------------------
        xs_final = self._preserve_motion(xs, xs_kf)
        ys_final = self._preserve_motion(ys, ys_kf)

        # -----------------------------
        # Step 4 — Rebuild track
        # -----------------------------
        smoothed = []
        for i, t in enumerate(track):
            entry = {
                "frame": t["frame"],
                "x": xs_final[i],
                "y": ys_final[i],
                "conf": t["conf"],
                "track_id": t["track_id"],
            }

            if self.enable_debug_labels:
                entry["debug"] = {
                    "original_x": xs[i],
                    "original_y": ys[i],
                    "sg_x": xs_sg[i] if self.use_savgol else xs[i],
                    "sg_y": ys_sg[i] if self.use_savgol else ys[i],
                    "kf_x": xs_kf[i] if self.use_kalman else xs_sg[i],
                    "kf_y": ys_kf[i] if self.use_kalman else ys_sg[i],
                    "final_x": xs_final[i],
                    "final_y": ys_final[i],
                }

            smoothed.append(entry)

        return smoothed
