import numpy as np


def savgol_smooth(y, window, poly):
    if window % 2 == 0:
        raise ValueError("window must be odd")

    if len(y) < window:
        return y

    half = window // 2
    x = np.arange(-half, half + 1)

    A = np.vander(x, poly + 1, increasing=True)
    ATA = A.T @ A
    ATA_inv = np.linalg.pinv(ATA)
    coeffs = ATA_inv @ A.T

    kernel = coeffs[0]
    return np.convolve(y, kernel[::-1], mode="same")


class TrajectorySmoother:
    """
    Story 4.4 — Advanced smoothing:
    - Pure NumPy Savitzky–Golay
    - Pure NumPy Kalman filter
    """

    def __init__(
        self,
        use_savgol=True,
        use_kalman=True,
        window=7,
        poly=3,
        kalman_process_var=1e-3,
        kalman_measure_var=1e-2,
    ):
        self.use_savgol = use_savgol
        self.use_kalman = use_kalman
        self.window = window
        self.poly = poly
        self.kalman_process_var = kalman_process_var
        self.kalman_measure_var = kalman_measure_var

    def _kalman_filter(self, arr):
        if len(arr) == 0:
            return arr

        x = arr[0]
        p = 1.0
        q = self.kalman_process_var
        r = self.kalman_measure_var

        out = [x]

        for z in arr[1:]:
            x_pred = x
            p_pred = p + q

            k = p_pred / (p_pred + r)
            x = x_pred + k * (z - x_pred)
            p = (1 - k) * p_pred

            out.append(x)

        return out

    def smooth(self, track):
        if not track:
            return track

        xs = np.array([t["x"] for t in track], dtype=float)
        ys = np.array([t["y"] for t in track], dtype=float)

        if self.use_savgol:
            xs = savgol_smooth(xs, self.window, self.poly)
            ys = savgol_smooth(ys, self.window, self.poly)

        if self.use_kalman:
            xs = self._kalman_filter(xs)
            ys = self._kalman_filter(ys)

        smoothed = []
        for i, t in enumerate(track):
            smoothed.append({
                "frame": t["frame"],
                "x": float(xs[i]),
                "y": float(ys[i]),
                "conf": t["conf"],
                "track_id": t["track_id"],
            })

        return smoothed
