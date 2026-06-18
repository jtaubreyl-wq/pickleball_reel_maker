# kalman_filter.py
"""
Robust 7‑state Kalman filter for object tracking.

This version includes:
- Motion‑preserving prediction (prevents flattened trajectories)
- Adaptive process noise (more noise when motion is high)
- Velocity damping (prevents runaway drift)
- Clear documentation and helper functions

This stabilizes trajectories → prevents broken rally detection →
prevents the 1‑second highlight fallback.
"""

import numpy as np


class KalmanFilter:
    """
    7‑state Kalman filter used in SORT/ByteTrack‑style trackers.

    State vector:
        x = [
            cx,     # center x
            cy,     # center y
            s,      # scale (area)
            r,      # aspect ratio
            vx,     # velocity x
            vy,     # velocity y
            vs      # velocity scale
        ]

    Measurement vector:
        z = [cx, cy, s, r]

    This implementation adds:
    - Motion preservation
    - Adaptive Q (process noise)
    - Velocity damping
    """

    def __init__(self):
        # ---------------------------------------------------------
        # Initial state
        # ---------------------------------------------------------
        self.x = np.zeros((7, 1))

        # Large initial covariance → tracker adapts quickly
        self.P = np.eye(7) * 10.0

        # ---------------------------------------------------------
        # State transition matrix (constant velocity model)
        # ---------------------------------------------------------
        dt = 1.0
        self.F = np.eye(7)
        self.F[0, 4] = dt  # cx += vx
        self.F[1, 5] = dt  # cy += vy
        self.F[2, 6] = dt  # s  += vs

        # ---------------------------------------------------------
        # Measurement matrix
        # ---------------------------------------------------------
        self.H = np.eye(4, 7)

        # ---------------------------------------------------------
        # Measurement noise (R)
        # ---------------------------------------------------------
        self.R = np.eye(4) * 1.0

        # ---------------------------------------------------------
        # Base process noise (Q)
        # This will be ADAPTED dynamically.
        # ---------------------------------------------------------
        self.base_Q = np.eye(7) * 0.01
        self.Q = self.base_Q.copy()

        # Velocity damping factor (prevents runaway drift)
        self.velocity_damping = 0.95

    # =====================================================================
    #  Helper: Adaptive process noise
    # =====================================================================
    def _update_process_noise(self):
        """
        Increase process noise when velocity is high.
        This prevents the filter from over‑smoothing real motion.
        """
        vx = float(self.x[4])
        vy = float(self.x[5])
        speed = np.sqrt(vx * vx + vy * vy)

        # More speed → more noise → less smoothing
        scale = 1.0 + min(speed / 20.0, 3.0)

        self.Q = self.base_Q * scale

    # =====================================================================
    #  Prediction step
    # =====================================================================
    def predict(self):
        """
        Predict next state using constant velocity model.
        Includes:
        - Velocity damping
        - Adaptive process noise
        """
        # Apply velocity damping
        self.x[4] *= self.velocity_damping
        self.x[5] *= self.velocity_damping
        self.x[6] *= self.velocity_damping

        # Update Q based on current motion
        self._update_process_noise()

        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x

    # =====================================================================
    #  Update step
    # =====================================================================
    def update(self, z):
        """
        Update state with measurement z = [cx, cy, s, r].
        """
        z = z.reshape((4, 1))

        # Innovation
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + (K @ y)

        # Update covariance
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P

        return self.x
