"""
kalman_filter.py — Constant-velocity Kalman Filter for Bounding-Box Tracking

State vector: [x, y, w, h, vx, vy, vw, vh]
  (x, y)   = bounding-box centroid
  (w, h)   = bounding-box width / height
  v-prefix = corresponding velocities

Measurement vector: [x, y, w, h]

This follows the SORT state representation (Bewley et al., 2016).
"""

import numpy as np


class KalmanBoxTracker:
    """Kalman filter that tracks a single bounding box in [x, y, w, h] space."""

    _count = 0  # class-level ID counter

    def __init__(self, bbox, dt=1.0):
        """
        Parameters
        ----------
        bbox : array-like, shape (4,)
            Initial bounding box [x1, y1, x2, y2] in pixel coordinates.
        dt   : float
            Time step (1 frame by default).
        """
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

        # Convert [x1, y1, x2, y2] → [cx, cy, w, h]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        # State transition (constant-velocity model)
        self.F = np.eye(8)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt

        # Measurement matrix — observe [cx, cy, w, h]
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Covariance matrices
        self.P = np.eye(8) * 10.0        # state covariance
        self.P[4:, 4:] *= 100.0          # high uncertainty on initial velocities
        self.Q = np.eye(8) * 1.0         # process noise
        self.Q[4:, 4:] *= 0.01           # velocities change slowly
        self.R = np.eye(4) * 1.0         # measurement noise

        # Bookkeeping
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    # ------------------------------------------------------------------
    def predict(self):
        """Advance state one time-step; return predicted [x1, y1, x2, y2]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox()

    # ------------------------------------------------------------------
    def update(self, bbox):
        """Correct state with a matched detection [x1, y1, x2, y2]."""
        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R         # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    # ------------------------------------------------------------------
    def get_state(self):
        """Return current bounding box as [x1, y1, x2, y2]."""
        return self._state_to_bbox()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _bbox_to_z(self, bbox):
        """Convert [x1, y1, x2, y2] → measurement [cx, cy, w, h]."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return np.array([cx, cy, w, h], dtype=np.float64)

    def _state_to_bbox(self):
        """Convert state [cx, cy, w, h, ...] → [x1, y1, x2, y2]."""
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @staticmethod
    def reset_count():
        """Reset the global ID counter (useful between experiments)."""
        KalmanBoxTracker._count = 0
