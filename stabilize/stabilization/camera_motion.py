"""Camera motion estimator using background feature points.

Tracks stationary ground features (buildings, poles, runway markings)
outside the aircraft bounding box to compute camera pan/tilt motion.

Uses Shi-Tomasi corner detection + Lucas-Kanade optical flow + RANSAC
affine estimation to separate camera motion from aircraft motion.
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class CameraMotionEstimator:
    """Estimate camera pan/tilt from background feature points.

    Detects and tracks feature points in the background (outside the
    aircraft bounding box). The dominant motion of these points reveals
    the camera's pan/tilt rotation between frames.
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.prev_points: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self._frames_since_detect: int = 999
        self._redetect_interval: int = config.camera_redetect_interval

        # LK params
        self._lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        # Feature params
        self._feature_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=15, blockSize=7,
        )

        # Cached full-frame gradient (for template matcher reuse)
        self.gradient_map: np.ndarray | None = None
        self.gradient_frame_idx: int = -1

    def estimate(
        self, frame_bgr: np.ndarray, aircraft_bbox: tuple | None
    ) -> tuple[float, float] | None:
        """Estimate camera translation (dx, dy) between previous and current frame.

        Args:
            frame_bgr: Current frame (uint8 BGR).
            aircraft_bbox: (x, y, w, h) of aircraft region to exclude.

        Returns:
            (dx_cam, dy_cam) camera translation in pixels, or None if
            insufficient background points.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Store gradient map for template matcher reuse
        self.gradient_map = gray
        self.gradient_frame_idx = getattr(self, 'gradient_frame_idx', -1) + 1

        # Re-detect features periodically or when first frame
        if (self.prev_points is None or self.prev_gray is None
                or self._frames_since_detect >= self._redetect_interval):
            self._detect_features(gray, aircraft_bbox)
            self._frames_since_detect = 0
            if self.prev_points is None:
                return None

        # Optical flow
        new_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self._lk_params
        )

        if new_points is None or len(new_points) < 8:
            self._detect_features(gray, aircraft_bbox)
            return None

        # Filter successfully tracked points
        status = status.ravel()
        ok_old = self.prev_points[status == 1]
        ok_new = new_points[status == 1]

        if len(ok_new) < 8:
            self._detect_features(gray, aircraft_bbox)
            return None

        # Compute robust translation via median of tracked deltas
        deltas = ok_new - ok_old
        dx = float(np.median(deltas[:, 0, 0]))
        dy = float(np.median(deltas[:, 0, 1]))

        # Update state
        self.prev_gray = gray
        self.prev_points = ok_new.reshape(-1, 1, 2)
        self._frames_since_detect += 1

        return (dx, dy)

    def _detect_features(
        self, gray: np.ndarray, aircraft_bbox: tuple | None
    ) -> None:
        """Detect Shi-Tomasi corners in background (outside aircraft)."""
        mask = None
        if aircraft_bbox is not None:
            x, y, w, h = aircraft_bbox
            # Exclude aircraft region plus margin
            mx, my = int(w * 0.15), int(h * 0.15)
            mask = np.ones_like(gray, dtype=np.uint8) * 255
            y1 = max(0, y - my)
            y2 = min(gray.shape[0], y + h + my)
            x1 = max(0, x - mx)
            x2 = min(gray.shape[1], x + w + mx)
            mask[y1:y2, x1:x2] = 0

        points = cv2.goodFeaturesToTrack(
            gray, mask=mask, **self._feature_params
        )
        if points is not None and len(points) >= 8:
            self.prev_points = points
            self.prev_gray = gray
            self._frames_since_detect = 0

    def build_gradient_map(self, gray: np.ndarray) -> np.ndarray:
        """Build Sobel gradient magnitude map for template matcher reuse.

        This avoids redundant Sobel computation — the template matcher
        can crop from this pre-computed map instead of recalculating.
        """
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = cv2.GaussianBlur(mag, (0, 0), self.config.edge_blur_sigma)
        mx = mag.max()
        if mx > 1e-6:
            mag /= mx
        self.gradient_map = mag
        return mag
