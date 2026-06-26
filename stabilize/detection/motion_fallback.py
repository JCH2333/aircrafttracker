"""Motion-based fallback detector using frame differencing.

This is a last-resort fallback when both the detector and tracker fail.
It identifies moving objects by differencing widely-spaced frames.
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig
from stabilize.detection.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class MotionFallbackDetector(BaseDetector):
    """Frame-differencing fallback for when primary detector fails."""

    def __init__(self, config: StabilizerConfig):
        super().__init__(config)
        self._prev_frame = None
        self._interval = 5  # frames between difference

    def warmup(self) -> None:
        """No-op: no model to load."""
        pass

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Find moving object via frame subtraction.

        Args:
            frame_bgr: uint8 BGR image.

        Returns:
            (x, y, w, h) of largest moving region or None.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_frame is None:
            self._prev_frame = gray
            return None

        # Absolute difference
        diff = cv2.absdiff(self._prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._prev_frame = gray
            return None

        # Pick largest contour that could be an aircraft (>1% of frame area)
        min_area = (w * h) * 0.01
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid:
            self._prev_frame = gray
            return None

        largest = max(valid, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        self._prev_frame = gray

        logger.debug("Motion fallback: box=(%d,%d,%d,%d)", x, y, bw, bh)
        return (x, y, bw, bh)
