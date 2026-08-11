"""Motion proposals used only to rank primary detector candidates."""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig
from stabilize.detection.base_detector import BaseDetector
from stabilize.tracking.models import DetectionCandidate

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

    def detect_candidates(self, frame_bgr: np.ndarray) -> list[DetectionCandidate]:
        """Propose moving regions via frame subtraction.

        Args:
            frame_bgr: uint8 BGR image.

        Returns:
            Motion proposals. They must pass temporal gating before use.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_frame is None:
            self._prev_frame = gray
            return []

        # Absolute difference
        diff = cv2.absdiff(self._prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._prev_frame = gray
            return []

        # Pick largest contour that could be an aircraft (>1% of frame area)
        min_area = (w * h) * 0.01
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid:
            self._prev_frame = gray
            return []

        self._prev_frame = gray
        candidates = []
        for contour in valid:
            x, y, bw, bh = cv2.boundingRect(contour)
            area_ratio = cv2.contourArea(contour) / max(float(w * h), 1.0)
            if area_ratio > 0.50:
                logger.debug(
                    "Motion proposal rejected before gating: %.1f%% of frame",
                    area_ratio * 100,
                )
                continue
            candidates.append(
                DetectionCandidate(
                    bbox=(x, y, bw, bh),
                    score=float(min(0.49, 0.2 + area_ratio)),
                    label=None,
                    source="motion",
                )
            )
        return candidates
