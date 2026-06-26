"""MIL tracker wrapper with periodic re-detection and IoU-based handoff.

Uses cv2.TrackerMIL (available in opencv-python-headless 4.13+).
For better accuracy, cv2.TrackerDaSiamRPN_create() can be substituted
if the ONNX model files are available.
"""

import logging
from typing import Callable

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection-over-Union of two (x, y, w, h) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class AircraftTracker:
    """MIL tracker for aircraft with periodic re-detection handoff.

    The tracker runs CSRT frame-to-frame and periodically calls
    a detector function to re-anchor the bounding box. An IoU-based
    handoff logic prevents bad detections from corrupting tracking.
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.reinit_interval = config.detection_interval
        self.timeout = config.tracker_quality_timeout
        self.tracker: cv2.Tracker | None = None
        self._initialized = False
        self._frames_since_reinit = 0
        self._current_bbox: tuple[float, float, float, float] | None = None
        self._current_centroid: tuple[float, float] | None = None
        self._lost = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def centroid(self) -> tuple[float, float] | None:
        """Current aircraft centroid (cx, cy)."""
        return self._current_centroid

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """Current bounding box (x, y, w, h)."""
        return self._current_bbox

    @property
    def lost(self) -> bool:
        """True if tracker has lost the aircraft."""
        return self._lost

    def init(self, frame_bgr: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> None:
        """Initialize tracker with a new bounding box."""
        # Use MIL tracker (available in opencv-python-headless 4.13+).
        # DaSiamRPN would be more accurate but requires extra ONNX model files.
        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(frame_bgr, tuple(bbox_xywh))
        self._current_bbox = tuple(float(v) for v in bbox_xywh)
        self._current_centroid = (
            bbox_xywh[0] + bbox_xywh[2] / 2,
            bbox_xywh[1] + bbox_xywh[3] / 2,
        )
        self._initialized = True
        self._frames_since_reinit = 0
        self._lost = False

    def update(self, frame_bgr: np.ndarray) -> bool:
        """Update tracker on a new frame.

        Returns:
            True if tracker update succeeded, False if lost.
        """
        if not self._initialized or self.tracker is None:
            return False

        self._frames_since_reinit += 1
        success, bbox = self.tracker.update(frame_bgr)

        if success:
            x, y, w, h = bbox
            self._current_bbox = (x, y, w, h)
            self._current_centroid = (x + w / 2, y + h / 2)
            self._lost = False
        else:
            self._lost = True

        return success

    def needs_redetection(self) -> bool:
        """Check if re-detection should be triggered."""
        return self._frames_since_reinit >= self.reinit_interval

    def is_stale(self) -> bool:
        """Check if tracker has gone too long without successful re-detection."""
        return self._frames_since_reinit >= self.timeout

    def try_handoff(
        self,
        detection_bbox: tuple[int, int, int, int] | None,
        frame_bgr: np.ndarray,
    ) -> bool:
        """Attempt to hand off tracking to a new detection.

        Args:
            detection_bbox: (x, y, w, h) from detector, or None.
            frame_bgr: Current frame for tracker re-initialization.

        Returns:
            True if tracker was re-initialized with new detection.
        """
        if detection_bbox is None:
            return False

        # If not initialized, accept any detection
        if not self._initialized or self._current_bbox is None:
            self.init(frame_bgr, detection_bbox)
            logger.info("Tracker initialized with detection: %s", detection_bbox)
            return True

        iou = compute_iou(detection_bbox, self._current_bbox)
        threshold = 0.2 if self.is_stale() else 0.3

        if iou > threshold:
            self.init(frame_bgr, detection_bbox)
            logger.debug("Tracker re-initialized (IoU=%.2f): %s", iou, detection_bbox)
            return True
        else:
            logger.debug("Detection skipped (IoU=%.2f < %.2f)", iou, threshold)
            return False

    def get_centroid_list(self) -> list[tuple[float, float]] | None:
        """Return current tracking data. Used by pipeline to build trajectory."""
        return None  # managed externally by pipeline
