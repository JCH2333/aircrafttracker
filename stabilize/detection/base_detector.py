"""Abstract base class for aircraft detectors."""

from abc import ABC, abstractmethod

import numpy as np

from stabilize.config import StabilizerConfig
from stabilize.tracking.models import DetectionCandidate


class BaseDetector(ABC):
    """Interface for aircraft detection backends.

    All detectors return candidate bounding boxes in (x, y, w, h) format.
    Temporal selection is owned by the tracking backend.
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.conf = config.detection_confidence

    @abstractmethod
    def detect_candidates(self, frame_bgr: np.ndarray) -> list[DetectionCandidate]:
        """Detect all plausible aircraft candidates in the frame.

        Args:
            frame_bgr: uint8 BGR image of shape (H, W, 3).

        Returns:
            Candidate detections. Temporal gating is handled by the pipeline.
        """
        ...

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Backward-compatible single-box detector API."""
        candidates = self.detect_candidates(frame_bgr)
        if not candidates:
            return None
        best = max(candidates, key=lambda candidate: candidate.score)
        return best.bbox

    @abstractmethod
    def warmup(self) -> None:
        """Pre-load model to avoid first-frame latency."""
        ...
