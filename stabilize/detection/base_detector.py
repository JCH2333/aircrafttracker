"""Abstract base class for aircraft detectors."""

from abc import ABC, abstractmethod

import numpy as np

from stabilize.config import StabilizerConfig


class BaseDetector(ABC):
    """Interface for aircraft detection backends.

    All detectors return bounding boxes in (x, y, w, h) format,
    or None if no aircraft is detected.
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.conf = config.detection_confidence

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect the largest aircraft in the frame.

        Args:
            frame_bgr: uint8 BGR image of shape (H, W, 3).

        Returns:
            (x, y, w, h) bounding box of the largest detected aircraft,
            or None if no aircraft meets the confidence threshold.
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Pre-load model to avoid first-frame latency."""
        ...
