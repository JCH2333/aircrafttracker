"""YOLOv8 aircraft detector (optional backend).

Requires: pip install ultralytics
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig
from stabilize.detection.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """YOLOv8-nano detector for aircraft."""

    def __init__(self, config: StabilizerConfig):
        super().__init__(config)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e

        self.model = YOLO("yolov8n.pt")
        self.device_str = "cuda" if "cuda" in str(config.device) else "cpu"
        self._warmed = False

    def warmup(self) -> None:
        """Run a dummy inference."""
        if self._warmed:
            return
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        logger.info("Warming up YOLOv8 model...")
        self.detect(dummy)
        self._warmed = True
        logger.info("Model warmup complete.")

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect the largest airplane in the frame."""
        h, w = frame_bgr.shape[:2]
        max_dim = self.config.analysis_downscale

        # Downscale for speed
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

        results = self.model(
            small,
            classes=[4],  # COCO airplane
            conf=self.conf,
            device=self.device_str,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy() / scale
        confs = boxes.conf.cpu().numpy()

        # Pick largest box
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best = xyxy[areas.argmax()]
        best_conf = confs[areas.argmax()]

        x1, y1, x2, y2 = best
        logger.debug(
            "YOLO detection: box=(%.0f,%.0f,%.0f,%.0f) conf=%.2f",
            x1, y1, x2, y2, best_conf,
        )

        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
