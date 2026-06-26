"""Torchvision Faster R-CNN aircraft detector.

Uses the COCO-pretrained Faster R-CNN ResNet-50 FPN V2 model.
COCO class 4 = airplane.
"""

import logging

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from stabilize.config import StabilizerConfig
from stabilize.detection.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class TorchvisionDetector(BaseDetector):
    """Faster R-CNN detector for aircraft using torchvision."""

    def __init__(self, config: StabilizerConfig):
        super().__init__(config)
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform from the weights
        self.transform = weights.transforms()
        self._warmed = False

    def warmup(self) -> None:
        """Run a dummy inference to trigger CUDA JIT compilation."""
        if self._warmed:
            return
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        logger.info("Warming up Faster R-CNN model (first run may take a moment)...")
        self.detect(dummy)
        self._warmed = True
        logger.info("Model warmup complete.")

    def detect(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect the largest aircraft-like object in the frame.

        Accepts multiple COCO vehicle classes since civil aviation aircraft
        may be misclassified from ground-level viewpoints (e.g., as "bus").

        Uses tiered confidence:
          - High confidence (self.conf): any accepted class
          - Low confidence (config.detection_confidence_low): any accepted class

        Args:
            frame_bgr: uint8 BGR image of shape (H, W, 3).

        Returns:
            (x, y, w, h) bounding box or None.
        """
        h, w = frame_bgr.shape[:2]
        max_dim = self.config.analysis_downscale

        # Downscale if needed for faster inference
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            small = frame_bgr

        # Convert to tensor with normalization
        tensor = torch.from_numpy(small).permute(2, 0, 1).float().div(255)
        tensor = self.transform(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)[0]

        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        boxes_all = pred["boxes"].cpu().numpy() / scale

        # Find detections matching accepted vehicle classes
        target_classes = set(self.config.detection_classes)
        candidates = []

        for i, label in enumerate(labels):
            if label not in target_classes:
                continue
            # Use tiered confidence
            score = float(scores[i])
            threshold = (
                self.config.detection_confidence_low
                if label != 4  # class 4 (airplane) uses main confidence
                else self.conf
            )
            if score >= threshold:
                box = boxes_all[i]
                area = (box[2] - box[0]) * (box[3] - box[1])
                candidates.append((box, area, score, int(label)))

        if not candidates:
            return None

        # Pick largest candidate box
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_box, area, score, label = candidates[0]
        coco_names = {4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat"}
        label_name = coco_names.get(label, str(label))

        logger.debug(
            "Detection: class=%s(%d), box=(%.0f,%.0f,%.0f,%.0f) area=%.0f conf=%.2f",
            label_name, label,
            best_box[0], best_box[1], best_box[2], best_box[3],
            area, score,
        )

        x1, y1, x2, y2 = best_box
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
