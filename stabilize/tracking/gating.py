"""Temporal gating for detector and fallback candidates."""

from __future__ import annotations

from collections import deque
from math import hypot, log

import numpy as np

from stabilize.tracking.models import (
    BBox,
    DetectionCandidate,
    Point,
    RejectedCandidate,
)


class CandidateGate:
    """Reject target candidates that violate recent motion and geometry."""

    def __init__(
        self,
        max_history: int = 45,
        max_area_ratio: float = 4.0,
        min_area_ratio: float = 0.25,
        max_frame_area: float = 0.50,
    ):
        self._areas: deque[float] = deque(maxlen=max_history)
        self._aspects: deque[float] = deque(maxlen=max_history)
        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.max_frame_area = max_frame_area

    @property
    def median_area(self) -> float | None:
        if not self._areas:
            return None
        return float(np.median(self._areas))

    @property
    def median_aspect(self) -> float | None:
        if not self._aspects:
            return None
        return float(np.median(self._aspects))

    def record_bbox(self, bbox: BBox | None) -> None:
        if bbox is None:
            return
        _, _, w, h = bbox
        area = float(max(w, 0) * max(h, 0))
        if area <= 0:
            return
        self._areas.append(area)
        self._aspects.append(w / max(float(h), 1.0))

    def select(
        self,
        candidates: list[DetectionCandidate],
        frame_shape: tuple[int, int],
        predicted_center: Point | None = None,
        predicted_bbox: BBox | None = None,
        manual: bool = False,
    ) -> tuple[DetectionCandidate | None, tuple[RejectedCandidate, ...]]:
        if not candidates:
            return None, ()

        if manual:
            return max(candidates, key=lambda c: c.score), ()

        frame_h, frame_w = frame_shape
        frame_area = float(frame_h * frame_w)
        median_area = self.median_area
        median_aspect = self.median_aspect
        rejected: list[RejectedCandidate] = []
        accepted: list[tuple[float, DetectionCandidate]] = []

        for candidate in candidates:
            reason = self._reject_reason(
                candidate,
                frame_area=frame_area,
                frame_shape=frame_shape,
                median_area=median_area,
                median_aspect=median_aspect,
                predicted_center=predicted_center,
                predicted_bbox=predicted_bbox,
            )
            if reason:
                rejected.append(RejectedCandidate(candidate, reason))
                continue

            accepted.append(
                (
                    self._rank_score(
                        candidate,
                        predicted_center=predicted_center,
                        median_area=median_area,
                        median_aspect=median_aspect,
                        frame_shape=frame_shape,
                    ),
                    candidate,
                )
            )

        if not accepted:
            return None, tuple(rejected)

        accepted.sort(key=lambda item: item[0], reverse=True)
        return accepted[0][1], tuple(rejected)

    def _reject_reason(
        self,
        candidate: DetectionCandidate,
        frame_area: float,
        frame_shape: tuple[int, int],
        median_area: float | None,
        median_aspect: float | None,
        predicted_center: Point | None,
        predicted_bbox: BBox | None,
    ) -> str | None:
        x, y, w, h = candidate.bbox
        frame_h, frame_w = frame_shape

        if w <= 0 or h <= 0:
            return "empty bbox"
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return "bbox outside frame"
        if candidate.area > frame_area * self.max_frame_area:
            return "bbox exceeds 50% of frame"

        if median_area:
            ratio = candidate.area / median_area
            if ratio > self.max_area_ratio:
                return "area increased more than 4x"
            if ratio < self.min_area_ratio:
                return "area decreased below 0.25x"

        if median_aspect:
            ratio = candidate.aspect_ratio / max(median_aspect, 1e-6)
            if ratio > 3.0 or ratio < 1.0 / 3.0:
                return "aspect ratio changed more than 3x"

        if predicted_center is not None:
            cx, cy = candidate.center
            distance = hypot(cx - predicted_center[0], cy - predicted_center[1])
            if predicted_bbox is not None:
                gate_radius = max(
                    60.0,
                    3.0 * hypot(predicted_bbox[2], predicted_bbox[3]),
                )
            elif median_area:
                gate_radius = max(60.0, 6.0 * np.sqrt(median_area))
            else:
                gate_radius = 0.35 * hypot(frame_w, frame_h)
            if distance > gate_radius:
                return "candidate outside predicted motion gate"

        return None

    @staticmethod
    def _rank_score(
        candidate: DetectionCandidate,
        predicted_center: Point | None,
        median_area: float | None,
        median_aspect: float | None,
        frame_shape: tuple[int, int],
    ) -> float:
        score = float(candidate.score) * 2.0

        if predicted_center is not None:
            frame_h, frame_w = frame_shape
            distance = hypot(
                candidate.center[0] - predicted_center[0],
                candidate.center[1] - predicted_center[1],
            )
            score -= distance / max(hypot(frame_w, frame_h), 1.0) * 5.0

        if median_area:
            score -= abs(log(max(candidate.area / median_area, 1e-6)))
        if median_aspect:
            score -= 0.5 * abs(
                log(max(candidate.aspect_ratio / median_aspect, 1e-6))
            )

        if candidate.label == 4:
            score += 0.25
        return score
