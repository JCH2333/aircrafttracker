"""Fixed-anchor feature tracking constrained by an object mask."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import degrees, hypot

import cv2
import numpy as np

from stabilize.config import StabilizerConfig
from stabilize.tracking.models import BBox, Point


@dataclass
class FeatureMeasurement:
    center: Point | None
    bbox: BBox | None
    confidence: float
    visibility: float
    inlier_count: int
    point_count: int


class MaskedFeatureTracker:
    """Track a fixed object anchor with masked LK points and RANSAC."""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.anchor: Point | None = None
        self.bbox: BBox | None = None
        self._mask_areas: deque[float] = deque(maxlen=45)

        self.lk_params = {
            "winSize": config.lk_win_size,
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.lk_max_iter,
                config.lk_epsilon,
            ),
        }

    @property
    def initialized(self) -> bool:
        return self.anchor is not None and self.prev_gray is not None

    @property
    def point_count(self) -> int:
        return len(self.prev_points) if self.prev_points is not None else 0

    @property
    def median_mask_area(self) -> float | None:
        if not self._mask_areas:
            return None
        return float(np.median(self._mask_areas))

    def initialize(
        self,
        frame_bgr: np.ndarray,
        bbox: BBox,
        mask: np.ndarray | None = None,
    ) -> FeatureMeasurement:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.bbox = _clip_bbox(bbox, gray.shape)
        x, y, w, h = self.bbox
        self.anchor = (x + w / 2.0, y + h / 2.0)
        point_mask = _build_point_mask(gray.shape, self.bbox, mask)
        self.prev_points = _extract_points(gray, point_mask, self.config)

        visibility, mask_quality, area = self._measure_mask(mask, gray.shape)
        if area > 0:
            self._mask_areas.append(area)
        confidence = max(mask_quality, 0.8 if self.point_count >= 8 else 0.55)
        return FeatureMeasurement(
            center=self.anchor,
            bbox=self.bbox,
            confidence=float(min(confidence, 1.0)),
            visibility=visibility,
            inlier_count=self.point_count,
            point_count=self.point_count,
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray | None,
        predicted_center: Point | None = None,
    ) -> FeatureMeasurement:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        visibility, mask_quality, area = self._measure_mask(mask, gray.shape)

        if (
            self.prev_gray is None
            or self.prev_points is None
            or self.anchor is None
            or self.bbox is None
            or len(self.prev_points) < 4
        ):
            recovered = self._recover_from_mask(
                gray,
                mask,
                visibility,
                mask_quality,
                area,
                predicted_center,
            )
            if recovered is not None:
                return recovered
            self.prev_gray = gray
            return FeatureMeasurement(
                center=None,
                bbox=self.bbox,
                confidence=0.0,
                visibility=visibility,
                inlier_count=0,
                point_count=self.point_count,
            )

        new_points, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            **self.lk_params,
        )
        if new_points is None or status_fwd is None:
            self.prev_gray = gray
            return FeatureMeasurement(
                center=None,
                bbox=self.bbox,
                confidence=0.0,
                visibility=visibility,
                inlier_count=0,
                point_count=self.point_count,
            )

        back_points, status_back, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.prev_gray,
            new_points,
            None,
            **self.lk_params,
        )
        if back_points is None or status_back is None:
            self.prev_gray = gray
            return FeatureMeasurement(
                center=None,
                bbox=self.bbox,
                confidence=0.0,
                visibility=visibility,
                inlier_count=0,
                point_count=self.point_count,
            )

        old_flat = self.prev_points.reshape(-1, 2)
        new_flat = new_points.reshape(-1, 2)
        back_flat = back_points.reshape(-1, 2)
        valid = (status_fwd.ravel() == 1) & (status_back.ravel() == 1)
        valid &= np.linalg.norm(old_flat - back_flat, axis=1) <= 1.5

        if mask is not None:
            mask_bool = mask.astype(bool)
            xs = np.clip(np.rint(new_flat[:, 0]).astype(int), 0, gray.shape[1] - 1)
            ys = np.clip(np.rint(new_flat[:, 1]).astype(int), 0, gray.shape[0] - 1)
            valid &= mask_bool[ys, xs]

        old_valid = old_flat[valid]
        new_valid = new_flat[valid]
        if len(new_valid) < 4:
            self.prev_points = new_valid.reshape(-1, 1, 2).astype(np.float32)
            self.prev_gray = gray
            return FeatureMeasurement(
                center=None,
                bbox=self.bbox,
                confidence=0.0,
                visibility=visibility,
                inlier_count=0,
                point_count=len(new_valid),
            )

        transform, inlier_mask = cv2.estimateAffinePartial2D(
            old_valid,
            new_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if transform is None or inlier_mask is None:
            self.prev_points = new_valid.reshape(-1, 1, 2).astype(np.float32)
            self.prev_gray = gray
            return FeatureMeasurement(
                center=None,
                bbox=self.bbox,
                confidence=0.0,
                visibility=visibility,
                inlier_count=0,
                point_count=len(new_valid),
            )

        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        inlier_ratio = inlier_count / max(len(new_valid), 1)
        if inlier_count < 4:
            self.prev_points = new_valid.reshape(-1, 1, 2).astype(np.float32)
            self.prev_gray = gray
            return FeatureMeasurement(
                center=None,
                bbox=self.bbox,
                confidence=0.0,
                visibility=visibility,
                inlier_count=inlier_count,
                point_count=len(new_valid),
            )

        old_inlier = old_valid[inliers]
        new_inlier = new_valid[inliers]
        translation_only = False
        if not self._is_plausible_transform(transform, gray.shape):
            translation = np.median(new_inlier - old_inlier, axis=0)
            if not self._is_plausible_translation(
                translation,
                old_inlier,
                new_inlier,
            ):
                self.prev_points = None
                self.prev_gray = gray
                return FeatureMeasurement(
                    center=None,
                    bbox=self.bbox,
                    confidence=0.0,
                    visibility=visibility,
                    inlier_count=inlier_count,
                    point_count=len(new_valid),
                )
            translation_only = True

        old_anchor = self.anchor
        if translation_only:
            dx, dy = (float(translation[0]), float(translation[1]))
            new_anchor = (old_anchor[0] + dx, old_anchor[1] + dy)
        else:
            new_anchor = _transform_point(old_anchor, transform)

        self.anchor = new_anchor
        self.bbox = _translate_bbox(
            self.bbox,
            new_anchor[0] - old_anchor[0],
            new_anchor[1] - old_anchor[1],
            gray.shape,
        )

        self.prev_points = new_inlier.reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = gray

        point_strength = min(1.0, inlier_count / 20.0)
        confidence = (
            0.50 * inlier_ratio
            + 0.30 * point_strength
            + 0.20 * mask_quality
        )
        if translation_only:
            confidence *= 0.85

        if confidence >= self.config.template_update_min_confidence:
            if area > 0:
                self._mask_areas.append(area)
            self._replenish(gray, mask)

        return FeatureMeasurement(
            center=self.anchor,
            bbox=self.bbox,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            visibility=visibility,
            inlier_count=inlier_count,
            point_count=self.point_count,
        )

    def _recover_from_mask(
        self,
        gray: np.ndarray,
        mask: np.ndarray | None,
        visibility: float,
        mask_quality: float,
        area: float,
        predicted_center: Point | None,
    ) -> FeatureMeasurement | None:
        if self.anchor is None or self.bbox is None or mask is None:
            return None

        recovery_confidence = (
            0.65 * mask_quality
            + 0.35 * visibility
        )
        if recovery_confidence < self.config.tracking_update_confidence:
            return None

        recovered_bbox = _bbox_from_mask_near(
            mask,
            predicted_center or self.anchor,
        )
        if recovered_bbox is None:
            return None

        if not self._is_plausible_recovery_bbox(
            recovered_bbox,
            predicted_center,
        ):
            return None

        old_anchor = self.anchor
        if predicted_center is not None:
            predicted_x = int(round(predicted_center[0]))
            predicted_y = int(round(predicted_center[1]))
            prediction_inside_mask = (
                0 <= predicted_x < mask.shape[1]
                and 0 <= predicted_y < mask.shape[0]
                and bool(mask[predicted_y, predicted_x])
            )
            if prediction_inside_mask:
                recovered_anchor = predicted_center
            else:
                component_center = (
                    recovered_bbox[0] + recovered_bbox[2] / 2.0,
                    recovered_bbox[1] + recovered_bbox[3] / 2.0,
                )
                recovered_anchor = (
                    0.25 * predicted_center[0] + 0.75 * component_center[0],
                    0.25 * predicted_center[1] + 0.75 * component_center[1],
                )
        else:
            old_x, old_y, old_w, old_h = self.bbox
            rel_x = (old_anchor[0] - old_x) / max(float(old_w), 1.0)
            rel_y = (old_anchor[1] - old_y) / max(float(old_h), 1.0)
            new_x, new_y, new_w, new_h = recovered_bbox
            recovered_anchor = (
                new_x + rel_x * new_w,
                new_y + rel_y * new_h,
            )

        recovered_tracking_bbox = _translate_bbox(
            self.bbox,
            recovered_anchor[0] - old_anchor[0],
            recovered_anchor[1] - old_anchor[1],
            gray.shape,
        )
        point_mask = _build_point_mask(
            gray.shape,
            recovered_tracking_bbox,
            mask,
        )
        recovered_points = _extract_points(gray, point_mask, self.config)
        if recovered_points is None or len(recovered_points) < 4:
            return None

        self.anchor = recovered_anchor
        self.bbox = recovered_tracking_bbox
        self.prev_points = recovered_points
        self.prev_gray = gray

        point_strength = min(1.0, self.point_count / 20.0)
        confidence = (
            0.75 * recovery_confidence
            + 0.25 * point_strength
        )
        if confidence >= self.config.tracking_update_confidence and area > 0:
            self._mask_areas.append(area)

        return FeatureMeasurement(
            center=self.anchor if self.point_count >= 4 else None,
            bbox=self.bbox,
            confidence=(
                float(np.clip(confidence, 0.0, 1.0))
                if self.point_count >= 4
                else 0.0
            ),
            visibility=visibility,
            inlier_count=self.point_count,
            point_count=self.point_count,
        )

    def _is_plausible_transform(
        self,
        transform: np.ndarray,
        shape: tuple[int, int],
    ) -> bool:
        if self.bbox is None:
            return False

        linear = transform[:, :2]
        scale_x = float(np.linalg.norm(linear[:, 0]))
        scale_y = float(np.linalg.norm(linear[:, 1]))
        scale = (scale_x + scale_y) * 0.5
        if scale <= 0.0:
            return False
        if abs(scale_x - scale_y) / scale > 0.20:
            return False
        if abs(scale - 1.0) > self.config.feature_max_scale_delta:
            return False

        rotation = degrees(float(np.arctan2(linear[1, 0], linear[0, 0])))
        if abs(rotation) > self.config.feature_max_rotation_degrees:
            return False

        proposed_anchor = _transform_point(self.anchor, transform)
        local_translation = np.asarray(
            (
                proposed_anchor[0] - self.anchor[0],
                proposed_anchor[1] - self.anchor[1],
            ),
            dtype=np.float64,
        )
        if not self._translation_within_gate(local_translation):
            return False
        return True

    def _is_plausible_translation(
        self,
        translation: np.ndarray,
        old_points: np.ndarray,
        new_points: np.ndarray,
    ) -> bool:
        if not self._translation_within_gate(translation):
            return False
        displacements = new_points - old_points
        residuals = np.linalg.norm(displacements - translation, axis=1)
        return (
            float(np.median(residuals))
            <= self.config.feature_translation_residual_px
            and float(np.percentile(residuals, 90))
            <= self.config.feature_translation_residual_px * 2.0
        )

    def _translation_within_gate(self, translation: np.ndarray) -> bool:
        if self.bbox is None:
            return False
        _, _, width, height = self.bbox
        max_translation = max(
            12.0,
            self.config.feature_max_translation_ratio
            * hypot(float(width), float(height)),
        )
        return float(np.linalg.norm(translation)) <= max_translation

    def _is_plausible_recovery_bbox(
        self,
        recovered_bbox: BBox,
        predicted_center: Point | None,
    ) -> bool:
        if self.bbox is None:
            return False

        old_area = max(float(self.bbox[2] * self.bbox[3]), 1.0)
        new_area = float(recovered_bbox[2] * recovered_bbox[3])
        area_ratio = new_area / old_area
        if not (0.10 <= area_ratio <= 8.0):
            return False

        old_aspect = self.bbox[2] / max(float(self.bbox[3]), 1.0)
        new_aspect = recovered_bbox[2] / max(float(recovered_bbox[3]), 1.0)
        aspect_ratio = new_aspect / max(old_aspect, 1e-6)
        if not (
            1.0 / 4.0
            <= aspect_ratio
            <= 4.0
        ):
            return False

        reference = predicted_center or (
            self.bbox[0] + self.bbox[2] / 2.0,
            self.bbox[1] + self.bbox[3] / 2.0,
        )
        candidate_center = (
            recovered_bbox[0] + recovered_bbox[2] / 2.0,
            recovered_bbox[1] + recovered_bbox[3] / 2.0,
        )
        distance = hypot(
            candidate_center[0] - reference[0],
            candidate_center[1] - reference[1],
        )
        gate = max(
            30.0,
            2.5 * hypot(float(self.bbox[2]), float(self.bbox[3])),
        )
        return distance <= gate

    def _measure_mask(
        self,
        mask: np.ndarray | None,
        shape: tuple[int, int],
    ) -> tuple[float, float, float]:
        if mask is None:
            return 1.0, 0.5, 0.0

        mask_bool = mask.astype(bool)
        area = float(mask_bool.sum())
        if area <= 0:
            return 0.0, 0.0, 0.0
        if area > shape[0] * shape[1] * 0.5:
            return 0.0, 0.0, area

        median_area = self.median_mask_area
        if median_area:
            area_ratio = min(area / median_area, median_area / area)
        else:
            area_ratio = 1.0

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bool.astype(np.uint8),
            connectivity=8,
        )
        largest = 0.0
        if count > 1:
            largest = float(stats[1:, cv2.CC_STAT_AREA].max())
        connectedness = largest / area if area > 0 else 0.0
        quality = 0.65 * float(np.clip(area_ratio, 0.0, 1.0)) + 0.35 * connectedness
        return (
            float(np.clip(area_ratio, 0.0, 1.0)),
            float(np.clip(quality, 0.0, 1.0)),
            area,
        )

    def _replenish(
        self,
        gray: np.ndarray,
        mask: np.ndarray | None,
    ) -> None:
        point_mask = _build_point_mask(gray.shape, self.bbox, mask)
        candidates = _extract_points(gray, point_mask, self.config)
        if candidates is None or len(candidates) == 0:
            return

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = candidates
            return

        existing = self.prev_points.reshape(-1, 2)
        new_points = candidates.reshape(-1, 2)
        kept = []
        for point in new_points:
            if np.linalg.norm(existing - point, axis=1).min() >= self.config.feature_min_distance:
                kept.append(point)
        if kept:
            merged = np.vstack([existing, np.asarray(kept, dtype=np.float32)])
            self.prev_points = merged[: self.config.feature_max_corners * 2].reshape(
                -1, 1, 2
            )


def bbox_from_mask(mask: np.ndarray | None) -> BBox | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def _bbox_from_mask_near(
    mask: np.ndarray | None,
    reference: Point | None,
) -> BBox | None:
    if mask is None:
        return None
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )
    if count <= 1:
        return None

    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    largest_area = float(component_areas.max())
    candidates = [
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= max(4.0, largest_area * 0.10)
    ]
    if not candidates:
        return None

    selected = max(candidates, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    if reference is not None:
        ref_x = int(round(reference[0]))
        ref_y = int(round(reference[1]))
        if 0 <= ref_x < mask.shape[1] and 0 <= ref_y < mask.shape[0]:
            reference_label = int(labels[ref_y, ref_x])
            if reference_label in candidates:
                selected = reference_label
            else:
                selected = min(
                    candidates,
                    key=lambda label: hypot(
                        float(centroids[label][0]) - reference[0],
                        float(centroids[label][1]) - reference[1],
                    ),
                )

    x = int(stats[selected, cv2.CC_STAT_LEFT])
    y = int(stats[selected, cv2.CC_STAT_TOP])
    width = int(stats[selected, cv2.CC_STAT_WIDTH])
    height = int(stats[selected, cv2.CC_STAT_HEIGHT])
    return (x, y, width, height)


def _build_point_mask(
    shape: tuple[int, int],
    bbox: BBox,
    object_mask: np.ndarray | None,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = _clip_bbox(bbox, shape)
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    x1 = min(max(x + margin_x, 0), shape[1])
    y1 = min(max(y + margin_y, 0), shape[0])
    x2 = min(max(x + w - margin_x, x1), shape[1])
    y2 = min(max(y + h - margin_y, y1), shape[0])
    mask[y1:y2, x1:x2] = 255

    if object_mask is not None:
        obj = object_mask.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        obj = cv2.erode(obj, kernel, iterations=1)
        mask = cv2.bitwise_and(mask, obj)
    return mask


def _extract_points(
    gray: np.ndarray,
    mask: np.ndarray,
    config: StabilizerConfig,
) -> np.ndarray | None:
    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=config.feature_max_corners,
        qualityLevel=config.feature_quality,
        minDistance=config.feature_min_distance,
        blockSize=7,
    )
    return points


def _transform_point(point: Point, transform: np.ndarray) -> Point:
    vector = np.array([point[0], point[1], 1.0], dtype=np.float64)
    result = transform @ vector
    return (float(result[0]), float(result[1]))


def _translate_bbox(
    bbox: BBox,
    dx: float,
    dy: float,
    shape: tuple[int, int],
) -> BBox:
    x, y, w, h = bbox
    return _clip_bbox(
        (
            int(round(x + dx)),
            int(round(y + dy)),
            w,
            h,
        ),
        shape,
    )


def _clip_bbox(bbox: BBox, shape: tuple[int, int]) -> BBox:
    x, y, w, h = bbox
    x1 = int(np.clip(x, 0, max(shape[1] - 1, 0)))
    y1 = int(np.clip(y, 0, max(shape[0] - 1, 0)))
    x2 = int(np.clip(x + w, x1 + 1, shape[1]))
    y2 = int(np.clip(y + h, y1 + 1, shape[0]))
    return (x1, y1, x2 - x1, y2 - y1)
