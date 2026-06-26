"""Feature-point optical flow aircraft tracker.

Tracks aircraft by incrementally maintaining Shi-Tomasi corner points
across frames via Lucas-Kanade optical flow. Points are NEVER fully
replaced — new points are added alongside surviving old ones to fill
gaps from occlusion/tracking loss.

Detection is used only for:
  - First-frame initialization
  - Periodic point replenishment (add new points in uncovered areas)
  - The centroid position is ALWAYS derived from optical flow deltas,
    never reset from detection bbox center.
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class FeatureTracker:
    """Incremental feature-point tracker for rigid aircraft.

    Maintains a set of feature points across frames. Points are
    tracked with Lucas-Kanade optical flow. Lost/inlier points are
    removed. New points are added from detection bbox as needed.

    Key invariant: centroid position comes ONLY from accumulated
    optical flow deltas. Detection bbox is used for point extraction
    region guidance, never for centroid reset.
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.prev_points: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.current_centroid: tuple[float, float] | None = None
        self._min_points = config.feature_redetect_min_points
        self._max_points = config.feature_max_corners * 2  # allow growth
        self.frames_since_replenish = 0

        # Lucas-Kanade parameters
        ws = config.lk_win_size
        self.lk_params = dict(
            winSize=ws,
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.lk_max_iter,
                config.lk_epsilon,
            ),
        )

        # Shi-Tomasi parameters
        self.feature_params = dict(
            maxCorners=config.feature_max_corners,
            qualityLevel=config.feature_quality,
            minDistance=config.feature_min_distance,
            blockSize=7,
        )

    # ── public API ──────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        return self.current_centroid is not None

    @property
    def point_count(self) -> int:
        return len(self.prev_points) if self.prev_points is not None else 0

    def init_from_detection(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """First-time initialization from a detection bbox.

        Extracts Shi-Tomasi corners and sets the initial centroid
        to the bbox center. Should only be called once at start.

        Args:
            frame_bgr: Current frame (uint8 BGR).
            bbox: Detection bounding box (x, y, w, h).
        """
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray

        # Extract points from inner region (avoid background edges)
        self.prev_points = _extract_points(
            gray, bbox, self.config.feature_bbox_margin, self.feature_params
        )
        self.current_centroid = (x + w / 2.0, y + h / 2.0)
        logger.debug(
            "Tracker init: %d points, centroid=(%.1f, %.1f)",
            self.point_count, self.current_centroid[0], self.current_centroid[1],
        )

    def replenish_points(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Add fresh feature points from a detection bbox.

        Merges new points with existing tracked points:
        - Points already tracked are kept (not replaced).
        - New points are added only in areas without existing coverage.
        - Centroid is NOT reset.

        Args:
            frame_bgr: Current frame (uint8 BGR).
            bbox: Detection bounding box (x, y, w, h).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Extract candidate new points
        candidates = _extract_points(
            gray, bbox, self.config.feature_bbox_margin,
            {**self.feature_params, "maxCorners": self.config.feature_max_corners // 2},
        )

        if candidates is None or len(candidates) == 0:
            return

        # Merge with existing points — exclude candidates too close to existing
        if self.prev_points is not None and len(self.prev_points) > 0:
            existing = self.prev_points.reshape(-1, 2)
            new_pts = candidates.reshape(-1, 2)

            # Simple distance-based dedup
            keep = np.ones(len(new_pts), dtype=bool)
            min_dist = self.config.feature_min_distance
            for i, pt in enumerate(new_pts):
                dists = np.linalg.norm(existing - pt, axis=1)
                if dists.min() < min_dist:
                    keep[i] = False

            new_filtered = new_pts[keep].reshape(-1, 1, 2)
            if len(new_filtered) > 0:
                self.prev_points = np.vstack([self.prev_points, new_filtered])
        else:
            self.prev_points = candidates

        # Trim if too many points
        if len(self.prev_points) > self._max_points:
            indices = np.random.choice(
                len(self.prev_points), self._max_points, replace=False
            )
            self.prev_points = self.prev_points[indices]

        # prev_gray stays as the CURRENT frame since we just added points here
        self.prev_gray = gray
        self.frames_since_replenish = 0

        logger.debug(
            "Replenish: +%d points -> %d total",
            len(candidates) if candidates is not None else 0,
            self.point_count,
        )

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Track aircraft to new frame via optical flow.

        Args:
            frame_bgr: Current frame (uint8 BGR).

        Returns:
            (cx, cy) updated centroid, or None if tracking lost.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_points is None or self.prev_gray is None:
            return None
        if len(self.prev_points) < 3:
            return None

        # Lucas-Kanade optical flow
        new_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if new_points is None:
            return None

        # Filter to successfully tracked points
        status = status.ravel()
        ok_old = self.prev_points[status == 1]
        ok_new = new_points[status == 1]

        if len(ok_new) < 3:
            logger.debug("OF lost: %d points", len(ok_new))
            return None

        # Robust displacement from tracked points
        deltas = ok_new - ok_old
        dx, dy, inliers = _robust_displacement(deltas)

        # Suppress unreasonable jumps (> 200 px in one frame is non-physical)
        max_jump = 200.0
        if abs(dx) > max_jump or abs(dy) > max_jump:
            logger.debug("OF jump suppressed: dx=%.1f, dy=%.1f", dx, dy)
            dx = np.clip(dx, -max_jump, max_jump)
            dy = np.clip(dy, -max_jump, max_jump)

        # Update centroid (accumulate OF delta — never reset from detection)
        cx, cy = self.current_centroid
        self.current_centroid = (cx + dx, cy + dy)

        # Keep inlier points for next frame + update prev_gray
        n_inliers = inliers.sum()
        if n_inliers >= 3:
            self.prev_points = ok_new[inliers].reshape(-1, 1, 2)
        else:
            self.prev_points = ok_new.reshape(-1, 1, 2)

        self.prev_gray = gray
        self.frames_since_replenish += 1
        return self.current_centroid

    def needs_replenish(self) -> bool:
        """Check if new feature points should be added."""
        n = self.point_count
        return n < self._min_points


def _extract_points(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_fraction: float,
    params: dict,
) -> np.ndarray | None:
    """Extract Shi-Tomasi corners within the inner region of a bbox.

    The margin excludes the bbox border to avoid background features.
    Falls back to a uniform grid if too few corners are found.
    """
    x, y, w, h = bbox
    mx = int(w * margin_fraction)
    my = int(h * margin_fraction)

    inner_x = max(0, x + mx)
    inner_y = max(0, y + my)
    inner_w = min(w - 2 * mx, gray.shape[1] - inner_x)
    inner_h = min(h - 2 * my, gray.shape[0] - inner_y)

    mask = np.zeros_like(gray)
    if inner_w > 20 and inner_h > 20:
        mask[inner_y : inner_y + inner_h, inner_x : inner_x + inner_w] = 255
    else:
        mask[y : y + h, x : x + w] = 255

    points = cv2.goodFeaturesToTrack(gray, mask=mask, **params)

    if points is None or len(points) < 4:
        points = _generate_grid_points(bbox, 8)

    return points


def _robust_displacement(
    deltas: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Compute robust displacement with MAD-based outlier rejection.

    Args:
        deltas: (N, 1, 2) array of per-point (dx, dy).

    Returns:
        (dx, dy, inlier_mask)
    """
    dx = deltas[:, 0, 0]
    dy = deltas[:, 0, 1]

    med_dx = np.median(dx)
    med_dy = np.median(dy)

    mad_dx = np.median(np.abs(dx - med_dx))
    mad_dy = np.median(np.abs(dy - med_dy))

    sigma_dx = mad_dx / 0.6745 + 1e-6
    sigma_dy = mad_dy / 0.6745 + 1e-6

    inliers = (
        (np.abs(dx - med_dx) < 3.0 * sigma_dx)
        & (np.abs(dy - med_dy) < 3.0 * sigma_dy)
    )

    if inliers.sum() >= 3:
        final_dx = np.median(dx[inliers])
        final_dy = np.median(dy[inliers])
    else:
        final_dx, final_dy = med_dx, med_dy

    return float(final_dx), float(final_dy), inliers


def _generate_grid_points(
    bbox: tuple[int, int, int, int], step: int = 8
) -> np.ndarray | None:
    """Fallback: uniform grid of points within bbox."""
    x, y, w, h = bbox
    points = []
    for py in range(y + step, y + h - step, step):
        for px in range(x + step, x + w - step, step):
            points.append([[px, py]])
    if not points:
        return None
    return np.array(points, dtype=np.float32)
