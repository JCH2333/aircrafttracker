"""Weighted-grid template-matching aircraft tracker.

Splits the aircraft into a 3x2 grid (6 regions), matches each
independently via Sobel contour NCC, then uses weighted-median
voting with physics-constrained outlier rejection.

Key features:
  - Tail regions (top row) have highest weight (2.0) — rigid anchor
  - Belly regions (bottom row) have lowest weight (0.5) — often occluded
  - Physics constraint: displacements far from predicted position are rejected
  - Vertical pole detection: column displacement mismatch triggers column exclusion
  - Coasting: when <2 regions valid, use velocity prediction
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class TemplateTracker:
    """3x2 weighted-grid contour tracker with physics constraints."""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.templates: list[np.ndarray] = []       # blurred edge map per region
        self.template_raws: list[np.ndarray] = []    # source grayscale per region
        self.template_bbox: tuple[int, int, int, int] | None = None
        self.current_centroid: tuple[float, float] | None = None
        self.last_match_score: float = 0.0
        self.frames_since_detect: int = 0
        self._frames_since_update: int = 0

        # Velocity tracking
        self._vx: float = 0.0
        self._vy: float = 0.0

        # Grid
        self._rows: int = config.grid_rows
        self._cols: int = config.grid_cols
        self._weights: list[float] = list(config.grid_weights)

        # Edge
        self._canny_low: int = config.canny_low_threshold
        self._canny_high: int = config.canny_high_threshold
        self._edge_sigma: float = config.edge_blur_sigma

        # Config
        self._velocity_alpha: float = config.template_velocity_alpha
        self._base_margin: int = config.template_search_margin
        self._match_threshold: float = config.template_match_threshold
        self._update_alpha: float = config.template_update_alpha
        self._max_jump_factor: float = config.template_max_jump_factor
        self._quality_score: float = config.template_quality_score
        self._update_interval: int = config.template_update_interval
        self._pole_threshold: float = config.pole_variance_threshold

        # Transition
        self._transition: dict | None = None
        self._transition_frames: int = config.transition_frames
        self._transition_threshold: float = config.transition_threshold

    # ── public ─────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        return self.current_centroid is not None

    @property
    def match_quality(self) -> float:
        return self.last_match_score

    @property
    def velocity(self) -> tuple[float, float]:
        return (self._vx, self._vy)

    def init_from_detection(
        self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> None:
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        patch = gray[y : y + h, x : x + w]

        target_cx = x + w / 2.0
        target_cy = y + h / 2.0

        # Start transition if jump is large
        if self.current_centroid is not None and self.frames_since_detect > 0:
            jump = np.sqrt((target_cx - self.current_centroid[0]) ** 2 +
                           (target_cy - self.current_centroid[1]) ** 2)
            if jump > self._transition_threshold:
                self._transition = {
                    "start_cx": self.current_centroid[0],
                    "start_cy": self.current_centroid[1],
                    "target_cx": target_cx, "target_cy": target_cy,
                    "total": self._transition_frames, "frame": 0,
                    "bbox": (x, y, w, h),
                    "gray": gray,
                }
                return

        self._init_regions(patch, gray[y:y+h, x:x+w], (x, y, w, h), target_cx, target_cy)

    def _init_regions(self, patch, raw_patch, bbox, cx, cy):
        """Split patch into grid regions and compute templates."""
        self.templates.clear()
        self.template_raws.clear()
        ph, pw = patch.shape

        for row in range(self._rows):
            for col in range(self._cols):
                y0 = row * ph // self._rows
                y1 = (row + 1) * ph // self._rows
                x0 = col * pw // self._cols
                x1 = (col + 1) * pw // self._cols
                r_patch = patch[y0:y1, x0:x1]
                r_raw = raw_patch[y0:y1, x0:x1]
                self.templates.append(self._contour_image(r_patch))
                self.template_raws.append(r_raw)

        self.template_bbox = bbox
        self.current_centroid = (cx, cy)
        self.last_match_score = 1.0
        self.frames_since_detect = 0
        self._frames_since_update = 0
        self._vx = 0.0
        self._vy = 0.0
        self._transition = None

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        if self._transition is not None:
            return self._update_transition()

        if not self.templates or self.template_bbox is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        tx_full, ty_full, tw, th = self.template_bbox

        # Predict
        cx_pred = self.current_centroid[0] + self._vx
        cy_pred = self.current_centroid[1] + self._vy

        # Adaptive margin
        speed = np.sqrt(self._vx ** 2 + self._vy ** 2)
        margin = int(np.clip(max(self._base_margin // 4, speed * 3.0),
                             self._base_margin // 4, self._base_margin * 2))

        max_jump = max(speed * self._max_jump_factor, self._base_margin * 0.5)

        # Search region in full-frame coords
        t_x1 = max(0, int(cx_pred - tw / 2.0) - margin)
        t_y1 = max(0, int(cy_pred - th / 2.0) - margin)
        t_x2 = min(fw, int(cx_pred + tw / 2.0) + margin)
        t_y2 = min(fh, int(cy_pred + th / 2.0) + margin)

        if t_x2 - t_x1 < tw or t_y2 - t_y1 < th:
            return None

        # Match each region in its own LOCAL search window
        region_dx: list[float] = []
        region_dy: list[float] = []
        region_scores: list[float] = []
        region_valid: list[bool] = []

        for i, tmpl in enumerate(self.templates):
            th_i, tw_i = tmpl.shape
            if th_i < 4 or tw_i < 4:
                region_valid.append(False)
                region_dx.append(0.0); region_dy.append(0.0); region_scores.append(0.0)
                continue

            row = i // self._cols
            col = i % self._cols

            # Expected region center in full-frame coords (based on prediction)
            exp_fx = (cx_pred - tw / 2.0) + col * tw / self._cols + tw / (self._cols * 2)
            exp_fy = (cy_pred - th / 2.0) + row * th / self._rows + th / (self._rows * 2)

            # Local search window around expected position
            local_margin = margin
            sx = max(0, int(exp_fx - tw_i / 2.0) - local_margin)
            sy = max(0, int(exp_fy - th_i / 2.0) - local_margin)
            ex = min(fw, int(exp_fx + tw_i / 2.0) + local_margin)
            ey = min(fh, int(exp_fy + th_i / 2.0) + local_margin)
            if ex - sx < tw_i or ey - sy < th_i:
                region_valid.append(False)
                region_dx.append(0.0); region_dy.append(0.0); region_scores.append(0.0)
                continue

            local_gray = gray[sy:ey, sx:ex]
            local_contour = self._contour_image(local_gray)

            result = cv2.matchTemplate(local_contour, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            score = float(max_val)

            # Matched center in full-frame coords
            match_fx = sx + max_loc[0] + tw_i / 2.0
            match_fy = sy + max_loc[1] + th_i / 2.0

            drx = match_fx - exp_fx
            dry = match_fy - exp_fy
            jump = np.sqrt(drx**2 + dry**2)

            if jump > max_jump and self.frames_since_detect > 0:
                region_valid.append(False)
            elif score < self._match_threshold:
                region_valid.append(False)
            else:
                region_valid.append(True)

            region_dx.append(drx)
            region_dy.append(dry)
            region_scores.append(score)

        # Pole detection: compare left vs right column displacement
        pole_col = self._detect_pole(region_dx, region_valid)
        weights = list(self._weights)
        if pole_col is not None:
            for i in range(len(weights)):
                if i % self._cols == pole_col:
                    weights[i] = 0.0

        # Weighted median
        valid_dx = []
        valid_dy = []
        valid_w = []
        for i in range(len(region_dx)):
            if region_valid[i] and weights[i] > 0:
                valid_dx.append(region_dx[i])
                valid_dy.append(region_dy[i])
                valid_w.append(weights[i])

        if len(valid_dx) >= 2:
            dx = _weighted_median(valid_dx, valid_w)
            dy = _weighted_median(valid_dy, valid_w)
            quality_ok = np.mean(region_scores) >= self._quality_score
            self.last_match_score = float(np.mean([s for j, s in enumerate(region_scores) if region_valid[j]] or [0]))
        else:
            # Coast: use predicted position
            dx = cx_pred - self.current_centroid[0]
            dy = cy_pred - self.current_centroid[1]
            quality_ok = False
            self.last_match_score = self.last_match_score * 0.9

        # Update centroid
        matched_cx = self.current_centroid[0] + dx
        matched_cy = self.current_centroid[1] + dy

        if quality_ok:
            actual_dx = matched_cx - self.current_centroid[0]
            actual_dy = matched_cy - self.current_centroid[1]
            self._vx = self._velocity_alpha * actual_dx + (1 - self._velocity_alpha) * self._vx
            self._vy = self._velocity_alpha * actual_dy + (1 - self._velocity_alpha) * self._vy

        self.current_centroid = (matched_cx, matched_cy)

        # Update template periodically
        self._frames_since_update += 1
        tx = max(0, min(fw - tw, int(matched_cx - tw / 2.0)))
        ty = max(0, min(fh - th, int(matched_cy - th / 2.0)))

        if quality_ok and self._frames_since_update >= self._update_interval:
            new_patch = gray[ty:ty+th, tx:tx+tw]
            self._refresh_templates(new_patch)
            self._frames_since_update = 0

        self.template_bbox = (tx, ty, tw, th)
        self.frames_since_detect += 1
        return self.current_centroid

    def needs_redetection(self) -> bool:
        if self._transition is not None:
            return False
        return self.last_match_score < self.config.template_redetect_score

    # ── internals ──────────────────────────────────────────────

    def _contour_image(self, gray: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude -> GaussianBlur -> normalize."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = cv2.GaussianBlur(mag, (0, 0), self._edge_sigma)
        mx = mag.max()
        if mx > 1e-6:
            mag /= mx
        return mag

    def _detect_pole(self, region_dx, region_valid) -> int | None:
        """Detect which column (0=left, 1=right) is occluded by a vertical pole.
        Returns None if no pole detected."""
        left_vals = []
        right_vals = []
        for i in range(len(region_dx)):
            if region_valid[i]:
                (left_vals if i % self._cols == 0 else right_vals).append(region_dx[i])
        if len(left_vals) >= 2 and len(right_vals) >= 2:
            lm = np.median(left_vals)
            rm = np.median(right_vals)
            mad_l = np.median(np.abs(np.array(left_vals) - lm)) / 0.6745 + 1e-6
            mad_r = np.median(np.abs(np.array(right_vals) - rm)) / 0.6745 + 1e-6
            diff = abs(lm - rm)
            if diff > self._pole_threshold * max(mad_l, mad_r):
                return 0 if mad_l > mad_r else 1
        return None

    def _refresh_templates(self, patch: np.ndarray):
        """Update region templates from a new patch."""
        old_templates = self.templates
        old_raws = self.template_raws
        self.templates = []
        self.template_raws = []
        ph, pw = patch.shape
        for row in range(self._rows):
            for col in range(self._cols):
                y0 = row * ph // self._rows
                y1 = (row + 1) * ph // self._rows
                x0 = col * pw // self._cols
                x1 = (col + 1) * pw // self._cols
                r_patch = patch[y0:y1, x0:x1]
                idx = row * self._cols + col
                if idx < len(old_raws) and old_raws[idx].shape == r_patch.shape:
                    r_patch = cv2.addWeighted(old_raws[idx], 1.0 - self._update_alpha, r_patch, self._update_alpha, 0)
                self.template_raws.append(r_patch)
                self.templates.append(self._contour_image(r_patch))

    def _update_transition(self) -> tuple[float, float]:
        t = self._transition
        t["frame"] += 1
        progress = min(1.0, t["frame"] / t["total"])
        ease = 1.0 - (1.0 - progress) ** 3
        cx = t["start_cx"] + (t["target_cx"] - t["start_cx"]) * ease
        cy = t["start_cy"] + (t["target_cy"] - t["start_cy"]) * ease
        if progress >= 1.0:
            x, y, w, h = t["bbox"]
            gray = t["gray"]
            patch = gray[y:y+h, x:x+w]
            self._init_regions(patch, patch, (x, y, w, h), t["target_cx"], t["target_cy"])
        else:
            self.current_centroid = (cx, cy)
            self.frames_since_detect += 1
        return self.current_centroid


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Compute weighted median."""
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(weights)
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= total_w / 2.0:
            return v
    return pairs[-1][0] if pairs else 0.0
