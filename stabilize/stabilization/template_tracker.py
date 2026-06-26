"""Template-matching aircraft tracker with boundary-aware cropping.

Uses NCC grayscale template matching with velocity-constrained search,
quality-gated coasting, and adaptive cropping when the aircraft
partially exits the frame.

When the template extends beyond the frame boundary, the visible
portion is cropped and matched independently. The centroid offset
is adjusted to account for the cropped region, allowing the tracker
to follow just the nose/cockpit when the tail is out of frame.
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class TemplateTracker:
    """NCC template matcher with velocity prediction and boundary cropping."""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.template: np.ndarray | None = None       # grayscale aircraft patch
        self.template_bbox: tuple[int, int, int, int] | None = None  # full bbox in frame coords
        self.current_centroid: tuple[float, float] | None = None
        self.last_match_score: float = 0.0
        self.frames_since_detect: int = 0

        # Velocity tracking (EWMA)
        self._vx: float = 0.0
        self._vy: float = 0.0

        # Config shortcuts
        self._velocity_alpha: float = config.template_velocity_alpha
        self._base_margin: int = config.template_search_margin
        self._match_threshold: float = config.template_match_threshold
        self._update_alpha: float = config.template_update_alpha
        self._max_jump_factor: float = config.template_max_jump_factor
        self._quality_score: float = config.template_quality_score

    # ── public API ──────────────────────────────────────────────

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
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Extract template from a detection bounding box."""
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.template = gray[y : y + h, x : x + w].copy()
        self.template_bbox = (x, y, w, h)
        self.current_centroid = (x + w / 2.0, y + h / 2.0)
        self.last_match_score = 1.0
        self.frames_since_detect = 0
        self._vx = 0.0
        self._vy = 0.0
        logger.debug(
            "Template init: %dx%d at (%d,%d), centroid=(%.1f, %.1f)",
            w, h, x, y, self.current_centroid[0], self.current_centroid[1],
        )

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Track aircraft with boundary-aware template matching.

        When the template extends beyond the frame edge, the visible
        portion is cropped and matched. The centroid is adjusted to
        compensate for the cropping offset.
        """
        if self.template is None or self.template_bbox is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        _, _, tw, th = self.template_bbox

        # ── Predict position ──
        cx_pred = self.current_centroid[0] + self._vx
        cy_pred = self.current_centroid[1] + self._vy

        # ── Adaptive search margin ──
        speed = np.sqrt(self._vx ** 2 + self._vy ** 2)
        margin = int(np.clip(
            max(self._base_margin, speed * 3.0),
            self._base_margin, self._base_margin * 3,
        ))

        # ── Check for boundary clipping ──
        tx_full = int(cx_pred - tw / 2.0)
        ty_full = int(cy_pred - th / 2.0)

        # Template region in frame coordinates
        t_x1 = max(0, tx_full)
        t_y1 = max(0, ty_full)
        t_x2 = min(fw, tx_full + tw)
        t_y2 = min(fh, ty_full + th)

        # Visible portion of template
        crop_x1 = t_x1 - tx_full  # offset into template
        crop_y1 = t_y1 - ty_full
        crop_x2 = tw - (tx_full + tw - t_x2)
        crop_y2 = th - (ty_full + th - t_y2)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w < 10 or crop_h < 10:
            return None  # too little visible

        # Crop template to visible portion
        if crop_w < tw or crop_h < th:
            vis_template = self.template[crop_y1:crop_y2, crop_x1:crop_x2]
            # Centroid of the VISIBLE portion relative to full template center
            vis_cx_offset = (crop_x1 + crop_x2) / 2.0 - tw / 2.0
            vis_cy_offset = (crop_y1 + crop_y2) / 2.0 - th / 2.0
        else:
            vis_template = self.template
            vis_cx_offset = 0.0
            vis_cy_offset = 0.0

        # ── Search region ──
        sx = max(0, t_x1 - margin)
        sy = max(0, t_y1 - margin)
        ex = min(fw, t_x2 + margin)
        ey = min(fh, t_y2 + margin)

        # Ensure search region is at least template-sized
        if (ex - sx < crop_w) or (ey - sy < crop_h):
            sx = max(0, min(sx, fw - crop_w))
            sy = max(0, min(sy, fh - crop_h))
            ex = min(fw, sx + crop_w)
            ey = min(fh, sy + crop_h)
            if (ex - sx < crop_w) or (ey - sy < crop_h):
                return None

        # ── Template matching ──
        search = gray[sy:ey, sx:ex]
        result = cv2.matchTemplate(
            search, vis_template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        self.last_match_score = float(max_val)

        if max_val < self._match_threshold:
            logger.debug("Match low: %.4f < %.2f", max_val, self._match_threshold)
            return None

        # Match position of visible template center
        matched_vis_cx = sx + max_loc[0] + crop_w / 2.0
        matched_vis_cy = sy + max_loc[1] + crop_h / 2.0

        # Convert to full template centroid (accounting for crop offset)
        matched_cx = matched_vis_cx - vis_cx_offset
        matched_cy = matched_vis_cy - vis_cy_offset

        # ── Jump detection ──
        jump_dx = matched_cx - cx_pred
        jump_dy = matched_cy - cy_pred
        jump_dist = np.sqrt(jump_dx ** 2 + jump_dy ** 2)
        max_jump = max(speed * self._max_jump_factor, self._base_margin * 0.5)

        quality_ok = max_val >= self._quality_score

        if jump_dist > max_jump and self.frames_since_detect > 0:
            logger.debug("Jump rejected: %.0fpx > %.0fpx", jump_dist, max_jump)
            quality_ok = False

        if not quality_ok:
            matched_cx = cx_pred
            matched_cy = cy_pred
        else:
            actual_dx = matched_cx - self.current_centroid[0]
            actual_dy = matched_cy - self.current_centroid[1]
            self._vx = (
                self._velocity_alpha * actual_dx
                + (1 - self._velocity_alpha) * self._vx
            )
            self._vy = (
                self._velocity_alpha * actual_dy
                + (1 - self._velocity_alpha) * self._vy
            )

        self.current_centroid = (matched_cx, matched_cy)

        # ── Update template ──
        tx = int(matched_cx - tw / 2.0)
        ty = int(matched_cy - th / 2.0)
        tx = max(0, min(fw - tw, tx))
        ty = max(0, min(fh - th, ty))

        if quality_ok:
            new_patch = gray[ty : ty + th, tx : tx + tw]
            if new_patch.shape == self.template.shape:
                self.template = cv2.addWeighted(
                    self.template, 1.0 - self._update_alpha,
                    new_patch, self._update_alpha, 0,
                )

        self.template_bbox = (tx, ty, tw, th)
        self.frames_since_detect += 1
        return self.current_centroid

    def needs_redetection(self) -> bool:
        return self.last_match_score < self.config.template_redetect_score
