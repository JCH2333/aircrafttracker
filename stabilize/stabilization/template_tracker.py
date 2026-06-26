"""NCC template-matching aircraft tracker with velocity-constrained search.

Tracks the aircraft by matching a template against a search region
in each new frame. Search region is centered on the *predicted* position
(based on recent velocity), preventing the matcher from jumping to
foreground objects when the aircraft is blurred or occluded.

Detection is used only for initialization and fallback recovery.
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class TemplateTracker:
    """Velocity-constrained NCC template matching tracker.

    Maintains a velocity estimate (EWMA of frame-to-frame displacement)
    to predict the aircraft position and constrain the search region.
    This prevents foreground contamination when the aircraft is blurred
    or partially occluded — the matcher can't jump to objects far from
    the physically-expected position.
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.template: np.ndarray | None = None
        self.template_bbox: tuple[int, int, int, int] | None = None
        self.current_centroid: tuple[float, float] | None = None
        self.last_match_score: float = 0.0
        self.frames_since_detect: int = 0

        # Velocity tracking (EWMA)
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._velocity_alpha: float = config.template_velocity_alpha
        self._base_margin: int = config.template_search_margin
        self._match_threshold: float = config.template_match_threshold
        self._update_alpha: float = config.template_update_alpha
        self._max_jump_factor: float = config.template_max_jump_factor

    # ── public API ──────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        return self.current_centroid is not None

    @property
    def match_quality(self) -> float:
        return self.last_match_score

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (px/frame)."""
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
        """Track aircraft with velocity-constrained template matching.

        1. Predict position from velocity
        2. Search around predicted position (adaptive margin)
        3. Reject matches that jump too far from prediction
        """
        if self.template is None or self.template_bbox is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, _, tw, th = self.template_bbox

        # Predict next position from current position + velocity
        cx_pred = self.current_centroid[0] + self._vx
        cy_pred = self.current_centroid[1] + self._vy

        # Adaptive search margin: at least base_margin, at most 3x base_margin
        speed = np.sqrt(self._vx ** 2 + self._vy ** 2)
        margin = int(np.clip(
            max(self._base_margin, speed * 3.0),
            self._base_margin,
            self._base_margin * 3,
        ))

        # Search region centered on predicted position.
        # Must be at least as large as the template.
        tx_pred = int(cx_pred - tw / 2.0)
        ty_pred = int(cy_pred - th / 2.0)
        sx = max(0, tx_pred - margin)
        sy = max(0, ty_pred - margin)
        ex = min(gray.shape[1], tx_pred + tw + margin)
        ey = min(gray.shape[0], ty_pred + th + margin)

        # Ensure search region is large enough for the template
        if (ex - sx < tw) or (ey - sy < th):
            # Expand to fit template, or fall back to full-frame search
            sx = max(0, min(sx, gray.shape[1] - tw))
            sy = max(0, min(sy, gray.shape[0] - th))
            ex = min(gray.shape[1], sx + tw)
            ey = min(gray.shape[0], sy + th)
            if (ex - sx < tw) or (ey - sy < th):
                return None  # can't fit template anywhere

        search = gray[sy:ey, sx:ex]

        # Template matching (search must be >= template in both dimensions)
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        self.last_match_score = float(max_val)

        if max_val < self._match_threshold:
            logger.debug("Template match low: %.4f < %.2f", max_val, self._match_threshold)
            return None

        # Match position in full-frame coords
        matched_tx = sx + max_loc[0]
        matched_ty = sy + max_loc[1]
        matched_cx = matched_tx + tw / 2.0
        matched_cy = matched_ty + th / 2.0

        # Check for physically impossible jump (foreground contamination)
        jump_dx = matched_cx - cx_pred
        jump_dy = matched_cy - cy_pred
        jump_dist = np.sqrt(jump_dx ** 2 + jump_dy ** 2)
        max_allowed_jump = max(speed * self._max_jump_factor, self._base_margin * 0.5)

        quality_ok = max_val >= self.config.template_quality_score

        if jump_dist > max_allowed_jump and self.frames_since_detect > 0:
            logger.debug(
                "Jump rejected: %.0fpx > %.0fpx, coasting",
                jump_dist, max_allowed_jump,
            )
            quality_ok = False  # force coasting

        if not quality_ok:
            # Low-quality match — coast at current velocity, don't update template
            matched_cx = cx_pred
            matched_cy = cy_pred
            # Velocity unchanged (coasting)

        else:
            # Good match — update velocity from actual displacement
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

        # Update centroid position
        self.current_centroid = (matched_cx, matched_cy)

        # Update template only on good-quality matches
        if quality_ok:
            new_tx = max(0, min(gray.shape[1] - tw, matched_tx))
            new_ty = max(0, min(gray.shape[0] - th, matched_ty))
            new_template = gray[new_ty : new_ty + th, new_tx : new_tx + tw]
            if new_template.shape == self.template.shape:
                self.template = cv2.addWeighted(
                    self.template, 1.0 - self._update_alpha,
                    new_template, self._update_alpha, 0,
                )
            self.template_bbox = (new_tx, new_ty, tw, th)
        else:
            # Update bbox position for search region (template unchanged)
            tx = int(matched_cx - tw / 2.0)
            ty = int(matched_cy - th / 2.0)
            self.template_bbox = (tx, ty, tw, th)

        self.frames_since_detect += 1
        return self.current_centroid

    def needs_redetection(self) -> bool:
        """Check if detection recovery is needed."""
        return self.last_match_score < self.config.template_redetect_score
