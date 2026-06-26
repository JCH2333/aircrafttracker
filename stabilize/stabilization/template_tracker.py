"""NCC template-matching aircraft tracker.

Tracks the aircraft by matching a template (grayscale aircraft patch)
against a search region in each new frame. Much more stable than
optical flow for rigid objects — no feature point extraction,
no outlier filtering needed.

Detection is used only for initialization and fallback recovery.
The centroid position comes exclusively from template matching.
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class TemplateTracker:
    """Tracks aircraft via NCC template matching.

    Maintains a template (grayscale patch of the aircraft) and searches
    for it in each new frame using cv2.matchTemplate with normalized
    cross-correlation (TM_CCOEFF_NORMED).

    The template is updated each frame to adapt to gradual appearance
    changes (lighting, angle). The search region is centered on the
    last known position with a configurable margin.

    Detection provides initialization and recovery from low-confidence
    matches (e.g., under occlusion).
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.template: np.ndarray | None = None     # grayscale aircraft patch
        self.template_bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)
        self.current_centroid: tuple[float, float] | None = None
        self.last_match_score: float = 0.0
        self.frames_since_detect: int = 0
        self._search_margin = config.template_search_margin
        self._match_threshold = config.template_match_threshold
        self._update_alpha = config.template_update_alpha

    # ── public API ──────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        return self.current_centroid is not None

    @property
    def match_quality(self) -> float:
        """Last frame's NCC match score (0..1, higher = better)."""
        return self.last_match_score

    def init_from_detection(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Extract template from a detection bounding box.

        Args:
            frame_bgr: Current frame (uint8 BGR).
            bbox: Detection bounding box (x, y, w, h).
        """
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.template = gray[y : y + h, x : x + w].copy()
        self.template_bbox = (x, y, w, h)
        self.current_centroid = (x + w / 2.0, y + h / 2.0)
        self.last_match_score = 1.0
        self.frames_since_detect = 0
        logger.debug(
            "Template init: %dx%d at (%d,%d), centroid=(%.1f, %.1f)",
            w, h, x, y, self.current_centroid[0], self.current_centroid[1],
        )

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Track aircraft to new frame via template matching.

        Args:
            frame_bgr: Current frame (uint8 BGR).

        Returns:
            (cx, cy) updated centroid, or None if match failed.
        """
        if self.template is None or self.template_bbox is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tx, ty, tw, th = self.template_bbox
        m = self._search_margin

        # Search region around last known position
        sx = max(0, tx - m)
        sy = max(0, ty - m)
        ex = min(gray.shape[1], tx + tw + m)
        ey = min(gray.shape[0], ty + th + m)

        if ex <= sx or ey <= sy:
            return None

        search = gray[sy:ey, sx:ex]

        # Normalized cross-correlation
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        self.last_match_score = float(max_val)

        if max_val < self._match_threshold:
            logger.debug("Template match low: %.4f < %.2f", max_val, self._match_threshold)
            return None

        # Convert from search-region coords to full-frame coords
        new_tx = sx + max_loc[0]
        new_ty = sy + max_loc[1]

        # Update centroid
        new_cx = new_tx + tw / 2.0
        new_cy = new_ty + th / 2.0
        self.current_centroid = (new_cx, new_cy)

        # Update template (blend with new patch for gradual adaptation)
        new_template = gray[new_ty : new_ty + th, new_tx : new_tx + tw]
        if new_template.shape == self.template.shape:
            alpha = self._update_alpha
            self.template = cv2.addWeighted(
                self.template, 1.0 - alpha, new_template, alpha, 0
            )
        self.template_bbox = (new_tx, new_ty, tw, th)
        self.frames_since_detect += 1

        return self.current_centroid

    def needs_redetection(self) -> bool:
        """Check if detection recovery is needed.

        Only triggers on match quality degradation (occlusion, extreme
        appearance change). Does NOT trigger on a timer — the template
        self-updates each frame, so periodic re-detection is unnecessary
        and causes centroid jumps.
        """
        return self.last_match_score < self.config.template_redetect_score

    def refresh_template(self, frame_bgr: np.ndarray) -> None:
        """Update template from current frame at current position.

        Use when detection is unavailable but the tracker has a good
        position — refreshes the template to adapt to appearance changes
        without disrupting the centroid.
        """
        if self.template_bbox is None:
            return
        tx, ty, tw, th = self.template_bbox
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        new_patch = gray[ty : ty + th, tx : tx + tw]
        if new_patch.shape == self.template.shape:
            self.template = new_patch.copy()
            self.frames_since_detect = 0
            logger.debug("Template refreshed from current position")
