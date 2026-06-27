"""Contour-based template-matching aircraft tracker.

Uses Canny edge detection + Gaussian blur to match the aircraft's
CONTOUR SHAPE rather than pixel texture. This makes tracking robust to:
  - Foreground occlusion (different shape = low match score)
  - Blur / defocus (edges soften but contour shape persists)
  - Partial visibility (visible edges still form the shape)
  - Lighting changes (edges are intensity-invariant)

The Gaussian blur converts sparse binary edges into soft "contour bands",
enabling dense NCC matching that focuses on structural shape similarity.

Includes velocity-constrained search, quality-gated coasting, and
boundary-aware cropping for when the aircraft exits the frame.
"""

import logging

import cv2
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class TemplateTracker:
    """Contour-based (edge-blurred) NCC template matching tracker.

    Template = Canny(gray) → GaussianBlur → normalized float32
    Search   = same pipeline on search region
    Match    = TM_CCOEFF_NORMED on these contour-band images
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.template: np.ndarray | None = None       # full aircraft contour
        self.template_raw: np.ndarray | None = None    # full aircraft grayscale
        self.tail_template: np.ndarray | None = None   # tail region contour (top ~40%)
        self.tail_template_raw: np.ndarray | None = None
        self.template_bbox: tuple[int, int, int, int] | None = None
        self.current_centroid: tuple[float, float] | None = None
        self.last_match_score: float = 0.0
        self.frames_since_detect: int = 0

        # Tail config
        self._tail_ratio: float = config.tail_template_ratio
        self._tail_threshold: float = config.tail_disagreement_threshold

        # Velocity tracking (EWMA)
        self._vx: float = 0.0
        self._vy: float = 0.0

        # Edge parameters
        self._canny_low: int = config.canny_low_threshold
        self._canny_high: int = config.canny_high_threshold
        self._edge_sigma: float = config.edge_blur_sigma

        # Smooth transition on detection re-init
        self._transition: dict | None = None  # {start_cx, start_cy, target_cx, target_cy, total, frame}
        self._transition_frames: int = config.transition_frames
        self._transition_threshold: float = config.transition_threshold

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
        """Extract contour template from a detection bounding box.

        If already tracking and the detection position differs significantly,
        starts a smooth transition instead of an instant jump.
        """
        x, y, w, h = bbox
        target_cx = x + w / 2.0
        target_cy = y + h / 2.0

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        patch = gray[y : y + h, x : x + w]
        new_template_raw = patch.copy()
        new_template = self._contour_image(patch)

        # Tail region (upper portion — rigid anchor)
        tail_h = max(10, int(h * self._tail_ratio))
        tail_patch = patch[:tail_h, :]
        new_tail_raw = tail_patch.copy()
        new_tail = self._contour_image(tail_patch)

        # Check if we need a smooth transition
        if self.current_centroid is not None and self.frames_since_detect > 0:
            old_cx, old_cy = self.current_centroid
            jump_dist = np.sqrt(
                (target_cx - old_cx) ** 2 + (target_cy - old_cy) ** 2
            )
            if jump_dist > self._transition_threshold:
                logger.debug(
                    "Smooth transition: %.0fpx over %d frames",
                    jump_dist, self._transition_frames,
                )
                self._transition = {
                    "start_cx": old_cx, "start_cy": old_cy,
                    "target_cx": target_cx, "target_cy": target_cy,
                    "total": self._transition_frames, "frame": 0,
                    "new_template_raw": new_template_raw,
                    "new_template": new_template,
                    "new_tail_raw": new_tail_raw,
                    "new_tail": new_tail,
                    "new_bbox": (x, y, w, h),
                }
                return

        # Direct init (first frame or small jump)
        self.template_raw = new_template_raw
        self.template = new_template
        self.tail_template_raw = new_tail_raw
        self.tail_template = new_tail
        self.template_bbox = (x, y, w, h)
        self.current_centroid = (target_cx, target_cy)
        self.last_match_score = 1.0
        self.frames_since_detect = 0
        self._vx = 0.0
        self._vy = 0.0
        self._transition = None
        logger.debug(
            "Template init: %dx%d at (%d,%d), centroid=(%.1f, %.1f)",
            w, h, x, y, target_cx, target_cy,
        )

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Track aircraft via contour-based template matching."""
        # ── Active transition: skip matching, just interpolate ──
        if self._transition is not None:
            return self._update_transition()

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

        # ── Boundary clipping ──
        tx_full = int(cx_pred - tw / 2.0)
        ty_full = int(cy_pred - th / 2.0)
        t_x1, t_y1 = max(0, tx_full), max(0, ty_full)
        t_x2, t_y2 = min(fw, tx_full + tw), min(fh, ty_full + th)
        crop_x1, crop_y1 = t_x1 - tx_full, t_y1 - ty_full
        crop_x2 = tw - (tx_full + tw - t_x2)
        crop_y2 = th - (ty_full + th - t_y2)
        crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1

        if crop_w < 10 or crop_h < 10:
            return None

        # Crop template (edge map) to visible portion
        if crop_w < tw or crop_h < th:
            vis_template = self.template[crop_y1:crop_y2, crop_x1:crop_x2]
            vis_cx_offset = (crop_x1 + crop_x2) / 2.0 - tw / 2.0
            vis_cy_offset = (crop_y1 + crop_y2) / 2.0 - th / 2.0
        else:
            vis_template = self.template
            vis_cx_offset = 0.0
            vis_cy_offset = 0.0

        # ── Search region ──
        sx, sy = max(0, t_x1 - margin), max(0, t_y1 - margin)
        ex, ey = min(fw, t_x2 + margin), min(fh, t_y2 + margin)
        if (ex - sx < crop_w) or (ey - sy < crop_h):
            sx = max(0, min(sx, fw - crop_w))
            sy = max(0, min(sy, fh - crop_h))
            ex, ey = min(fw, sx + crop_w), min(fh, sy + crop_h)
            if (ex - sx < crop_w) or (ey - sy < crop_h):
                return None

        # ── Full template matching (downscale for speed) ──
        scale = self.config.match_downscale
        search_patch = gray[sy:ey, sx:ex]
        if scale < 1.0:
            small_search = cv2.resize(search_patch, None, fx=scale, fy=scale)
            small_tmpl = cv2.resize(vis_template, None, fx=scale, fy=scale)
            sc = self._contour_image(small_search)
            result = cv2.matchTemplate(sc, small_tmpl, cv2.TM_CCOEFF_NORMED)
            _, full_score, _, full_loc = cv2.minMaxLoc(result)
            full_loc = (int(full_loc[0] / scale), int(full_loc[1] / scale))
        else:
            sc = self._contour_image(search_patch)
            result = cv2.matchTemplate(sc, vis_template, cv2.TM_CCOEFF_NORMED)
            _, full_score, _, full_loc = cv2.minMaxLoc(result)
        full_score = float(full_score)
        full_dx = sx + full_loc[0] + crop_w / 2.0 - vis_cx_offset - self.current_centroid[0]
        full_dy = sy + full_loc[1] + crop_h / 2.0 - vis_cy_offset - self.current_centroid[1]

        # ── Tail template matching ──
        tail_dx = full_dx
        tail_dy = full_dy
        tail_score = 0.0
        use_tail = False

        if self.tail_template is not None:
            th_t, tw_t = self.tail_template.shape
            if crop_h >= th_t:
                # Tail search region: narrower vertically, centered on upper portion
                tail_cy_pred = cy_pred - th * 0.3  # tail is above center
                tsy = max(0, int(tail_cy_pred - th_t / 2.0) - margin)
                tey = min(fh, int(tail_cy_pred + th_t / 2.0) + margin)
                tsx, tex = sx, ex  # same horizontal
                if tey - tsy >= th_t and tex - tsx >= tw_t:
                    tail_search = gray[tsy:tey, tsx:tex]
                    scale = self.config.match_downscale
                    if scale < 1.0:
                        ts_small = cv2.resize(tail_search, None, fx=scale, fy=scale)
                        tt_small = cv2.resize(self.tail_template, None, fx=scale, fy=scale)
                        tc = self._contour_image(ts_small)
                        tr = cv2.matchTemplate(tc, tt_small, cv2.TM_CCOEFF_NORMED)
                        _, tail_score, _, tloc = cv2.minMaxLoc(tr)
                        tl = (int(tloc[0] / scale), int(tloc[1] / scale))
                    else:
                        tc = self._contour_image(tail_search)
                        tr = cv2.matchTemplate(tc, self.tail_template, cv2.TM_CCOEFF_NORMED)
                        _, tail_score, _, tl = cv2.minMaxLoc(tr)
                    tail_score = float(tail_score)
                    # Tail centroid displacement
                    tail_match_cx = tsx + tl[0] + tw_t / 2.0
                    tail_match_cy = tsy + tl[1] + th_t / 2.0
                    # Full centroid = tail match + offset (tail center -> full center)
                    tail_full_cx = tail_match_cx  # tail centered horizontally = aircraft centered
                    tail_full_cy = tail_match_cy + th * (0.5 - self._tail_ratio / 2.0)
                    tail_dx = tail_full_cx - self.current_centroid[0]
                    tail_dy = tail_full_cy - self.current_centroid[1]

        # ── Choose between full and tail ──
        full_ok = full_score >= self._match_threshold
        tail_ok = tail_score >= self._match_threshold
        disagreement = abs(full_dx - tail_dx) + abs(full_dy - tail_dy)

        if full_ok and tail_ok and disagreement < self._tail_threshold:
            # Normal: full template (more pixels, more stable)
            self.last_match_score = full_score
            matched_cx = self.current_centroid[0] + full_dx
            matched_cy = self.current_centroid[1] + full_dy
        elif tail_ok and (not full_ok or disagreement >= self._tail_threshold):
            # Occlusion: tail anchor (rigid, above obstacles)
            self.last_match_score = tail_score
            matched_cx = self.current_centroid[0] + tail_dx
            matched_cy = self.current_centroid[1] + tail_dy
            use_tail = True
            logger.debug("Tail anchor: tail=%.3f full=%.3f diff=%.1f", tail_score, full_score, disagreement)
        elif full_ok:
            # Only full template valid
            self.last_match_score = full_score
            matched_cx = self.current_centroid[0] + full_dx
            matched_cy = self.current_centroid[1] + full_dy
        else:
            # Both bad — return None to trigger fallback
            logger.debug("Dual match low: full=%.3f tail=%.3f", full_score, tail_score)
            return None

        # ── Jump detection ──
        jump_dx = matched_cx - cx_pred
        jump_dy = matched_cy - cy_pred
        jump_dist = np.sqrt(jump_dx ** 2 + jump_dy ** 2)
        max_jump = max(speed * self._max_jump_factor, self._base_margin * 0.5)
        quality_ok = self.last_match_score >= self._quality_score

        if jump_dist > max_jump and self.frames_since_detect > 0:
            logger.debug("Jump rejected: %.0fpx > %.0fpx", jump_dist, max_jump)
            quality_ok = False

        if not quality_ok:
            matched_cx, matched_cy = cx_pred, cy_pred
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
            if self.template_raw is not None and new_patch.shape == self.template_raw.shape:
                self.template_raw = cv2.addWeighted(
                    self.template_raw, 1.0 - self._update_alpha,
                    new_patch, self._update_alpha, 0,
                )
                self.template = self._contour_image(self.template_raw)
            # Also update tail template
            if self.tail_template_raw is not None:
                tail_h = max(10, int(th * self._tail_ratio))
                tail_patch = new_patch[:tail_h, :]
                if tail_patch.shape == self.tail_template_raw.shape:
                    self.tail_template_raw = cv2.addWeighted(
                        self.tail_template_raw, 1.0 - self._update_alpha,
                        tail_patch, self._update_alpha, 0,
                    )
                    self.tail_template = self._contour_image(self.tail_template_raw)

        self.template_bbox = (tx, ty, tw, th)
        self.frames_since_detect += 1
        return self.current_centroid

    def _update_transition(self) -> tuple[float, float]:
        """Advance smooth transition by one frame. Returns interpolated centroid."""
        t = self._transition
        t["frame"] += 1
        progress = min(1.0, t["frame"] / t["total"])
        # Ease-out cubic
        ease = 1.0 - (1.0 - progress) ** 3
        cx = t["start_cx"] + (t["target_cx"] - t["start_cx"]) * ease
        cy = t["start_cy"] + (t["target_cy"] - t["start_cy"]) * ease

        if progress >= 1.0:
            # Finalize: apply detection template
            nt = self._transition
            self.template_raw = nt["new_template_raw"]
            self.template = nt["new_template"]
            self.tail_template_raw = nt["new_tail_raw"]
            self.tail_template = nt["new_tail"]
            self.template_bbox = nt["new_bbox"]
            self.current_centroid = (t["target_cx"], t["target_cy"])
            self._vx = 0.0
            self._vy = 0.0
            self.frames_since_detect = 0
            self.last_match_score = 1.0
            self._transition = None
            logger.debug("Transition complete")
        else:
            self.current_centroid = (cx, cy)
            self.frames_since_detect += 1

        return self.current_centroid

    def needs_redetection(self) -> bool:
        # Don't interrupt an active transition
        if self._transition is not None:
            return False
        return self.last_match_score < self.config.template_redetect_score

    # ── internal ────────────────────────────────────────────────

    def _contour_image(self, gray: np.ndarray) -> np.ndarray:
        """Convert grayscale to a gradient-magnitude contour image.

        Pipeline:
          1. Sobel gradients (dx, dy) in both directions
          2. Magnitude = sqrt(dx^2 + dy^2) — continuous edge strength
          3. GaussianBlur to spread gradients into soft contour bands
          4. Normalize to [0, 1]

        Unlike Canny (binary threshold), Sobel produces CONTINUOUS
        edge strength values at every pixel. This is much more stable
        across frames — no threshold sensitivity, no missing edges.
        The Gaussian blur creates soft "shape bands" for robust NCC.
        """
        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude (continuous, every pixel has a value)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        # Gaussian blur: spread edges into soft contour bands
        mag = cv2.GaussianBlur(mag, (0, 0), self._edge_sigma)

        # Normalize to [0, 1]
        mx = mag.max()
        if mx > 1e-6:
            mag /= mx
        return mag
