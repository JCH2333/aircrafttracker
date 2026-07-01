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

        # Orientation channel caches (full-scale + downscaled)
        self.template_channels: list[np.ndarray] = []
        self.template_channels_small: list[np.ndarray] = []
        self.tail_template_channels: list[np.ndarray] = []
        self.tail_template_channels_small: list[np.ndarray] = []
        self._channel_weights: list[float] = []  # per-channel energy weight
        self._channels_frame: int = 0  # throttle channel rebuild

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

        # Jump detection — aircraft motion must be continuous;
        # large frame-to-frame displacement = tracking failure
        self._max_jump_factor: float = config.template_max_jump_factor

        # Config shortcuts
        self._velocity_alpha: float = config.template_velocity_alpha
        self._base_margin: int = config.template_search_margin
        self._match_threshold: float = config.template_match_threshold
        self._update_alpha: float = config.template_update_alpha
        self._quality_score: float = config.template_quality_score
        self._use_edge_matching: bool = config.use_edge_matching
        self._density_suppress: bool = config.edge_density_suppress
        self._density_beta: float = config.edge_density_beta

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

    def _build_template_channels(self, gray_patch: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """Build orientation channel caches from a grayscale template patch.

        Returns:
            (full_scale_channels, small_scale_channels, channel_weights)
        """
        n_bins = self.config.orient_bins
        scale = self.config.match_downscale

        full_channels = self._orient_channels(gray_patch, n_bins)

        # Channel weights from per-channel energy (constant for template lifetime)
        weights = [float(ch.sum()) for ch in full_channels]
        total_w = sum(weights)
        if total_w > 1e-6:
            weights = [w / total_w for w in weights]
        else:
            weights = [1.0 / n_bins] * n_bins

        # Downscaled channels
        if scale < 1.0:
            small_patch = cv2.resize(gray_patch, None, fx=scale, fy=scale)
            small_channels = self._orient_channels(small_patch, n_bins)
        else:
            small_channels = full_channels

        return full_channels, small_channels, weights

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

        # Detection reset: if tracking centroid disagrees with detection by
        # a large margin, the tracker is following foreground/background.
        # Reset immediately to detection position — no smoothing.
        if self.current_centroid is not None and self.frames_since_detect > 0:
            old_cx, old_cy = self.current_centroid
            jump_dist = np.sqrt(
                (target_cx - old_cx) ** 2 + (target_cy - old_cy) ** 2
            )
            if jump_dist > 50:  # pixels — hard threshold for "teleport"
                logger.warning(
                    "Tracker LOST: detection jump %.0fpx → resetting immediately",
                    jump_dist,
                )

        # Direct init
        self.template_raw = new_template_raw
        self.template = new_template
        self.template_channels, self.template_channels_small, self._channel_weights = \
            self._build_template_channels(new_template_raw)
        self.tail_template_raw = new_tail_raw
        self.tail_template = new_tail
        self.tail_template_channels, self.tail_template_channels_small, _ = \
            self._build_template_channels(new_tail_raw)
        self.template_bbox = (x, y, w, h)
        self.current_centroid = (target_cx, target_cy)
        self.last_match_score = 1.0
        self.frames_since_detect = 0
        self._channels_frame = 0
        self._vx = 0.0
        self._vy = 0.0
        self._transition = None
        logger.debug(
            "Template init: %dx%d at (%d,%d), centroid=(%.1f, %.1f)",
            w, h, x, y, target_cx, target_cy,
        )

    def update(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Track aircraft via contour-based template matching."""

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

        # ── Full template matching ──
        scale = self.config.match_downscale
        search_patch = gray[sy:ey, sx:ex]

        # Choose matching method: orientation channels (experimental) or magnitude NCC
        use_orient = self._use_edge_matching and self.template_channels

        if use_orient:
            # Orientation-aware multi-channel matching
            n_bins = len(self.template_channels)
            if scale < 1.0:
                small_search = cv2.resize(search_patch, None, fx=scale, fy=scale)
                search_channels = self._orient_channels(small_search, n_bins)
                tmpl_channels = self.template_channels_small
            else:
                search_channels = self._orient_channels(search_patch, n_bins)
                tmpl_channels = self.template_channels

            # Crop template channels for boundary-clipped visible region
            if crop_w < tw or crop_h < th:
                if scale < 1.0:
                    sc_x1 = int(crop_x1 * scale)
                    sc_x2 = int(crop_x2 * scale)
                    sc_y1 = int(crop_y1 * scale)
                    sc_y2 = int(crop_y2 * scale)
                else:
                    sc_x1, sc_x2 = crop_x1, crop_x2
                    sc_y1, sc_y2 = crop_y1, crop_y2
                tmpl_channels = [
                    ch[sc_y1:sc_y2, sc_x1:sc_x2] for ch in tmpl_channels
                ]

            # Multi-channel NCC
            result = None
            for i in range(n_bins):
                ch_result = cv2.matchTemplate(
                    search_channels[i], tmpl_channels[i], cv2.TM_CCOEFF_NORMED,
                )
                w = self._channel_weights[i] if i < len(self._channel_weights) else 1.0 / n_bins
                if result is None:
                    result = ch_result * w
                else:
                    result += ch_result * w

            # Edge density suppression (on combined response)
            if self._density_suppress:
                ref_img = search_channels[0] if scale < 1.0 else search_channels[0]
                result = self._suppress_high_density(result, ref_img)
        else:
            # Magnitude-based NCC (default, proven)
            if scale < 1.0:
                small_search = cv2.resize(search_patch, None, fx=scale, fy=scale)
                small_tmpl = cv2.resize(vis_template, None, fx=scale, fy=scale)
                sc = self._contour_image(small_search)
                result = cv2.matchTemplate(sc, small_tmpl, cv2.TM_CCOEFF_NORMED)
                # Edge density suppression
                if self._density_suppress:
                    result = self._suppress_high_density(result, sc)
            else:
                sc = self._contour_image(search_patch)
                result = cv2.matchTemplate(sc, vis_template, cv2.TM_CCOEFF_NORMED)
                if self._density_suppress:
                    result = self._suppress_high_density(result, sc)

        _, full_score, _, full_loc = cv2.minMaxLoc(result)
        full_score = float(full_score)
        if scale < 1.0:
            full_loc = (int(full_loc[0] / scale), int(full_loc[1] / scale))

        # Store for ambiguity check
        small_search = cv2.resize(search_patch, None, fx=scale, fy=scale) if scale < 1.0 else search_patch

        full_dx = sx + full_loc[0] + crop_w / 2.0 - vis_cx_offset - self.current_centroid[0]
        full_dy = sy + full_loc[1] + crop_h / 2.0 - vis_cy_offset - self.current_centroid[1]

        # ── Ambiguity check (uses legacy contour for now) ──
        ambig_search = small_search
        if self.template is not None:
            ambig_tmpl = cv2.resize(self.template, None, fx=scale, fy=scale) if scale < 1.0 and self.template is not None else self.template
            ambig_x = int(full_loc[0] / scale) if scale < 1.0 else full_loc[0]
            ambig_y = int(full_loc[1] / scale) if scale < 1.0 else full_loc[1]
            if crop_w < tw or crop_h < th:
                ambig_tmpl = ambig_tmpl[crop_y1:crop_y2, crop_x1:crop_x2]
            if self._check_ambiguity(ambig_search, ambig_tmpl, full_score, ambig_x, ambig_y):
                full_ok = False

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
            # Both bad — coast with velocity prediction (don't return None)
            logger.debug("Coasting: full=%.3f tail=%.3f vel=(%.1f,%.1f)",
                         full_score, tail_score, self._vx, self._vy)
            matched_cx = self.current_centroid[0] + self._vx
            matched_cy = self.current_centroid[1] + self._vy

        # ── Jump detection ──
        jump_dx = matched_cx - cx_pred
        jump_dy = matched_cy - cy_pred
        jump_dist = np.sqrt(jump_dx ** 2 + jump_dy ** 2)
        # Aircraft + camera motion is continuous — large jumps = tracking failure
        max_jump = max(speed * self._max_jump_factor, 25.0)
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
                # Rebuild orientation channels every 5 frames (expensive: 2×N Sobel+GaussianBlur)
                if self._channels_frame % 5 == 0:
                    self.template_channels, self.template_channels_small, self._channel_weights = \
                        self._build_template_channels(self.template_raw)
                self._channels_frame += 1
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
                    if self._channels_frame % 5 == 1:  # offset from full template rebuild
                        self.tail_template_channels, self.tail_template_channels_small, _ = \
                            self._build_template_channels(self.tail_template_raw)

        self.template_bbox = (tx, ty, tw, th)
        self.frames_since_detect += 1
        return self.current_centroid

    def _check_ambiguity(self, search_img, tmpl, best_score, bx, by) -> bool:
        """Check if the NCC response has multiple competing peaks (ambiguity).

        When two distinct regions have similar high correlation (e.g. the
        aircraft and a tree), neither should be trusted. Returns True if
        the scene is ambiguous.
        """
        try:
            sc = self._contour_image(search_img)
            result = cv2.matchTemplate(sc, tmpl, cv2.TM_CCOEFF_NORMED)
            # Suppress a window around the best peak
            h, w = result.shape
            suppress_r = max(tmpl.shape[0] // 4, 10)
            suppress_c = max(tmpl.shape[1] // 4, 10)
            y1 = max(0, by - suppress_r)
            y2 = min(h, by + suppress_r)
            x1 = max(0, bx - suppress_c)
            x2 = min(w, bx + suppress_c)
            result[y1:y2, x1:x2] = -999
            # Find second-best peak
            _, second_best, _, _ = cv2.minMaxLoc(result)
            ratio = second_best / (best_score + 1e-8)
            if ratio > 0.85:
                logger.debug("Ambiguity: best=%.3f 2nd=%.3f ratio=%.2f",
                             best_score, second_best, ratio)
                return True
        except Exception:
            pass
        return False

    def _suppress_high_density(
        self, response: np.ndarray, edge_img: np.ndarray,
    ) -> np.ndarray:
        """Suppress NCC response in regions with anomalously high edge density.

        Foreground objects (trees, poles, buildings) are closer to the camera
        and produce much denser edge patterns than the distant aircraft. This
        method creates a density map from the edge image, then multiplies the
        NCC response by a suppression mask that penalises high-density regions.

        Args:
            response: NCC correlation result (h, w) float32.
            edge_img: Edge/contour image of the search region (H, W) float32.

        Returns:
            Suppressed response, same shape as input.
        """
        try:
            rh, rw = response.shape
            eh, ew = edge_img.shape

            # Edge density: large blur over edge image to capture cluster density
            density = cv2.GaussianBlur(edge_img, (0, 0), sigmaX=20.0)

            # Crop density map to match NCC response dimensions
            # (NCC result is H - th + 1, W - tw + 1 relative to search region)
            crop_h = min(rh, eh)
            crop_w = min(rw, ew)
            density_crop = density[:crop_h, :crop_w]

            # Normalise and create suppression mask
            dmx = density_crop.max()
            if dmx > 1e-6:
                density_crop = density_crop / dmx

            # Soft suppression: high density → low weight
            mask = 1.0 / (1.0 + self._density_beta * density_crop)

            # Apply to response (handle size mismatch gracefully)
            if crop_h == rh and crop_w == rw:
                return response * mask.astype(np.float32)
            else:
                # Resize mask to match response
                mask_resized = cv2.resize(mask, (rw, rh))
                return response * mask_resized.astype(np.float32)
        except Exception:
            return response

    def needs_redetection(self) -> bool:
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

    def _orient_channels(
        self, gray: np.ndarray, n_bins: int | None = None,
    ) -> list[np.ndarray]:
        """Decompose gradient magnitude into orientation channels.

        Each channel contains edges oriented in a specific direction range.
        Low-magnitude pixels (unreliable orientation) are zeroed in all
        channels. This makes NCC matching orientation-aware: an aircraft's
        structured edges (horizontal fuselage, diagonal wings) concentrate
        energy in a few channels, while tree branches (random orientations)
        spread energy evenly — producing lower combined correlation.

        Args:
            gray: Grayscale image (uint8 or float32).
            n_bins: Number of orientation bins (default: config.orient_bins).

        Returns:
            List of n_bins float32 arrays, each same shape as gray, values [0,1].
        """
        if n_bins is None:
            n_bins = self.config.orient_bins

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag_max = mag.max()
        if mag_max < 1e-6:
            return [np.zeros_like(gray, dtype=np.float32) for _ in range(n_bins)]

        mag_norm = mag / mag_max

        # Unsigned gradient orientation: [0, π)
        # arctan2 returns [-π, π]; modulo π maps opposite directions together
        orient = np.arctan2(gy, gx) % np.pi

        # Quantize orientation into n_bins
        bin_width = np.pi / n_bins
        bin_idx = np.floor(orient / bin_width).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # Only pixels with strong enough gradient get assigned an orientation
        reliable = mag_norm >= self.config.orient_mag_threshold

        channels = []
        for b in range(n_bins):
            ch = np.where((bin_idx == b) & reliable, mag_norm, 0.0).astype(np.float32)
            # Gaussian blur to spread edges into soft bands for NCC tolerance
            ch = cv2.GaussianBlur(ch, (0, 0), self._edge_sigma)
            # Per-channel normalize
            mx = ch.max()
            if mx > 1e-6:
                ch /= mx
            channels.append(ch)

        return channels
