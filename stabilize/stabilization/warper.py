"""Frame warping optimized for pure translation.

Uses np.roll for the bulk shift (milliseconds at 4K) instead of
cv2.warpAffine (~500ms at 4K), then fills exposed borders.

Supports 8-bit and 16-bit frames.
"""

import numpy as np


def translate_frame(
    frame: np.ndarray,
    dx: float,
    dy: float,
    border_mode: str = "reflect",
) -> np.ndarray:
    """Translate a multi-channel image by (dx, dy) pixels.

    Optimized for pure translation: uses np.roll for the integer
    part of the shift, then fills the exposed border strip.

    Args:
        frame: Input image of shape (H, W, C), any dtype.
        dx: Pixels to shift right (positive) or left (negative).
        dy: Pixels to shift down (positive) or up (negative).
        border_mode: "replicate" (extend edge pixel) or
                     "reflect" (mirror edge strip).

    Returns:
        Shifted image with same shape and dtype as input.
    """
    h, w = frame.shape[:2]
    xi = int(round(dx))
    yi = int(round(dy))

    # No-op for zero shift
    if xi == 0 and yi == 0:
        return frame.copy()

    # Bulk shift via np.roll (fast, O(N) memory move)
    shifted = np.roll(frame, (yi, xi), axis=(0, 1))

    # Fill exposed borders
    _fill_borders(shifted, xi, yi, border_mode)

    return shifted


def _fill_borders(
    img: np.ndarray,
    dx: int,
    dy: int,
    mode: str,
) -> None:
    """Fill borders exposed by integer shift (dx, dy). Modifies img in-place.

    Args:
        img: Shifted image (from np.roll), modified in-place.
        dx: Horizontal shift (positive = right, negative = left).
        dy: Vertical shift (positive = down, negative = up).
        mode: "constant" (zeros/black), "replicate" (stretch edge),
              "reflect" (mirror edge region).
    """
    h, w = img.shape[:2]

    if mode == "constant":
        # Black borders — zero fill
        if dx > 0:
            img[:, :dx] = 0
        elif dx < 0:
            img[:, dx:] = 0
        if dy > 0:
            img[:dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        return

    # Non-constant modes: fill from image content
    if dx > 0:
        if mode == "reflect":
            _reflect_fill_x(img, dx, 0, dx)
        else:  # replicate
            img[:, :dx] = img[:, dx : dx + 1]

    elif dx < 0:
        d = abs(dx)
        if mode == "reflect":
            _reflect_fill_x(img, d, w - d, w)
        else:  # replicate
            img[:, -d:] = img[:, -d - 1 : -d]

    if dy > 0:
        if mode == "reflect":
            _reflect_fill_y(img, dy, 0, dy)
        else:  # replicate
            img[:dy, :] = img[dy : dy + 1, :]

    elif dy < 0:
        d = abs(dy)
        if mode == "reflect":
            _reflect_fill_y(img, d, h - d, h)
        else:  # replicate
            img[-d:, :] = img[-d - 1 : -d, :]


def _reflect_fill_x(img: np.ndarray, width: int, src_start: int, dst_end: int) -> None:
    """Fill a horizontal strip by mirroring from adjacent region."""
    for i in range(width):
        img[:, i] = img[:, src_start + (width - i)]


def _reflect_fill_y(img: np.ndarray, height: int, src_start: int, dst_end: int) -> None:
    """Fill a vertical strip by mirroring from adjacent region."""
    for i in range(height):
        img[i, :] = img[src_start + (height - i), :]


def compute_transforms(
    centroids_smooth: list[tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> list[tuple[float, float]]:
    """Compute translation to center the aircraft at the frame center.

    Args:
        centroids_smooth: Smoothed centroid positions (cx, cy) per frame.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        List of (dx, dy) per frame. dx > 0 shifts image right
        (moving aircraft toward screen center).
    """
    frame_cx = frame_width / 2.0
    frame_cy = frame_height / 2.0

    transforms = []
    for cx, cy in centroids_smooth:
        dx = frame_cx - cx  # pixels to shift the image right
        dy = frame_cy - cy  # pixels to shift the image down
        transforms.append((dx, dy))

    return transforms


def compute_jitter_correction(
    centroids_raw: list[tuple[float, float]],
    centroids_smooth: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Compute translation to cancel frame-to-frame jitter.

    The correction shifts the image so the aircraft stays at its
    smoothed (noise-filtered) position, keeping it in its natural
    region of the frame instead of forcing it to the center.

    Args:
        centroids_raw: Raw per-frame aircraft centroids from tracker.
        centroids_smooth: Savitzky-Golay filtered centroids.
            Must be same length as centroids_raw.

    Returns:
        List of (dx, dy) per frame. dx > 0 shifts image right
        (moving aircraft left relative to frame).
    """
    if len(centroids_raw) != len(centroids_smooth):
        raise ValueError(
            f"Length mismatch: raw={len(centroids_raw)}, smooth={len(centroids_smooth)}"
        )

    transforms = []
    for (cx_raw, cy_raw), (cx_smooth, cy_smooth) in zip(
        centroids_raw, centroids_smooth
    ):
        # Shift image so aircraft moves from raw position to smoothed position
        dx = cx_smooth - cx_raw
        dy = cy_smooth - cy_raw
        transforms.append((dx, dy))

    return transforms
