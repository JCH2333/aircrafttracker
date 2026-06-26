"""Trajectory smoothing using Savitzky-Golay or Gaussian filters.

All filters are zero-phase (forward-backward or symmetric) to avoid
introducing temporal lag in the output video.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def smooth_trajectory(
    centroids: list[tuple[float, float]],
    window: int = 61,
    method: str = "savgol",
    polyorder: int = 2,
) -> list[tuple[float, float]]:
    """Apply zero-phase low-pass filter to centroid trajectory.

    Args:
        centroids: List of (cx, cy) tuples, one per frame.
        window: Smoothing window in frames (odd for savgol).
        method: "savgol" or "gaussian".
        polyorder: Polynomial order for Savitzky-Golay filter.

    Returns:
        List of (cx_smooth, cy_smooth) same length as input.

    Raises:
        ValueError: If centroids is empty or method is unknown.
    """
    if not centroids:
        raise ValueError("centroids list is empty")

    n = len(centroids)
    cx = np.array([c[0] for c in centroids], dtype=np.float64)
    cy = np.array([c[1] for c in centroids], dtype=np.float64)

    # Validate window
    actual_window = min(window, n)
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window < 3:
        actual_window = 3

    if actual_window != window:
        logger.info(
            "Smoothing window adjusted: %d -> %d (%d frames total)", window, actual_window, n
        )

    if method == "savgol":
        from scipy.signal import savgol_filter

        cx_s = savgol_filter(cx, actual_window, polyorder, mode="nearest")
        cy_s = savgol_filter(cy, actual_window, polyorder, mode="nearest")

    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        sigma = actual_window / 4.0
        cx_s = gaussian_filter1d(cx, sigma, mode="nearest")
        cy_s = gaussian_filter1d(cy, sigma, mode="nearest")

    else:
        raise ValueError(f"Unknown smoother method: {method}")

    # Compute jitter reduction stats
    raw_jitter = np.std(np.diff(cx)) + np.std(np.diff(cy))
    smooth_jitter = np.std(np.diff(cx_s)) + np.std(np.diff(cy_s))
    reduction = (1 - smooth_jitter / raw_jitter) * 100 if raw_jitter > 0 else 0
    logger.info(
        "Jitter reduction: %.1f%% (raw=%.2f, smooth=%.2f px/frame std-dev)",
        reduction, raw_jitter, smooth_jitter,
    )

    return list(zip(cx_s.tolist(), cy_s.tolist()))
