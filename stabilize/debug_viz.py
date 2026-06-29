"""Debug visualization — overlay tracking info on frames for diagnosis.

Draws bounding boxes, centroids, frame numbers, and match scores
directly on Pass 1 analysis frames. Saved as PNG sequence for
frame-by-frame inspection.
"""

from pathlib import Path

import cv2
import numpy as np


def draw_overlay(
    frame_bgr: np.ndarray,
    frame_idx: int,
    tracker_state: dict,
) -> np.ndarray:
    """Draw tracking debug overlay on a BGR frame.

    Args:
        frame_bgr: Original frame (uint8 BGR).
        frame_idx: Frame number (0-based).
        tracker_state: Dict with keys:
            - bbox: (x, y, w, h) full template bbox or None
            - tail_bbox: (x, y, w, h) tail region bbox or None
            - centroid: (cx, cy) current tracking centroid
            - pred_centroid: (cx, cy) predicted centroid or None
            - match_score: float NCC score
            - match_source: str "full"/"tail"/"anchor"/"coast"/"fallback"
            - velocity: (vx, vy) px/frame or (0,0)
            - detection_used: bool — was detection used this frame?

    Returns:
        Frame with overlay drawn (new array, original unchanged).
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # ── Colors ──
    GREEN = (0, 255, 0)
    BLUE = (255, 140, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 230, 255)
    WHITE = (255, 255, 255)
    GRAY = (180, 180, 180)
    ORANGE = (0, 140, 255)

    bbox = tracker_state.get("bbox")
    tail_bbox = tracker_state.get("tail_bbox")
    centroid = tracker_state.get("centroid")
    pred = tracker_state.get("pred_centroid")
    score = tracker_state.get("match_score", 0.0)
    source = tracker_state.get("match_source", "?")
    vel = tracker_state.get("velocity", (0, 0))
    det_used = tracker_state.get("detection_used", False)

    # Bounding boxes
    if bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), GREEN, 2)
    if tail_bbox is not None:
        x, y, tw_, th_ = tail_bbox
        cv2.rectangle(out, (x, y), (x + tw_, y + th_), BLUE, 1)

    # Centroid
    if centroid is not None:
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.drawMarker(out, (cx, cy), RED, cv2.MARKER_CROSS, 20, 2)
        cv2.circle(out, (cx, cy), 5, RED, -1)

    # Predicted centroid
    if pred is not None:
        px, py = int(pred[0]), int(pred[1])
        cv2.drawMarker(out, (px, py), YELLOW, cv2.MARKER_TILTED_CROSS, 12, 1)

    # ── Text overlays ──
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 22
    y0 = 30

    # Frame number (top-left, large)
    cv2.putText(out, f"Frame: {frame_idx}", (10, y0), font, 0.8, WHITE, 2)

    # Detection indicator
    if det_used:
        cv2.putText(out, "DETECT", (10, y0 + line_h), font, 0.6, ORANGE, 2)

    # Match info (top-right)
    texts = [
        f"Score: {score:.3f}",
        f"Source: {source}",
        f"Vel: ({vel[0]:.1f}, {vel[1]:.1f})",
    ]
    for i, txt in enumerate(texts):
        (tw_, th_), _ = cv2.getTextSize(txt, font, 0.5, 1)
        cv2.putText(out, txt, (w - tw_ - 10, y0 + i * line_h), font, 0.5, GRAY, 1)

    # Centroid coords (bottom-left)
    if centroid is not None:
        cv2.putText(
            out, f"CX:{centroid[0]:.1f} CY:{centroid[1]:.1f}",
            (10, h - 15), font, 0.5, GRAY, 1,
        )

    # Legend (bottom-right, small)
    legend = [
        ("Green", GREEN, "Full bbox"),
        ("Blue", BLUE, "Tail bbox"),
        ("Red +", RED, "Centroid"),
        ("Yellow x", YELLOW, "Predicted"),
    ]
    ly = h - 15 - len(legend) * 18
    for label, color, desc in legend:
        cv2.putText(out, f"{label}: {desc}", (w - 220, ly), font, 0.4, color, 1)
        ly += 18

    return out


class DebugVizWriter:
    """Writes debug visualization frames to disk as PNG sequence."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, frame_bgr: np.ndarray, frame_idx: int, tracker_state: dict):
        """Render overlay and save to PNG."""
        out = draw_overlay(frame_bgr, frame_idx, tracker_state)
        path = self.output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(path), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        self._count += 1

    def close(self):
        pass
