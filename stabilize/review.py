"""Manual correction requests and a CLI OpenCV reviewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from stabilize.tracking.models import FlaggedSegment, TrackObservation
from stabilize.tracking.track_file import ManualAnchor


@dataclass
class ReviewRequest:
    video_path: Path
    segments: list[FlaggedSegment]
    predicted_centers: list[tuple[float, float]]
    existing_anchors: list[ManualAnchor]
    width: int
    height: int
    frame_rate: float
    observations: list[TrackObservation] = field(default_factory=list)
    masks: dict[int, object] = field(default_factory=dict)


def review_with_opencv(request: ReviewRequest) -> list[ManualAnchor]:
    """Review flagged segments with OpenCV's native ROI selector."""
    capture = cv2.VideoCapture(str(request.video_path))
    if not capture.isOpened():
        return []

    anchors: list[ManualAnchor] = []
    window_name = "Aircraft tracking review"
    try:
        for segment in request.segments:
            frame_idx = (segment.start_frame + segment.end_frame) // 2
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue

            mask = request.masks.get(frame_idx)
            if mask is not None:
                mask = cv2.resize(
                    mask.astype("uint8"),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                overlay = frame.copy()
                overlay[mask] = (40, 200, 80)
                frame = cv2.addWeighted(frame, 0.72, overlay, 0.28, 0)

            display, scale = _fit_frame(frame, 1280, 800)
            center = request.predicted_centers[frame_idx]
            _draw_review_context(display, request, frame_idx, scale)
            cv2.drawMarker(
                display,
                (int(center[0] * scale), int(center[1] * scale)),
                (0, 220, 255),
                cv2.MARKER_TILTED_CROSS,
                20,
                2,
            )
            cv2.putText(
                display,
                (
                    f"Frames {segment.start_frame}-{segment.end_frame}: "
                    "drag aircraft box, Enter=accept, Esc=skip"
                ),
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            x, y, w, h = cv2.selectROI(
                window_name,
                display,
                showCrosshair=True,
                fromCenter=False,
            )
            if w > 0 and h > 0:
                anchors.append(
                    ManualAnchor(
                        frame_idx=frame_idx,
                        bbox=(
                            int(round(x / scale)),
                            int(round(y / scale)),
                            int(round(w / scale)),
                            int(round(h / scale)),
                        ),
                    )
                )
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
    return anchors


def _draw_review_context(
    frame,
    request: ReviewRequest,
    frame_idx: int,
    scale: float,
) -> None:
    start = max(0, frame_idx - 15)
    end = min(len(request.predicted_centers), frame_idx + 16)
    points = [
        (
            int(request.predicted_centers[idx][0] * scale),
            int(request.predicted_centers[idx][1] * scale),
        )
        for idx in range(start, end)
    ]
    if len(points) >= 2:
        cv2.polylines(
            frame,
            [np.asarray(points, dtype=np.int32)],
            False,
            (0, 220, 255),
            2,
        )

    if 0 <= frame_idx < len(request.observations):
        bbox = request.observations[frame_idx].bbox
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(
                frame,
                (int(x * scale), int(y * scale)),
                (int((x + w) * scale), int((y + h) * scale)),
                (80, 255, 80),
                2,
            )


def _fit_frame(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame, scale
