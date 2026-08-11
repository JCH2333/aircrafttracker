"""Shared tracking data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


BBox = tuple[int, int, int, int]
Point = tuple[float, float]


class TrackingState(str, Enum):
    TRACKING = "tracking"
    OCCLUDED = "occluded"
    LOST = "lost"
    MANUAL_ANCHOR = "manual_anchor"


@dataclass(frozen=True)
class DetectionCandidate:
    bbox: BBox
    score: float
    label: int | None = None
    source: str = "detector"

    @property
    def area(self) -> float:
        return float(max(0, self.bbox[2]) * max(0, self.bbox[3]))

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def aspect_ratio(self) -> float:
        return self.bbox[2] / max(float(self.bbox[3]), 1.0)


@dataclass
class RejectedCandidate:
    candidate: DetectionCandidate
    reason: str


@dataclass
class TrackObservation:
    frame_idx: int
    center: Point | None
    bbox: BBox | None
    confidence: float
    visibility: float
    state: TrackingState
    source: str
    predicted_center: Point | None = None
    measured: bool = True
    rejected_candidates: tuple[RejectedCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


@dataclass(frozen=True)
class FlaggedSegment:
    start_frame: int
    end_frame: int
    reason: str
    max_gap: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
