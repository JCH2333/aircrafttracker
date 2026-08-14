"""Persistent manual anchors and review metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stabilize.tracking.models import BBox, FlaggedSegment, Point


@dataclass(order=True)
class ManualAnchor:
    frame_idx: int
    bbox: BBox
    source: str = "manual"
    reference_point: Point | None = None


@dataclass
class TrackFile:
    version: int = 1
    source_path: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = "hybrid"
    anchors: list[ManualAnchor] = field(default_factory=list)
    flagged_segments: list[FlaggedSegment] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str) -> "TrackFile":
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            version=int(data.get("version", 1)),
            source_path=str(data.get("source_path", "")),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            fps=float(data.get("fps", 0.0)),
            backend=str(data.get("backend", "hybrid")),
            anchors=[
                ManualAnchor(
                    frame_idx=int(item["frame_idx"]),
                    bbox=tuple(int(value) for value in item["bbox"]),
                    source=str(item.get("source", "manual")),
                    reference_point=(
                        tuple(float(value) for value in item["reference_point"])
                        if item.get("reference_point") is not None
                        else None
                    ),
                )
                for item in data.get("anchors", [])
            ],
            flagged_segments=[
                FlaggedSegment(
                    start_frame=int(item["start_frame"]),
                    end_frame=int(item["end_frame"]),
                    reason=str(item.get("reason", "low confidence")),
                    max_gap=int(item.get("max_gap", 0)),
                )
                for item in data.get("flagged_segments", [])
            ],
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "source_path": self.source_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "backend": self.backend,
            "anchors": [asdict(anchor) for anchor in sorted(self.anchors)],
            "flagged_segments": [
                segment.to_dict() for segment in self.flagged_segments
            ],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def is_compatible(
        self,
        source_path: Path | str,
        width: int,
        height: int,
        fps: float | None = None,
    ) -> bool:
        if not self.source_path:
            return True
        compatible = (
            Path(self.source_path).resolve() == Path(source_path).resolve()
            and self.width == width
            and self.height == height
        )
        if fps is not None and self.fps > 0:
            compatible = compatible and abs(self.fps - fps) <= 0.01
        return compatible

    def add_anchor(self, anchor: ManualAnchor) -> None:
        self.anchors = [
            existing
            for existing in self.anchors
            if existing.frame_idx != anchor.frame_idx
        ]
        self.anchors.append(anchor)
        self.anchors.sort()

    def remove_last_anchor(self) -> ManualAnchor | None:
        if not self.anchors:
            return None
        return self.anchors.pop()

    def anchor_map(self) -> dict[int, ManualAnchor]:
        return {anchor.frame_idx: anchor for anchor in self.anchors}
