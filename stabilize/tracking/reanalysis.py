"""Build bounded reanalysis jobs around manual correction anchors."""

from __future__ import annotations

from dataclasses import dataclass

from stabilize.tracking.models import TrackObservation, TrackingState
from stabilize.tracking.track_file import ManualAnchor


@dataclass(frozen=True)
class ReanalysisRange:
    start_frame: int
    end_frame: int
    anchors: tuple[ManualAnchor, ...]


def build_reanalysis_ranges(
    observations: list[TrackObservation],
    existing_anchors: list[ManualAnchor],
    new_anchors: list[ManualAnchor],
) -> list[ReanalysisRange]:
    """Bound each correction by the nearest reliable observations."""
    if not observations or not new_anchors:
        return []

    raw_ranges: list[tuple[int, int]] = []
    final_frame = len(observations) - 1
    for anchor in new_anchors:
        if anchor.frame_idx < 0 or anchor.frame_idx > final_frame:
            continue
        start = _find_reliable(
            observations,
            anchor.frame_idx - 1,
            step=-1,
        )
        end = _find_reliable(
            observations,
            anchor.frame_idx + 1,
            step=1,
        )
        raw_ranges.append(
            (
                0 if start is None else start,
                final_frame if end is None else end,
            )
        )

    merged = _merge_ranges(raw_ranges)
    ranges = []
    for start, end in merged:
        anchor_map = {
            anchor.frame_idx: anchor
            for anchor in existing_anchors
            if start <= anchor.frame_idx <= end
        }
        for anchor in new_anchors:
            if start <= anchor.frame_idx <= end:
                anchor_map[anchor.frame_idx] = anchor

        for boundary in (start, end):
            observation = observations[boundary]
            if (
                boundary not in anchor_map
                and _is_reliable(observation)
                and observation.bbox is not None
            ):
                anchor_map[boundary] = ManualAnchor(
                    frame_idx=boundary,
                    bbox=observation.bbox,
                    source="boundary",
                )

        ranges.append(
            ReanalysisRange(
                start_frame=start,
                end_frame=end,
                anchors=tuple(
                    sorted(anchor_map.values(), key=lambda item: item.frame_idx)
                ),
            )
        )
    return ranges


def _find_reliable(
    observations: list[TrackObservation],
    start: int,
    step: int,
) -> int | None:
    idx = start
    while 0 <= idx < len(observations):
        if _is_reliable(observations[idx]):
            return idx
        idx += step
    return None


def _is_reliable(observation: TrackObservation) -> bool:
    return (
        observation.center is not None
        and observation.bbox is not None
        and observation.confidence >= 0.65
        and observation.state
        in (TrackingState.TRACKING, TrackingState.MANUAL_ANCHOR)
    )


def _merge_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]
