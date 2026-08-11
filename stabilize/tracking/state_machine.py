"""Occlusion-aware tracking state transitions."""

from __future__ import annotations

from stabilize.tracking.models import BBox, Point, TrackObservation, TrackingState


class TrackingStateMachine:
    def __init__(
        self,
        low_threshold: float = 0.45,
        recovery_threshold: float = 0.65,
        low_frames: int = 2,
        recovery_frames: int = 3,
        lost_frames: int = 90,
    ):
        self.low_threshold = low_threshold
        self.recovery_threshold = recovery_threshold
        self.low_frames = low_frames
        self.recovery_frames = recovery_frames
        self.lost_frames = lost_frames
        self.state = TrackingState.LOST
        self._low_count = 0
        self._high_count = 0
        self._missing_count = 0
        self._recovering = True
        self._last_center: Point | None = None
        self._velocity: Point = (0.0, 0.0)

    @property
    def predicted_center(self) -> Point | None:
        if self._last_center is None:
            return None
        prediction_steps = max(1, self._missing_count + 1)
        return (
            self._last_center[0] + self._velocity[0] * prediction_steps,
            self._last_center[1] + self._velocity[1] * prediction_steps,
        )

    @property
    def velocity(self) -> Point:
        return self._velocity

    def update(
        self,
        frame_idx: int,
        center: Point | None,
        bbox: BBox | None,
        confidence: float,
        visibility: float,
        source: str,
        manual: bool = False,
        reset_motion: bool = False,
        rejected_candidates=(),
    ) -> TrackObservation:
        predicted = self.predicted_center
        missing_before_update = self._missing_count

        if manual:
            self.state = TrackingState.MANUAL_ANCHOR
            self._low_count = 0
            self._high_count = self.recovery_frames
            self._missing_count = 0
            self._recovering = False
            self._reset_center(center)
            return TrackObservation(
                frame_idx=frame_idx,
                center=center,
                bbox=bbox,
                confidence=1.0,
                visibility=1.0,
                state=self.state,
                source=source,
                predicted_center=predicted,
                measured=True,
                rejected_candidates=tuple(rejected_candidates),
            )

        reliable_center = center if center is not None and confidence >= self.low_threshold else None

        if reliable_center is None:
            self._low_count += 1
            self._high_count = 0
            self._missing_count += 1
            self._recovering = True
            if self._missing_count >= self.lost_frames:
                self.state = TrackingState.LOST
            elif self._low_count >= self.low_frames:
                self.state = TrackingState.OCCLUDED
        else:
            self._missing_count = 0
            self._low_count = 0
            if confidence >= self.recovery_threshold:
                self._high_count += 1
            else:
                self._high_count = 0
            if self._high_count >= self.recovery_frames:
                self._recovering = False

            if self.state in (TrackingState.LOST, TrackingState.OCCLUDED):
                if self._high_count >= self.recovery_frames:
                    self.state = TrackingState.TRACKING
                else:
                    self.state = TrackingState.OCCLUDED
            else:
                self.state = TrackingState.TRACKING
            if reset_motion or missing_before_update > 0:
                self._reset_center(reliable_center)
            else:
                self._accept_center(reliable_center)

        return TrackObservation(
            frame_idx=frame_idx,
            center=reliable_center,
            bbox=bbox,
            confidence=float(max(0.0, min(confidence, 1.0))),
            visibility=float(max(0.0, min(visibility, 1.0))),
            state=self.state,
            source=source,
            predicted_center=predicted,
            measured=reliable_center is not None and not self._recovering,
            rejected_candidates=tuple(rejected_candidates),
        )

    def _accept_center(self, center: Point | None) -> None:
        if center is None:
            return
        if self._last_center is not None:
            dx = center[0] - self._last_center[0]
            dy = center[1] - self._last_center[1]
            self._velocity = (
                0.5 * dx + 0.5 * self._velocity[0],
                0.5 * dy + 0.5 * self._velocity[1],
            )
        self._last_center = center

    def _reset_center(self, center: Point | None) -> None:
        if center is None:
            return
        self._last_center = center
        self._velocity = (0.0, 0.0)
