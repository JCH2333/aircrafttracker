"""Confidence-aware Kalman filtering and RTS trajectory smoothing."""

from __future__ import annotations

import numpy as np

from stabilize.stabilization.smoother import smooth_trajectory
from stabilize.tracking.models import FlaggedSegment, TrackObservation, TrackingState


def smooth_observations(
    observations: list[TrackObservation],
    frame_width: int,
    frame_height: int,
    smoother_window: int = 15,
    smoother_method: str = "savgol",
    smoother_polyorder: int = 2,
) -> tuple[list[tuple[float, float]], list[FlaggedSegment]]:
    if not observations:
        raise ValueError("observations list is empty")

    measured_indices = [
        i for i, obs in enumerate(observations)
        if _is_trusted_observation(obs)
    ]
    if not measured_indices:
        raise RuntimeError(
            "No reliable target observations; add a manual aircraft anchor"
        )

    first = measured_indices[0]
    last = measured_indices[-1]
    middle = _kalman_rts(observations[first : last + 1])
    result: list[tuple[float, float] | None] = [None] * len(observations)
    result[first : last + 1] = middle

    _fill_leading(result, first)
    _fill_trailing(result, last)

    clamped = [
        (
            float(np.clip(point[0], 0.0, frame_width - 1.0)),
            float(np.clip(point[1], 0.0, frame_height - 1.0)),
        )
        for point in result
    ]
    if len(clamped) >= 3 and smoother_window >= 3:
        clamped = smooth_trajectory(
            clamped,
            window=smoother_window,
            method=smoother_method,
            polyorder=smoother_polyorder,
        )
        clamped = [
            (
                float(np.clip(point[0], 0.0, frame_width - 1.0)),
                float(np.clip(point[1], 0.0, frame_height - 1.0)),
            )
            for point in clamped
        ]
        for idx, observation in enumerate(observations):
            if (
                observation.source == "sam2_mask_recovery"
                and observation.confidence >= 0.65
                and observation.center is not None
            ) or (
                observation.state == TrackingState.MANUAL_ANCHOR
                and observation.center is not None
            ):
                clamped[idx] = observation.center

    return clamped, find_flagged_segments(observations)


def find_flagged_segments(
    observations: list[TrackObservation],
    minimum_length: int = 2,
) -> list[FlaggedSegment]:
    segments: list[FlaggedSegment] = []
    start: int | None = None
    reasons: set[str] = set()

    def flush(end: int) -> None:
        nonlocal start, reasons
        if start is None:
            return
        length = end - start + 1
        if (
            length < minimum_length
            and "missing measurement" not in reasons
        ):
            start = None
            reasons = set()
            return
        segments.append(
            FlaggedSegment(
                start_frame=start,
                end_frame=end,
                reason=_segment_reason(reasons, length),
                max_gap=length,
            )
        )
        start = None
        reasons = set()

    for idx, obs in enumerate(observations):
        flagged = (
            obs.center is None
            or not obs.measured
            or obs.confidence < 0.65
            or obs.state in (TrackingState.OCCLUDED, TrackingState.LOST)
        )
        if flagged:
            if start is None:
                start = idx
            if obs.center is None:
                reasons.add("missing measurement")
            if not obs.measured:
                reasons.add("recovery verification")
            if obs.state == TrackingState.OCCLUDED:
                reasons.add("occluded")
            if obs.state == TrackingState.LOST:
                reasons.add("lost")
            if obs.confidence < 0.65:
                reasons.add("low confidence")
        elif start is not None:
            flush(idx - 1)

    if start is not None:
        flush(len(observations) - 1)
    return segments


def _segment_reason(reasons: set[str], length: int) -> str:
    if length > 90:
        reasons = set(reasons)
        reasons.add("manual confirmation required")
    return ", ".join(sorted(reasons)) or "low confidence"


def _kalman_rts(
    observations: list[TrackObservation],
) -> list[tuple[float, float]]:
    n = len(observations)
    first_center = next(
        obs.center
        for obs in observations
        if _is_trusted_observation(obs)
    )

    f = np.array(
        [
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    h = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        dtype=np.float64,
    )
    q = np.diag([0.04, 0.04, 0.15, 0.15, 0.3, 0.3])
    identity = np.eye(6, dtype=np.float64)

    filtered_x = np.zeros((n, 6), dtype=np.float64)
    filtered_p = np.zeros((n, 6, 6), dtype=np.float64)
    predicted_x = np.zeros((n, 6), dtype=np.float64)
    predicted_p = np.zeros((n, 6, 6), dtype=np.float64)

    x = np.array([first_center[0], first_center[1], 0, 0, 0, 0], dtype=np.float64)
    p = np.diag([25.0, 25.0, 100.0, 100.0, 25.0, 25.0])

    for idx, obs in enumerate(observations):
        if idx > 0:
            x = f @ x
            p = f @ p @ f.T + q
        predicted_x[idx] = x
        predicted_p[idx] = p

        if _is_trusted_observation(obs):
            confidence = max(obs.confidence, 0.05)
            if obs.state == TrackingState.MANUAL_ANCHOR:
                variance = 1e-4
            elif obs.source == "sam2_mask_recovery":
                variance = 1e-4 + 0.01 * (1.0 - confidence) ** 2
            else:
                variance = 1.0 + 36.0 * (1.0 - confidence) ** 2
            r = np.eye(2, dtype=np.float64) * variance
            z = np.asarray(obs.center, dtype=np.float64)
            innovation = z - h @ x
            innovation_cov = h @ p @ h.T + r
            gain = p @ h.T @ np.linalg.pinv(innovation_cov)
            x = x + gain @ innovation
            p = (identity - gain @ h) @ p

        filtered_x[idx] = x
        filtered_p[idx] = p

    smoothed_x = filtered_x.copy()
    smoothed_p = filtered_p.copy()
    for idx in range(n - 2, -1, -1):
        gain = (
            filtered_p[idx]
            @ f.T
            @ np.linalg.pinv(predicted_p[idx + 1])
        )
        smoothed_x[idx] = filtered_x[idx] + gain @ (
            smoothed_x[idx + 1] - predicted_x[idx + 1]
        )
        smoothed_p[idx] = filtered_p[idx] + gain @ (
            smoothed_p[idx + 1] - predicted_p[idx + 1]
        ) @ gain.T

    return [(float(row[0]), float(row[1])) for row in smoothed_x]


def _is_trusted_observation(observation: TrackObservation) -> bool:
    if observation.center is None:
        return False
    if observation.state == TrackingState.MANUAL_ANCHOR:
        return True
    return (
        observation.state == TrackingState.TRACKING
        and observation.measured
        and observation.confidence >= 0.65
    )


def _fill_leading(
    result: list[tuple[float, float] | None],
    first: int,
) -> None:
    if first <= 0:
        return
    first_point = result[first]
    next_point = result[first + 1] if first + 1 < len(result) else first_point
    vx = next_point[0] - first_point[0]
    vy = next_point[1] - first_point[1]
    for idx in range(first - 1, -1, -1):
        distance = min(first - idx, 30)
        result[idx] = (
            first_point[0] - vx * distance,
            first_point[1] - vy * distance,
        )


def _fill_trailing(
    result: list[tuple[float, float] | None],
    last: int,
) -> None:
    if last >= len(result) - 1:
        return
    last_point = result[last]
    prev_point = result[last - 1] if last > 0 else last_point
    vx = last_point[0] - prev_point[0]
    vy = last_point[1] - prev_point[1]
    for idx in range(last + 1, len(result)):
        distance = min(idx - last, 30)
        result[idx] = (
            last_point[0] + vx * distance,
            last_point[1] + vy * distance,
        )
