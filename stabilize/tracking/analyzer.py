"""Legacy and SAM 2 hybrid analysis backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stabilize.config import StabilizerConfig
from stabilize.debug_viz import DebugVizWriter
from stabilize.detection.motion_fallback import MotionFallbackDetector
from stabilize.io.reader import VideoReader
from stabilize.stabilization.template_tracker import TemplateTracker
from stabilize.tracking.analysis_cache import AnalysisFrameCache
from stabilize.tracking.gating import CandidateGate
from stabilize.tracking.masked_feature_tracker import (
    MaskedFeatureTracker,
    bbox_from_mask,
)
from stabilize.tracking.models import (
    DetectionCandidate,
    RejectedCandidate,
    TrackObservation,
    TrackingState,
)
from stabilize.tracking.sam2_adapter import Sam2MaskProvider, Sam2Prompt
from stabilize.tracking.state_machine import TrackingStateMachine
from stabilize.tracking.track_file import ManualAnchor

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    observations: list[TrackObservation]
    width: int
    height: int
    frame_rate: float
    backend: str
    review_masks: dict[int, np.ndarray] = field(default_factory=dict)


def analyze_legacy(
    config: StabilizerConfig,
    detector,
    anchors: list[ManualAnchor],
    progress_cb=None,
    debug_viz_path: Path | None = None,
    frame_range: tuple[int, int] | None = None,
) -> AnalysisResult:
    reader = VideoReader(config.input_path, mode="analysis")
    total = reader.total_frames
    start_frame, end_frame = _normalize_frame_range(total, frame_range)
    tracker = TemplateTracker(config)
    motion_detector = MotionFallbackDetector(config)
    gate = CandidateGate()
    state = _make_state_machine(config)
    anchor_map = {anchor.frame_idx: anchor for anchor in anchors}
    observations: list[TrackObservation] = []

    debug_writer = None
    if debug_viz_path is not None:
        debug_writer = DebugVizWriter(
            output_path=debug_viz_path,
            frame_rate=reader.frame_rate,
            width=reader.width,
            height=reader.height,
        )

    pbar = tqdm(
        reader,
        total=total or None,
        desc="Pass 1 (legacy track)",
        unit="f",
        colour="blue",
    )
    try:
        for frame_bgr, idx in pbar:
            if idx < start_frame:
                continue
            if end_frame is not None and idx > end_frame:
                break
            if progress_cb:
                current = idx - start_frame + 1
                expected = (
                    end_frame - start_frame + 1
                    if end_frame is not None
                    else total or current
                )
                progress_cb(current, expected)

            manual_anchor = anchor_map.get(idx)
            rejected = ()
            detection_used = False
            motion_hints = motion_detector.detect_candidates(frame_bgr)

            if manual_anchor is not None:
                tracker.init_from_detection(frame_bgr, manual_anchor.bbox)
                gate.record_bbox(manual_anchor.bbox)
                observation = state.update(
                    frame_idx=idx,
                    center=_bbox_center(manual_anchor.bbox),
                    bbox=manual_anchor.bbox,
                    confidence=1.0,
                    visibility=1.0,
                    source="manual",
                    manual=True,
                )
                detection_used = True
            else:
                need_detect = (
                    not tracker.initialized
                    or tracker.needs_redetection()
                    or state.state in (TrackingState.OCCLUDED, TrackingState.LOST)
                    or idx % max(config.detection_interval, 1) == 0
                )
                selected = None
                if need_detect:
                    candidates = _apply_motion_hints(
                        detector.detect_candidates(frame_bgr),
                        motion_hints,
                    )
                    selected, rejected = gate.select(
                        candidates,
                        frame_shape=frame_bgr.shape[:2],
                        predicted_center=state.predicted_center,
                        predicted_bbox=tracker.template_bbox,
                    )
                    rejected = rejected + tuple(
                        RejectedCandidate(
                            candidate=hint,
                            reason="motion hint only; cannot initialize track",
                        )
                        for hint in motion_hints
                    )

                should_reinitialize = selected is not None
                if should_reinitialize:
                    tracker.init_from_detection(frame_bgr, selected.bbox)
                    gate.record_bbox(selected.bbox)
                    observation = state.update(
                        frame_idx=idx,
                        center=selected.center,
                        bbox=selected.bbox,
                        confidence=max(0.70, selected.score),
                        visibility=1.0,
                        source=selected.source,
                        reset_motion=True,
                        rejected_candidates=rejected,
                    )
                    detection_used = True
                else:
                    center = tracker.update(frame_bgr) if tracker.initialized else None
                    confidence = tracker.last_match_score if center is not None else 0.0
                    observation = state.update(
                        frame_idx=idx,
                        center=center,
                        bbox=tracker.template_bbox,
                        confidence=confidence,
                        visibility=1.0 if center is not None else 0.0,
                        source="template" if center is not None else "missing",
                        rejected_candidates=rejected,
                    )
                    if (
                        observation.center is not None
                        and observation.confidence
                        >= config.tracking_update_confidence
                    ):
                        gate.record_bbox(observation.bbox)

            observations.append(observation)
            if debug_writer is not None:
                debug_writer.write(
                    frame_bgr,
                    idx,
                    _debug_state(
                        observation,
                        velocity=state.velocity,
                        detection_used=detection_used,
                    ),
                )
            pbar.set_postfix_str(
                f"state={observation.state.value} conf={observation.confidence:.2f}"
            )
    finally:
        reader.close()
        pbar.close()
        if debug_writer is not None:
            debug_writer.close()

    return AnalysisResult(
        observations=observations,
        width=reader.width,
        height=reader.height,
        frame_rate=reader.frame_rate,
        backend="legacy",
        review_masks={},
    )


def analyze_hybrid(
    config: StabilizerConfig,
    detector,
    anchors: list[ManualAnchor],
    progress_cb=None,
    debug_viz_path: Path | None = None,
    frame_range: tuple[int, int] | None = None,
) -> AnalysisResult:
    start_frame = frame_range[0] if frame_range else 0
    end_frame = frame_range[1] if frame_range else None
    with AnalysisFrameCache(
        config.input_path,
        max_dimension=config.analysis_downscale,
        jpeg_quality=config.analysis_jpeg_quality,
        start_frame=start_frame,
        end_frame=end_frame,
    ) as cache:
        cache.build(
            progress_cb=(
                lambda current, total: progress_cb(
                    int(current * 0.25),
                    total,
                )
                if progress_cb
                else None
            )
        )
        if cache.total_frames == 0:
            raise RuntimeError("Input video contains no frames")

        local_anchors = [
            ManualAnchor(
                frame_idx=anchor.frame_idx - cache.frame_offset,
                bbox=anchor.bbox,
                source=anchor.source,
            )
            for anchor in anchors
            if (
                cache.frame_offset
                <= anchor.frame_idx
                < cache.frame_offset + cache.total_frames
            )
        ]
        prompt_candidates = _build_sam_prompts(
            cache,
            detector,
            local_anchors,
            config,
        )
        provider = Sam2MaskProvider(
            model_id=config.sam2_model_id,
            device=config.device,
            offload_video_to_cpu=config.sam2_offload_video_to_cpu,
            offload_state_to_cpu=config.sam2_offload_state_to_cpu,
        )

        feature_tracker = MaskedFeatureTracker(config)
        state = _make_state_machine(config)
        anchor_map = {
            anchor.frame_idx: ManualAnchor(
                frame_idx=anchor.frame_idx,
                bbox=cache.to_analysis_bbox(anchor.bbox),
                source=anchor.source,
            )
            for anchor in local_anchors
        }
        auto_prompt_map = {
            prompt.frame_idx: prompt for prompt in prompt_candidates
            if prompt.source == "auto"
        }
        observations: list[TrackObservation | None] = [None] * cache.total_frames
        review_masks: dict[int, np.ndarray] = {}

        debug_writer = None
        if debug_viz_path is not None:
            debug_writer = DebugVizWriter(
                output_path=debug_viz_path,
                frame_rate=cache.frame_rate,
                width=cache.width,
                height=cache.height,
            )

        last_processed = -1
        try:
            for sam_mask in provider.propagate(cache.path, prompt_candidates):
                idx = sam_mask.frame_idx
                if idx < 0 or idx >= cache.total_frames or idx <= last_processed:
                    continue
                source_idx = idx + cache.frame_offset

                for missing_idx in range(last_processed + 1, idx):
                    missing_obs = state.update(
                        frame_idx=missing_idx + cache.frame_offset,
                        center=None,
                        bbox=feature_tracker.bbox,
                        confidence=0.0,
                        visibility=0.0,
                        source="missing",
                    )
                    observations[missing_idx] = _to_source_observation(
                        missing_obs, cache
                    )

                frame_bgr = cache.read(idx)
                valid_mask = _sanitize_mask(
                    sam_mask.mask,
                    feature_tracker.median_mask_area,
                )
                manual_anchor = anchor_map.get(idx)
                auto_prompt = auto_prompt_map.get(idx)
                if (
                    manual_anchor is None
                    and auto_prompt is None
                    and feature_tracker.initialized
                    and not _mask_matches_prediction(
                        valid_mask,
                        feature_tracker.bbox,
                        predicted_center=state.predicted_center,
                        gate_multiplier=(
                            8.0
                            if state.state
                            in (TrackingState.OCCLUDED, TrackingState.LOST)
                            else 3.0
                        ),
                    )
                ):
                    valid_mask = np.zeros_like(valid_mask, dtype=bool)

                if manual_anchor is not None:
                    measurement = feature_tracker.initialize(
                        frame_bgr,
                        manual_anchor.bbox,
                        valid_mask,
                    )
                    observation = state.update(
                        frame_idx=source_idx,
                        center=measurement.center,
                        bbox=measurement.bbox,
                        confidence=1.0,
                        visibility=1.0,
                        source="manual",
                        manual=True,
                    )
                elif auto_prompt is not None:
                    consistency = _mask_bbox_iou(
                        valid_mask,
                        auto_prompt.bbox,
                    )
                    if consistency < 0.05:
                        observation = state.update(
                            frame_idx=source_idx,
                            center=None,
                            bbox=None,
                            confidence=0.0,
                            visibility=0.0,
                            source="sam2_prompt_rejected",
                        )
                    else:
                        measurement = feature_tracker.initialize(
                            frame_bgr,
                            auto_prompt.bbox,
                            valid_mask,
                        )
                        confidence = (
                            0.45 * measurement.confidence
                            + 0.25 * sam_mask.quality
                            + 0.20 * auto_prompt.score
                            + 0.10 * consistency
                        )
                        observation = state.update(
                            frame_idx=source_idx,
                            center=measurement.center,
                            bbox=measurement.bbox,
                            confidence=confidence,
                            visibility=measurement.visibility,
                            source="sam2_init",
                            reset_motion=True,
                        )
                elif not feature_tracker.initialized:
                    initial_bbox = bbox_from_mask(valid_mask)
                    if initial_bbox is None:
                        measurement = None
                        observation = state.update(
                            frame_idx=source_idx,
                            center=None,
                            bbox=None,
                            confidence=0.0,
                            visibility=0.0,
                            source="sam2_missing",
                        )
                    else:
                        measurement = feature_tracker.initialize(
                            frame_bgr,
                            initial_bbox,
                            valid_mask,
                        )
                        observation = state.update(
                            frame_idx=source_idx,
                            center=measurement.center,
                            bbox=measurement.bbox,
                            confidence=0.70 * measurement.confidence
                            + 0.30 * sam_mask.quality,
                            visibility=measurement.visibility,
                            source="sam2_mask_init",
                        )
                else:
                    measurement = feature_tracker.update(
                        frame_bgr,
                        valid_mask,
                        predicted_center=state.predicted_center,
                    )
                    confidence = (
                        0.85 * measurement.confidence
                        + 0.15 * sam_mask.quality
                    )
                    observation = state.update(
                        frame_idx=source_idx,
                        center=measurement.center,
                        bbox=measurement.bbox,
                        confidence=confidence,
                        visibility=measurement.visibility,
                        source=(
                            "sam2_lk"
                            if measurement.center is not None
                            else "occluded"
                        ),
                    )

                observations[idx] = _to_source_observation(observation, cache)
                if (
                    observation.center is None
                    or observation.confidence < 0.65
                    or observation.state
                    in (TrackingState.OCCLUDED, TrackingState.LOST)
                ):
                    review_masks[source_idx] = valid_mask.copy()
                if debug_writer is not None:
                    debug_writer.write(
                        frame_bgr,
                        source_idx,
                        _debug_state(
                            observation,
                            velocity=state.velocity,
                            detection_used=(
                                manual_anchor is not None or auto_prompt is not None
                            ),
                        ),
                    )

                last_processed = idx
                if progress_cb:
                    progress_cb(
                        int(cache.total_frames * 0.25 + (idx + 1) * 0.75),
                        cache.total_frames,
                    )

            for missing_idx in range(last_processed + 1, cache.total_frames):
                missing_obs = state.update(
                    frame_idx=missing_idx + cache.frame_offset,
                    center=None,
                    bbox=feature_tracker.bbox,
                    confidence=0.0,
                    visibility=0.0,
                    source="missing",
                )
                observations[missing_idx] = _to_source_observation(
                    missing_obs, cache
                )
        finally:
            if debug_writer is not None:
                debug_writer.close()

        return AnalysisResult(
            observations=[obs for obs in observations if obs is not None],
            width=cache.source_width,
            height=cache.source_height,
            frame_rate=cache.frame_rate,
            backend="hybrid",
            review_masks=review_masks,
        )


def _build_sam_prompts(
    cache: AnalysisFrameCache,
    detector,
    anchors: list[ManualAnchor],
    config: StabilizerConfig,
) -> list[Sam2Prompt]:
    prompts = [
        Sam2Prompt(
            frame_idx=anchor.frame_idx,
            bbox=cache.to_analysis_bbox(anchor.bbox),
            source=anchor.source,
            score=1.0,
        )
        for anchor in anchors
        if 0 <= anchor.frame_idx < cache.total_frames
    ]

    has_early_manual = any(prompt.frame_idx <= 5 for prompt in prompts)
    if not has_early_manual:
        gate = CandidateGate()
        motion_detector = MotionFallbackDetector(config)
        scan_limit = min(cache.total_frames, 91)
        step = max(1, config.detection_interval // 2)
        for frame_idx in range(0, scan_limit, step):
            frame_bgr = cache.read(frame_idx)
            motion_hints = motion_detector.detect_candidates(frame_bgr)
            candidate, _ = gate.select(
                _apply_motion_hints(
                    detector.detect_candidates(frame_bgr),
                    motion_hints,
                ),
                frame_shape=frame_bgr.shape[:2],
            )
            if candidate is not None:
                prompts.append(
                    Sam2Prompt(
                        frame_idx=frame_idx,
                        bbox=candidate.bbox,
                        source="auto",
                        score=candidate.score,
                    )
                )
                break

    if not prompts:
        raise RuntimeError(
            "Could not initialize aircraft tracking in the first 90 frames"
        )

    prompts.sort(key=lambda prompt: prompt.frame_idx)
    return prompts


def _make_state_machine(config: StabilizerConfig) -> TrackingStateMachine:
    return TrackingStateMachine(
        low_threshold=config.tracking_low_confidence,
        recovery_threshold=config.tracking_recovery_confidence,
        low_frames=config.tracking_low_frames,
        recovery_frames=config.tracking_recovery_frames,
        lost_frames=config.tracker_quality_timeout,
    )


def _sanitize_mask(
    mask: np.ndarray,
    median_area: float | None,
) -> np.ndarray:
    mask = mask.astype(bool)
    area = float(mask.sum())
    if area <= 0:
        return np.zeros_like(mask, dtype=bool)
    if area > mask.shape[0] * mask.shape[1] * 0.50:
        return np.zeros_like(mask, dtype=bool)
    if median_area and area > median_area * 4.0:
        return np.zeros_like(mask, dtype=bool)
    return mask


def _mask_matches_prediction(
    mask: np.ndarray,
    predicted_bbox: tuple[int, int, int, int] | None,
    predicted_center: tuple[float, float] | None = None,
    gate_multiplier: float = 3.0,
) -> bool:
    if predicted_bbox is None:
        return bool(mask.any())
    mask_bbox = bbox_from_mask(mask)
    if mask_bbox is None:
        return False
    mask_area = float(mask_bbox[2] * mask_bbox[3])
    predicted_area = max(float(predicted_bbox[2] * predicted_bbox[3]), 1.0)
    if mask_area / predicted_area > 8.0:
        return False

    mask_aspect = mask_bbox[2] / max(float(mask_bbox[3]), 1.0)
    predicted_aspect = predicted_bbox[2] / max(float(predicted_bbox[3]), 1.0)
    aspect_ratio = mask_aspect / max(predicted_aspect, 1e-6)
    if aspect_ratio > 4.0 or aspect_ratio < 1.0 / 4.0:
        return False

    predicted_center = predicted_center or _bbox_center(predicted_bbox)
    mask_center = _bbox_center(mask_bbox)
    distance = float(np.hypot(
        mask_center[0] - predicted_center[0],
        mask_center[1] - predicted_center[1],
    ))
    gate_radius = max(
        30.0,
        gate_multiplier
        * float(np.hypot(predicted_bbox[2], predicted_bbox[3])),
    )
    return (
        distance <= gate_radius
        and _mask_bbox_iou(mask, predicted_bbox) >= 0.01
    )


def _mask_bbox_iou(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    mask_bbox = bbox_from_mask(mask)
    if mask_bbox is None:
        return 0.0
    ax, ay, aw, ah = mask_bbox
    bx, by, bw, bh = bbox
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / max(float(union), 1.0)


def _apply_motion_hints(
    candidates: list[DetectionCandidate],
    motion_hints: list[DetectionCandidate],
) -> list[DetectionCandidate]:
    if not candidates or not motion_hints:
        return candidates
    boosted = []
    for candidate in candidates:
        overlap = max(
            _bbox_iou(candidate.bbox, hint.bbox)
            for hint in motion_hints
        )
        boosted.append(
            DetectionCandidate(
                bbox=candidate.bbox,
                score=min(1.0, candidate.score + 0.05 * overlap),
                label=candidate.label,
                source=candidate.source,
            )
        )
    return boosted


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / max(float(union), 1.0)


def _normalize_frame_range(
    total_frames: int,
    frame_range: tuple[int, int] | None,
) -> tuple[int, int | None]:
    if frame_range is None:
        return 0, total_frames - 1 if total_frames > 0 else None
    start, end = frame_range
    start = max(0, start)
    if total_frames > 0:
        start = min(start, total_frames - 1)
        end = min(max(end, start), total_frames - 1)
    else:
        end = max(end, start)
    return start, end


def _to_source_observation(
    observation: TrackObservation,
    cache: AnalysisFrameCache,
) -> TrackObservation:
    return TrackObservation(
        frame_idx=observation.frame_idx,
        center=cache.to_source_point(observation.center),
        bbox=cache.to_source_bbox(observation.bbox),
        confidence=observation.confidence,
        visibility=observation.visibility,
        state=observation.state,
        source=observation.source,
        predicted_center=cache.to_source_point(observation.predicted_center),
        measured=observation.measured,
        rejected_candidates=observation.rejected_candidates,
    )


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _debug_state(
    observation: TrackObservation,
    velocity: tuple[float, float],
    detection_used: bool,
) -> dict:
    return {
        "bbox": observation.bbox,
        "centroid": observation.center,
        "pred_centroid": observation.predicted_center,
        "match_score": observation.confidence,
        "match_source": observation.source,
        "velocity": velocity,
        "detection_used": detection_used,
        "tracking_state": observation.state.value,
        "visibility": observation.visibility,
        "rejected_candidates": observation.rejected_candidates,
    }
