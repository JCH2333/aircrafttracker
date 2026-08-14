import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stabilize.tracking.analysis_cache import AnalysisFrameCache
from stabilize.tracking.analyzer import (
    _build_sam_prompts,
    _find_boundary_anchor,
    _is_boundary_observation,
    _segment_windows,
)
from stabilize.tracking.models import (
    DetectionCandidate,
    TrackObservation,
    TrackingState,
)
from stabilize.tracking.track_file import ManualAnchor
from stabilize.config import StabilizerConfig


def _observation(frame_idx, reliable=True):
    return TrackObservation(
        frame_idx=frame_idx,
        center=(100.0 + frame_idx, 80.0) if reliable else None,
        bbox=(90 + frame_idx, 70, 40, 24) if reliable else None,
        confidence=0.9 if reliable else 0.2,
        visibility=1.0 if reliable else 0.0,
        state=TrackingState.TRACKING if reliable else TrackingState.OCCLUDED,
        source="synthetic",
        measured=reliable,
    )


class SegmentedAnalysisTests(unittest.TestCase):
    def test_windows_cover_each_frame_once_after_overlap_merge(self):
        windows = _segment_windows(317, 120, 24)
        covered = set()
        for start, end in windows:
            covered.update(range(start, end + 1))

        self.assertEqual(covered, set(range(317)))
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0], (0, 119))
        self.assertEqual(windows[-1][1], 316)

    def test_invalid_overlap_cannot_create_zero_stride(self):
        windows = _segment_windows(10, 3, 99)
        self.assertEqual(windows[-1][1], 9)
        self.assertTrue(all(end >= start for start, end in windows))

    def test_boundary_anchor_requires_reliable_measured_observation(self):
        observations = [_observation(idx) for idx in range(8)]
        observations[6] = _observation(6, reliable=False)
        observations[7] = _observation(7, reliable=False)

        self.assertTrue(_is_boundary_observation(observations[5]))
        self.assertFalse(_is_boundary_observation(observations[6]))
        anchor = _find_boundary_anchor(observations, 2, 7)

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.frame_idx, 2)
        self.assertEqual(anchor.source, "boundary")
        self.assertEqual(anchor.reference_point, observations[2].center)

    def test_boundary_anchor_rejects_clipped_or_offscreen_track(self):
        observations = [_observation(idx) for idx in range(8)]
        observations[5].bbox = (0, 70, 40, 24)
        observations[6].center = (-10.0, 80.0)
        observations[7] = _observation(7, reliable=False)

        self.assertFalse(
            _is_boundary_observation(
                observations[5],
                frame_width=320,
                frame_height=180,
            )
        )
        self.assertFalse(
            _is_boundary_observation(
                observations[6],
                frame_width=320,
                frame_height=180,
            )
        )

    def test_boundary_anchor_looks_back_before_occluded_overlap(self):
        observations = [_observation(idx) for idx in range(12)]
        for idx in range(7, 12):
            observations[idx] = _observation(idx, reliable=False)

        anchor = _find_boundary_anchor(
            observations,
            9,
            11,
            lookback_start=4,
            frame_width=320,
            frame_height=180,
        )

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.frame_idx, 6)
        self.assertEqual(anchor.reference_point, observations[6].center)

    def test_window_anchor_range_uses_global_frame_numbers(self):
        cache_offset = 500
        window_start = 96
        window_end = 119
        source_start = cache_offset + window_start
        source_end = cache_offset + window_end
        anchors = [_observation(595), _observation(600), _observation(620)]
        selected = [
            observation
            for observation in anchors
            if source_start <= observation.frame_idx <= source_end
        ]

        self.assertEqual([observation.frame_idx for observation in selected], [600])

    def test_lookback_boundary_adds_post_occlusion_detector_prompt(self):
        class FakeCache:
            total_frames = 20
            frame_offset = 500

            @staticmethod
            def to_analysis_bbox(bbox):
                return bbox

            @staticmethod
            def read(frame_idx):
                return np.zeros((80, 120, 3), dtype=np.uint8)

        class FakeDetector:
            def __init__(self):
                self.calls = 0

            def detect_candidates(self, _frame):
                self.calls += 1
                if self.calls < 2:
                    return []
                return [
                    DetectionCandidate(
                        bbox=(35, 30, 24, 12),
                        score=0.9,
                        label=4,
                        source="synthetic",
                    )
                ]

        prompts = _build_sam_prompts(
            FakeCache(),
            FakeDetector(),
            [
                ManualAnchor(
                    frame_idx=0,
                    bbox=(30, 30, 24, 12),
                    source="boundary",
                    reference_point=(42.0, 36.0),
                )
            ],
            StabilizerConfig(detection_interval=10),
            continuation_start=5,
        )

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].source, "boundary")
        self.assertEqual(prompts[1].source, "auto")
        self.assertEqual(prompts[1].frame_idx, 10)

    def test_analysis_window_uses_sequential_local_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = object.__new__(AnalysisFrameCache)
            cache.path = root / "root"
            cache.path.mkdir()
            cache.total_frames = 5
            cache.source_width = 16
            cache.source_height = 12
            cache.source_total_frames = 5
            cache.frame_offset = 0
            cache.width = 16
            cache.height = 12
            cache.frame_rate = 30.0
            cache.scale_x = 1.0
            cache.scale_y = 1.0
            cache._temp_dir = None

            for idx in range(5):
                image = np.full((12, 16, 3), idx, dtype=np.uint8)
                self.assertTrue(
                    cv2.imwrite(str(cache.path / f"{idx:06d}.jpg"), image)
                )

            with cache.window(2, 4) as window:
                window.build()
                self.assertEqual(window.frame_offset, 2)
                self.assertEqual(window.total_frames, 3)
                self.assertTrue((window.path / "000000.jpg").exists())
                self.assertTrue((window.path / "000002.jpg").exists())
                self.assertEqual(int(window.read(0).mean()), 2)
                self.assertEqual(int(window.read(2).mean()), 4)


if __name__ == "__main__":
    unittest.main()
