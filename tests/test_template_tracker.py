import unittest
from unittest.mock import patch

import numpy as np

from stabilize.config import StabilizerConfig
from stabilize.stabilization.template_tracker import TemplateTracker


class TemplateTrackerTests(unittest.TestCase):
    def test_gated_redetection_reinitializes_immediately(self):
        frame = np.zeros((240, 360, 3), dtype=np.uint8)
        frame[30:80, 20:100] = 80
        frame[140:200, 220:310] = 220

        tracker = TemplateTracker(StabilizerConfig())
        tracker.init_from_detection(frame, (20, 30, 80, 50))
        tracker.frames_since_detect = 40

        tracker.init_from_detection(frame, (220, 140, 90, 60))

        self.assertEqual(tracker.template_bbox, (220, 140, 90, 60))
        self.assertEqual(tracker.current_centroid, (265.0, 170.0))
        self.assertEqual(tracker.frames_since_detect, 0)
        self.assertEqual(tracker.last_match_score, 1.0)
        self.assertFalse(hasattr(tracker, "_transition"))

    def test_template_is_frozen_below_update_confidence(self):
        config = StabilizerConfig(
            template_search_margin=0,
            template_match_threshold=0.40,
            template_quality_score=0.55,
            template_update_min_confidence=0.80,
        )
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        frame[30:80, 50:130] = 100
        tracker = TemplateTracker(config)
        tracker.init_from_detection(frame, (50, 30, 80, 50))
        original_template = tracker.template_raw.copy()

        changed = frame.copy()
        changed[30:80, 50:130] = 220
        with patch(
            "stabilize.stabilization.template_tracker.cv2.matchTemplate",
            return_value=np.ones((1, 1), dtype=np.float32),
        ), patch(
            "stabilize.stabilization.template_tracker.cv2.minMaxLoc",
            return_value=(0.0, 0.70, (0, 0), (0, 0)),
        ):
            center = tracker.update(changed)

        self.assertIsNotNone(center)
        self.assertTrue(np.array_equal(tracker.template_raw, original_template))

    def test_high_score_match_outside_motion_gate_is_rejected(self):
        config = StabilizerConfig(
            template_search_margin=120,
            template_match_threshold=0.40,
            template_quality_score=0.55,
        )
        frame = np.zeros((160, 260, 3), dtype=np.uint8)
        frame[50:100, 80:160] = 180
        tracker = TemplateTracker(config)
        tracker.init_from_detection(frame, (80, 50, 80, 50))

        with patch(
            "stabilize.stabilization.template_tracker.cv2.matchTemplate",
            return_value=np.ones((121, 121), dtype=np.float32),
        ), patch(
            "stabilize.stabilization.template_tracker.cv2.minMaxLoc",
            return_value=(0.0, 0.99, (0, 0), (160, 90)),
        ):
            center = tracker.update(frame)

        self.assertIsNone(center)
        self.assertEqual(tracker.last_match_score, 0.0)
        self.assertEqual(tracker.current_centroid, (120.0, 75.0))


if __name__ == "__main__":
    unittest.main()
