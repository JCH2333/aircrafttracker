import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stabilize.config import StabilizerConfig
from stabilize.tracking.masked_feature_tracker import MaskedFeatureTracker


class MaskedFeatureTrackerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.frame = np.zeros((240, 360, 3), dtype=np.uint8)
        for _ in range(80):
            x = int(rng.integers(90, 250))
            y = int(rng.integers(80, 165))
            cv2.circle(self.frame, (x, y), 2, (255, 255, 255), -1)
        self.bbox = (80, 70, 190, 110)
        self.mask = np.zeros((240, 360), dtype=bool)
        self.mask[70:180, 80:270] = True

    def test_tracks_fixed_anchor_through_translation(self):
        tracker = MaskedFeatureTracker(StabilizerConfig())
        initial = tracker.initialize(self.frame, self.bbox, self.mask)

        transform = np.float32([[1, 0, 6], [0, 1, 4]])
        shifted = cv2.warpAffine(self.frame, transform, (360, 240))
        shifted_mask = cv2.warpAffine(
            self.mask.astype(np.uint8),
            transform,
            (360, 240),
        ).astype(bool)
        measurement = tracker.update(shifted, shifted_mask)

        self.assertIsNotNone(measurement.center)
        self.assertAlmostEqual(
            measurement.center[0], initial.center[0] + 6, delta=1.0
        )
        self.assertAlmostEqual(
            measurement.center[1], initial.center[1] + 4, delta=1.0
        )
        self.assertGreaterEqual(measurement.inlier_count, 4)

    def test_rejects_fully_occluded_points(self):
        tracker = MaskedFeatureTracker(StabilizerConfig())
        tracker.initialize(self.frame, self.bbox, self.mask)
        occluded_mask = np.zeros_like(self.mask)
        measurement = tracker.update(self.frame, occluded_mask)
        self.assertIsNone(measurement.center)
        self.assertEqual(measurement.confidence, 0.0)

    def test_recovers_fixed_anchor_when_mask_returns(self):
        tracker = MaskedFeatureTracker(StabilizerConfig())
        initial = tracker.initialize(self.frame, self.bbox, self.mask)
        tracker.update(self.frame, np.zeros_like(self.mask))

        transform = np.float32([[1, 0, 12], [0, 1, 7]])
        shifted = cv2.warpAffine(self.frame, transform, (360, 240))
        shifted_mask = cv2.warpAffine(
            self.mask.astype(np.uint8),
            transform,
            (360, 240),
        ).astype(bool)
        recovered = tracker.update(shifted, shifted_mask)

        self.assertIsNotNone(recovered.center)
        self.assertAlmostEqual(
            recovered.center[0], initial.center[0] + 12, delta=2.0
        )
        self.assertAlmostEqual(
            recovered.center[1], initial.center[1] + 7, delta=2.0
        )
        self.assertGreaterEqual(recovered.confidence, 0.65)
        self.assertEqual(recovered.bbox[2:], initial.bbox[2:])

    def test_rejects_implausible_scale_instead_of_resizing_box(self):
        tracker = MaskedFeatureTracker(StabilizerConfig())
        initial = tracker.initialize(self.frame, self.bbox, self.mask)

        transform = np.float32([[1.8, 0, 0], [0, 1.8, 0]])
        scaled = cv2.warpAffine(self.frame, transform, (360, 240))
        scaled_mask = cv2.warpAffine(
            self.mask.astype(np.uint8),
            transform,
            (360, 240),
        ).astype(bool)
        measurement = tracker.update(scaled, scaled_mask)

        self.assertIsNone(measurement.center)
        self.assertEqual(measurement.bbox[2:], initial.bbox[2:])

    def test_affine_scale_does_not_accumulate_into_tracking_box(self):
        tracker = MaskedFeatureTracker(StabilizerConfig())
        initial = tracker.initialize(self.frame, self.bbox, self.mask)

        transform = np.float32([[1.05, 0, 3], [0, 1.05, 2]])
        scaled = cv2.warpAffine(self.frame, transform, (360, 240))
        scaled_mask = cv2.warpAffine(
            self.mask.astype(np.uint8),
            transform,
            (360, 240),
        ).astype(bool)
        measurement = tracker.update(scaled, scaled_mask)

        self.assertIsNotNone(measurement.center)
        self.assertEqual(measurement.bbox[2:], initial.bbox[2:])

    def test_high_quality_mask_corrects_flow_drift_before_recovery_state(self):
        tracker = MaskedFeatureTracker(StabilizerConfig())
        initial = tracker.initialize(self.frame, self.bbox, self.mask)

        transform = np.float32([[1, 0, -50], [0, 1, 0]])
        shifted = cv2.warpAffine(self.frame, transform, (360, 240))
        shifted_mask = cv2.warpAffine(
            self.mask.astype(np.uint8),
            transform,
            (360, 240),
        ).astype(bool)

        old_points = tracker.prev_points.reshape(-1, 1, 2)
        bad_points = old_points + np.float32([20.0, 0.0])
        statuses = np.ones((len(old_points), 1), dtype=np.uint8)
        errors = np.zeros((len(old_points), 1), dtype=np.float32)
        bad_transform = np.float32([[1, 0, 20], [0, 1, 0]])
        inliers = np.ones((len(old_points), 1), dtype=np.uint8)

        with patch(
            "stabilize.tracking.masked_feature_tracker.cv2.calcOpticalFlowPyrLK",
            return_value=(bad_points, statuses, errors),
        ), patch(
            "stabilize.tracking.masked_feature_tracker.cv2.estimateAffinePartial2D",
            return_value=(bad_transform, inliers),
        ):
            measurement = tracker.update(
                shifted,
                shifted_mask,
                predicted_center=initial.center,
                recovery_mode=False,
            )

        self.assertTrue(measurement.mask_recovery)
        self.assertAlmostEqual(
            measurement.center[0],
            initial.center[0] - 50,
            delta=2.0,
        )
        self.assertAlmostEqual(
            measurement.center[1],
            initial.center[1],
            delta=2.0,
        )

    def test_mask_recovery_can_coast_without_features(self):
        tracker = MaskedFeatureTracker(StabilizerConfig())
        tracker.initialize(self.frame, self.bbox, self.mask)
        tracker.prev_points = None

        with patch(
            "stabilize.tracking.masked_feature_tracker._extract_points",
            return_value=None,
        ):
            measurement = tracker.update(
                self.frame,
                self.mask,
                predicted_center=tracker.anchor,
                recovery_mode=True,
            )

        self.assertIsNotNone(measurement.center)
        self.assertTrue(measurement.mask_recovery)
        self.assertEqual(measurement.point_count, 0)
        self.assertLess(measurement.confidence, 0.65)


if __name__ == "__main__":
    unittest.main()
