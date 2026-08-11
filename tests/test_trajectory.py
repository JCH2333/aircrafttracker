import unittest

from stabilize.tracking.models import (
    TrackObservation,
    TrackingState,
)
from stabilize.tracking.trajectory import smooth_observations


def build_linear_observations(total, gap_start, gap_length):
    observations = []
    gap_end = gap_start + gap_length
    for idx in range(total):
        missing = gap_start <= idx < gap_end
        observations.append(
            TrackObservation(
                frame_idx=idx,
                center=None if missing else (200.0 + idx * 3.0, 400.0 - idx),
                bbox=None,
                confidence=0.0 if missing else 0.95,
                visibility=0.0 if missing else 1.0,
                state=(
                    TrackingState.OCCLUDED
                    if missing
                    else TrackingState.TRACKING
                ),
                source="synthetic",
            )
        )
    return observations


class TrajectoryTests(unittest.TestCase):
    def test_refuses_to_fabricate_frame_center_without_observations(self):
        observations = build_linear_observations(20, 0, 20)
        with self.assertRaisesRegex(RuntimeError, "No reliable target"):
            smooth_observations(observations, 1920, 1080)

    def test_repairs_short_medium_and_long_occlusions(self):
        for gap_length in (15, 45, 90):
            with self.subTest(gap_length=gap_length):
                observations = build_linear_observations(
                    total=160,
                    gap_start=30,
                    gap_length=gap_length,
                )
                centers, segments = smooth_observations(
                    observations,
                    frame_width=1920,
                    frame_height=1080,
                )
                for idx in range(30, 30 + gap_length):
                    self.assertAlmostEqual(
                        centers[idx][0],
                        200.0 + idx * 3.0,
                        delta=1.5,
                    )
                    self.assertAlmostEqual(
                        centers[idx][1],
                        400.0 - idx,
                        delta=1.5,
                    )
                self.assertEqual(len(segments), 1)
                self.assertEqual(segments[0].length, gap_length)

    def test_manual_anchor_has_high_weight(self):
        observations = build_linear_observations(40, 10, 20)
        observations[20] = TrackObservation(
            frame_idx=20,
            center=(500.0, 300.0),
            bbox=(450, 270, 100, 60),
            confidence=1.0,
            visibility=1.0,
            state=TrackingState.MANUAL_ANCHOR,
            source="manual",
        )
        centers, _ = smooth_observations(observations, 1920, 1080)
        self.assertAlmostEqual(centers[20][0], 500.0, delta=1.0)
        self.assertAlmostEqual(centers[20][1], 300.0, delta=1.0)

    def test_long_gap_requires_manual_confirmation(self):
        observations = build_linear_observations(160, 20, 91)
        _, segments = smooth_observations(observations, 1920, 1080)
        self.assertEqual(len(segments), 1)
        self.assertIn("manual confirmation required", segments[0].reason)

    def test_single_missing_measurement_is_flagged(self):
        observations = build_linear_observations(20, 10, 1)
        _, segments = smooth_observations(observations, 1920, 1080)
        self.assertEqual(len(segments), 1)
        self.assertEqual(
            (segments[0].start_frame, segments[0].end_frame),
            (10, 10),
        )

    def test_occluded_center_does_not_pull_rts_trajectory(self):
        observations = build_linear_observations(40, 15, 5)
        observations[15] = TrackObservation(
            frame_idx=15,
            center=(900.0, 100.0),
            bbox=None,
            confidence=0.55,
            visibility=0.2,
            state=TrackingState.OCCLUDED,
            source="occluded",
            measured=True,
        )
        centers, _ = smooth_observations(observations, 1920, 1080)

        self.assertAlmostEqual(centers[15][0], 200.0 + 15 * 3.0, delta=2.0)
        self.assertAlmostEqual(centers[15][1], 400.0 - 15, delta=2.0)

    def test_mask_recovery_is_a_strong_identity_anchor(self):
        observations = build_linear_observations(40, 0, 0)
        observations[20] = TrackObservation(
            frame_idx=20,
            center=(500.0, 300.0),
            bbox=(450, 270, 100, 60),
            confidence=0.75,
            visibility=0.8,
            state=TrackingState.TRACKING,
            source="sam2_mask_recovery",
            measured=True,
        )
        observations[21] = TrackObservation(
            frame_idx=21,
            center=(480.0, 300.0),
            bbox=(430, 270, 100, 60),
            confidence=0.75,
            visibility=0.8,
            state=TrackingState.TRACKING,
            source="sam2_mask_recovery",
            measured=True,
        )

        centers, _ = smooth_observations(observations, 1920, 1080)

        self.assertAlmostEqual(centers[20][0], 500.0, delta=20.0)
        self.assertAlmostEqual(centers[21][0], 480.0, delta=20.0)


if __name__ == "__main__":
    unittest.main()
