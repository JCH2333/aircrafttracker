import unittest

from stabilize.tracking.models import TrackObservation, TrackingState
from stabilize.tracking.reanalysis import build_reanalysis_ranges
from stabilize.tracking.track_file import ManualAnchor


def _observation(frame_idx, reliable=True):
    return TrackObservation(
        frame_idx=frame_idx,
        center=(100.0 + frame_idx, 80.0) if reliable else None,
        bbox=(80 + frame_idx, 60, 40, 40) if reliable else None,
        confidence=0.95 if reliable else 0.0,
        visibility=1.0 if reliable else 0.0,
        state=(
            TrackingState.TRACKING
            if reliable
            else TrackingState.OCCLUDED
        ),
        source="synthetic",
    )


class ReanalysisTests(unittest.TestCase):
    def test_manual_anchor_recomputes_only_between_reliable_boundaries(self):
        observations = [
            _observation(idx, reliable=not 30 <= idx <= 50)
            for idx in range(80)
        ]
        manual = ManualAnchor(40, (120, 70, 50, 30))

        ranges = build_reanalysis_ranges(
            observations,
            existing_anchors=[manual],
            new_anchors=[manual],
        )

        self.assertEqual(len(ranges), 1)
        self.assertEqual(
            (ranges[0].start_frame, ranges[0].end_frame),
            (29, 51),
        )
        anchors = {anchor.frame_idx: anchor for anchor in ranges[0].anchors}
        self.assertEqual(anchors[40].source, "manual")
        self.assertEqual(anchors[29].source, "boundary")
        self.assertEqual(anchors[51].source, "boundary")

    def test_overlapping_manual_ranges_are_merged(self):
        observations = [
            _observation(idx, reliable=not 20 <= idx <= 60)
            for idx in range(90)
        ]
        anchors = [
            ManualAnchor(30, (100, 60, 50, 30)),
            ManualAnchor(50, (130, 60, 50, 30)),
        ]

        ranges = build_reanalysis_ranges(
            observations,
            existing_anchors=anchors,
            new_anchors=anchors,
        )

        self.assertEqual(len(ranges), 1)
        self.assertEqual(
            (ranges[0].start_frame, ranges[0].end_frame),
            (19, 61),
        )


if __name__ == "__main__":
    unittest.main()
