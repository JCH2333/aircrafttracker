import unittest

import numpy as np

from stabilize.tracking.gating import CandidateGate
from stabilize.tracking.analyzer import (
    _apply_motion_hints,
    _mask_matches_prediction,
)
from stabilize.tracking.models import DetectionCandidate


class CandidateGateTests(unittest.TestCase):
    def test_rejects_full_frame_motion_candidate(self):
        gate = CandidateGate()
        gate.record_bbox((500, 300, 400, 120))

        selected, rejected = gate.select(
            [
                DetectionCandidate(
                    (0, 0, 1920, 1080),
                    0.99,
                    source="motion",
                ),
                DetectionCandidate(
                    (520, 310, 410, 125),
                    0.80,
                    label=4,
                ),
            ],
            frame_shape=(1080, 1920),
            predicted_center=(710, 370),
            predicted_bbox=(500, 300, 400, 120),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.bbox, (520, 310, 410, 125))
        self.assertTrue(
            any("50%" in item.reason or "4x" in item.reason for item in rejected)
        )

    def test_rejects_distant_candidate(self):
        gate = CandidateGate()
        gate.record_bbox((100, 100, 200, 80))
        selected, rejected = gate.select(
            [DetectionCandidate((1500, 800, 200, 80), 0.95, label=4)],
            frame_shape=(1080, 1920),
            predicted_center=(200, 140),
            predicted_bbox=(100, 100, 200, 80),
        )
        self.assertIsNone(selected)
        self.assertEqual(len(rejected), 1)
        self.assertIn("motion gate", rejected[0].reason)

    def test_manual_candidate_bypasses_gating(self):
        gate = CandidateGate()
        selected, rejected = gate.select(
            [DetectionCandidate((0, 0, 1920, 1080), 1.0, source="manual")],
            frame_shape=(1080, 1920),
            manual=True,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(rejected, ())

    def test_motion_hint_cannot_create_a_detection(self):
        motion = DetectionCandidate(
            (100, 100, 300, 120),
            0.49,
            source="motion",
        )
        self.assertEqual(_apply_motion_hints([], [motion]), [])

    def test_recovery_gate_allows_detached_sam_component(self):
        mask = np.zeros((240, 360), dtype=bool)
        mask[100:140, 40:120] = True

        self.assertTrue(
            _mask_matches_prediction(
                mask,
                predicted_bbox=(160, 100, 80, 40),
                predicted_center=(200.0, 120.0),
                gate_multiplier=8.0,
            )
        )
        self.assertFalse(
            _mask_matches_prediction(
                mask,
                predicted_bbox=(160, 100, 80, 40),
                predicted_center=(200.0, 120.0),
                gate_multiplier=3.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
