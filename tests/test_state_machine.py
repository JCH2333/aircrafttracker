import unittest

from stabilize.tracking.models import TrackingState
from stabilize.tracking.state_machine import TrackingStateMachine


class TrackingStateMachineTests(unittest.TestCase):
    def test_occlusion_and_recovery_use_hysteresis(self):
        machine = TrackingStateMachine(
            low_threshold=0.45,
            recovery_threshold=0.65,
            low_frames=2,
            recovery_frames=3,
            lost_frames=6,
        )

        for idx in range(3):
            observation = machine.update(
                idx,
                (100.0 + idx, 50.0),
                (80 + idx, 40, 40, 20),
                0.9,
                1.0,
                "test",
            )
        self.assertEqual(observation.state, TrackingState.TRACKING)

        observation = machine.update(3, None, None, 0.0, 0.0, "missing")
        self.assertEqual(observation.state, TrackingState.TRACKING)
        observation = machine.update(4, None, None, 0.0, 0.0, "missing")
        self.assertEqual(observation.state, TrackingState.OCCLUDED)

        for idx in range(5, 7):
            observation = machine.update(
                idx,
                (100.0 + idx, 50.0),
                (80 + idx, 40, 40, 20),
                0.9,
                1.0,
                "test",
            )
            self.assertEqual(observation.state, TrackingState.OCCLUDED)

        observation = machine.update(
            7,
            (107.0, 50.0),
            (87, 40, 40, 20),
            0.9,
            1.0,
            "test",
        )
        self.assertEqual(observation.state, TrackingState.TRACKING)

    def test_marks_long_gap_lost(self):
        machine = TrackingStateMachine(lost_frames=4)
        machine.update(0, (10.0, 10.0), (0, 0, 20, 20), 0.9, 1.0, "test")
        for idx in range(1, 5):
            observation = machine.update(
                idx, None, None, 0.0, 0.0, "missing"
            )
        self.assertEqual(observation.state, TrackingState.LOST)

    def test_manual_anchor_resets_stale_velocity(self):
        machine = TrackingStateMachine()
        machine.update(0, (10.0, 10.0), (0, 0, 20, 20), 0.9, 1.0, "test")
        machine.update(1, (30.0, 10.0), (20, 0, 20, 20), 0.9, 1.0, "test")
        self.assertNotEqual(machine.velocity, (0.0, 0.0))

        machine.update(
            2,
            (300.0, 200.0),
            (280, 180, 40, 40),
            1.0,
            1.0,
            "manual",
            manual=True,
        )

        self.assertEqual(machine.velocity, (0.0, 0.0))
        self.assertEqual(machine.predicted_center, (300.0, 200.0))

    def test_prediction_advances_through_missing_frames(self):
        machine = TrackingStateMachine()
        machine.update(0, (10.0, 10.0), (0, 0, 20, 20), 0.9, 1.0, "test")
        machine.update(1, (20.0, 10.0), (10, 0, 20, 20), 0.9, 1.0, "test")
        one_step = machine.predicted_center
        machine.update(2, None, None, 0.0, 0.0, "missing")
        two_steps = machine.predicted_center

        self.assertGreater(two_steps[0], one_step[0])
        self.assertEqual(two_steps[1], one_step[1])

    def test_recovery_after_gap_resets_stale_velocity(self):
        machine = TrackingStateMachine()
        machine.update(0, (10.0, 10.0), (0, 0, 20, 20), 0.9, 1.0, "test")
        machine.update(1, (30.0, 10.0), (20, 0, 20, 20), 0.9, 1.0, "test")
        machine.update(2, None, None, 0.0, 0.0, "missing")

        machine.update(
            3,
            (200.0, 80.0),
            (180, 60, 40, 40),
            0.9,
            1.0,
            "recovered",
        )

        self.assertEqual(machine.velocity, (0.0, 0.0))
        self.assertEqual(machine.predicted_center, (200.0, 80.0))


if __name__ == "__main__":
    unittest.main()
