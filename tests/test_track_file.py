import tempfile
import unittest
from pathlib import Path

from stabilize.tracking.models import FlaggedSegment
from stabilize.tracking.track_file import ManualAnchor, TrackFile


class TrackFileTests(unittest.TestCase):
    def test_round_trip_and_replace_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clip.track.json"
            track_file = TrackFile(
                source_path=str(Path(temp_dir) / "clip.mov"),
                width=1920,
                height=1080,
                fps=29.97,
                anchors=[ManualAnchor(100, (10, 20, 30, 40))],
                flagged_segments=[
                    FlaggedSegment(90, 110, "occluded", 21)
                ],
            )
            track_file.add_anchor(ManualAnchor(100, (11, 22, 33, 44)))
            track_file.save(path)

            loaded = TrackFile.load(path)
            self.assertEqual(len(loaded.anchors), 1)
            self.assertEqual(loaded.anchors[0].bbox, (11, 22, 33, 44))
            self.assertEqual(loaded.flagged_segments[0].length, 21)
            self.assertTrue(
                loaded.is_compatible(
                    Path(temp_dir) / "clip.mov",
                    1920,
                    1080,
                    fps=29.97,
                )
            )
            self.assertFalse(
                loaded.is_compatible(
                    Path(temp_dir) / "clip.mov",
                    1920,
                    1080,
                    fps=25.0,
                )
            )


if __name__ == "__main__":
    unittest.main()
