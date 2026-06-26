"""PyAV-based video reader with dual mode support.

- analysis mode: yields uint8 BGR frames for detection/tracking (8-bit)
- render mode: yields uint16 RGB frames for final output warping (10-bit in 16-bit container)
"""

from pathlib import Path

import av
import numpy as np


class VideoReader:
    """Read video frames using PyAV with mode-dependent output format.

    Attributes:
        frame_rate: Average frame rate of the video stream.
        total_frames: Total number of video frames (0 if unknown).
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, path: Path | str, mode: str = "analysis"):
        """
        Args:
            path: Path to the input video file.
            mode: "analysis" for uint8 BGR, "render" for uint16 RGB (rgb48le).
        """
        if mode not in ("analysis", "render"):
            raise ValueError(f"mode must be 'analysis' or 'render', got '{mode}'")

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Input video not found: {self.path}")

        self.mode = mode
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"  # multi-threaded decode

        self.width = self.stream.width
        self.height = self.stream.height
        self.total_frames = self.stream.frames or 0
        self.frame_rate = float(self.stream.average_rate) if self.stream.average_rate else 30.0

    @property
    def audio_streams(self):
        """Return list of audio streams for pass-through."""
        return [s for s in self.container.streams if s.type == "audio"]

    @property
    def video_duration(self) -> float:
        """Estimated duration in seconds."""
        if self.total_frames > 0 and self.frame_rate > 0:
            return self.total_frames / self.frame_rate
        return 0.0

    def __iter__(self):
        """Iterate over frames, yielding (ndarray, frame_index).

        Uses a manual frame counter since PyAV's VideoFrame does not
        have a reliable .index attribute across all codec/container combos.
        """
        self.container.seek(0)
        counter = 0
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                if self.mode == "analysis":
                    yield frame.to_ndarray(format="bgr24"), counter
                else:  # render
                    yield frame.to_ndarray(format="rgb48le"), counter
                counter += 1

    def read_frame(self, frame_index: int):
        """Read a single frame by index (inefficient for sequential access)."""
        self.container.seek(0)
        for img, idx in self:
            if idx == frame_index:
                return img
        raise IndexError(f"Frame {frame_index} not found")

    def close(self):
        """Release the container."""
        if self.container:
            self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return (
            f"VideoReader('{self.path.name}', mode='{self.mode}', "
            f"{self.width}x{self.height}, {self.total_frames} frames)"
        )
