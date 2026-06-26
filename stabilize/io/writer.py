"""PyAV-based video writer with 10-bit 4:2:2 output and audio pass-through."""

import logging
from pathlib import Path

import av
import numpy as np

from stabilize.config import StabilizerConfig

logger = logging.getLogger(__name__)


class VideoWriter:
    """Write stabilized frames to output video, preserving audio streams.

    Frames are expected as uint16 RGB (rgb48le) arrays of shape (H, W, 3).
    Output is encoded as yuv422p10le matching the original color space.
    """

    def __init__(
        self,
        output_path: Path | str,
        config: StabilizerConfig,
        reader: "VideoReader",  # noqa: F821
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.reader = reader

        # Output container
        self.output_container = av.open(str(self.output_path), "w")

        # Video stream — use exact input frame rate (as Fraction)
        rate = reader.stream.average_rate
        self.video_stream = self.output_container.add_stream(
            config.video_codec, rate=rate
        )
        self.video_stream.width = reader.width
        self.video_stream.height = reader.height
        self.video_stream.pix_fmt = "yuv422p10le"
        self.video_stream.options = {
            "preset": config.preset,
            "crf": str(config.crf),
            "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        }

        # Audio pass-through
        self.audio_streams = []
        if config.copy_audio:
            self._setup_audio_pass_through()

        self._frame_count = 0

    def _setup_audio_pass_through(self):
        """Create output audio streams matching input audio streams.

        Uses PyAV's template-based stream copy where supported,
        falling back to manual codec parameter copying.
        """
        input_container = av.open(str(self.reader.path))
        for in_stream in input_container.streams:
            if in_stream.type != "audio":
                continue
            try:
                # PyAV >= 9.0 supports template= for stream copy
                out_stream = self.output_container.add_stream(
                    template=in_stream
                )
                self.audio_streams.append((in_stream, out_stream))
            except TypeError:
                # Older PyAV: manually copy codec parameters
                try:
                    codec_name = (
                        in_stream.codec_context.name
                        if in_stream.codec_context
                        else "aac"
                    )
                    out_stream = self.output_container.add_stream(codec_name)
                    # Copy time_base from input
                    out_stream.time_base = in_stream.time_base
                    self.audio_streams.append((in_stream, out_stream))
                except Exception as e:
                    logger.warning(
                        "Could not create audio stream #%d: %s",
                        in_stream.index, e,
                    )
        input_container.close()

    def write(self, frame: np.ndarray, frame_index: int) -> None:
        """Write a single frame to the output.

        Args:
            frame: uint16 RGB array of shape (H, W, 3), dtype np.uint16.
            frame_index: Frame index for PTS assignment.
        """
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb48le")
        av_frame.pts = frame_index
        for packet in self.video_stream.encode(av_frame):
            self.output_container.mux(packet)
        self._frame_count += 1

    def write_audio(self) -> None:
        """Copy all audio packets from input to output.

        Call this after all video frames have been written.
        """
        if not self.config.copy_audio:
            return

        input_container = av.open(str(self.reader.path))
        input_audio_streams = [
            s for s in input_container.streams if s.type == "audio"
        ]

        if not input_audio_streams or not self.audio_streams:
            input_container.close()
            return

        # Map input audio stream index -> output stream
        stream_map = {}
        for in_stream, out_stream in self.audio_streams:
            stream_map[in_stream.index] = out_stream

        # Re-read and copy audio packets
        for in_stream in input_audio_streams:
            if in_stream.index not in stream_map:
                continue
            out_stream = stream_map[in_stream.index]
            input_container.seek(0)
            for packet in input_container.demux(in_stream):
                if packet.dts is None:
                    continue
                packet.stream = out_stream
                self.output_container.mux(packet)

        input_container.close()

    def close(self) -> None:
        """Flush encoder and close the output container."""
        # Flush video encoder
        for packet in self.video_stream.encode(None):
            self.output_container.mux(packet)

        # Copy audio
        self.write_audio()

        self.output_container.close()
        logger.info(
            "Wrote %d frames to %s (%dx%d)",
            self._frame_count,
            self.output_path.name,
            self.video_stream.width,
            self.video_stream.height,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
