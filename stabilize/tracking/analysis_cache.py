"""Temporary downscaled frame cache for model-based analysis."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2

from stabilize.io.reader import VideoReader


class AnalysisFrameCache:
    def __init__(
        self,
        input_path: Path | str,
        max_dimension: int,
        jpeg_quality: int = 95,
        start_frame: int = 0,
        end_frame: int | None = None,
    ):
        self.input_path = Path(input_path)
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.start_frame = max(0, start_frame)
        self.end_frame = end_frame
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None
        self.source_width = 0
        self.source_height = 0
        self.source_total_frames = 0
        self.frame_offset = self.start_frame
        self.width = 0
        self.height = 0
        self.frame_rate = 0.0
        self.total_frames = 0
        self.scale_x = 1.0
        self.scale_y = 1.0

    def build(self, progress_cb=None) -> "AnalysisFrameCache":
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="aircraft_tracker_frames_"
        )
        self.path = Path(self._temp_dir.name)

        reader = VideoReader(self.input_path, mode="analysis")
        self.source_width = reader.width
        self.source_height = reader.height
        self.source_total_frames = reader.total_frames
        self.frame_rate = reader.frame_rate
        if reader.total_frames > 0:
            self.start_frame = min(
                self.start_frame,
                reader.total_frames - 1,
            )
            if self.end_frame is None:
                self.end_frame = reader.total_frames - 1
            else:
                self.end_frame = min(
                    max(self.end_frame, self.start_frame),
                    reader.total_frames - 1,
                )
        self.frame_offset = self.start_frame
        scale = min(
            self.max_dimension / max(reader.width, reader.height),
            1.0,
        )
        self.width = max(1, int(round(reader.width * scale)))
        self.height = max(1, int(round(reader.height * scale)))
        self.scale_x = self.width / reader.width
        self.scale_y = self.height / reader.height

        count = 0
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        for frame_bgr, idx in reader:
            if idx < self.start_frame:
                continue
            if self.end_frame is not None and idx > self.end_frame:
                break
            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                )
            frame_path = self.path / f"{count:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame_bgr, encode_params):
                reader.close()
                raise RuntimeError(f"Could not write analysis frame: {frame_path}")
            count += 1
            if progress_cb:
                expected = (
                    self.end_frame - self.start_frame + 1
                    if self.end_frame is not None
                    else reader.total_frames or count
                )
                progress_cb(count, expected)

        reader.close()
        self.total_frames = count
        return self

    def read(self, frame_idx: int):
        if self.path is None:
            raise RuntimeError("Analysis cache has not been built")
        frame = cv2.imread(str(self.path / f"{frame_idx:06d}.jpg"))
        if frame is None:
            raise IndexError(f"Analysis frame {frame_idx} not found")
        return frame

    def to_analysis_bbox(
        self,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        return (
            int(round(x * self.scale_x)),
            int(round(y * self.scale_y)),
            max(1, int(round(w * self.scale_x))),
            max(1, int(round(h * self.scale_y))),
        )

    def to_source_bbox(
        self,
        bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        x, y, w, h = bbox
        return (
            int(round(x / self.scale_x)),
            int(round(y / self.scale_y)),
            max(1, int(round(w / self.scale_x))),
            max(1, int(round(h / self.scale_y))),
        )

    def to_source_point(
        self,
        point: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if point is None:
            return None
        return (point[0] / self.scale_x, point[1] / self.scale_y)

    def to_analysis_point(
        self,
        point: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if point is None:
            return None
        return (point[0] * self.scale_x, point[1] * self.scale_y)

    def close(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self.path = None

    def window(
        self,
        start_frame: int,
        end_frame: int,
    ) -> "AnalysisFrameWindow":
        """Create a contiguous temporary view of a cached frame interval.

        SAM 2 accepts a directory of sequentially named JPEG frames. Long
        videos are therefore analyzed through bounded windows while retaining
        one root cache, rather than decoding the original 6K clip again for
        every window. The window uses NTFS hard links when available and falls
        back to copies on filesystems that do not support links.
        """
        return AnalysisFrameWindow(self, start_frame, end_frame)

    def __enter__(self) -> "AnalysisFrameCache":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class AnalysisFrameWindow:
    """A short, sequentially named view into an ``AnalysisFrameCache``."""

    def __init__(
        self,
        parent: AnalysisFrameCache,
        start_frame: int,
        end_frame: int,
    ):
        if parent.path is None or parent.total_frames <= 0:
            raise RuntimeError("Analysis cache has not been built")
        self.parent = parent
        self.start_frame = max(0, int(start_frame))
        self.end_frame = min(int(end_frame), parent.total_frames - 1)
        if self.end_frame < self.start_frame:
            raise ValueError("Analysis window end precedes its start")

        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self.path: Path | None = None
        self.source_width = parent.source_width
        self.source_height = parent.source_height
        self.source_total_frames = parent.source_total_frames
        self.frame_offset = parent.frame_offset + self.start_frame
        self.width = parent.width
        self.height = parent.height
        self.frame_rate = parent.frame_rate
        self.total_frames = self.end_frame - self.start_frame + 1
        self.scale_x = parent.scale_x
        self.scale_y = parent.scale_y

    def build(self) -> "AnalysisFrameWindow":
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="aircraft_tracker_window_"
        )
        self.path = Path(self._temp_dir.name)
        assert self.parent.path is not None
        for local_idx, parent_idx in enumerate(
            range(self.start_frame, self.end_frame + 1)
        ):
            source = self.parent.path / f"{parent_idx:06d}.jpg"
            target = self.path / f"{local_idx:06d}.jpg"
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        return self

    def read(self, frame_idx: int):
        if self.path is None:
            raise RuntimeError("Analysis window has not been built")
        frame = cv2.imread(str(self.path / f"{frame_idx:06d}.jpg"))
        if frame is None:
            raise IndexError(f"Analysis frame {frame_idx} not found")
        return frame

    def to_analysis_bbox(
        self,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        return self.parent.to_analysis_bbox(bbox)

    def to_source_bbox(
        self,
        bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        return self.parent.to_source_bbox(bbox)

    def to_source_point(
        self,
        point: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        return self.parent.to_source_point(point)

    def to_analysis_point(
        self,
        point: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        return self.parent.to_analysis_point(point)

    def close(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self.path = None

    def __enter__(self) -> "AnalysisFrameWindow":
        return self

    def __exit__(self, *args) -> None:
        self.close()
