"""Two-pass stabilization pipeline orchestrator.

Pass 1 (Analysis): Detect aircraft centroid on every frame
    (no tracker — avoids re-initialization jumps), smooth the
    trajectory, compute per-frame centering translation.

Pass 2 (Render): Decode frames at 16-bit precision, apply
    translation with black borders, encode output.
"""

import logging
import os
import subprocess
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stabilize.config import StabilizerConfig
from stabilize.detection.torchvision_detector import TorchvisionDetector
from stabilize.detection.yolo_detector import YOLODetector
from stabilize.io.reader import VideoReader
from stabilize.io.writer import VideoWriter
from stabilize.review import ReviewRequest, review_with_opencv
from stabilize.stabilization.warper import compute_transforms, translate_frame
from stabilize.tracking.analyzer import AnalysisResult, analyze_hybrid, analyze_legacy
from stabilize.tracking.models import FlaggedSegment, TrackObservation
from stabilize.tracking.reanalysis import build_reanalysis_ranges
from stabilize.tracking.sam2_adapter import Sam2Unavailable
from stabilize.tracking.track_file import TrackFile
from stabilize.tracking.trajectory import smooth_observations

logger = logging.getLogger(__name__)


class StabilizationPipeline:
    """Two-pass video stabilization pipeline.

    Pass 1: Detect aircraft on every frame, smooth centroids,
            compute centering transforms.
    Pass 2: Render stabilized frames with black borders.
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.transforms: list[tuple[float, float]] = []
        self.centroids_raw: list[tuple[float, float]] = []
        self.observations: list[TrackObservation] = []
        self.flagged_segments: list[FlaggedSegment] = []
        self._progress_cb: callable | None = None
        self._review_cb: callable | None = None

    def set_progress_callback(self, cb: callable) -> None:
        """Set a callback for progress updates.

        Args:
            cb: callable(phase, current, total) where phase is 1 or 2.
        """
        self._progress_cb = cb

    def set_review_callback(self, cb: callable) -> None:
        """Set a callback that returns manual anchors for flagged segments."""
        self._review_cb = cb

    def _report_progress(self, phase: int, current: int, total: int) -> None:
        if self._progress_cb:
            try:
                self._progress_cb(phase, current, total)
            except Exception:
                pass

    def _make_debug_viz_path(self) -> Path:
        """Compute the debug video output path from config.

        Uses config.debug_viz_dir if set, otherwise config.output_dir.
        Filename is {input_stem}_debug.MOV.
        """
        input_stem = Path(self.config.input_path).stem
        viz_dir = self.config.debug_viz_dir
        if viz_dir:
            out_dir = Path(viz_dir)
        else:
            out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_stem}_debug.MOV"

    def _resolve_track_file_path(self) -> Path:
        if self.config.track_file:
            return Path(self.config.track_file)
        return (
            Path(self.config.output_dir)
            / f"{Path(self.config.input_path).stem}.track.json"
        )

    def run(self) -> Path:
        """Execute both passes and return the output path."""
        output_path = self.config.resolve_output_path()

        logger.info("=" * 60)
        logger.info("Aircraft Video Stabilizer")
        logger.info("=" * 60)
        logger.info("Input:  %s", self.config.input_path)
        logger.info("Output: %s", output_path)

        self._run_pass1()

        # Write video-only to temp file, then mux audio
        temp_video = output_path.with_suffix(".video_only.MOV")
        self._run_pass2(temp_video)
        self._mux_audio(temp_video, output_path)

        try:
            temp_video.unlink()
        except OSError:
            logger.warning("Could not remove temp file: %s", temp_video)

        logger.info("Done: %s", output_path)
        return output_path

    def _mux_audio(self, video_path: Path, output_path: Path) -> None:
        """Copy audio streams from original input using FFmpeg."""
        logger.info("--- Muxing audio ---")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(self.config.input_path),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a?",
            "-map_metadata", "1",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Audio muxed successfully")
        except FileNotFoundError:
            logger.warning("FFmpeg not found - output will have no audio.")
            logger.info("Install FFmpeg: https://ffmpeg.org/download.html")
            import shutil
            shutil.copy2(str(video_path), str(output_path))
        except subprocess.CalledProcessError as e:
            logger.warning("FFmpeg audio mux failed: %s", e.stderr)
            import shutil
            shutil.copy2(str(video_path), str(output_path))

    def _run_pass1(self) -> None:
        """Analyze target motion, repair occlusions, and compute transforms."""
        logger.info("--- Pass 1: Analysis ---")

        cv2.setNumThreads(min(8, os.cpu_count() or 4))
        cv2.setUseOptimized(True)

        metadata_reader = VideoReader(self.config.input_path, mode="analysis")
        width = metadata_reader.width
        height = metadata_reader.height
        frame_rate = metadata_reader.frame_rate
        total_frames = metadata_reader.total_frames
        metadata_reader.close()
        logger.info(
            "%d×%d, %.2f fps, ~%d frames",
            width, height, frame_rate, total_frames,
        )

        detector = self._make_detector()
        detector.warmup()
        track_file_path = self._resolve_track_file_path()
        track_file = TrackFile.load(track_file_path)
        if not track_file.is_compatible(
            self.config.input_path,
            width,
            height,
            fps=frame_rate,
        ):
            logger.warning(
                "Ignoring incompatible track file: %s",
                track_file_path,
            )
            track_file = TrackFile()

        debug_viz_path = (
            self._make_debug_viz_path() if self.config.debug_viz else None
        )
        result = self._analyze(
            detector=detector,
            anchors=track_file.anchors,
            debug_viz_path=debug_viz_path,
        )
        centers, segments = smooth_observations(
            result.observations,
            result.width,
            result.height,
            smoother_window=self.config.smoother_window,
            smoother_method=self.config.smoother_method,
            smoother_polyorder=self.config.smoother_polyorder,
        )

        if self.config.review and segments:
            request = ReviewRequest(
                video_path=Path(self.config.input_path),
                segments=segments,
                predicted_centers=centers,
                existing_anchors=track_file.anchors,
                width=result.width,
                height=result.height,
                frame_rate=result.frame_rate,
                observations=result.observations,
                masks=result.review_masks,
            )
            review_cb = self._review_cb or review_with_opencv
            new_anchors = review_cb(request) or []
            if new_anchors:
                for anchor in new_anchors:
                    track_file.add_anchor(anchor)
                track_file.save(track_file_path)
                logger.info(
                    "Re-analyzing corrected intervals with %d manual anchor(s)",
                    len(new_anchors),
                )
                result = self._reanalyze_corrected_ranges(
                    detector=detector,
                    result=result,
                    existing_anchors=track_file.anchors,
                    new_anchors=new_anchors,
                )
                centers, segments = smooth_observations(
                    result.observations,
                    result.width,
                    result.height,
                    smoother_window=self.config.smoother_window,
                    smoother_method=self.config.smoother_method,
                    smoother_polyorder=self.config.smoother_polyorder,
                )

        self.observations = result.observations
        self.flagged_segments = segments
        self.centroids_raw = [
            observation.center
            or observation.predicted_center
            or centers[observation.frame_idx]
            for observation in result.observations
        ]

        track_file.source_path = str(Path(self.config.input_path).resolve())
        track_file.width = result.width
        track_file.height = result.height
        track_file.fps = result.frame_rate
        track_file.backend = result.backend
        track_file.flagged_segments = segments
        track_file.save(track_file_path)

        if segments:
            logger.warning(
                "Review recommended for %d low-confidence segment(s): %s",
                len(segments),
                track_file_path,
            )

        self.transforms = compute_transforms(
            centers,
            result.width,
            result.height,
        )

        c_arr = np.array(centers)
        dx_frame = np.diff(c_arr[:, 0])
        dy_frame = np.diff(c_arr[:, 1])
        logger.info(
            "Smoothed frame motion: dx_std=%.2f, dy_std=%.2f px",
            float(np.std(dx_frame)), float(np.std(dy_frame)),
        )

        dx_arr = np.array([t[0] for t in self.transforms])
        dy_arr = np.array([t[1] for t in self.transforms])
        logger.info(
            "Centering range: dx=[%.0f, %.0f], dy=[%.0f, %.0f] pixels",
            dx_arr.min(), dx_arr.max(), dy_arr.min(), dy_arr.max(),
        )

    def _make_detector(self):
        if self.config.detector_backend == "yolo":
            return YOLODetector(self.config)
        return TorchvisionDetector(self.config)

    def _analyze(
        self,
        detector,
        anchors,
        debug_viz_path,
        frame_range: tuple[int, int] | None = None,
    ):
        progress = lambda current, total: self._report_progress(
            1, current, total
        )
        if self.config.tracking_backend == "hybrid":
            try:
                return analyze_hybrid(
                    self.config,
                    detector,
                    anchors,
                    progress_cb=progress,
                    debug_viz_path=debug_viz_path,
                    frame_range=frame_range,
                )
            except Sam2Unavailable as exc:
                if not self.config.hybrid_fallback_to_legacy:
                    raise
                logger.warning(
                    "Hybrid tracking unavailable (%s); falling back to legacy. "
                    "Occlusion recovery quality will be lower.",
                    exc,
                )
            except RuntimeError as exc:
                if not self.config.hybrid_fallback_to_legacy:
                    raise
                logger.warning(
                    "Hybrid tracking failed (%s); falling back to legacy. "
                    "Occlusion recovery quality will be lower.",
                    exc,
                )

        return analyze_legacy(
            self.config,
            detector,
            anchors,
            progress_cb=progress,
            debug_viz_path=debug_viz_path,
            frame_range=frame_range,
        )

    def _reanalyze_corrected_ranges(
        self,
        detector,
        result: AnalysisResult,
        existing_anchors,
        new_anchors,
    ) -> AnalysisResult:
        ranges = build_reanalysis_ranges(
            result.observations,
            existing_anchors=existing_anchors,
            new_anchors=new_anchors,
        )
        observations = list(result.observations)
        review_masks = dict(result.review_masks)
        backend = result.backend

        for reanalysis_range in ranges:
            logger.info(
                "Re-analyzing frames %d-%d",
                reanalysis_range.start_frame,
                reanalysis_range.end_frame,
            )
            partial = self._analyze(
                detector=detector,
                anchors=list(reanalysis_range.anchors),
                debug_viz_path=None,
                frame_range=(
                    reanalysis_range.start_frame,
                    reanalysis_range.end_frame,
                ),
            )
            for observation in partial.observations:
                observations[observation.frame_idx] = observation
            for frame_idx in range(
                reanalysis_range.start_frame,
                reanalysis_range.end_frame + 1,
            ):
                review_masks.pop(frame_idx, None)
            review_masks.update(partial.review_masks)
            backend = partial.backend

        return AnalysisResult(
            observations=observations,
            width=result.width,
            height=result.height,
            frame_rate=result.frame_rate,
            backend=backend,
            review_masks=review_masks,
        )

    def _run_pass2(self, output_path: Path) -> None:
        """Render pass: warp frames with black borders and encode."""
        logger.info("--- Pass 2: Render ---")

        reader = VideoReader(self.config.input_path, mode="render")
        self.config.copy_audio = False
        writer = VideoWriter(output_path, self.config, reader)

        total = reader.total_frames or len(self.transforms)
        pbar = tqdm(
            zip(reader, self.transforms),
            total=total,
            desc="Pass 2 (render)",
            unit="f",
            colour="green",
        )

        for (frame_rgb48, idx), (dx, dy) in pbar:
            self._report_progress(2, idx + 1, total)
            if idx >= len(self.transforms):
                break

            # Apply translation with black (constant) borders
            warped = translate_frame(
                frame_rgb48,
                dx, dy,
                border_mode="constant",
            )
            writer.write(warped, idx)

        reader.close()
        writer.close()
        pbar.close()

    def save_debug_data(self, output_dir: Path) -> None:
        """Save intermediate data for debugging."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.centroids_raw:
            np.save(
                output_dir / "centroids_raw.npy",
                np.array(self.centroids_raw, dtype=np.float32),
            )
        if self.transforms:
            np.save(
                output_dir / "transforms.npy",
                np.array(self.transforms, dtype=np.float32),
            )
        if self.observations:
            with (output_dir / "observations.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    [
                        observation.to_dict()
                        for observation in self.observations
                    ],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        logger.info("Debug data saved to %s", output_dir)
