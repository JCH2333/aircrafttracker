"""Two-pass stabilization pipeline orchestrator.

Pass 1 (Analysis): Detect aircraft centroid on every frame
    (no tracker — avoids re-initialization jumps), smooth the
    trajectory, compute per-frame centering translation.

Pass 2 (Render): Decode frames at 16-bit precision, apply
    translation with black borders, encode output.
"""

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from stabilize.config import StabilizerConfig
from stabilize.detection.motion_fallback import MotionFallbackDetector
from stabilize.detection.torchvision_detector import TorchvisionDetector
from stabilize.io.reader import VideoReader
from stabilize.io.writer import VideoWriter
from stabilize.stabilization.warper import compute_transforms, translate_frame

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
            "-map", "1:a",
            "-map_metadata", "0",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Audio muxed successfully")
        except subprocess.CalledProcessError as e:
            logger.warning("FFmpeg audio mux failed: %s", e.stderr)
            import shutil
            shutil.copy2(str(video_path), str(output_path))

    def _run_pass1(self) -> None:
        """Analysis pass: track aircraft, compute centering transforms.

        Uses FeatureTracker (Shi-Tomasi + Lucas-Kanade optical flow)
        for smooth sub-pixel frame-to-frame tracking. Detection is
        used for initialization and periodic drift correction.
        """
        logger.info("--- Pass 1: Analysis ---")

        reader = VideoReader(self.config.input_path, mode="analysis")
        logger.info(
            "%d×%d, %.2f fps, ~%d frames",
            reader.width, reader.height,
            reader.frame_rate, reader.total_frames,
        )

        total = reader.total_frames if reader.total_frames > 0 else sys.maxsize

        # Initialize detection and optical flow tracker
        detector = TorchvisionDetector(self.config)
        detector.warmup()
        fallback = MotionFallbackDetector(self.config)

        from stabilize.stabilization.template_tracker import TemplateTracker
        tracker = TemplateTracker(self.config)

        centroids = []
        n_detections = 0
        n_optical_flow = 0
        n_fallback = 0
        last_bbox = None

        pbar = tqdm(reader, total=total, desc="Pass 1 (track)", unit="f", colour="blue")

        for frame_bgr, idx in pbar:
            centroid = None

            # Decide: use detection or template matching?
            need_detect = (
                not tracker.initialized
                or tracker.needs_redetection()
            )

            if need_detect:
                bbox = detector.detect(frame_bgr)
                if bbox is None:
                    bbox = fallback.detect(frame_bgr)

                if bbox is not None:
                    last_bbox = bbox
                    # Only use detection for first init or fallback recovery.
                    # Normal operation: template matching self-updates, no
                    # centroid reset from detection (avoids jumps).
                    tracker.init_from_detection(frame_bgr, bbox)
                    centroid = tracker.current_centroid
                    n_detections += 1

            # If detection wasn't needed or failed, use optical flow
            if centroid is None and tracker.initialized:
                centroid = tracker.update(frame_bgr)
                if centroid is not None:
                    n_optical_flow += 1

            # Fallback: use last known detection centroid
            if centroid is None and last_bbox is not None:
                x, y, w, h = last_bbox
                centroid = (x + w / 2.0, y + h / 2.0)
                n_fallback += 1

            # Last resort: frame center
            if centroid is None:
                centroid = (reader.width / 2.0, reader.height / 2.0)
                n_fallback += 1

            centroids.append(centroid)
            pbar.set_postfix_str(
                f"det={n_detections} of={n_optical_flow} fb={n_fallback}"
            )

        reader.close()
        pbar.close()

        self.centroids_raw = centroids

        logger.info(
            "Tracking: %d detection, %d optical flow, %d fallback (%d frames)",
            n_detections, n_optical_flow, n_fallback, len(centroids),
        )

        if not centroids:
            raise RuntimeError("No centroids computed")

        # Direct centering — each frame pinned to center
        self.transforms = compute_transforms(
            centroids,
            reader.width,
            reader.height,
        )

        # Compute jitter stats (frame-to-frame centroid variation)
        c_arr = np.array(centroids)
        dx_frame = np.diff(c_arr[:, 0])
        dy_frame = np.diff(c_arr[:, 1])
        logger.info(
            "Frame-to-frame jitter: dx_std=%.2f, dy_std=%.2f px",
            float(np.std(dx_frame)), float(np.std(dy_frame)),
        )

        dx_arr = np.array([t[0] for t in self.transforms])
        dy_arr = np.array([t[1] for t in self.transforms])
        logger.info(
            "Centering range: dx=[%.0f, %.0f], dy=[%.0f, %.0f] pixels",
            dx_arr.min(), dx_arr.max(), dy_arr.min(), dy_arr.max(),
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
        logger.info("Debug data saved to %s", output_dir)
