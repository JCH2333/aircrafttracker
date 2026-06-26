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
from stabilize.stabilization.smoother import smooth_trajectory
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
        """Analysis pass: detect aircraft on every frame, smooth, compute transforms."""
        logger.info("--- Pass 1: Analysis ---")

        reader = VideoReader(self.config.input_path, mode="analysis")
        logger.info(
            "%d×%d, %.2f fps, ~%d frames",
            reader.width, reader.height,
            reader.frame_rate, reader.total_frames,
        )

        total = reader.total_frames if reader.total_frames > 0 else sys.maxsize

        # Initialize detector (no tracker — detect every frame to avoid jumps)
        detector = TorchvisionDetector(self.config)
        detector.warmup()
        fallback = MotionFallbackDetector(self.config)

        centroids = []
        detection_ok = 0
        detection_miss = 0
        last_bbox = None  # fallback: use last known bbox for centroid

        pbar = tqdm(reader, total=total, desc="Pass 1 (detect)", unit="f", colour="blue")

        for frame_bgr, idx in pbar:
            bbox = detector.detect(frame_bgr)

            if bbox is None:
                bbox = fallback.detect(frame_bgr)

            if bbox is not None:
                x, y, w, h = bbox
                centroids.append((x + w / 2.0, y + h / 2.0))
                last_bbox = bbox
                detection_ok += 1
                pbar.set_postfix_str(f"ok={detection_ok} miss={detection_miss}")
            elif last_bbox is not None:
                # Detection lost: reuse last known centroid
                x, y, w, h = last_bbox
                centroids.append((x + w / 2.0, y + h / 2.0))
                detection_miss += 1
                pbar.set_postfix_str(f"ok={detection_ok} miss={detection_miss}")
            else:
                # No detections at all yet — use frame center
                centroids.append((reader.width / 2.0, reader.height / 2.0))
                detection_miss += 1

        reader.close()
        pbar.close()

        self.centroids_raw = centroids

        logger.info(
            "Detection: %d/%d frames (%.1f%% success)",
            detection_ok, len(centroids),
            100 * detection_ok / len(centroids) if centroids else 0,
        )

        if not centroids:
            raise RuntimeError("No centroids computed")

        # Smooth trajectory
        logger.info("Smoothing trajectory...")
        centroids_smooth = smooth_trajectory(
            centroids,
            window=self.config.smoother_window,
            method=self.config.smoother_method,
            polyorder=self.config.smoother_polyorder,
        )

        # Compute transforms to center the aircraft
        self.transforms = compute_transforms(
            centroids_smooth,
            reader.width,
            reader.height,
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
