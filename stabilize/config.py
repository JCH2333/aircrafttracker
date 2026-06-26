"""Configuration dataclass for the stabilization pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StabilizerConfig:
    # I/O
    input_path: Path | str = ""
    output_path: Path | str | None = None  # defaults to input_stem + "_stabilized.MOV"
    output_dir: Path | str = Path("处理结果")

    # Detection
    detector_backend: str = "torchvision"  # "torchvision" | "yolo"
    detection_confidence: float = 0.5
    detection_interval: int = 30  # frames between re-detection
    # COCO classes to accept as potential aircraft:
    # 4=airplane, 5=bus, 6=train, 7=truck, 8=boat
    # Civil aviation aircraft may be misclassified as "bus" or "train"
    # from certain ground-level angles.
    detection_classes: tuple[int, ...] = (4, 5, 6, 7, 8)
    detection_confidence_low: float = 0.3  # min confidence for secondary classes

    # Tracking
    tracker_quality_timeout: int = 90  # max frames without successful re-detect

    # Smoothing
    smoother_method: str = "savgol"  # "savgol" | "gaussian"
    smoother_window: int = 61  # frames, must be odd for savgol
    smoother_polyorder: int = 2

    # Warping
    border_mode: str = "constant"  # "constant" | "reflect" | "replicate"

    # Encoding
    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "slow"
    copy_audio: bool = True

    # Runtime
    device: str = field(default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu")
    preview: bool = False  # show preview window during analysis pass
    analysis_downscale: int = 1280  # max dimension for detection inference

    def resolve_output_path(self) -> Path:
        """Resolve the output path from input_path and output_dir."""
        input_p = Path(self.input_path)
        if self.output_path:
            return Path(self.output_path)
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_p.stem}_stabilized.MOV"
