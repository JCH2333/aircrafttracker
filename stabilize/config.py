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

    # Feature tracking (Lucas-Kanade optical flow)
    feature_max_corners: int = 100     # Shi-Tomasi max corners
    feature_quality: float = 0.01      # qualityLevel threshold
    feature_min_distance: int = 10     # min pixel distance between corners
    feature_redetect_min_points: int = 15  # re-detect below this count
    lk_win_size: tuple[int, int] = (21, 21)  # optical flow search window
    lk_max_iter: int = 30              # optical flow max iterations
    lk_epsilon: float = 0.01           # optical flow convergence threshold
    feature_bbox_margin: float = 0.10  # fraction of bbox to exclude from edges

    # Template matching tracker
    template_search_margin: int = 200  # base pixels to search around predicted position
    template_match_threshold: float = 0.40  # min NCC score to accept match
    template_redetect_score: float = 0.50   # re-detect if score drops below this
    template_update_alpha: float = 0.3      # blend factor for template update
    template_velocity_alpha: float = 0.5    # EWMA alpha for velocity estimate
    template_max_jump_factor: float = 5.0   # reject match if jump > factor * speed
    template_quality_score: float = 0.55    # coast (no template/velocity update) below this

    # Edge detection (Canny) for contour-based matching
    canny_low_threshold: int = 30    # Canny low threshold (blended with auto-tune)
    canny_high_threshold: int = 90   # Canny high threshold (blended with auto-tune)
    edge_blur_sigma: float = 5.0     # Gaussian blur sigma for contour bands

    # Smooth transition on detection re-init
    transition_frames: int = 10       # frames to smooth over when jumping
    transition_threshold: float = 30.0  # min jump distance (px) to trigger transition

    # Encoding
    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
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
