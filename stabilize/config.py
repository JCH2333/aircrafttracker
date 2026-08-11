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
    detection_interval: int = 10  # frames between re-detection
    # COCO classes to accept as potential aircraft:
    # 4=airplane, 5=bus, 6=train, 7=truck, 8=boat
    # Civil aviation aircraft may be misclassified as "bus" or "train"
    # from certain ground-level angles.
    detection_classes: tuple[int, ...] = (4, 5, 6, 7, 8)
    detection_confidence_low: float = 0.3  # min confidence for secondary classes

    # Tracking
    tracking_backend: str = "hybrid"  # "hybrid" | "legacy"
    tracker_quality_timeout: int = 90  # max frames without successful re-detect
    tracking_low_confidence: float = 0.45
    tracking_recovery_confidence: float = 0.65
    tracking_update_confidence: float = 0.80
    tracking_low_frames: int = 2
    tracking_recovery_frames: int = 3

    # Smoothing
    smoother_method: str = "savgol"  # "savgol" | "gaussian"
    smoother_window: int = 15  # frames, must be odd for savgol
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
    feature_max_scale_delta: float = 0.18
    feature_max_rotation_degrees: float = 12.0
    feature_max_translation_ratio: float = 0.75
    feature_translation_residual_px: float = 4.5

    # Template matching tracker
    template_search_margin: int = 200  # base pixels to search around predicted position
    template_match_threshold: float = 0.40  # min NCC score to accept match
    template_redetect_score: float = 0.50   # re-detect if score drops below this
    template_update_alpha: float = 0.3      # blend factor for template update
    template_velocity_alpha: float = 0.5    # EWMA alpha for velocity estimate
    template_max_jump_factor: float = 2.0   # reject match if jump > factor * speed
    template_quality_score: float = 0.55    # coast (no template/velocity update) below this
    template_update_min_confidence: float = 0.80

    # Edge detection (Canny) for contour-based matching
    canny_low_threshold: int = 30    # Canny low threshold (blended with auto-tune)
    canny_high_threshold: int = 90   # Canny high threshold (blended with auto-tune)
    edge_blur_sigma: float = 5.0     # Gaussian blur sigma for contour bands

    # Orientation-aware matching (experimental — use_edge_matching to enable)
    orient_bins: int = 4              # number of orientation bins (0–π quantized)
    orient_mag_threshold: float = 0.1 # min normalized magnitude for reliable orientation
    use_edge_matching: bool = False    # True = orientation channels; False = magnitude NCC

    # Edge density suppression (experimental)
    edge_density_suppress: bool = False   # suppress high edge-density regions in NCC
    edge_density_beta: float = 3.0       # suppression strength (higher = more aggressive)

    # Dual-template: full aircraft + tail anchor
    tail_template_ratio: float = 0.40     # fraction of template height for tail region
    tail_disagreement_threshold: float = 15.0
    match_downscale: float = 0.5  # downscale search region before Sobel (speed)

    # Encoding
    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    copy_audio: bool = True

    # Runtime
    device: str = field(default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu")
    preview: bool = False  # show preview window during analysis pass
    analysis_downscale: int = 1280  # max dimension for detection inference
    analysis_jpeg_quality: int = 95

    # SAM 2 video segmentation (optional hybrid backend)
    sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
    sam2_offload_video_to_cpu: bool = True
    sam2_offload_state_to_cpu: bool = True
    hybrid_fallback_to_legacy: bool = True

    # Review and persistent manual anchors
    review: bool = False
    track_file: Path | str | None = None

    # Debug
    debug_viz: bool = False
    debug_viz_dir: str = ""  # override output dir for debug video (default: use output_dir)

    def resolve_output_path(self) -> Path:
        """Resolve the output path from input_path and output_dir."""
        input_p = Path(self.input_path)
        if self.output_path:
            return Path(self.output_path)
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_p.stem}_stabilized.MOV"
