#!/usr/bin/env python3
"""Aircraft video stabilization — CLI entry point.

Usage:
    python -m stabilize.main input.MOV
    python -m stabilize.main input.MOV -o output.MOV --preview
    python -m stabilize.main input.MOV --detector yolo --smooth-window 31
"""

import argparse
import logging
import sys
from pathlib import Path

from stabilize.config import StabilizerConfig
from stabilize.pipeline import StabilizationPipeline
from stabilize.utils.video_info import print_video_info


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="stabilize",
        description="Stabilize videos of civil aviation aircraft. "
        "Preserves original resolution, color space, and bit depth.",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to input .MOV file",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (default: {input}_stabilized.MOV in --output-dir)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("处理结果"),
        help="Output directory (default: 处理结果)",
    )

    # Detection options
    det_group = parser.add_argument_group("Detection")
    det_group.add_argument(
        "--detector",
        choices=["torchvision", "yolo"],
        default="torchvision",
        help="Detection backend (default: torchvision)",
    )
    det_group.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    det_group.add_argument(
        "--reinterval",
        type=int,
        default=30,
        help="Frames between re-detection (default: 30)",
    )

    # Smoothing options
    smooth_group = parser.add_argument_group("Smoothing")
    smooth_group.add_argument(
        "--smooth-window",
        type=int,
        default=61,
        help="Smoothing filter window in frames, must be odd (default: 61)",
    )
    smooth_group.add_argument(
        "--smooth-method",
        choices=["savgol", "gaussian"],
        default="savgol",
        help="Smoothing algorithm (default: savgol)",
    )

    # Encoding options
    enc_group = parser.add_argument_group("Encoding")
    enc_group.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 CRF quality, lower is better (default: 18)",
    )
    enc_group.add_argument(
        "--preset",
        choices=["ultrafast", "fast", "medium", "slow", "slower"],
        default="slow",
        help="x264 encoding preset (default: slow)",
    )
    enc_group.add_argument(
        "--border",
        choices=["reflect", "replicate"],
        default="reflect",
        help="Border fill mode (default: reflect)",
    )

    # Runtime options
    run_group = parser.add_argument_group("Runtime")
    run_group.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV preview window during analysis pass",
    )
    run_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and save intermediate data",
    )
    run_group.add_argument(
        "--info",
        action="store_true",
        help="Print video info and exit",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    # Info mode
    if args.info:
        print_video_info(args.input)
        return 0

    # Build config
    config = StabilizerConfig(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.output_dir,
        detector_backend=args.detector,
        detection_confidence=args.conf,
        detection_interval=args.reinterval,
        smoother_method=args.smooth_method,
        smoother_window=args.smooth_window,
        border_mode=args.border,
        crf=args.crf,
        preset=args.preset,
        preview=args.preview,
    )

    # Print summary
    print_video_info(args.input)
    print()

    # Run pipeline
    pipeline = StabilizationPipeline(config)
    try:
        output_path = pipeline.run()

        if args.debug:
            pipeline.save_debug_data(Path("debug_data"))

        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as e:
        logging.error("Pipeline failed: %s", e, exc_info=args.debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
