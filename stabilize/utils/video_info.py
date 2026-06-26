"""Video information probe using PyAV."""

from pathlib import Path

import av


def probe_video(path: Path | str) -> dict:
    """Read video metadata and return as a dictionary.

    Args:
        path: Path to the video file.

    Returns:
        Dictionary with keys: path, width, height, fps, frames, duration,
        codec, pix_fmt, bit_depth, color_space, has_audio, audio_codecs.
    """
    container = av.open(str(path))
    video_stream = container.streams.video[0]
    audio_streams = [s for s in container.streams if s.type == "audio"]

    info = {
        "path": str(path),
        "width": video_stream.width,
        "height": video_stream.height,
        "fps": float(video_stream.average_rate) if video_stream.average_rate else 0,
        "frames": video_stream.frames or 0,
        "duration": float(video_stream.duration * video_stream.time_base)
        if video_stream.duration
        else 0,
        "codec": video_stream.codec_context.name if video_stream.codec_context else "unknown",
        "pix_fmt": video_stream.pix_fmt or "unknown",
        "bit_depth": _bit_depth_from_pix_fmt(video_stream.pix_fmt),
        "color_space": video_stream.codec_context.color_range if video_stream.codec_context else "unknown",
        "has_audio": len(audio_streams) > 0,
        "audio_codecs": [s.codec_context.name for s in audio_streams if s.codec_context],
    }

    container.close()
    return info


def _bit_depth_from_pix_fmt(pix_fmt: str | None) -> int:
    """Extract bit depth from pixel format string.

    Examples: 'yuv422p10le' -> 10, 'rgb24' -> 8, 'yuv420p' -> 8.
    In FFmpeg, 'p' followed by digits gives the bit depth for planar formats.
    For packed formats like 'rgb24', the digits directly indicate bit depth.
    """
    if pix_fmt is None:
        return 0
    import re
    # Match 'p' + digits (e.g., p10le, p16be)
    match = re.search(r"p(\d+)", pix_fmt)
    if match:
        return int(match.group(1))
    # Fallback: find any digit group not in a subsampling pattern
    match = re.search(r"(\d+)", pix_fmt)
    if match:
        depth = int(match.group(1))
        return depth if depth <= 64 else 8
    return 8


def print_video_info(path: Path | str) -> None:
    """Pretty-print video information."""
    info = probe_video(path)
    print(f"File:      {info['path']}")
    print(f"Codec:     {info['codec']}")
    print(f"Resolution:{info['width']}×{info['height']}")
    print(f"FPS:       {info['fps']:.2f}")
    print(f"Frames:    {info['frames']}")
    print(f"Duration:  {info['duration']:.1f}s")
    print(f"Pixel fmt: {info['pix_fmt']} ({info['bit_depth']}-bit)")
    print(f"Audio:     {'Yes' if info['has_audio'] else 'No'} ({', '.join(info['audio_codecs']) or 'N/A'})")
