#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract frames from 360 equirectangular video at specified FPS."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_video_info(video_path: str | Path) -> dict:
    """Get video metadata using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with video info: width, height, duration, fps, frame_count.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError("No video stream found")

    # Parse frame rate (can be "30/1" or "29.97")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = map(float, fps_str.split("/"))
        fps = num / den
    else:
        fps = float(fps_str)

    duration = float(data.get("format", {}).get("duration", 0))

    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "duration": duration,
        "fps": fps,
        "frame_count": int(video_stream.get("nb_frames", duration * fps)),
        "codec": video_stream.get("codec_name", "unknown"),
    }


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    output_format: str = "jpg",
    quality: int = 2,
) -> list[Path]:
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        fps: Frames per second to extract (default: 2).
        start_sec: Start time in seconds (default: 0).
        duration_sec: Duration to extract in seconds (None = full video).
        output_format: Output image format (jpg, png).
        quality: JPEG quality (1-31, lower is better).

    Returns:
        List of paths to extracted frame images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video info for validation
    info = get_video_info(video_path)
    print(f"Video: {video_path.name}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  FPS: {info['fps']:.2f}")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]

    # Input seeking (fast)
    if start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])

    cmd.extend(["-i", str(video_path)])

    # Duration limit
    if duration_sec is not None:
        cmd.extend(["-t", str(duration_sec)])

    # Output settings
    output_pattern = output_dir / f"frame_%06d.{output_format}"

    # Video filter for frame rate
    cmd.extend(["-vf", f"fps={fps}"])

    # Quality settings
    if output_format == "jpg":
        cmd.extend(["-q:v", str(quality)])
    elif output_format == "png":
        cmd.extend(["-compression_level", "6"])

    cmd.append(str(output_pattern))

    print(f"Extracting frames at {fps} fps...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    # Collect extracted frames
    frames = sorted(output_dir.glob(f"frame_*.{output_format}"))
    print(f"Extracted {len(frames)} frames to {output_dir}")

    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames from 360 video")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract (default: 2)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: full video)")
    parser.add_argument("--format", type=str, default="jpg", choices=["jpg", "png"], help="Output format")
    parser.add_argument("--quality", type=int, default=2, help="JPEG quality 1-31 (lower=better)")
    parser.add_argument("--info-only", action="store_true", help="Only print video info, don't extract")

    args = parser.parse_args()

    try:
        if args.info_only:
            info = get_video_info(args.video)
            print(json.dumps(info, indent=2))
            return 0

        if not args.output:
            parser.error("--output is required for frame extraction")

        frames = extract_frames(
            video_path=args.video,
            output_dir=args.output,
            fps=args.fps,
            start_sec=args.start,
            duration_sec=args.duration,
            output_format=args.format,
            quality=args.quality,
        )

        # Write frame list
        frame_list_path = Path(args.output) / "frame_list.txt"
        with open(frame_list_path, "w") as f:
            for frame in frames:
                f.write(f"{frame.name}\n")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
