#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main orchestration script for 360 video to 3DGS pipeline.

Pipeline stages:
1. Frame extraction from 360 video
2. OpenSfM reconstruction (spherical camera model)
3. Export to COLMAP format
4. 3DGUT training with equirectangular dataset
5. PLY export

Usage:
    python scripts/pipeline_360_to_3dgs.py \
        --video /path/to/360_video.mp4 \
        --output runs/my_scene \
        --fps 2

Test mode (5 seconds, 1000 iterations):
    python scripts/pipeline_360_to_3dgs.py \
        --video /path/to/360_video.mp4 \
        --output runs/my_scene \
        --test-mode
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PipelineConfig:
    """Configuration for the 360 to 3DGS pipeline."""

    video_path: Path
    output_dir: Path
    fps: float = 2.0
    start_sec: float = 0.0
    duration_sec: Optional[float] = None
    n_iterations: int = 30000
    downsample_factor: int = 1
    test_split_interval: int = 8
    with_gui: bool = False
    skip_sfm: bool = False
    skip_training: bool = False


def run_command(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and optionally check for errors."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=cwd, capture_output=False)

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")

    return result


def stage_extract_frames(config: PipelineConfig) -> Path:
    """Stage 1: Extract frames from 360 video.

    Returns:
        Path to extracted frames directory.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: Frame Extraction")
    print("=" * 70)

    frames_dir = config.output_dir / "frames"

    cmd = [
        sys.executable,
        "scripts/video360_to_frames.py",
        str(config.video_path),
        "--output",
        str(frames_dir),
        "--fps",
        str(config.fps),
    ]

    if config.start_sec > 0:
        cmd.extend(["--start", str(config.start_sec)])

    if config.duration_sec is not None:
        cmd.extend(["--duration", str(config.duration_sec)])

    project_root = Path(__file__).parent.parent
    run_command(cmd, cwd=project_root)

    # Verify frames were extracted
    frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames extracted to {frames_dir}")

    print(f"Extracted {len(frames)} frames")
    return frames_dir


def stage_run_opensfm(config: PipelineConfig, frames_dir: Path) -> Path:
    """Stage 2: Run OpenSfM for Structure from Motion.

    Returns:
        Path to COLMAP-format output directory.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: OpenSfM Reconstruction")
    print("=" * 70)

    opensfm_dir = config.output_dir / "opensfm"
    colmap_dir = config.output_dir / "colmap_format"
    debug_dir = config.output_dir / "debug"

    cmd = [
        sys.executable,
        "scripts/run_opensfm.py",
        "--images",
        str(frames_dir),
        "--project",
        str(opensfm_dir),
        "--output",
        str(colmap_dir),
        "--export-ply",
        str(debug_dir / "sparse_cloud.ply"),
    ]

    if config.skip_sfm:
        cmd.append("--skip-sfm")

    project_root = Path(__file__).parent.parent
    result = run_command(cmd, cwd=project_root, check=False)

    if result.returncode != 0:
        print("WARNING: OpenSfM failed. Saving debug info...")
        save_debug_info(config, frames_dir, opensfm_dir, debug_dir)
        raise RuntimeError("OpenSfM reconstruction failed")

    # Verify COLMAP format was created
    sparse_dir = colmap_dir / "sparse" / "0"
    if not (sparse_dir / "images.txt").exists():
        print("WARNING: COLMAP export incomplete. Saving debug info...")
        save_debug_info(config, frames_dir, opensfm_dir, debug_dir)
        raise RuntimeError("COLMAP format export failed")

    return colmap_dir


def stage_train_3dgut(config: PipelineConfig, data_dir: Path) -> Path:
    """Stage 3: Train 3DGUT on equirectangular dataset.

    Returns:
        Path to checkpoint file.
    """
    print("\n" + "=" * 70)
    print("STAGE 3: 3DGUT Training")
    print("=" * 70)

    training_dir = config.output_dir / "3dgut"

    cmd = [
        sys.executable,
        "scripts/train_3dgut_equirect.py",
        "--data",
        str(data_dir),
        "--output",
        str(training_dir),
        "--iterations",
        str(config.n_iterations),
        "--downsample",
        str(config.downsample_factor),
        "--test-interval",
        str(config.test_split_interval),
    ]

    if config.with_gui:
        cmd.append("--gui")

    project_root = Path(__file__).parent.parent
    run_command(cmd, cwd=project_root)

    checkpoint = training_dir / "ckpt_last.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    return checkpoint


def stage_export_ply(config: PipelineConfig, checkpoint: Path) -> Path:
    """Stage 4: Export trained model to PLY.

    Returns:
        Path to output PLY file.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: PLY Export")
    print("=" * 70)

    output_ply = config.output_dir / "output.ply"

    cmd = [
        sys.executable,
        "scripts/train_3dgut_equirect.py",
        "--data",
        str(config.output_dir / "colmap_format"),
        "--output",
        str(config.output_dir / "3dgut"),
        "--iterations",
        "0",  # No training, just export
        "--export-ply",
        str(output_ply),
    ]

    project_root = Path(__file__).parent.parent
    run_command(cmd, cwd=project_root, check=False)

    # Also try to export PLY using the model directly
    try:
        export_gaussians_to_ply(checkpoint, output_ply)
    except Exception as e:
        print(f"PLY export warning: {e}")

    return output_ply


def export_gaussians_to_ply(checkpoint_path: Path, output_path: Path):
    """Export Gaussian model to PLY format.

    Args:
        checkpoint_path: Path to model checkpoint.
        output_path: Output PLY file path.
    """
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract Gaussian parameters
    model_state = checkpoint.get("model", checkpoint.get("state_dict", {}))

    # Find position data
    xyz = None
    for key in ["_xyz", "xyz", "means", "_means"]:
        if key in model_state:
            xyz = model_state[key].numpy()
            break

    if xyz is None:
        print("Could not find position data in checkpoint")
        return

    # Find color data (SH coefficients or direct RGB)
    colors = None
    for key in ["_features_dc", "features_dc", "_sh_dc", "sh_dc"]:
        if key in model_state:
            sh_dc = model_state[key].numpy()
            # Convert SH DC to RGB (SH0 * C0 where C0 = 0.28209479177387814)
            C0 = 0.28209479177387814
            colors = (sh_dc[:, 0, :] * C0 + 0.5).clip(0, 1) * 255
            colors = colors.astype("uint8")
            break

    if colors is None:
        colors = (128 * np.ones((len(xyz), 3))).astype("uint8")

    # Write PLY
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(xyz, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    print(f"Exported {len(xyz)} Gaussians to {output_path}")


def save_debug_info(config: PipelineConfig, frames_dir: Path, opensfm_dir: Path, debug_dir: Path):
    """Save debug information when pipeline fails."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = debug_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "video_path": str(config.video_path),
                "output_dir": str(config.output_dir),
                "fps": config.fps,
                "start_sec": config.start_sec,
                "duration_sec": config.duration_sec,
            },
            f,
            indent=2,
        )

    # List frames
    frames = list(frames_dir.glob("*.*"))
    frames_list = debug_dir / "frames_list.txt"
    with open(frames_list, "w") as f:
        for frame in sorted(frames):
            f.write(f"{frame.name}\n")

    # Copy OpenSfM logs if they exist
    for log_file in opensfm_dir.glob("*.log"):
        shutil.copy2(log_file, debug_dir / log_file.name)

    print(f"Debug info saved to {debug_dir}")


def run_pipeline(config: PipelineConfig):
    """Run the full 360 to 3DGS pipeline."""
    print("=" * 70)
    print("360 Video to 3DGS Pipeline")
    print("=" * 70)
    print(f"Video: {config.video_path}")
    print(f"Output: {config.output_dir}")
    print(f"FPS: {config.fps}")
    if config.duration_sec:
        print(f"Duration: {config.duration_sec}s")
    print(f"Iterations: {config.n_iterations}")
    print("=" * 70)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "pipeline_config.json", "w") as f:
        json.dump(
            {
                "video_path": str(config.video_path),
                "fps": config.fps,
                "start_sec": config.start_sec,
                "duration_sec": config.duration_sec,
                "n_iterations": config.n_iterations,
                "downsample_factor": config.downsample_factor,
            },
            f,
            indent=2,
        )

    try:
        # Stage 1: Extract frames
        frames_dir = stage_extract_frames(config)

        # Stage 2: OpenSfM
        colmap_dir = stage_run_opensfm(config, frames_dir)

        # Stage 3: Training (optional skip)
        if not config.skip_training:
            checkpoint = stage_train_3dgut(config, colmap_dir)

            # Stage 4: Export
            output_ply = stage_export_ply(config, checkpoint)

            print("\n" + "=" * 70)
            print("Pipeline Complete!")
            print("=" * 70)
            print(f"Checkpoint: {checkpoint}")
            print(f"Output PLY: {output_ply}")
        else:
            print("\n" + "=" * 70)
            print("Pipeline Complete (training skipped)")
            print("=" * 70)
            print(f"COLMAP format: {colmap_dir}")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="360 Video to 3DGS Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (5 seconds, quick training)
  python scripts/pipeline_360_to_3dgs.py --video input.mp4 --output runs/test --test-mode

  # Full pipeline
  python scripts/pipeline_360_to_3dgs.py --video input.mp4 --output runs/full --fps 2

  # Skip training (only SfM)
  python scripts/pipeline_360_to_3dgs.py --video input.mp4 --output runs/sfm --skip-training
""",
    )

    parser.add_argument("--video", "-v", type=str, required=True, help="Path to 360 video file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate (default: 2)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: full)")
    parser.add_argument("--iterations", "-n", type=int, default=30000, help="Training iterations")
    parser.add_argument("--downsample", type=int, default=1, help="Image downsample factor")
    parser.add_argument("--test-interval", type=int, default=8, help="Validation split interval")
    parser.add_argument("--gui", action="store_true", help="Enable training GUI")
    parser.add_argument("--skip-sfm", action="store_true", help="Skip OpenSfM (use existing)")
    parser.add_argument("--skip-training", action="store_true", help="Skip 3DGUT training")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: 5 sec duration, 1000 iterations",
    )

    args = parser.parse_args()

    # Test mode overrides
    duration = args.duration
    iterations = args.iterations

    if args.test_mode:
        duration = 5.0
        iterations = 1000
        print("Test mode enabled: 5 seconds, 1000 iterations")

    config = PipelineConfig(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        fps=args.fps,
        start_sec=args.start,
        duration_sec=duration,
        n_iterations=iterations,
        downsample_factor=args.downsample,
        test_split_interval=args.test_interval,
        with_gui=args.gui,
        skip_sfm=args.skip_sfm,
        skip_training=args.skip_training,
    )

    return run_pipeline(config)


if __name__ == "__main__":
    sys.exit(main())
