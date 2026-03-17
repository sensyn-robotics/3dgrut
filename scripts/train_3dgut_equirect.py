#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train 3DGUT using equirectangular dataset.

This is a convenience wrapper around train.py with the equirect_3dgut config.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_3dgut_equirect(
    data_dir: str | Path,
    output_dir: str | Path,
    n_iterations: int = 30000,
    with_gui: bool = False,
    downsample_factor: int = 1,
    test_split_interval: int = 8,
    extra_args: list[str] | None = None,
) -> int:
    """Train 3DGUT on equirectangular dataset.

    Args:
        data_dir: Path to dataset in COLMAP format (with equirect images).
        output_dir: Output directory for checkpoints and exports.
        n_iterations: Number of training iterations.
        with_gui: Enable interactive GUI during training.
        downsample_factor: Image downsample factor.
        test_split_interval: Every N-th frame goes to validation.
        extra_args: Additional arguments to pass to train.py.

    Returns:
        Exit code from training.
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--config-name",
        "apps/equirect_3dgut.yaml",
        f"path={data_dir}",
        f"out_dir={output_dir.parent}",
        f"experiment_name={output_dir.name}",
        f"n_iterations={n_iterations}",
        f"dataset.downsample_factor={downsample_factor}",
        f"dataset.test_split_interval={test_split_interval}",
    ]

    if with_gui:
        cmd.append("with_gui=True")

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")

    # Run from project root
    project_root = Path(__file__).parent.parent
    result = subprocess.run(cmd, cwd=project_root)

    return result.returncode


def export_ply(checkpoint_path: str | Path, output_ply: str | Path) -> int:
    """Export trained model to PLY format.

    Args:
        checkpoint_path: Path to the checkpoint file.
        output_ply: Output PLY file path.

    Returns:
        Exit code.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    output_ply = Path(output_ply).resolve()
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    # Use render.py with export option
    cmd = [
        sys.executable,
        "render.py",
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(output_ply.parent),
        "--export-ply",
        str(output_ply),
    ]

    print(f"Running: {' '.join(cmd)}")

    project_root = Path(__file__).parent.parent
    result = subprocess.run(cmd, cwd=project_root)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train 3DGUT on equirectangular dataset")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--iterations", "-n", type=int, default=30000, help="Number of iterations")
    parser.add_argument("--gui", action="store_true", help="Enable training GUI")
    parser.add_argument("--downsample", type=int, default=1, help="Image downsample factor")
    parser.add_argument("--test-interval", type=int, default=8, help="Test split interval")
    parser.add_argument("--export-ply", type=str, help="Export final model to PLY")
    parser.add_argument("extra_args", nargs="*", help="Additional args for train.py")

    args = parser.parse_args()

    # Run training
    ret = train_3dgut_equirect(
        data_dir=args.data,
        output_dir=args.output,
        n_iterations=args.iterations,
        with_gui=args.gui,
        downsample_factor=args.downsample,
        test_split_interval=args.test_interval,
        extra_args=args.extra_args if args.extra_args else None,
    )

    if ret != 0:
        print(f"Training failed with code {ret}")
        return ret

    # Export PLY if requested
    if args.export_ply:
        checkpoint = Path(args.output) / "ckpt_last.pt"
        if checkpoint.exists():
            ret = export_ply(checkpoint, args.export_ply)
            if ret != 0:
                print(f"PLY export failed with code {ret}")
                return ret
        else:
            print(f"Checkpoint not found: {checkpoint}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
