#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run OpenSfM with spherical camera model for 360 equirectangular images.

If OpenSfM is not installed, can generate synthetic circular camera poses
for testing purposes.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Default OpenSfM config for spherical cameras
DEFAULT_OPENSFM_CONFIG = """
# OpenSfM configuration for 360 spherical cameras

# Use spherical camera model for equirectangular images
camera_models_overrides:
    ".*": spherical

# Feature extraction
feature_type: SIFT
feature_process_size: 2048
feature_min_frames: 4000

# Matching
matching_gps_neighbors: 0
matching_gps_distance: 0
matching_time_neighbors: 10
matching_order_neighbors: 5
matcher_type: FLANN

# Reconstruction
depthmap_min_consistent_views: 2
optimize_camera_parameters: false
triangulation_min_ray_angle: 1.0

# Bundle adjustment
bundle_use_gps: false
bundle_compensate_gps_bias: false
"""


def is_opensfm_available() -> bool:
    """Check if OpenSfM is installed and available."""
    try:
        result = subprocess.run(["opensfm", "--help"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_opensfm_project(
    image_dir: str | Path,
    project_dir: str | Path,
    config_overrides: dict | None = None,
) -> Path:
    """Create an OpenSfM project directory structure.

    Args:
        image_dir: Directory containing equirectangular images.
        project_dir: Output project directory.
        config_overrides: Additional config overrides.

    Returns:
        Path to the project directory.
    """
    image_dir = Path(image_dir)
    project_dir = Path(project_dir)

    # Create project structure
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "images").mkdir(exist_ok=True)

    # Link or copy images
    image_dest = project_dir / "images"
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix in image_extensions:
            dest_path = image_dest / img_path.name
            if not dest_path.exists():
                # Create symlink if possible, otherwise copy
                try:
                    dest_path.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, dest_path)

    # Write config
    config_path = project_dir / "config.yaml"
    config_content = DEFAULT_OPENSFM_CONFIG

    if config_overrides:
        config_content += "\n# Additional overrides\n"
        for key, value in config_overrides.items():
            config_content += f"{key}: {value}\n"

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Created OpenSfM project at {project_dir}")
    print(f"  Images: {len(list(image_dest.glob('*')))}")

    return project_dir


def run_opensfm_command(project_dir: str | Path, command: str) -> bool:
    """Run an OpenSfM command.

    Args:
        project_dir: Path to the OpenSfM project.
        command: OpenSfM command to run (e.g., 'extract_metadata').

    Returns:
        True if successful, False otherwise.
    """
    cmd = ["opensfm", command, str(project_dir)]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running '{command}':")
        print(result.stderr)
        return False

    if result.stdout:
        print(result.stdout)

    return True


def run_opensfm_pipeline(project_dir: str | Path) -> bool:
    """Run the full OpenSfM reconstruction pipeline.

    Args:
        project_dir: Path to the OpenSfM project.

    Returns:
        True if successful, False otherwise.
    """
    project_dir = Path(project_dir)

    # OpenSfM pipeline steps
    steps = [
        "extract_metadata",
        "detect_features",
        "match_features",
        "create_tracks",
        "reconstruct",
    ]

    for step in steps:
        print(f"\n{'=' * 60}")
        print(f"Step: {step}")
        print("=" * 60)

        if not run_opensfm_command(project_dir, step):
            print(f"Pipeline failed at step: {step}")
            return False

    print("\nOpenSfM reconstruction complete!")
    return True


def generate_synthetic_circular_poses(
    image_dir: str | Path,
    output_dir: str | Path,
    radius: float = 2.0,
    height: float = 0.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bool:
    """Generate synthetic camera poses in a circular pattern.

    This is a fallback when OpenSfM is not available. Assumes the camera
    moves in a horizontal circle around a central point.

    Args:
        image_dir: Directory containing images.
        output_dir: Output directory for COLMAP format.
        radius: Radius of the circular path.
        height: Height of cameras above the center.
        center: Center point of the scene (x, y, z).

    Returns:
        True if successful.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions])

    if not images:
        print(f"No images found in {image_dir}")
        return False

    n_images = len(images)
    print(f"Generating synthetic poses for {n_images} images")
    print(f"  Radius: {radius}, Height: {height}")

    # Create output directories
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    images_out = output_dir / "images"
    images_out.mkdir(exist_ok=True)

    # Link images
    for img_path in images:
        dest = images_out / img_path.name
        if not dest.exists():
            try:
                dest.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dest)

    # Get image dimensions
    from PIL import Image

    with Image.open(images[0]) as img:
        width, height_img = img.size

    # Write cameras.txt
    cameras_path = sparse_dir / "cameras.txt"
    focal = width / (2 * np.pi)  # Placeholder for equirectangular
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 SIMPLE_PINHOLE {width} {height_img} {focal:.6f} {width/2:.6f} {height_img/2:.6f}\n")

    # Generate circular poses
    # Camera moves in a circle, always looking at center
    images_path = sparse_dir / "images.txt"
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, img_path in enumerate(images):
            # Angle around the circle
            angle = 2 * np.pi * idx / n_images

            # Camera position (looking inward)
            cam_x = center[0] + radius * np.cos(angle)
            cam_y = center[1] + height
            cam_z = center[2] + radius * np.sin(angle)

            # Camera looks at center
            # Forward direction (Z in camera space) points to center
            forward = np.array([center[0] - cam_x, center[1] - cam_y, center[2] - cam_z])
            forward = forward / np.linalg.norm(forward)

            # Up direction (Y in camera space, but Y is down, so we use -Y up)
            up = np.array([0.0, -1.0, 0.0])

            # Right direction
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)

            # Recompute up to be orthogonal
            up = np.cross(right, forward)

            # Rotation matrix (camera to world)
            # Camera convention: X-right, Y-down, Z-forward
            R_c2w = np.column_stack([right, -up, forward])

            # World to camera rotation
            R_w2c = R_c2w.T

            # Translation in world-to-camera
            t_w2c = -R_w2c @ np.array([cam_x, cam_y, cam_z])

            # Convert rotation matrix to quaternion
            # Using Shepperd's method
            trace = np.trace(R_w2c)
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                qw = 0.25 / s
                qx = (R_w2c[2, 1] - R_w2c[1, 2]) * s
                qy = (R_w2c[0, 2] - R_w2c[2, 0]) * s
                qz = (R_w2c[1, 0] - R_w2c[0, 1]) * s
            elif R_w2c[0, 0] > R_w2c[1, 1] and R_w2c[0, 0] > R_w2c[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_w2c[0, 0] - R_w2c[1, 1] - R_w2c[2, 2])
                qw = (R_w2c[2, 1] - R_w2c[1, 2]) / s
                qx = 0.25 * s
                qy = (R_w2c[0, 1] + R_w2c[1, 0]) / s
                qz = (R_w2c[0, 2] + R_w2c[2, 0]) / s
            elif R_w2c[1, 1] > R_w2c[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R_w2c[1, 1] - R_w2c[0, 0] - R_w2c[2, 2])
                qw = (R_w2c[0, 2] - R_w2c[2, 0]) / s
                qx = (R_w2c[0, 1] + R_w2c[1, 0]) / s
                qy = 0.25 * s
                qz = (R_w2c[1, 2] + R_w2c[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R_w2c[2, 2] - R_w2c[0, 0] - R_w2c[1, 1])
                qw = (R_w2c[1, 0] - R_w2c[0, 1]) / s
                qx = (R_w2c[0, 2] + R_w2c[2, 0]) / s
                qy = (R_w2c[1, 2] + R_w2c[2, 1]) / s
                qz = 0.25 * s

            tx, ty, tz = t_w2c

            f.write(f"{idx + 1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} ")
            f.write(f"{tx:.10f} {ty:.10f} {tz:.10f} 1 {img_path.name}\n")
            f.write("\n")

    # Write empty points3D.txt (random initialization will be used)
    points_path = sparse_dir / "points3D.txt"
    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        # Generate some random points around the center for initialization
        n_points = 1000
        for i in range(n_points):
            # Random points in a sphere around center
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-np.pi / 2, np.pi / 2)
            r = np.random.uniform(0.1, radius * 0.8)
            x = center[0] + r * np.cos(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi)
            z = center[2] + r * np.cos(phi) * np.sin(theta)
            f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} 128 128 128 0.0\n")

    print(f"Generated synthetic poses for {n_images} images")
    print(f"  Output: {output_dir}")
    return True


def load_opensfm_reconstruction(project_dir: str | Path) -> dict | None:
    """Load OpenSfM reconstruction results.

    Args:
        project_dir: Path to the OpenSfM project.

    Returns:
        Reconstruction data or None if not found.
    """
    recon_path = Path(project_dir) / "reconstruction.json"

    if not recon_path.exists():
        print(f"Reconstruction not found: {recon_path}")
        return None

    with open(recon_path) as f:
        data = json.load(f)

    if not data:
        print("Empty reconstruction")
        return None

    # Take the first reconstruction (usually the main one)
    recon = data[0]

    print(f"Loaded reconstruction:")
    print(f"  Cameras: {len(recon.get('cameras', {}))}")
    print(f"  Shots: {len(recon.get('shots', {}))}")
    print(f"  Points: {len(recon.get('points', {}))}")

    return recon


def export_to_colmap_format(
    project_dir: str | Path,
    output_dir: str | Path,
    image_dir: str | Path | None = None,
) -> bool:
    """Export OpenSfM reconstruction to COLMAP format for 3DGUT.

    This creates a COLMAP-compatible sparse reconstruction that can be
    loaded by the existing ColmapDataset or a custom equirect dataset.

    Args:
        project_dir: Path to the OpenSfM project.
        output_dir: Output directory for COLMAP format files.
        image_dir: Optional path to images (uses project images if None).

    Returns:
        True if successful, False otherwise.
    """
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)

    recon = load_opensfm_reconstruction(project_dir)
    if recon is None:
        return False

    # Create output directories
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Setup images directory
    if image_dir is None:
        image_dir = project_dir / "images"
    else:
        image_dir = Path(image_dir)

    images_out = output_dir / "images"
    images_out.mkdir(exist_ok=True)

    # Link images
    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            dest = images_out / img_path.name
            if not dest.exists():
                try:
                    dest.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, dest)

    # Export cameras.txt
    # For spherical cameras, we use a placeholder pinhole camera since
    # we'll handle the equirectangular projection in the dataset loader
    cameras_path = sparse_dir / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        # Get image dimensions from first image
        first_image = list(images_out.glob("*"))[0]
        from PIL import Image

        with Image.open(first_image) as img:
            width, height = img.size

        # Use SIMPLE_PINHOLE as placeholder (actual rays generated in dataset)
        # The focal length is a placeholder since we generate spherical rays
        focal = width / (2 * np.pi)  # Approximate for 360 degree horizontal FOV
        f.write(f"1 SIMPLE_PINHOLE {width} {height} {focal:.6f} {width/2:.6f} {height/2:.6f}\n")

    # Export images.txt
    images_path = sparse_dir / "images.txt"
    shots = recon.get("shots", {})

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, (name, shot) in enumerate(sorted(shots.items()), start=1):
            rotation = shot.get("rotation", [0, 0, 0])
            translation = shot.get("translation", [0, 0, 0])

            # Convert axis-angle rotation to quaternion
            angle = np.linalg.norm(rotation)
            if angle > 1e-10:
                axis = np.array(rotation) / angle
                qw = np.cos(angle / 2)
                qx, qy, qz = axis * np.sin(angle / 2)
            else:
                qw, qx, qy, qz = 1, 0, 0, 0

            tx, ty, tz = translation

            f.write(f"{idx} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} ")
            f.write(f"{tx:.10f} {ty:.10f} {tz:.10f} 1 {name}\n")
            f.write("\n")  # Empty line for 2D points (not needed for our use case)

    # Export points3D.txt (sparse point cloud)
    points_path = sparse_dir / "points3D.txt"
    points = recon.get("points", {})

    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

        for point_id, point in points.items():
            coords = point.get("coordinates", [0, 0, 0])
            color = point.get("color", [128, 128, 128])
            x, y, z = coords
            r, g, b = color
            f.write(f"{point_id} {x:.10f} {y:.10f} {z:.10f} {r} {g} {b} 0.0\n")

    print(f"Exported COLMAP format to {output_dir}")
    print(f"  Cameras: 1 (spherical)")
    print(f"  Images: {len(shots)}")
    print(f"  Points: {len(points)}")

    return True


def export_sparse_ply(project_dir: str | Path, output_path: str | Path) -> bool:
    """Export sparse point cloud as PLY file for debugging.

    Args:
        project_dir: Path to OpenSfM project.
        output_path: Output PLY file path.

    Returns:
        True if successful, False otherwise.
    """
    recon = load_opensfm_reconstruction(project_dir)
    if recon is None:
        return False

    points = recon.get("points", {})
    if not points:
        print("No points to export")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect point data
    vertices = []
    colors = []

    for point_id, point in points.items():
        coords = point.get("coordinates", [0, 0, 0])
        color = point.get("color", [128, 128, 128])
        vertices.append(coords)
        colors.append(color)

    vertices = np.array(vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    # Write PLY
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    print(f"Exported {len(vertices)} points to {output_path}")
    return True


def visualize_reconstruction(
    project_dir: str | Path,
    output_dir: str | Path,
    num_views: int = 4,
) -> list[Path]:
    """Generate debug visualizations of the reconstruction.

    Args:
        project_dir: Path to OpenSfM project.
        output_dir: Output directory for images.
        num_views: Number of views to render.

    Returns:
        List of generated image paths.
    """
    # This is a placeholder - actual visualization would use Open3D or similar
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recon = load_opensfm_reconstruction(project_dir)
    if recon is None:
        return []

    # Export PLY for external visualization
    ply_path = output_dir / "sparse_cloud.ply"
    export_sparse_ply(project_dir, ply_path)

    # Export camera positions as separate PLY
    shots = recon.get("shots", {})
    cam_positions = []

    for name, shot in shots.items():
        translation = shot.get("translation", [0, 0, 0])
        rotation = shot.get("rotation", [0, 0, 0])

        # Camera position in world coordinates
        # OpenSfM stores camera-to-world as rotation + translation
        # Position = -R^T * t
        angle = np.linalg.norm(rotation)
        if angle > 1e-10:
            axis = np.array(rotation) / angle
            # Rodrigues formula for rotation matrix
            K = np.array(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ]
            )
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        else:
            R = np.eye(3)

        position = -R.T @ np.array(translation)
        cam_positions.append(position)

    # Write camera positions PLY
    cam_ply_path = output_dir / "camera_positions.ply"
    cam_positions = np.array(cam_positions, dtype=np.float32)

    with open(cam_ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(cam_positions)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p in cam_positions:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 255 0 0\n")

    print(f"Exported camera positions to {cam_ply_path}")

    return [ply_path, cam_ply_path]


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenSfM for 360 equirectangular images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
If OpenSfM is not installed, use --synthetic-poses to generate circular camera
poses for testing.

To install OpenSfM:
  # Clone and build from source
  git clone --recursive https://github.com/mapillary/OpenSfM.git
  cd OpenSfM
  pip install -r requirements.txt
  python setup.py build

Or use Docker:
  docker pull opendronemap/opensfm
""",
    )
    parser.add_argument("--images", "-i", type=str, required=True, help="Input images directory")
    parser.add_argument("--project", "-p", type=str, required=True, help="OpenSfM project directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory for COLMAP format export")
    parser.add_argument("--export-ply", type=str, help="Export sparse point cloud to PLY")
    parser.add_argument("--skip-sfm", action="store_true", help="Skip SfM, only export")
    parser.add_argument(
        "--synthetic-poses",
        action="store_true",
        help="Generate synthetic circular poses (fallback when OpenSfM unavailable)",
    )
    parser.add_argument("--radius", type=float, default=2.0, help="Radius for synthetic circular poses (default: 2.0)")
    parser.add_argument("--height", type=float, default=0.0, help="Camera height for synthetic poses (default: 0.0)")

    args = parser.parse_args()

    try:
        # Check if we should use synthetic poses
        use_synthetic = args.synthetic_poses

        if not use_synthetic and not args.skip_sfm:
            if not is_opensfm_available():
                print("WARNING: OpenSfM is not installed.")
                print("Falling back to synthetic circular poses.")
                print("For better results, install OpenSfM or use --synthetic-poses explicitly.")
                print()
                use_synthetic = True

        if use_synthetic:
            # Generate synthetic poses
            if args.output:
                success = generate_synthetic_circular_poses(
                    image_dir=args.images,
                    output_dir=args.output,
                    radius=args.radius,
                    height=args.height,
                )
                if not success:
                    return 1
            else:
                print("ERROR: --output is required for synthetic poses")
                return 1
        else:
            # Use OpenSfM
            project_dir = create_opensfm_project(args.images, args.project)

            # Run SfM pipeline
            if not args.skip_sfm:
                if not run_opensfm_pipeline(project_dir):
                    print("OpenSfM pipeline failed")
                    return 1

            # Export to COLMAP format
            if args.output:
                if not export_to_colmap_format(project_dir, args.output, args.images):
                    print("COLMAP export failed")
                    return 1

            # Export PLY
            if args.export_ply:
                if not export_sparse_ply(project_dir, args.export_ply):
                    print("PLY export failed")
                    return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
