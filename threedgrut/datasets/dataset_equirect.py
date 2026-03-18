# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset loader for equirectangular (360) images with spherical ray generation."""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .utils import (
    create_pixel_coords,
    get_center_and_diag,
    get_worker_id,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
    read_colmap_intrinsics_binary,
    read_colmap_intrinsics_text,
)


def generate_equirect_rays(width: int, height: int, ray_jitter=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray directions for equirectangular projection.

    For each pixel (u, v) in an equirectangular image:
    - theta = (u / width) * 2 * pi - pi      # longitude: -pi to pi
    - phi = (v / height) * pi - pi/2         # latitude: -pi/2 to pi/2

    Ray direction in camera space (Y-down, Z-forward):
    - x = cos(phi) * sin(theta)
    - y = -sin(phi)
    - z = cos(phi) * cos(theta)

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        ray_jitter: Optional ray jitter function.

    Returns:
        Tuple of (rays_ori, rays_dir) tensors, each of shape [1, H, W, 3].
    """
    # Generate pixel coordinates with center offset
    if ray_jitter is not None:
        n_pixels = width * height
        jitter = ray_jitter((n_pixels, 2)).numpy()
        jitter_u = jitter[:, 0].reshape(height, width)
        jitter_v = jitter[:, 1].reshape(height, width)
    else:
        jitter_u = 0.5
        jitter_v = 0.5

    # Create coordinate grids
    u_coords = np.arange(width, dtype=np.float32)
    v_coords = np.arange(height, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords, indexing="xy")

    # Add jitter (or center offset)
    u_grid = u_grid + jitter_u
    v_grid = v_grid + jitter_v

    # Normalize to [0, 1]
    u_norm = u_grid / width
    v_norm = v_grid / height

    # Convert to spherical coordinates
    # theta: longitude, -pi to pi (left to right)
    # phi: latitude, -pi/2 to pi/2 (top to bottom in image, but poles in sphere)
    theta = (u_norm * 2 - 1) * np.pi  # [-pi, pi]
    phi = (v_norm - 0.5) * np.pi  # [-pi/2, pi/2]

    # Convert to Cartesian coordinates
    # Camera convention: Y-down, Z-forward (looking at +Z)
    cos_phi = np.cos(phi)
    x = cos_phi * np.sin(theta)
    y = -np.sin(phi)  # Y is down
    z = cos_phi * np.cos(theta)

    # Stack into ray directions
    rays_dir = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Ray origins are at camera center (0, 0, 0 in camera space)
    rays_ori = np.zeros_like(rays_dir)

    # Add batch dimension [1, H, W, 3]
    rays_ori = torch.from_numpy(rays_ori).unsqueeze(0)
    rays_dir = torch.from_numpy(rays_dir).unsqueeze(0)

    return rays_ori, rays_dir


class EquirectDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    """Dataset for equirectangular (360) images with spherical ray generation.

    This dataset reads camera poses from COLMAP format (compatible with OpenSfM export)
    but generates spherical rays for equirectangular images instead of using
    the stored camera intrinsics.
    """

    def __init__(
        self,
        path: str,
        device: str = "cuda",
        split: str = "train",
        downsample_factor: int = 1,
        test_split_interval: int = 8,
        ray_jitter=None,
    ):
        """Initialize the equirectangular dataset.

        Args:
            path: Path to dataset root (COLMAP format with sparse/0 and images/).
            device: Device for GPU tensors.
            split: Dataset split ("train" or "val").
            downsample_factor: Factor to downsample images (1 = no downsampling).
            test_split_interval: Every N-th frame goes to validation split.
            ray_jitter: Optional ray jitter function for training.
        """
        self.path = path
        self.device = device
        self.split = split
        self.downsample_factor = downsample_factor
        self.ray_jitter = ray_jitter
        self.test_split_interval = test_split_interval

        # Worker-based GPU cache for multiprocessing compatibility
        self._worker_gpu_cache = {}

        # Load data
        self.reload()

    def reload(self):
        """(Re)load the dataset."""
        # Load intrinsics and extrinsics from COLMAP format
        self._load_colmap_data()

        # Build camera index mapping
        sorted_camera_ids = sorted(self.cam_intrinsics.keys())
        self._camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(sorted_camera_ids)}

        self.n_frames = len(self.cam_extrinsics)

        # Load camera data and poses
        self._load_camera_data()

        # Apply train/val split
        indices = np.arange(self.n_frames)

        if self.test_split_interval > 0:
            if self.split == "train":
                indices = np.mod(indices, self.test_split_interval) != 0
            else:
                indices = np.mod(indices, self.test_split_interval) == 0

        # Filter by split
        self.cam_extrinsics = [self.cam_extrinsics[i] for i in np.where(indices)[0]]
        self.poses = self.poses[indices].astype(np.float32)
        self.image_paths = self.image_paths[indices]
        self.camera_centers = self.camera_centers[indices]

        # Compute spatial extents
        self.center, self.length_scale, self.scene_bbox = self._compute_spatial_extents()

        # Update frame count
        self.n_frames = self.poses.shape[0]

        # Clear GPU cache
        self._worker_gpu_cache.clear()

        # Pre-compute equirectangular rays for the image resolution
        self._precompute_rays()

    def _load_colmap_data(self):
        """Load COLMAP intrinsics and extrinsics."""
        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)
        except Exception:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

    def _get_images_folder(self) -> str:
        """Get the images folder name with downsample suffix.

        Falls back to 'images/' when the downsampled folder doesn't exist,
        since __getitem__ handles on-the-fly downsampling.
        """
        if self.downsample_factor > 1:
            ds_folder = os.path.join(self.path, f"images_{self.downsample_factor}")
            if os.path.isdir(ds_folder):
                return f"images_{self.downsample_factor}"
            logger.info(f"Downsampled folder not found ({ds_folder}), falling back to 'images/' with on-the-fly resize")
        return "images"

    def _load_camera_data(self):
        """Load camera poses and image paths."""
        self.poses = []
        self.image_paths = []
        cam_centers = []

        images_folder = self._get_images_folder()

        for extr in logger.track(
            self.cam_extrinsics,
            description=f"Load Equirect Dataset ({self.split})",
            color="salmon1",
        ):
            # Build camera-to-world matrix from COLMAP extrinsics
            R = qvec_to_so3(extr.qvec)
            T = np.array(extr.tvec)
            W2C = np.zeros((4, 4), dtype=np.float32)
            W2C[:3, 3] = T
            W2C[:3, :3] = R
            W2C[3, 3] = 1.0
            C2W = np.linalg.inv(W2C)

            self.poses.append(C2W)
            cam_centers.append(C2W[:3, 3])

            image_path = os.path.join(self.path, images_folder, extr.name)
            self.image_paths.append(image_path)

        self.camera_centers = np.array(cam_centers)
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.poses = np.stack(self.poses)
        self.image_paths = np.array(self.image_paths, dtype=str)

    def _precompute_rays(self):
        """Pre-compute equirectangular rays for the image resolution."""
        # Get dimensions from first image
        first_image_path = self.image_paths[0]
        with Image.open(first_image_path) as img:
            width, height = img.size

        # Apply downsample factor
        if self.downsample_factor > 1:
            width = width // self.downsample_factor
            height = height // self.downsample_factor

        self.image_width = width
        self.image_height = height

        # Generate equirectangular rays (CPU tensors)
        self._rays_ori_cpu, self._rays_dir_cpu = generate_equirect_rays(width, height, self.ray_jitter)

        # Create pixel coordinates
        self._pixel_coords_cpu = create_pixel_coords(width, height)

        logger.info(f"Equirect rays pre-computed: {width}x{height}")

    def _get_worker_rays(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get GPU rays for the current worker."""
        worker_id = get_worker_id()

        if worker_id not in self._worker_gpu_cache:
            # Transfer to GPU for this worker
            rays_ori = self._rays_ori_cpu.to(self.device, non_blocking=True)
            rays_dir = self._rays_dir_cpu.to(self.device, non_blocking=True)
            pixel_coords = self._pixel_coords_cpu.to(self.device, non_blocking=True)
            self._worker_gpu_cache[worker_id] = (rays_ori, rays_dir, pixel_coords)

        return self._worker_gpu_cache[worker_id]

    @torch.no_grad()
    def _compute_spatial_extents(self):
        """Compute scene center and scale from camera positions."""
        camera_origins = torch.FloatTensor(self.poses[:, :, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self) -> torch.Tensor:
        return self.length_scale

    def get_center(self) -> torch.Tensor:
        return self.center

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self) -> float:
        return self.cameras_extent

    def get_observer_points(self) -> np.ndarray:
        return self.camera_centers

    def get_poses(self) -> np.ndarray:
        """Get camera poses as 4x4 camera-to-world transformation matrices."""
        return self.poses

    def get_camera_idx(self, frame_idx: int) -> int:
        """Return 0-based camera index for a given frame index."""
        colmap_camera_id = self.cam_extrinsics[frame_idx].camera_id
        return self._camera_id_to_idx[colmap_camera_id]

    def get_frames_per_camera(self) -> list[int]:
        """Return list of frame counts per camera."""
        num_cameras = len(self.cam_intrinsics)
        counts = [0] * num_cameras
        for extr in self.cam_extrinsics:
            camera_idx = self._camera_id_to_idx[extr.camera_id]
            counts[camera_idx] += 1
        return counts

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with 'data', 'pose', 'camera_idx', 'frame_idx'.
        """
        # Load image
        image_data = np.asarray(Image.open(self.image_paths[idx]))

        # Apply downsampling if needed
        if self.downsample_factor > 1:
            h, w = image_data.shape[:2]
            new_h = h // self.downsample_factor
            new_w = w // self.downsample_factor
            # Use PIL for high-quality downsampling
            pil_img = Image.fromarray(image_data)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            image_data = np.asarray(pil_img)

        assert image_data.dtype == np.uint8, "Image data must be uint8"

        output_dict = {
            "data": torch.tensor(image_data).unsqueeze(0),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "camera_idx": self.get_camera_idx(idx),
            "frame_idx": idx,
        }

        return output_dict

    def get_gpu_batch_with_intrinsics(self, batch: dict) -> Batch:
        """Add equirectangular rays to the batch and move data to GPU.

        This is the key method that differs from other datasets - instead of
        using stored camera intrinsics, we use pre-computed spherical rays.
        """
        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)

        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        # Get pre-computed equirectangular rays for this worker
        rays_ori, rays_dir, pixel_coords = self._get_worker_rays()

        # Provide wide-FOV pinhole intrinsics for tile-based culling/sorting.
        # Actual rendering uses pre-computed equirect rays, but the tracer needs
        # camera parameters for its spatial acceleration structure.
        # Use a very short focal length (~179° FOV) so no Gaussians are culled.
        w, h = self.image_width, self.image_height
        focal = 84.0  # ~170° FOV to minimize culling for equirectangular
        cx, cy = w / 2.0, h / 2.0
        intrinsics = [focal, focal, cx, cy]

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            "camera_idx": batch["camera_idx"],
            "frame_idx": batch["frame_idx"],
            "pixel_coords": pixel_coords,
            "intrinsics": intrinsics,
        }

        return Batch(**sample)

    def create_dataset_camera_visualization(self):
        """Create a visualization of the dataset cameras using polyscope."""
        import polyscope as ps

        cam_list = []

        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # Camera convention rotation
            camera_convention_rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            # Load image for visualization
            image_data = np.asarray(Image.open(self.image_paths[i_cam]))
            h, w = image_data.shape[:2]

            # For equirectangular, use 360 degree horizontal FOV
            fov_w = 2 * np.pi  # 360 degrees
            fov_h = np.pi  # 180 degrees

            assert image_data.dtype == np.uint8
            rgb = image_data.reshape(h, w, 3) / np.float32(255.0)

            cam_list.append(
                {
                    "ext_mat": trans_mat_world_to_camera,
                    "w": w,
                    "h": h,
                    "fov_w": fov_w,
                    "fov_h": fov_h,
                    "rgb_img": rgb,
                    "split": self.split,
                }
            )

        # Register cameras in polyscope
        for i_cam, cam in enumerate(cam_list):
            ps_cam_param = ps.CameraParameters(
                ps.CameraIntrinsics(
                    fov_vertical_deg=np.degrees(cam["fov_h"]),
                    fov_horizontal_deg=np.degrees(cam["fov_w"]),
                ),
                ps.CameraExtrinsics(mat=cam["ext_mat"]),
            )

            cam_color = (1.0, 1.0, 1.0)
            if cam["split"] == "train":
                cam_color = (1.0, 0.7, 0.7)
            elif cam["split"] == "val":
                cam_color = (0.7, 0.1, 0.7)

            ps_cam = ps.register_camera_view(f"{cam['split']}_view_{i_cam:03d}", ps_cam_param, widget_color=cam_color)
            ps_cam.add_color_image_quantity("target image", cam["rgb_img"][:, :, :3], enabled=True)
