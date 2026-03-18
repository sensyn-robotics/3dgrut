"""Convert a 3DGUT checkpoint to standard 3DGS PLY format for viewing in SuperSplat."""

import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def ckpt_to_ply(checkpoint_path: str, output_path: str | None = None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    positions = ckpt["positions"].detach().numpy()
    density = ckpt["density"].detach().numpy()
    scale = ckpt["scale"].detach().numpy()
    rotation = ckpt["rotation"].detach().numpy()
    features_albedo = ckpt["features_albedo"].detach().numpy()
    features_specular = ckpt["features_specular"].detach().numpy()

    n = positions.shape[0]
    normals = np.zeros((n, 3), dtype=np.float32)

    # Rearrange specular: (N, num_speculars*3) stored as (N, num_speculars, 3) -> (N, 3, num_speculars) -> (N, num_speculars*3)
    max_n_features = ckpt["max_n_features"]
    num_speculars = (max_n_features + 1) ** 2 - 1
    spec = features_specular.reshape(n, num_speculars, 3).transpose(0, 2, 1).reshape(n, num_speculars * 3)

    # Build attribute names
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(3)]
    attrs += [f"f_rest_{i}" for i in range(spec.shape[1])]
    attrs += ["opacity"]
    attrs += [f"scale_{i}" for i in range(3)]
    attrs += [f"rot_{i}" for i in range(4)]

    dtype = [(a, "f4") for a in attrs]
    data = np.concatenate([positions, normals, features_albedo, spec, density, scale, rotation], axis=1)

    elements = np.empty(n, dtype=dtype)
    elements[:] = list(map(tuple, data))

    out = output_path or str(Path(checkpoint_path).with_suffix(".ply"))
    PlyData([PlyElement.describe(elements, "vertex")]).write(out)
    print(f"Wrote {n} gaussians to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> [output.ply]")
        sys.exit(1)
    ckpt_to_ply(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
