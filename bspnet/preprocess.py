"""Exact voxelizer used for the HY3D training records."""
import numpy as np
import trimesh
from scipy.ndimage import binary_fill_holes

def voxelize(mesh):
    center = mesh.bounds.mean(axis=0)
    extent = float(mesh.extents.max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Degenerate mesh")
    scale = (254.0 / 256.0) / extent
    # Shift the rasterizer's integer lattice to BSP's cell-center convention.
    working = trimesh.Trimesh(vertices=(mesh.vertices - center) * scale + 0.5 - 0.5/256,
                              faces=mesh.faces, process=False)
    surface = working.voxelized(pitch=1/256, method="subdivide")
    idx = np.rint(surface.points * 256).astype(np.int32)
    if np.any(idx < 0) or np.any(idx >= 256):
        raise ValueError("Surface outside normalized grid")
    fine = np.zeros((256, 256, 256), np.uint8)
    fine[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    fine = binary_fill_holes(fine).astype(np.uint8)
    if fine.sum() == 0:
        raise ValueError("Empty voxelization")
    return fine, center, scale
