"""Official grid-based convex selection and BSP extraction; see THIRD_PARTY.md."""
import colorsys
import math
import numpy as np
import torch
import trimesh
from .bspt import get_mesh_watertight

PLANE_COUNT = 4096
CONVEX_COUNT = 256
GRID_SIZE = 64
ACTIVE_THRESHOLD = 0.01


def visible_color(index: int) -> np.ndarray:
    hue = (index * 0.6180339887498949) % 1.0
    saturation = 0.68 + 0.20 * ((index % 3) / 2.0)
    value = 0.82 + 0.16 * (((index // 3) % 2))
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return np.asarray([*(round(channel * 255) for channel in rgb), 255], dtype=np.uint8)

def grid_chunks(device: torch.device) -> list[torch.Tensor]:
    """Build the same eight interleaved 32^3 query batches as upstream."""

    chunks: list[torch.Tensor] = []
    base = torch.arange(0, GRID_SIZE, 2, device=device, dtype=torch.float32)
    for offset_x in range(2):
        for offset_y in range(2):
            for offset_z in range(2):
                x, y, z = torch.meshgrid(
                    base + offset_x,
                    base + offset_y,
                    base + offset_z,
                    indexing="ij",
                )
                xyz = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
                xyz = (xyz + 0.5) / GRID_SIZE - 0.5
                chunks.append(
                    torch.cat(
                        (xyz, torch.ones((len(xyz), 1), device=device)), dim=1
                    )
                )
    return chunks


def select_convexes(
    plane_matrix: torch.Tensor,
    connection_mask: torch.Tensor,
    query_chunks: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    minimum = torch.full(
        (CONVEX_COUNT,), math.inf, device=plane_matrix.device, dtype=torch.float32
    )
    inside_chunks: list[np.ndarray] = []
    for points in query_chunks:
        plane_values = torch.clamp(points @ plane_matrix, min=0.0)
        convex_values = plane_values @ connection_mask
        minimum = torch.minimum(minimum, convex_values.amin(dim=0))
        inside_chunks.append((convex_values < ACTIVE_THRESHOLD).cpu().numpy())
    raw_active = torch.nonzero(
        minimum < ACTIVE_THRESHOLD, as_tuple=False
    ).flatten().cpu().numpy()

    # Match upstream test_bsp/test_mesh_obj_material: remove a convex only when
    # every occupied grid sample remains covered by another retained convex.
    inside = np.concatenate(inside_chunks, axis=0)
    coverage = inside.sum(axis=1, dtype=np.int16)
    retained: list[int] = []
    for slot in range(CONVEX_COUNT):
        slot_inside = inside[:, slot]
        if not np.any(slot_inside):
            continue
        if np.min(coverage - slot_inside.astype(np.int16) * 2) >= 0:
            coverage -= slot_inside
        else:
            retained.append(slot)
    return raw_active, np.asarray(retained, dtype=np.int64)


def triangulate(polygons: list[list[int]]) -> np.ndarray:
    triangles: list[list[int]] = []
    for polygon in polygons:
        for index in range(1, len(polygon) - 1):
            triangles.append([polygon[0], polygon[index], polygon[index + 1]])
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def extract_parts(
    plane_matrix: np.ndarray,
    connection_mask: np.ndarray,
    active: np.ndarray,
) -> tuple[list[trimesh.Trimesh], list[int]]:
    parts: list[trimesh.Trimesh] = []
    source_slots: list[int] = []
    for slot in active.tolist():
        selected = np.flatnonzero(connection_mask[:, slot])
        if len(selected) == 0:
            continue
        # Upstream defines a*x+b*y+c*z+d > 0 as the convex interior and
        # negates the decoder output before passing it to the BSP extractor.
        box = np.ascontiguousarray(-plane_matrix[:, selected].T, dtype=np.float32)
        vertices, polygons = get_mesh_watertight([box])
        faces = triangulate(polygons)
        if len(vertices) < 4 or len(faces) < 4:
            continue
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=faces,
            process=False,
            validate=False,
        )
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
            continue
        color = visible_color(slot)
        mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))
        parts.append(mesh)
        source_slots.append(slot)
    return parts, source_slots
