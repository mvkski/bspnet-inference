"""Raw mesh to independent colored convexes, using the HY3D training conventions."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import trimesh

from .model import bsp_network
from .preprocess import voxelize
from .fill import side_fill, block_max
from .geometry import grid_chunks, select_convexes, extract_parts
from verify_weights import verify, EXPECTED_SHA256


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid triangle mesh: {path}")
    return mesh


def configure(device):
    random.seed(2191997)
    np.random.seed(2191997)
    torch.manual_seed(2191997)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for CPU inference")
        torch.cuda.manual_seed_all(2191997)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", nargs="+", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=Path("/weights"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if len({p.stem for p in args.meshes}) != len(args.meshes):
        parser.error("Input mesh stems must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f"{p.stem}.glb").exists() for p in args.meshes):
        raise FileExistsError("Refusing to overwrite an exported mesh")
    started = time.perf_counter()
    device = torch.device(args.device)
    configure(device)
    checkpoint = verify(args.weights_dir)
    model = bsp_network(phase=1, ef_dim=32, p_dim=4096, c_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    mask = (model.generator.convex_layer_weights.detach() > 0.01).float()
    mask_numpy = mask.cpu().numpy().astype(bool)
    queries = grid_chunks(device)
    with torch.inference_mode():
        dummy = torch.zeros((1, 1, 64, 64, 64), device=device)
        for _ in range(2):
            model(dummy, None, None, None, is_training=False)
        synchronize(device)
    setup_seconds = time.perf_counter() - started
    results = []
    for path in args.meshes:
        start = time.perf_counter()
        mesh = load_mesh(path)
        loaded = time.perf_counter()
        fine, center, scale = voxelize(mesh)
        before = int(fine.sum())
        side_fill(fine)
        added = int(fine.sum()) - before
        voxels = block_max(fine, 64)
        del fine
        prepared = time.perf_counter()
        with torch.inference_mode():
            tensor = torch.from_numpy(voxels.copy())[None, None].to(device).float()
            _, planes, _, _ = model(tensor, None, None, None, is_training=False)
            synchronize(device)
            forwarded = time.perf_counter()
            raw, retained = select_convexes(planes[0], mask, queries)
            planes_numpy = planes[0].cpu().numpy()
        selected = time.perf_counter()
        parts, slots = extract_parts(planes_numpy, mask_numpy, retained)
        if not parts:
            raise RuntimeError(f"No valid convexes for {path}")
        scene = trimesh.Scene()
        for part, slot in zip(parts, slots):
            if not np.isfinite(part.vertices).all():
                raise ValueError(f"Nonfinite convex: {slot}")
            part.vertices = part.vertices / scale + center
            scene.add_geometry(part, geom_name=f"convex_{slot:03d}", node_name=f"convex_{slot:03d}")
        extracted = time.perf_counter()
        output = args.output / f"{path.stem}.glb"
        temporary = output.with_suffix(".tmp.glb")
        scene.export(temporary)
        temporary.replace(output)
        exported = time.perf_counter()
        reloaded = trimesh.load(output, force="scene", process=False)
        if len(reloaded.geometry) != len(parts) or not all(np.isfinite(p.vertices).all() for p in reloaded.geometry.values()):
            raise RuntimeError("Export verification failed")
        result = {"shape": path.stem, "input": str(path), "output": output.name,
                  "parts": len(parts), "raw_active_slots": len(raw), "exported_slots": slots,
                  "retained_slots": retained.tolist(), "source_center": center.tolist(), "source_scale": scale,
                  "coordinates": "original input mesh coordinates", "checkpoint_sha256": EXPECTED_SHA256,
                  "occupied_64": int(voxels.sum()), "side_fill_added_voxels": added,
                  "input_bounds": mesh.bounds.tolist(), "output_bounds": scene.bounds.tolist(),
                  "mesh_load_seconds": loaded-start, "preprocess_seconds": prepared-loaded,
                  "nn_forward_seconds": forwarded-prepared, "convex_selection_seconds": selected-forwarded,
                  "bsp_extraction_seconds": extracted-selected, "export_seconds": exported-extracted,
                  "end_to_end_seconds": exported-start, "status": "success"}
        results.append(result)
        atomic_json(output.with_suffix(".stats.json"), result)
        atomic_json(args.output / "summary.json", {"completed": len(results), "requested": len(args.meshes),
                    "results": results, "setup_seconds": setup_seconds, "device": str(device),
                    "mean_parts": sum(r["parts"] for r in results)/len(results),
                    "mean_end_to_end_seconds": sum(r["end_to_end_seconds"] for r in results)/len(results),
                    "timing_note": "Per-shape mesh loading through GLB export; excludes one-time setup, verification and JSON writes. See run.json for command wall time."})
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
