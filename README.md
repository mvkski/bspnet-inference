# BSP-Net: input mesh to convex decomposition

Standalone inference for our **HY3D-trained BSP-Net autoencoder**. Supply an
OFF, OBJ, PLY, STL, GLB or GLTF mesh and obtain independently colored convex
components in a GLB, in the input mesh's original coordinates.

All inference source is included. No local research repository, training data,
chord bank, SDF, or private container image is required. The trained weights are
downloaded separately from a private Hugging Face repository.

The network has **4,096 planes and 256 candidate convex slots**. Unused/redundant
convexes are removed; the exported count varies by mesh. This is not a model with
a configurable requested output part count.

## Requirements

- Linux x86-64, Git, Python 3.10+ and Docker.
- For GPU inference: NVIDIA driver supporting CUDA 12.8 and NVIDIA Container
  Toolkit (`docker run --gpus all` must work). RTX 5090 is the test target.
- Access to the private Hugging Face weights repository below.
- Allow approximately 15–25 GB for the initial Docker image/build cache plus
  170 MiB for weights. No credentials are passed into the inference container.

CPU inference is supported explicitly with `--device cpu`; it still uses Docker
and is substantially slower. The Docker image uses public Python and PyTorch
CUDA 12.8 packages. GPU architecture support is whatever those PyTorch wheels provide.

## 1. Clone and build

```bash
git clone https://github.com/mvkski/bspnet-inference.git
cd bspnet-inference
bash build.sh
```

The initial build downloads dependencies and compiles the small Cython BSP
extractor. Subsequent runs reuse `bspnet-inference:1`.

## 2. Download weights

Install the Hugging Face CLI if necessary, then log in interactively:

```bash
python3 -m venv .venv
.venv/bin/pip install 'huggingface_hub==1.2.3'
source .venv/bin/activate
hf auth login
hf download my-little-perceptron/bspnet-hy3d BSP_AE_final.pth config.json training_config.json checksums.json README.md --revision 2941e02dd51f5d00c6cc5a6124e22efbedc3fc6b --local-dir weights
python3 verify_weights.py weights
```

Enter a token with read access when prompted; never put it into source files.
The explicit filenames/revision download only the weight release, not the prepared
training data. A 401/403/404 usually means your account lacks access. The wrapper
verifies the exact tested checkpoint hash before inference.

## 3. Run on a mesh

```bash
python3 infer.py /absolute/path/to/camel.off --output-dir results/camel
```

Several inputs can share model initialization:

```bash
python3 infer.py /path/to/camel.off /path/to/hornbug.off /path/to/elephant.off --output-dir results/examples
```

Optional: `--weights-dir /path/to/weights`, `--gpu 1`, or `--device cpu`.
Inputs must have distinct filename stems. Output directories must be new or
empty; existing results are never overwritten. Quote paths with spaces. Commas
and newlines in mount paths are rejected. For GLTF, keep referenced assets next
to the file; self-contained GLB is easier to transfer. Scene transforms and
instances are applied when loading meshes.

## Output and timing

```text
results/camel/
  camel.glb
  camel.stats.json
  summary.json
  inference.log
  run.json
```

The GLB has one named geometry per exported convex. Stats include part count,
original-coordinate bounds, preprocessing, forward-pass, convex-selection,
extraction and export timings. `summary.json` includes per-shape averages.

`end_to_end_seconds` covers raw mesh loading through GLB export, excluding
one-time model loading/warmup, post-export verification and JSON writes.
`run.json:command_wall_seconds` includes container startup and all work after
the host-side checkpoint check. The first shape's preprocessing includes Numba
JIT compilation; do not interpret it as steady-state network latency.
No quality metrics are computed by this repository.

## Exact preprocessing / export convention

1. Load without mesh processing, bbox-center, scale longest extent to `254/256`.
2. Subdivision voxelize at 256³ using cell-center alignment. Apply SciPy solid
   filling and the official IM-NET side-view filling operation.
3. Block-max pool to 64³ and run the original phase-1 BSP-Net in FP32, no TF32.
4. Use connections greater than 0.01 and 64³ occupancy queries to identify active
   convexes and remove redundant convexes in the original sequential order.
5. Extract each remaining convex with the official Cython BSP implementation,
   undo normalization, color and export. No additional hull merge or 3-sigma filter.

The encoder always receives 64³ occupancy, not surface points or depth maps.
This is the voxelizer used for our HY3D training run, not identical to HSP's
ShapeNet voxelization. Side-view filling can alter concavities. Small parts can
be missed by the 64³ convex-activity test. Overlap is permitted by the model.
Raw meshes are not repaired with Mesh2SDF automatically. Training used repaired
meshes; broken inputs may produce different results.

Weights were trained for 1,986,754 optimizer updates: phase 0 at 16³, 32³ and 64³
supervision, followed by phase 1 at 64³. They are final weights, not an authors'
original ShapeNet checkpoint. See the HF model card for the full protocol.

## Tests

```bash
python3 -m unittest tests.test_cli -v
docker run --rm --network none --entrypoint python bspnet-inference:1 -m unittest tests.test_geometry -v
```

See `VALIDATION.md` for integration test results and `THIRD_PARTY.md` for pinned
upstream revisions, source changes and licenses.
