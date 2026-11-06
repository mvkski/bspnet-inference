# Release validation — November 6, 2026

- Built the supplied Dockerfile from its public base image and public dependencies,
  not from the research machine's private BSP-Net image (7.57 GB resulting image).
- Downloaded the weights from the pinned private HF release and verified SHA-256.
- Five host CLI tests and six container geometry/network tests passed.
- Checked that the four network classes, voxelizer, side fill, block-max pooling,
  query grid, convex selection and extraction functions have identical Python
  syntax trees to the corresponding training/benchmark functions.
- Ran the documented `python3 infer.py ...` command. Only the downloaded weights,
  input mesh directories and output directory were mounted. Container networking
  was disabled. No training data or external code paths were mounted.

RTX 5090, PyTorch 2.7.0+cu128, NumPy 2.2.6, Trimesh 4.12.2:

| VHACD input | Exported convexes | Per-shape mesh-to-GLB seconds |
| --- | ---: | ---: |
| camel | 26 | 2.305 |
| hornbug | 10 | 1.797 |
| elephant | 27 | 2.122 |

Compared to the earlier research wrapper's exports, all three outputs had identical
active/retained slots, voxel counts, triangle indices, colors, and exported vertex
coordinates (maximum vertex difference **0.0**).

CPU camel inference also passed: 26 convexes, 9.899 seconds. CPU/GPU floating-point
coordinates differ slightly, as expected; cross-device bitwise equality is not promised.

Times exclude one-time model loading/warmup and output verification, and include
first-use Numba compilation on camel. These are smoke tests, not a full benchmark.
Other GPU models, operating systems and arbitrary meshes have not all been tested.
