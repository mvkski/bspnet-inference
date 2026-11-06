# Third-party source provenance

## BSP-Net (MIT)

Upstream: https://github.com/czq142857/BSP-NET-pytorch

Revision: `4b48b3c1ff13d803567cc5150f2cdcbf943feffb`

- `bspnet/model.py`: the four network classes from upstream `modelAE.py`,
  unmodified, with training driver and unused imports omitted.
- `bspnet/bspt.pyx`: official BSP polytope extractor. The uint8 scratch buffer
  is initialized to zero rather than 32768, reproducing the old NumPy wraparound
  behavior without NumPy 2's overflow error. Floating-point sentinels are unchanged.
- `bspnet/geometry.py`: wrapper implementing upstream's phase-1 64³ active-convex
  query and sequential redundant-convex removal, followed by per-part BSP extraction.

See `third_party/BSP-NET-LICENSE` for the original copyright and license.
Paper: *BSP-Net: Generating Compact Meshes via Binary Space Partitioning*, CVPR 2020.

## IM-NET (MIT)

Upstream: https://github.com/czq142857/IM-NET

Revision: `ff2403f79e2a8259165c7de26299cd4429fe934c`

`bspnet/fill.py` adapts the side-view filling operation from
`point_sampling/2_gather_256vox_16_32_64.py` to Numba, as used in this checkpoint's
training preprocessing. See `third_party/IM-NET-LICENSE`.

The wrapper, Docker packaging, and HY3D voxelizer are adaptations for this
checkpoint, not an official release by the BSP-Net authors. Dependency licenses,
weight access restrictions, and the original mesh licenses remain separate.
