import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import trimesh
from bspnet.geometry import extract_parts, grid_chunks, triangulate
from bspnet.preprocess import voxelize
from bspnet.fill import side_fill, block_max
from bspnet.run import load_mesh
from bspnet.model import bsp_network


class Geometry(unittest.TestCase):
    def test_cube_extraction(self):
        # Decoder inequalities a*x+b*y+c*z+d <= 0.
        planes = np.array([[1,0,0,-.25],[-1,0,0,-.25],[0,1,0,-.25],
                           [0,-1,0,-.25],[0,0,1,-.25],[0,0,-1,-.25]], dtype=np.float32).T
        parts, slots = extract_parts(planes, np.ones((6,1), dtype=bool), np.array([0]))
        self.assertEqual(slots, [0])
        np.testing.assert_allclose(parts[0].bounds, [[-.25]*3,[.25]*3], atol=1e-6)
        self.assertTrue(parts[0].is_watertight)

    def test_grid(self):
        xyz = torch.cat(grid_chunks(torch.device('cpu'))).numpy()
        self.assertEqual(xyz.shape, (64**3, 4))
        self.assertEqual(len(np.unique(xyz[:,:3], axis=0)), 64**3)
        self.assertTrue(np.all(xyz[:,3] == 1))

    def test_triangulate(self):
        np.testing.assert_array_equal(triangulate([[0,1,2,3]]), [[0,1,2],[0,2,3]])

    def test_voxel_normalization(self):
        box = trimesh.creation.box(extents=[2,1,1])
        box.apply_translation([4,5,6])
        fine, center, scale = voxelize(box)
        np.testing.assert_allclose(center, [4,5,6])
        self.assertAlmostEqual(scale, (254/256)/2)
        self.assertEqual(fine.shape, (256,256,256))
        side_fill(fine)
        coarse = block_max(fine, 64)
        self.assertTrue(coarse.any())

    def test_scene_transforms(self):
        scene = trimesh.Scene()
        transform = np.eye(4)
        transform[:3,3] = [5,6,7]
        scene.add_geometry(trimesh.creation.box(), transform=transform)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'scene.glb'
            scene.export(path)
            loaded = load_mesh(path)
            np.testing.assert_allclose(loaded.bounds, [[4.5,5.5,6.5],[5.5,6.5,7.5]])

    def test_architecture(self):
        model = bsp_network(1,32,4096,256).eval()
        with torch.inference_mode():
            latent, planes, _, _ = model(torch.zeros((1,1,64,64,64)),None,None,None)
        self.assertEqual(tuple(latent.shape), (1,256))
        self.assertEqual(tuple(planes.shape), (1,4,4096))


if __name__ == '__main__':
    unittest.main()
