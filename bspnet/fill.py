"""IM-NET side filling, Numba adaptation; see THIRD_PARTY.md."""
import numpy as np
from numba import njit

@njit(cache=True)
def side_fill(fine):
    """The official 'carve the voxels from side views' operation, in-place."""
    n = fine.shape[0]
    top = np.zeros((n, n), np.uint8)
    left_min = np.full((n, n), n, np.int32)
    left_max = np.full((n, n), -1, np.int32)
    front_min = np.full((n, n), n, np.int32)
    front_max = np.full((n, n), -1, np.int32)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if fine[i, j, k]:
                    top[i, k] = 1
                    left_min[j, k] = min(left_min[j, k], i)
                    left_max[j, k] = max(left_max[j, k], i)
                    front_min[i, j] = min(front_min[i, j], k)
                    front_max[i, j] = max(front_max[i, j], k)
    for i in range(n):
        for k in range(n):
            if top[i, k]:
                fill = False
                for j in range(n - 1, -1, -1):
                    if fine[i, j, k]:
                        fill = True
                    elif left_min[j, k] < i < left_max[j, k] and front_min[i, j] < k < front_max[i, j]:
                        if fill:
                            fine[i, j, k] = 1
                    else:
                        fill = False


def block_max(fine, resolution):
    stride = fine.shape[0] // resolution
    return fine.reshape(resolution, stride, resolution, stride, resolution, stride).max(axis=(1, 3, 5))

