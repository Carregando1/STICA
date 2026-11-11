"""Export a voxel lattice (e.g. a STICA trajectory) as a watertight binary STL of its exposed faces.

Voxel grid[i, j, k] is the unit cube [j, j+1] x [k, k+1] x [i, i+1] (x, y, z): the first axis (time) is vertical.
Any non-zero state is solid. Normals point outwards; the mesh has 2 triangles per exposed face.
"""
from __future__ import annotations

import os

import numpy as np

_STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def exposed_face_triangles(grid) -> tuple[np.ndarray, np.ndarray]:
    """Triangles (F, 3, 3) in (x, y, z) and their unit normals (F, 3)."""
    solid = np.pad(np.asarray(grid) != 0, 1).astype(np.int8)
    eye = np.eye(3)
    tris, normals = [], []
    for a in range(3):                        # face normal along index axis a
        u, v = (a + 1) % 3, (a + 2) % 3        # e_u x e_v = e_a
        d = np.diff(solid, axis=a)             # +1: empty -> solid (normal -e_a), -1: solid -> empty (normal +e_a)
        for sign in (1, -1):
            idx = np.argwhere(d == sign).astype(np.float32)
            if not len(idx):
                continue
            base = idx.copy()
            base[:, u] -= 1                    # undo the padding on the in-plane axes
            base[:, v] -= 1
            p0, p1, p2, p3 = base, base + eye[u], base + eye[u] + eye[v], base + eye[v]
            quad = (p0, p1, p2, p3) if sign == -1 else (p0, p3, p2, p1)
            tris += [np.stack([quad[0], quad[1], quad[2]], 1), np.stack([quad[0], quad[2], quad[3]], 1)]
            nrm = np.tile(eye[a] * (1.0 if sign == -1 else -1.0), (2 * len(idx), 1))
            normals.append(nrm)
    if not tris:
        return np.zeros((0, 3, 3), np.float32), np.zeros((0, 3), np.float32)
    tris = np.concatenate(tris)[..., [1, 2, 0]]            # index axes (i, j, k) -> (x, y, z) = (j, k, i)
    normals = np.concatenate(normals)[:, [1, 2, 0]]
    return tris.astype(np.float32), normals.astype(np.float32)


def lattice_to_stl(grid, path: str | os.PathLike, voxel_size: float = 1.0) -> int:
    """Write a binary STL; returns the number of triangles."""
    tris, normals = exposed_face_triangles(grid)
    data = np.zeros(len(tris), _STL_DTYPE)
    data["vertices"] = tris * voxel_size
    data["normal"] = normals
    with open(path, "wb") as f:
        f.write(b"STICA voxel lattice".ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(data.tobytes())
    return len(tris)
