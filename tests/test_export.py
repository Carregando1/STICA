import numpy as np

from stica.export import exposed_face_triangles, lattice_to_stl
from reference import surface_area_reference


def signed_volume(tris):
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)


def test_mesh_is_closed_and_outward(tmp_path):
    rng = np.random.default_rng(0)
    for _ in range(3):
        g = rng.integers(0, 3, (5, 6, 4))
        tris, normals = exposed_face_triangles(g)
        assert len(tris) == 2 * surface_area_reference(g)
        # divergence theorem: a closed, outward-oriented mesh encloses exactly the solid voxels
        assert np.isclose(signed_volume(tris), np.count_nonzero(g))
        geo = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        assert np.all(np.einsum("ij,ij->i", geo, normals) > 0)
        n = lattice_to_stl(g, tmp_path / "x.stl")
        assert (tmp_path / "x.stl").stat().st_size == 84 + 50 * n


def test_empty_lattice(tmp_path):
    assert lattice_to_stl(np.zeros((3, 3, 3)), tmp_path / "e.stl") == 0
