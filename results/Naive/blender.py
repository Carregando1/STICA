"""
module for rendering 3-dimensional binary arrays in blender.
"""

from typing import List

import bpy
from bpy.types import Object
from mathutils import Vector
from numpy import int32
from numpy._typing import NDArray
from typing_extensions import Self

bpy.ops.object.select_all(action="DESELECT")


def clear_initial():
    """
    Clears all of the initial objects that are loaded onto blender, as well as all non-default collections.
    """
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action="DESELECT")

    for coll in bpy.data.collections:
        # Unlink collection from any parent collections or scenes
        for scene in bpy.data.scenes:
            if coll.name in scene.collection.children:
                scene.collection.children.unlink(coll)

        # Remove collection from any other collections
        for parent_coll in bpy.data.collections:
            if coll.name in parent_coll.children:
                parent_coll.children.unlink(coll)

        # Delete the collection
        bpy.data.collections.remove(coll)


def cell_template() -> Object:
    """
    Generates a template for a cubic cell which can be used to generate all other cubes in a 3x3 lattice.
    The template itself will not be linked to any collection and remain invisible.
    """
    loc = Vector((0, 0, 1))
    size = Vector((1, 1, 1))
    bpy.ops.mesh.primitive_cube_add(location=loc + 0.5 * size, size=1)
    assert bpy.context is not None and bpy.context.object is not None, "Impossible"
    cube = bpy.context.object
    cube.name = "CellTemplate"
    bpy.context.scene.collection.objects.unlink(cube)
    return cube


def copy_cell(template: Object, location: tuple[int, int, int]) -> Object:
    """
    Copy a given cell to a specified coordinate.
    """
    cell = template.copy()
    cell.location = Vector(location) + Vector((0.5, 0.5, 1.5))
    cell.scale = (1, 1, 1)
    cell.name = f"Cell({location[0]},{location[1]},{location[2]})"
    assert bpy.context is not None, "Impossible"
    bpy.context.scene.collection.objects.link(cell)
    return cell


class Lattice:
    """
    Class that implements the 3d lattice and will select and unselect objects as necessary.
    """

    def __init__(
        self, dim: tuple[int, int, int], template: Object | None = None
    ) -> None:
        assert (
            dim[0] >= 1 and dim[1] >= 1 and dim[2] >= 1
        ), "All three dimensions must be positive integers!"
        self.template = cell_template() if template is None else template
        self.dim = dim
        self.cubes: List[Object] = []
        self.mesh: Object | None = None

    def clear_lattice(self) -> Self:
        """
        Clears a created lattice of cubes.
        """
        if self.mesh is not None:
            bpy.ops.object.select_all(action="DESELECT")
            self.mesh.select_set(True)
            bpy.ops.object.delete()
            self.mesh = None
        if self.cubes:
            bpy.ops.object.select_all(action="DESELECT")
            for cube in self.cubes:
                cube.select_set(True)
            bpy.ops.object.delete()
            self.cubes = []
        return self

    def update_selected(self, states: NDArray[int32]):
        """
        Updates the cubes that are selected by the lattice.

        :param states - a numpy array with the same shape as self.dim
        """
        assert bpy.context is not None, "blender context must actually exist."
        self.cubes = []
        assert self.dim[0] * self.dim[1] * self.dim[2] > 0
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                for k in range(self.dim[2]):
                    if states[k][i][j] != 0:
                        cube = copy_cell(self.template, (i, j, k))
                        cube.select_set(True)
                        self.cubes.append(cube)

    def export_obj(self, filepath: str):
        bpy.ops.wm.obj_export(filepath=filepath)

    def export_stl(self, filepath: str):
        bpy.ops.object.select_all(action="DESELECT")
        for cube in self.cubes:
            cube.select_set(True)
        bpy.ops.export_mesh.stl(filepath=filepath, use_selection=True)
