from copy import deepcopy
from pathlib import Path, PurePath

import pytest
from coremaker.core import Core
from coremaker.elements import BoxTree
from coremaker.example import (
    U_mixture,
    aluminium,
    example_core,
    hafnium,
    hafnium_block_size,
    heavy_water,
    lattice_shape,
    site_size,
)
from coremaker.geometries import Annulus, Box
from coremaker.grids import NullGrid
from coremaker.materials.aluminium import al1050, al6061
from coremaker.materials.water import make_light_water
from coremaker.transform import Transform
from coremaker.tree import ChildType, Node, Tree
from more_itertools import first
from ramp_core import TemporaryDirectory

from openmcadapter.plotting import plot_core, plot_element, plot_geometry


@pytest.mark.regression
def test_plot_geometry(image_regression):
    outer_radius = 1.0
    geometry = Annulus(
        center=(0.0, 0.0, 0.0), inner_radius=0.5, outer_radius=outer_radius, length=20.0, axis=(0.0, 0.0, 1.0)
    )
    with TemporaryDirectory() as d:
        path = Path(d) / Path("plot")
        plot_geometry(geometry, path=path, width=(2 * outer_radius,) * 2)
        image_regression.check(path.read_bytes())


@pytest.fixture()
def colors():
    colors = {heavy_water: "blue", aluminium: "gray", U_mixture: "red", hafnium: "black"}
    return colors


def test_plot_element(image_regression, colors):
    key = first(example_core.grid.keys())
    element = example_core.grid[key]
    assert set(colors.keys()) >= set(comp.mixture for path, comp in element.named_components())
    with TemporaryDirectory() as d:
        path = Path(d) / "plot"
        plot_element(element, path=path, width=(site_size,) * 2, colors=colors)
        image_regression.check(path.read_bytes())


@pytest.mark.regression
@pytest.mark.parametrize("basis", ["xy", "yz", "xz"])
def test_plot_core(image_regression, colors, basis):
    assert set(colors.keys()) >= set(comp.mixture for path, comp in example_core.named_components)
    with TemporaryDirectory() as d:
        path = Path(d) / "plot"
        # noinspection PyTypeChecker
        plot_core(
            example_core,
            path=path,
            width=tuple(3 / 2 * shape * site_size + hafnium_block_size for shape in lattice_shape),
            colors=colors,
            basis=basis,
        )
        image_regression.check(path.read_bytes(), basename=f"test_plot_core_{basis}")


@pytest.mark.regression
def test_plot_core_with_null_grid_by_regression(image_regression):
    b1 = BoxTree((0.1, 5, 10), al6061, PurePath("B1"))
    b2 = BoxTree((0.4, 5, 30), al1050, PurePath("B2"))
    pool = make_light_water(40.0)
    pool_name = PurePath("Pool")
    outer_geo = Box(center=(0, 0, 0), dimensions=(10, 10, 100))
    pool_node = Node(geometry=outer_geo, mixture=pool)
    core_tree = Tree()
    core_tree.nodes[pool_name] = pool_node
    core_tree.graft(b1, pool_name, ChildType.exclusive)
    core_tree.graft(b2, pool_name, ChildType.exclusive)
    core_tree.nodes[pool_name / "B2"].transform = Transform(translation=(-1, 0, -5))
    core = Core(grid=NullGrid(), aliases={}, tree=core_tree, outer_geometry=outer_geo)
    with TemporaryDirectory() as d:
        path = Path(d) / "plot"
        plot_core(core, path=path)
        image_regression.check(path.read_bytes())


def test_color_empty_grid_slots(image_regression, colors):
    core = deepcopy(example_core)
    for site in core.grid.sites():
        if "A" in site:
            del core.grid[site]
    with TemporaryDirectory() as d:
        path = Path(d) / "plot"
        # noinspection PyTypeChecker
        plot_core(
            core,
            path=path,
            width=tuple(shape * site_size for shape in lattice_shape),
            colors=colors,
            basis="xy",
            grid_color="purple",
        )
        image_regression.check(path.read_bytes())


def test_plotting_with_large_aspect(image_regression, colors):
    core = deepcopy(example_core)
    axes = plot_core(
        core,
        width=(site_size * lattice_shape[0], site_size / 60),
        resolution=site_size / 120,
        colors=colors,
        basis="xy",
    )
    fig = axes.get_figure()
    with TemporaryDirectory() as d:
        path = Path(d) / "figure.png"
        fig.savefig(path)
        image_regression.check(path.read_bytes())
