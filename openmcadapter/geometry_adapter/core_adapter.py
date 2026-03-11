"""
This modules contains functions to transform a RAMP Core to an openmc model
"""

from pathlib import PurePath
from typing import Callable, Iterable, Optional, Sequence

import coremaker.geometries.box
import numpy as np
import openmc
from coremaker.component import ConcreteComponent as Component
from coremaker.geometries.box import Rectangle
from coremaker.grids import CartesianLattice
from coremaker.protocols.core import Core
from coremaker.protocols.element import Element
from coremaker.protocols.grid import Grid
from coremaker.surfaces.surfacecache import SurfaceCache
from coremaker.transform import Transform

from openmcadapter.geometry_adapter.component_adapter import (
    openmc_component,
    openmc_region,
)
from openmcadapter.geometry_adapter.element_adapter import openmc_universe_from_element


def openmc_lattice(
    lattice: CartesianLattice,
    transform: Transform,
    path: PurePath,
    surface_cache: SurfaceCache = None,
    library_path=None,
) -> openmc.RectLattice:
    """
    function to create an openmc lattice form a RAMP lattice and its transform
    """
    lat = openmc.RectLattice(name=str(path))
    default_component = Component(lattice.mixture, lattice.inner_geometry)
    default_cell = openmc_component(
        component=default_component,
        surface_cache=surface_cache,
        library_path=library_path,
    )
    lat.pitch = np.round(lattice.inner_geometry.dimensions, 4)
    if isinstance(lattice.geometry, Rectangle):
        lat.lower_left = np.round(
            -lattice.dimensions * np.array(lattice.shape) / 2 + transform.translation.flatten()[:2],
            4,
        )
        shape = lattice.shape[::-1]
    else:
        lat.lower_left = np.round(
            -np.hstack([lattice.dimensions, [lattice.height]]) * np.hstack([(np.array(lattice.shape) / 2), [1 / 2]])
            + transform.translation.flatten(),
            4,
        )
        shape = np.hstack([[1], lattice.shape[::-1]])
    lat.universes = np.tile(openmc.Universe(cells=[default_cell]), shape)
    lat.outer = openmc.Universe(
        cells=[
            openmc_component(
                Component(lattice.mixture, lattice.geometry),
                surface_cache=surface_cache,
                library_path=library_path,
            )
        ]
    )
    return lat


def place_rod_in_lattice(
    rod: Element,
    grid: Grid,
    site: str,
    cells: dict,
    lattices: Sequence[openmc.Lattice],
    surface_cache: SurfaceCache = None,
    library_path=None,
):
    """
    function used to place a rod in an openmc lattice, it figures out where
    to place the rods and places it.
    The operation is a little bit different for 2d and 3d geometries.
    Parameters
    ----------
    core - the Core
    site - the site of the rod to place
    cells - dict of cells ids
    lattices - the sequence of openmc lattices which contains the lattice in
                which the rod should be inserted.
    """
    lattice, (i, j) = grid.site_index(site)
    for l_index, lat in enumerate(grid.lattices):
        if lat == lattice:
            break
    else:
        raise KeyError("Lattice was not found")
    universe, cells_ids = openmc_universe_from_element(rod, site, surface_cache, library_path=library_path)
    cells.update(cells_ids)
    if len((lat_universes := lattices[l_index].universes).shape) == 3:
        lat_universes[0, lat_universes.shape[1] - i - 1, j] = universe
    else:
        lat_universes[lat_universes.shape[0] - i - 1, j] = universe


def add_boundary_condition(surfaces: Iterable[openmc.Surface], boundary_condition: str):
    """
    function that adds a boundary condition to surfaces
    """
    for surface in surfaces:
        surface.boundary_type = boundary_condition


def construct_box_region(
    lower_left: np.array,
    upper_right: np.array,
    surface_cache: SurfaceCache,
    boundary_condition: Optional[str] = None,
) -> openmc.Region:
    """
    function that constructs a region of a box from its lower left and upper right vertices.
    If the box is infinite in the z direction a 2d region is returned
    """
    if (len(lower_left) < 3 or lower_left[2] == -np.inf) and (len(upper_right) < 3 or upper_right[2] == np.inf):
        geo = Rectangle(
            tuple(lower_left[:2] / 2 + upper_right[:2] / 2),
            upper_right[:2] - lower_left[:2],
        )
    else:
        geo = coremaker.geometries.box.Box(tuple(lower_left / 2 + upper_right / 2), upper_right - lower_left)
    region = openmc_region(geo, surface_cache)
    boundaries = region.get_surfaces().values()
    if boundary_condition:
        add_boundary_condition(boundaries, boundary_condition)
    return region


def _default_source_func(model) -> openmc.IndependentSource:
    """
    The default source is uniform in the bounding box of the core lattices.
    If there are no lattices, the source is uniform in the (-1, -1, -1) to (1, 1, 1) box.
    It is not taken to be uniform on the entire core as it takes too much
    time to create an initial source of that shape.
    """
    if not model.geometry.get_all_lattices():
        return openmc.IndependentSource(
            space=openmc.stats.Box((-1, -1, -1), (1, 1, 1)), constraints={"fissionable": True}
        )
    lower_left = np.min(
        np.vstack([lattice.outer.bounding_box[0] for lattice in model.geometry.get_all_lattices().values()]),
        axis=0,
    )
    upper_right = np.max(
        np.vstack([lattice.outer.bounding_box[1] for lattice in model.geometry.get_all_lattices().values()]),
        axis=0,
    )
    if lower_left[-1] == -np.inf:
        lower_left[-1] = -1
    if upper_right[-1] == np.inf:
        upper_right[-1] = 1
    return openmc.IndependentSource(
        space=openmc.stats.Box(lower_left.flatten(), upper_right.flatten()), constraints={"fissionable": True}
    )


def _false(*_, **__):
    return False


def openmc_core_to_model(
    core: Core,
    boundary_condition: str,
    source_func: Callable[[openmc.model.Model], openmc.IndependentSource] = _default_source_func,
    temperature_method="interpolation",
    library_path=None,
) -> tuple[openmc.model.Model, dict, SurfaceCache]:
    """
    function that creates an openmc model which represents a RAMP core.
    Parameters
    ----------
    core - the core to be represented
    boundary_condition - the boundary condition, it should be one of
                    'transmission', 'vacuum', 'reflective', 'periodic', 'white'
    source_func- a function that determines the source according to the model.
    temperature_method - Literal['nearest', 'interpolation']
        The method used to determine materials' temperature,
        default is set to `interpolation`.

    Returns
    -------
    A tuple of an openmc model object and a dict whose values are the ids of
    the cells in the model
    """
    surface_cache = SurfaceCache(_false)
    model = openmc.model.Model()
    cells = {}
    openmc_lattices = [
        openmc_lattice(lattice, transform, path, surface_cache, library_path)
        for path, transform, lattice in core.lattices()
    ]

    root_universe = openmc.Universe()
    exo_surfaces_ids = {}
    exo_universe = openmc.Universe()
    for element_name, element in core.free_elements:
        universe, element_cells = openmc_universe_from_element(element, element_name, surface_cache, library_path)
        if element_cells:
            cells.update(element_cells)
            dx = element.outer_geometry.bounding_box().dimensions
            center = element.outer_geometry.bounding_box().center
            lower_left = np.array(center) - np.array(dx) / 2
            upper_right = np.array(center) + np.array(dx) / 2
            cell = openmc.Cell(
                fill=universe,
                region=construct_box_region(lower_left, upper_right, surface_cache),
            )
            cells[PurePath(element_name)] = cell.id
            exo_universe.add_cell(cell)
    exo_cell = openmc.Cell(fill=exo_universe)
    exo_region = ""
    for site, rod in core.grid.items():
        place_rod_in_lattice(
            rod,
            core.grid,
            site,
            cells,
            openmc_lattices,
            surface_cache,
            library_path=library_path,
        )
    for mc_lattice in openmc_lattices:
        region = construct_box_region(
            mc_lattice.lower_left,
            mc_lattice.lower_left + mc_lattice.shape * mc_lattice.pitch,
            surface_cache=surface_cache,
        )
        exo_surfaces_ids.update(region.get_surfaces())
        cell = openmc.Cell(fill=mc_lattice, region=region)
        root_universe.add_cell(cell)
        cells[mc_lattice.name] = cell.id
        exo_region = exo_region + f" ~{cell.region}"
        exo_surfaces_ids.update(cell.region.get_surfaces())
    region = openmc_region(core.outer_geometry, surface_cache)
    add_boundary_condition(region.get_surfaces().values(), boundary_condition)
    exo_surfaces_ids.update(region.get_surfaces())
    exo_region = exo_region + f" {region}"
    exo_cell.region = openmc.Region.from_expression(exo_region, exo_surfaces_ids)
    if len(exo_universe.cells.values()) > 0:
        root_universe.add_cell(exo_cell)
        cells[PurePath("exo_universe")] = exo_cell.id
    root_cell = openmc.Cell(fill=root_universe, region=region)
    model.geometry.root_universe = openmc.Universe(cells=[root_cell])
    cells[PurePath("root_universe")] = root_cell.id
    model.materials = root_universe.get_all_materials().values()
    model.materials.cross_sections = library_path
    model.settings.source = source_func(model)
    model.settings.temperature = dict(method=temperature_method)
    model.settings.survival_biasing = True
    return model, cells, surface_cache
