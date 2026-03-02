import warnings
from contextlib import contextmanager
from typing import Sequence, Optional, Type

import numpy as np
import openmc
from coremaker.geometries.box import Box
from coremaker.protocols.component import Component
from coremaker.component import ConcreteComponent
from coremaker.protocols.geometry import Geometry, HoledGeometry, UnionGeometry
from coremaker.surfaces.surfacecache import SurfaceCache
from isotopes import U235
from coremaker.materials.mixture import Mixture
from openmc import IDWarning

from openmcadapter.geometry_adapter.surface_adapter import openmc_halfspace
from openmcadapter.mixture_adapter import openmc_material, add_burning_isotopes


def negate_surface(expresion_format: str, surface_num):
    """
    This function gets a format that specifies the relation between a
    region and its bounding surfaces and gets an index (from 0 to N-1 where
    N is the number of surfaces in the region) and changes the format so that
    the surface specified by surface_num will be accompanied by a minus
    sign.
    """
    split_format = expresion_format.split('%d')
    split_format[surface_num] += '-'
    return '%d'.join(split_format)


def construct_region(expression_format: str,
                     half_spaces: Sequence[tuple[int, openmc.Halfspace]]) -> openmc.Region:
    """
    this function gets a format and an iterable of half spaces and
    computes the region specified by them. Since in the ramp the
    specification of the chosen half space of the plane is different from
    the specification of the inner or outer half spaces of a cylinder/ball,
    these two cases need to be treated differently
    """
    for surface_num, (index, hs) in enumerate(half_spaces):
        to_negate = False
        if index and index < 0:
            to_negate = True
        to_negate = to_negate ^ (hs.side == "-")
        if to_negate:
            expression_format = negate_surface(expression_format, surface_num)
    region_expression = expression_format % tuple([surface.surface.id
                                                   for _, surface in half_spaces])
    return openmc.Region.from_expression(region_expression,
                                         {hs.surface.id: hs.surface for _, hs in half_spaces})


@contextmanager
def ignore_warnings(warn_type: Type[Warning]):
    warnings.filterwarnings("ignore", category=warn_type)
    yield
    try:
        # noinspection PyUnresolvedReferences
        warnings.filters.pop()
    except IndexError:
        pass


def openmc_region(geometry: Geometry, surface_cache: Optional[SurfaceCache] = None) -> openmc.Region:
    """
    function to build the region defining a geometry.

    Parameters
    ----------
    geometry: Geometry
     The geometry
    surface_cache: Optional[SurfaceCache]
     A surface cache that can be used to find the surfaces used in the region

    Returns
    -------
    openmc.Region
    """
    if surface_cache:
        surface_to_use = lambda surface: surface_cache.find_surface(surface, None)
    else:
        surface_to_use = lambda surface: (None, surface)
    surfaces_to_use = [surface_to_use(surface) for surface in geometry.surfaces]

    # OpenMC emits an IDWarning when comparing with cached Surface ids, such as:
    # ...openmc/mixin.py:70: IDWarning: Another Surface instance already exists with id=311
    # We want to override the ids as to not create duplicates of surfaces when converting
    # RAMP cores to OpenMC models more than once. Hence, we ignore the emitted warning.
    with ignore_warnings(IDWarning):
        return construct_region(format_creation(geometry),
                                [(index, openmc_halfspace(surface, abs(index) if index else index)) for
                                 index, surface in
                                 surfaces_to_use])


def openmc_component(component: Component, name: str = None,
                     surface_cache: SurfaceCache = None, library_path=None) -> openmc.Cell:
    """
    This function creates an openmc cell from a RAMP component and returns it
    Parameters
    ----------
    component - the componenet
    name - the name of the component, which is given to the cell

    Returns
    -------
    The cell
    """
    region = openmc_region(component.geometry, surface_cache)
    add_burning_isotopes(component.mixture)
    cell = openmc.Cell(region=region, fill=openmc_material(component.mixture, library_path=library_path),
                       name=name)
    return cell


def format_creation(geometry: Geometry | HoledGeometry | UnionGeometry):
    if isinstance(geometry, HoledGeometry):
        return ("(" + format_creation(geometry.inclusive) + "~" + "(" + "|".join(
            f"{format_creation(hole)}" for hole in geometry.exclusives))[:-1] + ") ) "
    elif isinstance(geometry, UnionGeometry):
        return ("(" + "|".join(
            f"{format_creation(geo)}" for geo in geometry.geometries))[:-1] + ") "
    if len(geometry.surfaces) > 0:
        return ("(" + "%d " * len(geometry.surfaces))[:-1] + ") "
    return ""


def test_box_adaptation():
    mixture = Mixture({U235: 1}, 293)
    geometry = Box(np.zeros(3), 2 * np.ones(3))
    comp = ConcreteComponent(mixture, geometry)
    assert str(
        openmc_component(comp).region) == "(1 2 3 X X 6)"
