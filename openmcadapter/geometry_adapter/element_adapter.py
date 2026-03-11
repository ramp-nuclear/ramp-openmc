from pathlib import PurePath

import openmc
from coremaker.protocols.element import Element
from coremaker.surfaces.surfacecache import SurfaceCache

from openmcadapter.geometry_adapter.component_adapter import openmc_component


def openmc_universe_from_element(
    element: Element,
    element_name: str,
    surface_cache: SurfaceCache = None,
    library_path=None,
) -> tuple[openmc.Universe, dict[PurePath, int]]:
    """
    This function creates an openmc universe from a RAMP Tree object
    """
    cells = [
        openmc_component(comp, str(PurePath(element_name) / path), surface_cache, library_path=library_path)
        for path, comp in element.named_components()
    ]
    cells_id = {PurePath(element_name) / path: cell.id for (path, _), cell in zip(element.named_components(), cells)}
    universe = openmc.Universe(cells=cells, name=element_name)
    return universe, cells_id
