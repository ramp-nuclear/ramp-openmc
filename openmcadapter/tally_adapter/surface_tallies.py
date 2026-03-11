from itertools import pairwise
from pathlib import PurePath

import numpy as np
import openmc
from corecompute.query import Query, SurfaceCurrentQuery
from corecompute.result import KResult, MeshResult, VolumeResult
from coremaker.protocols.component import Component
from coremaker.surfaces.surfacecache import SurfaceCache
from multipledispatch import dispatch

from openmcadapter.tally_adapter.burnup_tallies import openmc_energies, openmc_particle

Result = KResult | MeshResult | VolumeResult


def surface_current_tally(
    query: SurfaceCurrentQuery,
    cells_ids: dict[PurePath, int],
    named_components: dict[PurePath, Component],
    geometry: openmc.Geometry,
    surface_cache: SurfaceCache,
) -> openmc.Tally:
    """
    function that creates an openmc tally from a RAMP SurfaceCurrentQuery and returns it.

    Parameters
    ----------
    query: SurfaceCurrentQuery
        Query to tally
    cells_ids: dict[PurePath, int]
        A dict which contains the ids of cell according to their name.
    named_components: dict[PurePath, Component]
        A dictionary from paths to components. The result of iterating through a core object's components.
    surface_cache: SurfaceCache
        Cache for surfaces we've already seen

    Returns
    -------
    tally that answers the query
    """
    tally = openmc.Tally()
    tally.filters = [openmc.ParticleFilter(bins=[openmc_particle(query.particle)])]
    if query.from_component:
        tally.filters.append(openmc.CellFromFilter(cells_ids[query.from_component]))
        component_contains_surface = named_components[query.from_component]
    if query.to_component:
        tally.filters.append(openmc.CellFilter(cells_ids[query.to_component]))
        component_contains_surface = named_components[query.to_component]
    if not query.to_component and not query.from_component:
        raise NotImplementedError(
            "Computation of partial currents without specifying components containing the "
            "surface is not implemented yet"
        )
    for index, comp_surface in enumerate(component_contains_surface.geometry.surfaces):
        if query.surface.isclose(comp_surface) or query.surface.isclose(-comp_surface):
            surface_index = abs(surface_cache.find_surface(comp_surface, None)[0])
    try:
        surface_filter = openmc.SurfaceFilter(surface_index)
    except UnboundLocalError:
        raise ValueError(f"The surface {query.surface} isn't defined in the geometry.")
    tally.filters.append(surface_filter)
    tally.scores = ["current"]
    if len(query.energies) > 0:
        tally.filters += [openmc.EnergyFilter(values=openmc_energies(query.energies))]
    return tally


@dispatch(object, object, SurfaceCurrentQuery, object)
def get_result_from_statepoint(tallies: dict[Query, openmc.Tally], _, query: SurfaceCurrentQuery, __) -> list[Result]:
    answers = []

    def _gather_value(value, energy_bin):
        filters = [openmc.ParticleFilter]
        filter_bins = [(openmc_particle(query.particle),)]
        if energy_bin != (0, np.inf):
            filters += [openmc.EnergyFilter]
            filter_bins += [(energy_bin,)]
        return abs(tally.get_values(value=value, filters=filters, filter_bins=filter_bins))

    tally = tallies[query]
    for energy_bin in pairwise(openmc_energies(query.energies)):
        mean, std = np.array([_gather_value(value, energy_bin) for value in ["mean", "std_dev"]]).flatten()
        data = {
            "surface": str(query.surface),
            "score": "current",
            "value": mean,
            "error": std,
            "particle": query.particle.name.lower(),
        }
        if energy_bin != (0, np.inf):
            lower, upper = energy_bin
            data["lower_energy"] = lower
            data["upper_energy"] = upper
        answers.append(data)
    return answers
