"""
This file handels only the tallies needed for a burnup calculation using
a Batman
"""

import time
from itertools import chain, pairwise, product
from pathlib import PurePath
from typing import Iterable, Sequence, TypeVar

import numpy as np
import openmc
from corecompute.query import KQuery, Query, ReactionScore, Score, SurfaceTracksQuery, TabulatedScore, VolumeQuery
from corecompute.result import KResult, SurfaceTracksResult, VolumeResult
from coremaker.protocols.component import Component
from multipledispatch import dispatch
from reactions import Electron, Neutron, Photon, Positron, Proton, Typus
from reactions.particle import NamedParticle, Particle
from reactions.reaction import eV
from reactions.typus import ProdTypus
from uncertainties import ufloat

from openmcadapter.mixture_adapter import openmc_name

T = TypeVar("T")

scores_dict = {
    Typus.NFission: "fission",
    Typus.NGamma: "(n,gamma)",
    Typus.NElastic: "elastic",
    Typus.NInelastic: "(n,nc)",
    Typus.NAnything: "total",
    Typus.N2ND: "(n,2nd)",
    Typus.N2N: "(n,2n)",
    Typus.N3N: "(n,3n)",
    Typus.NNAlpha: "(n,na)",
    Typus.NN3Alpha: "(n,n3a)",
    Typus.N3NAlpha: "(n,3na)",
    Typus.N2NAlpha: "(n,2na)",
    Typus.NNP: "(n,np)",
    Typus.NN2Alpha: "(n,n2a)",
    Typus.N2N2Alpha: "(2n2a)",
    Typus.NND: "(n,nd)",
    Typus.NNT: "(n,nt)",
    Typus.NNHe3: "(n,n3He)",
    Typus.NND2Alpha: "(n,nd2a)",
    Typus.NNT2Alpha: "(n,nt2a)",
    Typus.N4N: "(n,4n)",
    Typus.N2NP: "(n,2np)",
    Typus.N3NP: "(n,3np)",
    Typus.NN2P: "(n,n2p)",
    Typus.NP: "(n,p)",
    Typus.ND: "(n,d)",
    Typus.NT: "(n,t)",
    Typus.NHe3: "(n,He3)",
    Typus.NAlpha: "(n,a)",
    Typus.N2Alpha: "(n,2a)",
    Typus.N3Alpha: "(n,3a)",
    Typus.N2P: "(n,2p)",
    Typus.NPAlpha: "(n,pa)",
    Typus.NT2Alpha: "(n,t2a)",
    Typus.ND2Alpha: "(n,d2a)",
    Typus.NPD: "(n,pd)",
    Typus.NPT: "(n,pt)",
    Typus.NDAlpha: "(n,da)",
    Typus.N2NPAlpha: "(n,2npa)",
    ProdTypus.NTtot: "(n,Xt)",
    ProdTypus.NPtot: "(n,Xp)",
    ProdTypus.NDtot: "(n,Xd)",
    ProdTypus.NHe3tot: "(n,XHe3)",
    ProdTypus.NAlphatot: "(n,XHe4)",
}


def tabulated_filter(scores: Sequence[Score]) -> Sequence[openmc.Filter]:
    """
    Checks if a sequence of scores contains a unique TabulatedScore, if it does a corresponding
    Energy Function filter is returned, if it doesn't an empty list is returned.

    Parameters
    ----------
    scores: Sequence[Score]
     A sequence of scores

    Returns
    -------
    Sequence[openmc.Filter]
     A list containing an Energy Function filter or an empty list.
    """
    for score in scores:
        if isinstance(score, TabulatedScore):
            if len(scores) > 1:
                raise NotImplementedError("Query with a Tabulated score must have a unique score")
            return [openmc.EnergyFunctionFilter(score.energy, score.score)]
    return []


def openmc_score(score: Score) -> str:
    match score:
        case Score():
            return score.name
        case ReactionScore():
            score: ReactionScore
            # TODO: consider mapping through mt number
            return scores_dict[score.reaction.typus]
        case TabulatedScore():
            return "flux"


openmc_particles = {Neutron: "neutron", Photon: "photon", Electron: "electron", Proton: "proton", Positron: "positron"}


def openmc_particle(particle: Particle):
    try:
        return openmc_particles[particle]
    except KeyError:
        raise ValueError(f"{particle} calculations are not supported by the openmc adapter.")


def openmc_energies(energies: Sequence[eV]) -> Sequence[eV]:
    if len(energies) == 0:
        # noinspection PyTypeChecker
        return [0.0, np.inf]
    else:
        return energies


def openmc_query_tally(query: VolumeQuery, cells_ids: dict[PurePath, int]) -> openmc.Tally:
    """
    function that creates an openmc tally from a RAMP query and returns it.
    The supported scores are the ones appearing in the score_dict above.
    Parameters
    ----------
    query - a volume query
    cells_ids - a dict which contains the ids of cell according to their name.

    Returns
    -------
    Tally that represents the query
    """
    tally = openmc.Tally()
    tally.filters = [
        openmc.CellFilter(bins=[cells_ids[name] for name in query.names]),
        openmc.ParticleFilter(bins=[openmc_particle(query.particle)]),
    ]
    tally.scores = list(set(map(openmc_score, query.scores)))
    tally.nuclides = list(
        set(openmc_name(score.reaction.parent) for score in query.scores if isinstance(score, ReactionScore))
    )
    if len(query.energies) > 0:
        tally.filters += [openmc.EnergyFilter(values=openmc_energies(query.energies))]
    tally.filters += tabulated_filter(query.scores)
    return tally


@dispatch(object, object, KQuery, object)
def get_result_from_statepoint(tallies: dict[str, tuple[ufloat, T]], _, __, ___) -> tuple[KResult, T]:
    k, k_series = tallies["k"]
    return KResult(k.nominal_value, k.std_dev), k_series


def gather_score_from_tally(
    tally: openmc.Tally,
    score: str,
    cell_ids: tuple[int],
    particle: str,
    energy_bins: tuple[tuple[float, float]],
    nuclides: list,
    division_factors: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = [score]
    filters = [openmc.CellFilter, openmc.ParticleFilter]
    filter_bins = [cell_ids, (particle,)]
    if energy_bins != ((0.0, np.inf),):
        filters.append(openmc.EnergyFilter)
        filter_bins.append(energy_bins)
    shape = (len(cell_ids), len(energy_bins))
    mean_vals = tally.get_values(scores, filters, filter_bins, nuclides, "mean").reshape(shape)
    error_vals = tally.get_values(scores, filters, filter_bins, nuclides, "std_dev").reshape(shape)
    if division_factors is not None:
        broadcasting_shape = (shape[0], *(1 for _ in shape[1:]))
        division_factors = division_factors.reshape(broadcasting_shape)
        mean_vals = mean_vals / division_factors
        error_vals = error_vals / division_factors
    return mean_vals, error_vals


def result_dict(
    score: Score, path: PurePath, energy_bin: tuple[eV, eV], particle: NamedParticle, mean: float, error: float
) -> dict:
    data = {"component": path, "score": score, "value": mean, "error": error, "particle": particle.name.lower()}
    if energy_bin != (0.0, np.inf):
        lower, upper = energy_bin
        data["lower_energy"] = lower
        data["upper_energy"] = upper
    return data


def get_score_results(
    tally: openmc.Tally,
    score: Score,
    query: VolumeQuery,
    cell_ids: dict[PurePath, int],
    named_components: dict[PurePath, Component],
) -> Iterable[dict]:
    cell_ids = tuple(cell_ids[path] for path in query.names)
    nuclides = [openmc_name(score.reaction.parent)] if isinstance(score, ReactionScore) else []
    energy_bins = tuple(pairwise(openmc_energies(query.energies)))
    particle = query.particle
    division_factors = np.ones(len(cell_ids))
    if score.volume_specific:
        division_factors *= np.fromiter((named_components[path].geometry.volume for path in query.names), dtype=float)
    if isinstance(score, ReactionScore):
        if score.density_specific:
            division_factors *= np.fromiter(
                (named_components[path].mixture.isotopes[score.reaction.parent] for path in query.names), dtype=float
            )
    division_factors = division_factors if (division_factors != 1).any() else None
    mean, error = gather_score_from_tally(
        tally, openmc_score(score), cell_ids, openmc_particle(particle), energy_bins, nuclides, division_factors
    )
    for (i, path), (j, energy_bin) in product(enumerate(query.names), enumerate(energy_bins)):
        yield result_dict(score, path, energy_bin, particle, mean=mean[i, j], error=error[i, j])


@dispatch(object, object, VolumeQuery, object)
def get_result_from_statepoint(  # noqa
    tallies: dict[Query, openmc.Tally],
    cells_ids: dict[PurePath, int],
    query: VolumeQuery,
    named_components: dict[PurePath, Component],
) -> list[VolumeResult]:
    tally = tallies[query]
    return list(chain(*(get_score_results(tally, score, query, cells_ids, named_components) for score in query.scores)))


@dispatch(object, object, SurfaceTracksQuery, object)
def get_result_from_statepoint(_, __, query: SurfaceTracksQuery, ___) -> SurfaceTracksResult:  # noqa
    return SurfaceTracksResult(query.path, time.localtime(), SurfaceTracksResult.compute_filehash(query.path))
