from dataclasses import dataclass
from pathlib import PurePath

import numpy as np
import openmc.stats
from coremaker.core import Core
from coremaker.geometries import Box
from coremaker.protocols.geometry import Geometry
from coremaker.units import cm
from multipledispatch import dispatch
from openmc.source import Source as OpenMCSource
from reactions.particle import Particle, Neutron

from openmcadapter.tally_adapter.burnup_tallies import openmc_particle


@dispatch(Geometry)
def uniform_distribution(geometry: Geometry) -> openmc.stats.Spatial:
    raise NotImplemented(f"A uniform distribution over a geometry of type {type(geometry)} is not implemented")


@dispatch(Box)
def uniform_distribution(box: Box) -> openmc.stats.Spatial:
    """
    Generate a uniform distribution in the region of a box.
    For the case the box is rotated, sample uniformly over the smallest bounding box that is not rotated, instead.
    """
    _box = box.bounding_box()
    c, d = map(np.asarray, (_box.center, _box.dimensions))
    ll = c - d / 2
    ur = c + d / 2
    return openmc.stats.Box(ll, ur)


@dataclass(eq=True, unsafe_hash=True)
class PointSource:
    """A points source that spews particles isotropically at some energy distribution.

    Parameters
    ----------
    location: Distribution[(cm, cm, cm)]
        The location of the source
    energy: Distribution[eV]
        The emission energy, in eV.
    particle: Particle
        source particle
    """
    location: tuple[cm, cm, cm]
    energy: openmc.stats.Univariate
    particle: Particle = Neutron


@dataclass(eq=True, unsafe_hash=True)
class PointsSource:
    locations: dict[tuple[cm, cm, cm], float]
    energy: openmc.stats.Univariate
    particle: Particle = Neutron


@dataclass(eq=True, unsafe_hash=True)
class ComponentSource:
    """A points source that spews particles isotropically at some energy distribution.

        Parameters
        ----------
        component: dict[PurePath, float] | PurePath
            The component to sample uniformly in.
        energy: Distribution[eV]
            The emission energy, in eV.
        particle: Particle
            source particle
        """
    component: PurePath
    energy: openmc.stats.Univariate
    particle: Particle = Neutron


@dataclass(eq=True, unsafe_hash=True)
class ComponentDependentSource:
    """A points source that spews particles isotropically at some energy distribution.

        Parameters
        ----------
        components: dict[PurePath, float] | PurePath
            Mapping between the components to sample uniformly in and their corresponding relative weights
        energy: Distribution[eV]
            The emission energy, in eV.
        particle: Particle
            source particle
        """
    components: dict[PurePath, float]
    energy: openmc.stats.Univariate
    particle: Particle = Neutron


@dataclass(eq=True, unsafe_hash=True)
class UniformSource:
    """A points source that spews particles isotropically at some energy distribution.

        Parameters
        ----------
        mesh: dict[Geometry, float]
            Mapping between the components to sample uniformly in and their corresponding relative weights
        energy: Distribution[eV]
            The emission energy, in eV.
        particle: Particle
            source particle
        """
    geometry: Geometry
    energy: openmc.stats.Univariate
    particle: Particle = Neutron


@dataclass(eq=True, unsafe_hash=True)
class MeshSource:
    """A points source that spews particles isotropically at some energy distribution.

        Parameters
        ----------
        mesh: dict[Geometry, float]
            Mapping between the components to sample uniformly in and their corresponding relative weights
        energy: Distribution[eV]
            The emission energy, in eV.
        particle: Particle
            source particle
        """
    mesh: dict[Geometry, float]
    energy: openmc.stats.Univariate
    particle: Particle = Neutron


Source = OpenMCSource | PointSource | PointsSource | ComponentSource | ComponentDependentSource | UniformSource \
         | MeshSource


@dispatch(OpenMCSource, Core)
def openmc_source(source: OpenMCSource, _) -> OpenMCSource:
    return source


@dispatch(list, Core)
def openmc_source(sources: list[Source], _) -> list[OpenMCSource]:
    return list(map(openmc_source, sources))


@dispatch(PointSource, Core)
def openmc_source(source: PointSource, _) -> OpenMCSource:
    s = openmc.Source()
    s.space = openmc.stats.Point(xyz=source.location)
    s.energy = source.energy
    s.particle = openmc_particle(source.particle)
    return s


@dispatch(PointsSource, Core)
def openmc_source(source: PointsSource, _) -> list[OpenMCSource]:
    def _source(location: tuple[cm, cm, cm]) -> Source:
        s = openmc_source(PointSource(location, source.energy, source.particle), _)
        s.strength = source.locations[location]
        return s
    return list(map(_source, source.locations.keys()))


@dispatch(UniformSource, Core)
def openmc_source(source: UniformSource, _):
    s = openmc.Source()
    s.space = uniform_distribution(source.geometry)
    s.energy = source.energy
    s.particle = openmc_particle(source.particle)
    return s


@dispatch(MeshSource, Core)
def openmc_source(source: MeshSource, _):
    def _source(geo: Geometry) -> Source:
        s = openmc_source(UniformSource(geo, source.energy, source.particle), _)
        s.strength = source.mesh[geo]
        return s
    return list(map(_source, source.mesh.keys()))


@dispatch(ComponentSource, Core)
def openmc_source(source: ComponentSource, core: Core) -> OpenMCSource:
    s = openmc.Source()
    s.energy = source.energy
    s.particle = openmc_particle(source.particle)
    s.space = uniform_distribution(core.geometry_of(source.component))
    return s


@dispatch(ComponentDependentSource, Core)
def openmc_source(source: ComponentDependentSource, core: Core) -> list[OpenMCSource]:

    def _source(comp: PurePath) -> Source:
        s = openmc_source(ComponentSource(comp, source.energy, source.particle), core)
        s.strength = source.components[comp]
        return s
    return list(map(_source, source.components.keys()))

