from itertools import pairwise
from typing import Dict

import numpy as np
import openmc
import pandas as pd
import xarray as xr
from coremaker.mesh import CartesianMesh, SphericalMesh, CylindricalMesh
from coremaker.transform import identity
from multipledispatch import dispatch
from openmcadapter.tally_adapter.burnup_tallies import openmc_score, openmc_energies, openmc_particle, \
    tabulated_filter
from ramp.transport import MeshQuery, Query

cm = float


def xyz_mesh_tally(model: openmc.model.Model, cells: dict, lower_left: tuple[cm, cm, cm],
                   pitch: tuple[cm, cm, cm], cells_amount: tuple[int, int, int], energy_mesh: np.ndarray,
                   scores=('flux')) -> tuple[openmc.model.Model, dict]:
    """
    Adds a mesh tally to the model
    """
    mesh = openmc.RegularMesh()
    mesh.lower_left = lower_left
    mesh.width = pitch
    mesh.dimension = cells_amount
    tally = openmc.Tally()
    tally.scores = scores
    tally.filters = [openmc.EnergyFilter(energy_mesh), openmc.MeshFilter(mesh)]
    model.tallies.append(tally)
    return model, dict(mesh_tally=tally.id)


def _single_volume_specific(query: MeshQuery) -> bool:
    vss = {s.volume_specific for s in query.scores}
    return len(vss) == 1


def meshtally_from_meshquery(query: MeshQuery) \
        -> openmc.Tally:
    """
    function that creates an openmc tally from a RAMP  mesh query and returns it.
    Parameters
    ----------
    query - a mesh query

    Returns
    -------
    Tally that represents the query
    """
    if query.transform != identity:
        raise NotImplementedError("There is no support for mesh queries with non trivial support in"
                             "the openmc adapter")
    if not _single_volume_specific(query):
        raise NotImplementedError("A single query with scores with varying \"volume specific\" definition are not"
                             "supported")
    tally = openmc.Tally()
    mesh = openmc_mesh(query.mesh)
    tally.filters = [
        openmc.MeshFilter(mesh),
        openmc.ParticleFilter(bins=[openmc_particle(query.particle)])]
    tally.scores = list(map(openmc_score, query.scores))
    if len(query.energies) > 0:
        tally.filters += \
            [openmc.EnergyFilter(values=openmc_energies(query.energies))]
    tally.filters += tabulated_filter(query.scores)
    return tally


@dispatch(CartesianMesh)
def openmc_mesh(mesh: CartesianMesh) -> openmc.RectilinearMesh | openmc.RegularMesh:
    """
    function to create openmc meshes from RAMP meshes.

    Parameters
    ----------
    mesh: CartesianMesh

    Returns
    -------
    openmc.RectilinearMesh | openmc.RegularMesh
     openmc mesh, if it can it returns a Regular mesh, otherwise it returns a cartesian mesh
    """
    if np.allclose(np.diff(np.diff(mesh.x)), 0) and np.allclose(np.diff(np.diff(mesh.y)), 0) and np.allclose(
            np.diff(np.diff(mesh.z)), 0):
        openmc_mesh = openmc.RegularMesh()
        openmc_mesh.lower_left = (mesh.x[0], mesh.y[0], mesh.z[0])
        openmc_mesh.upper_right = (mesh.x[-1], mesh.y[-1], mesh.z[-1])
        openmc_mesh.dimension = (len(mesh.x) - 1, len(mesh.y) - 1, len(mesh.z) - 1)
    else:
        openmc_mesh = openmc.RectilinearMesh()
        openmc_mesh.x_grid = mesh.x
        openmc_mesh.y_grid = mesh.y
        openmc_mesh.z_grid = mesh.z
    return openmc_mesh


@dispatch(SphericalMesh)
def openmc_mesh(mesh: SphericalMesh) -> openmc.SphericalMesh:
    """
    function to create openmc meshes from RAMP meshes.

    Parameters
    ----------
    mesh: SphericalMesh

    Returns
    -------
    openmc.SphericalMesh
     openmc spherical mesh
    """
    openmc_mesh = openmc.SphericalMesh()
    openmc_mesh.r_grid = mesh.r
    openmc_mesh.theta_grid = mesh.theta
    openmc_mesh.phi_grid = mesh.phi
    return openmc_mesh


@dispatch(CylindricalMesh)
def openmc_mesh(mesh: CylindricalMesh) -> openmc.CylindricalMesh:
    """
    function to create openmc meshes from RAMP meshes.

    Parameters
    ----------
    mesh: CylindricalMesh

    Returns
    -------
    openmc.CylindricalMesh
     openmc cylindrical mesh
    """
    openmc_mesh = openmc.CylindricalMesh()
    openmc_mesh.r_grid = mesh.r
    openmc_mesh.phi_grid = mesh.theta
    openmc_mesh.z_grid = mesh.z
    return openmc_mesh


def _interpret_cartesian_mesh(tally_df: pd.DataFrame, query: MeshQuery,
                              mesh_id: int, volume_specific: bool) -> xr.Dataset:
    #TODO: Improve this shitty function, it is disgusting
    cartesian_directions = ['x', 'y', 'z']
    other_parameters = ['energy low [eV]', 'energy high [eV]',
                        'particle', 'energyfunction', 'nuclide', 'score']
    tally_df.columns = tally_df.columns.to_flat_index()
    names = {**{(f'mesh {mesh_id}', direction): direction for direction in ['x', 'y', 'z']},
             **{(name, ''): name for name in other_parameters + ['mean']},
             ('std. dev.', ''): 'std'}
    # Renaming the stupid OpenMC names
    tally_df.rename(names, axis='columns', inplace=True)
    # Dropping the redundant data that is inferable from the query
    data_columns = ['mean', 'std']
    redundant_columns = tally_df.nunique() == 1 & \
                        (tally_df.columns.to_series().apply(lambda x: x not in [*data_columns, *cartesian_directions]))
    tally_df.drop(tally_df.columns[redundant_columns], axis='columns', inplace=True)
    # Translating mesh indices to physical coordinates using the data supplied in the query
    mesh: CartesianMesh = query.mesh
    if volume_specific:
        tally_df = tally_df.assign(volume=1.)
    for direction in filter(lambda x: x in tally_df.columns, cartesian_directions):
        limits = getattr(mesh, direction)
        centers = dict(enumerate((limits[:-1] + limits[1:]) / 2, start=1))
        #TODO: Check that the translation between OpenMC cell num and this coordinates is accurate
        coords = tally_df[direction].apply(lambda i: centers[i])
        if volume_specific:
            size = dict(enumerate((limits[1:] - limits[:-1]), start=1))
            sizes = tally_df[direction].apply(lambda i: size[i])
            tally_df['volume'] *= sizes
        tally_df = tally_df.assign(**{direction: coords})
    if volume_specific:
        tally_df[['mean', 'std']] = tally_df[['mean', 'std']].div(tally_df['volume'], axis='index')
        tally_df.drop('volume', axis='columns', inplace=True)
    # Dropping energy low [eV] data
    tally_df.drop('energy low [eV]', axis='columns', errors='ignore', inplace=True)
    tally_df.rename({'energy high [eV]': 'e'}, axis='columns', inplace=True)
    # Setting the relevant phase space parameters as labels
    index_labels = list(filter(lambda x: x not in ['mean', 'std'], tally_df.columns))
    return tally_df.set_index(index_labels).to_xarray()


def mesh_id(tally: openmc.Tally) -> int:
    # Assuming there is only one mesh in the tally
    for filt in tally.filters:
        if isinstance(filt, openmc.MeshFilter):
            return filt.mesh.id


@dispatch(object, object, MeshQuery, object)
def get_result_from_statepoint(tallies: dict[Query, openmc.Tally], cells_ids: Dict,
                               query: MeshQuery, named_components: Dict) -> tuple[xr.Dataset]:
    tally = tallies[query]
    volume_specific = all(score.volume_specific is True
                          for score in query.scores)
    df = tally.get_pandas_dataframe()
    match query.mesh:
        case CartesianMesh():
            res = _interpret_cartesian_mesh(df, query, mesh_id(tally), volume_specific)
        case SphericalMesh():
            if volume_specific:
                raise NotImplementedError("volume specific spherical mesh tally is not supported yet")
            # TODO: implement this option in a real way
            # A temporary default implementation
            res = df.to_xarray()
        case CylindricalMesh():
            if volume_specific:
                raise NotImplementedError("volume specific cylindrical mesh tally is not supported yet")
            # TODO: implement this option in a real way
            # A temporary default implementation
            res = df.to_xarray()
    return res,
