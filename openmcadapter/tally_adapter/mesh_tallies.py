import numpy as np
import openmc
import pandas as pd
import xarray as xr
from corecompute.query import MeshQuery, Query
from coremaker.mesh import CartesianMesh, CylindricalMesh, SphericalMesh
from coremaker.transform import identity
from multipledispatch import dispatch

from openmcadapter.tally_adapter.burnup_tallies import openmc_energies, openmc_particle, openmc_score, tabulated_filter

cm = float


def _single_volume_specific(query: MeshQuery) -> bool:
    vss = {s.volume_specific for s in query.scores}
    return len(vss) == 1


def meshtally_from_meshquery(query: MeshQuery) -> openmc.Tally:
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
        raise NotImplementedError(
            "Support for translated or rotated mesh queries is not supports, "
            "as it is also not trivially supported in OpenMC"
        )
    if not _single_volume_specific(query):
        raise NotImplementedError(
            'A single query with scores with varying "volume specific" definition are not supported'
        )
    tally = openmc.Tally()
    mesh = openmc_mesh(query.mesh)
    tally.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter(bins=[openmc_particle(query.particle)])]
    tally.scores = list(map(openmc_score, query.scores))
    if len(query.energies) > 0:
        tally.filters += [openmc.EnergyFilter(values=openmc_energies(query.energies))]
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
    if (
        np.allclose(np.diff(np.diff(mesh.x)), 0)
        and np.allclose(np.diff(np.diff(mesh.y)), 0)
        and np.allclose(np.diff(np.diff(mesh.z)), 0)
    ):
        omesh = openmc.RegularMesh()
        omesh.lower_left = (mesh.x[0], mesh.y[0], mesh.z[0])
        omesh.upper_right = (mesh.x[-1], mesh.y[-1], mesh.z[-1])
        omesh.dimension = (len(mesh.x) - 1, len(mesh.y) - 1, len(mesh.z) - 1)
    else:
        omesh = openmc.RectilinearMesh()
        omesh.x_grid = mesh.x
        omesh.y_grid = mesh.y
        omesh.z_grid = mesh.z
    return omesh


@dispatch(SphericalMesh)
def openmc_mesh(mesh: SphericalMesh) -> openmc.SphericalMesh:  # noqa
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
    omesh = openmc.SphericalMesh()
    omesh.r_grid = mesh.r
    omesh.theta_grid = mesh.theta
    omesh.phi_grid = mesh.phi
    return omesh


@dispatch(CylindricalMesh)
def openmc_mesh(mesh: CylindricalMesh) -> openmc.CylindricalMesh:  # noqa
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
    omesh = openmc.CylindricalMesh()
    omesh.r_grid = mesh.r
    omesh.phi_grid = mesh.theta
    omesh.z_grid = mesh.z
    return omesh


def _interpret_cartesian_mesh(
    tally_df: pd.DataFrame,
    query: MeshQuery,
    mesh_id: int,
    volume_specific: bool,
) -> xr.Dataset:
    # TODO: Improve this shitty function, it is disgusting
    cartesian_directions = ["x", "y", "z"]
    other_parameters = ["energy low [eV]", "energy high [eV]", "particle", "energyfunction", "nuclide", "score"]
    tally_df.columns = tally_df.columns.to_flat_index()
    names = {
        **{(f"mesh {mesh_id}", direction): direction for direction in ["x", "y", "z"]},
        **{(name, ""): name for name in other_parameters + ["mean"]},
        ("std. dev.", ""): "std",
    }
    # Renaming the stupid OpenMC names
    tally_df.rename(names, axis="columns", inplace=True)
    # Dropping the redundant data that is inferable from the query
    data_columns = ["mean", "std"]
    redundant_columns = tally_df.nunique() == 1 & (
        tally_df.columns.to_series().apply(lambda x: x not in [*data_columns, *cartesian_directions])
    )
    tally_df.drop(tally_df.columns[redundant_columns], axis="columns", inplace=True)
    # Translating mesh indices to physical coordinates using the data supplied in the query
    mesh: CartesianMesh = query.mesh
    if volume_specific:
        tally_df = tally_df.assign(volume=1.0)
    for direction in filter(lambda x: x in tally_df.columns, cartesian_directions):
        limits = getattr(mesh, direction)
        centers = dict(enumerate((limits[:-1] + limits[1:]) / 2, start=1))
        # TODO: Check that the translation between OpenMC cell num and this coordinates is accurate
        coords = tally_df[direction].apply(lambda i: centers[i])
        if volume_specific:
            size = dict(enumerate((limits[1:] - limits[:-1]), start=1))
            sizes = tally_df[direction].apply(lambda i: size[i])
            tally_df["volume"] *= sizes
        tally_df = tally_df.assign(**{direction: coords})
    if volume_specific:
        tally_df[["mean", "std"]] = tally_df[["mean", "std"]].div(tally_df["volume"], axis="index")
        tally_df.drop("volume", axis="columns", inplace=True)
    # Dropping energy low [eV] data
    tally_df.drop("energy low [eV]", axis="columns", errors="ignore", inplace=True)
    tally_df.rename({"energy high [eV]": "e"}, axis="columns", inplace=True)
    # Setting the relevant phase space parameters as labels
    index_labels = list(filter(lambda x: x not in ["mean", "std"], tally_df.columns))
    return tally_df.set_index(index_labels).to_xarray()


def tally_mesh_id(tally: openmc.Tally) -> int | None:
    """Gets the ID of the mesh associated with the tally, if it has a corresponding filter.

    Parameters
    ----------
    tally: openmc.Tally
        The tally for which there may be a MeshFilter to get its mesh ID.

    Returns
    -------
    int or None
        The mesh id, if found. None otherwise.

    """
    # Assuming there is only one mesh in the tally
    for filt in tally.filters:
        if isinstance(filt, openmc.MeshFilter):
            return filt.mesh.id


@dispatch(object, object, MeshQuery, object)
def get_result_from_statepoint(tallies: dict[Query, openmc.Tally], _, query: MeshQuery, __) -> tuple[xr.Dataset]:
    tally = tallies[query]
    volume_specific = all(score.volume_specific is True for score in query.scores)
    df = tally.get_pandas_dataframe()
    match query.mesh:
        case CartesianMesh():
            res = _interpret_cartesian_mesh(df, query, tally_mesh_id(tally), volume_specific)
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
        case _:
            raise NotImplementedError(f"Other case {type(query.mesh)} not supported yet for meshquery fulfillment")
    return (res,)
