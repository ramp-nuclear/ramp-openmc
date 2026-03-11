"""An openmc based oracle"""

import logging
import os
import pickle
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path, PurePath
from typing import Any, Callable, Literal, Optional, Sequence

import openmc
from corecompute.oracle import OracleResult
from corecompute.query import (
    HeatingRateQuery,
    KQuery,
    MeshQuery,
    Query,
    SurfaceCurrentQuery,
    SurfaceTracksQuery,
    VolumeQuery,
)
from coremaker.core import Core
from coremaker.protocols.component import Component
from coreoperator.operational_state import OperationalState
from distributed import get_worker
from ramp_core import TemporaryDirectory

from openmcadapter.geometry_adapter.core_adapter import openmc_core_to_model
from openmcadapter.source import Source, openmc_source
from openmcadapter.tally_adapter.burnup_tallies import get_result_from_statepoint, openmc_query_tally
from openmcadapter.tally_adapter.mesh_tallies import meshtally_from_meshquery
from openmcadapter.tally_adapter.power_computation_tallies import power_tally
from openmcadapter.tally_adapter.surface_tallies import surface_current_tally

logger = logging.getLogger(__name__)


def _workspace(path: Optional[Path]) -> tuple[Optional[Path], bool]:
    return (path, False) if path is not None else (None, True)


def _verify_extra_settings(extra_settings: dict[str, Any]):
    coinciding_attributes = [key for key in extra_settings if hasattr(Settings, key)]
    if len(coinciding_attributes) > 0:
        raise ValueError(f"The attributes: {', '.join(coinciding_attributes)} have repeated definitions.")


@dataclass(frozen=True)
class Settings:
    histories: int
    cycles: int
    passive_cycles: int = 0
    temperature_method: Literal["nearest", "interpolation"] = "interpolation"
    photon_transport: bool = False
    surface_source_path: Path = None
    # Should be given as absolute path.
    library_xml_path: Path = None
    # Used to set attributes of the `openmc.settings.Settings` object
    extra_settings: dict[str, Any] = field(default_factory=dict)
    create_tallies_txt_file: bool = True
    source: Source = None
    run_mode: Literal["eigenvalue", "fixed source"] = "eigenvalue"

    def __post_init__(self) -> None:
        if self.passive_cycles > self.cycles:
            passive_cycles, cycles = self.passive_cycles, self.cycles
            raise ValueError(f"number of {passive_cycles=} is larger than number of total {cycles=}")
        _verify_extra_settings(self.extra_settings)


class OpenMCOracle:
    """A transport Oracle that uses openmc for the underlying calculation.

    Parameters
    ----------
    settings - settings to use for kcode and the like.
    save_workspace - Path to save the workspace in.
    boundary_conditions - the default boundary conditions to be used around the core
    tasks - The number of thread tasks to run in OpenMC.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        save_workspace: Optional[Path] = None,
        tasks: int = 1,
        boundary_condition="vacuum",
    ):
        self.default_boundary_condition = boundary_condition
        self.settings = settings or Settings(10000, 150, run_mode="eigenvalue", passive_cycles=30)
        self.tasks = tasks
        d, cleanup = _workspace(save_workspace)
        self.direct = partial(
            run_openmc,
            settings=self.settings,
            directory=d,
            boundary_condition=boundary_condition,
            cleanup=cleanup,
        )

    @property
    def _workspace(self) -> Optional[Path]:
        return self.direct.keywords["directory"]

    def __call__(
        self,
        state: OperationalState,
        *queries: Query,
        tasks: Optional[int] = None,
        save_workspace: Optional[Path] = None,
        temp_workspace: Optional[Path] = None,
        boundary_condition: Optional[str] = None,
        extra_tallies: Optional[Callable] = None,
        prefix: str = "",
        **kwargs,
    ) -> OracleResult:
        """
        All kwargs are passed as attribute definitions for the settings. This does not change the default settings
        for future calls.
        """
        tasks = tasks or self.tasks
        boundary_condition = boundary_condition or self.default_boundary_condition
        extra = {"directory": temp_workspace, "cleanup": True} if temp_workspace is not None else {}
        extra = {"directory": save_workspace, "cleanup": True} if save_workspace is not None else extra

        return self.direct(
            state,
            *queries,
            tasks=tasks,
            prefix=prefix,
            boundary_condition=boundary_condition,
            extra_tallies=extra_tallies,
            settings=replace(self.settings, **kwargs),
            **extra,
        )


def _model(
    core: Core,
    queries: Sequence[Query],
    settings: Settings,
    boundary_condition: str,
    extra_tallies: Optional[Callable[[openmc.model.Model, dict], tuple[openmc.model.Model, dict]]] = None,
) -> tuple[openmc.model.Model, dict, dict, dict, list, None | Path]:
    """
    This function's purpose is to combine all methods of interpreting RAMP objects as OpenMC objects.
    """
    model, cells_ids, surface_cache = openmc_core_to_model(
        core, boundary_condition, temperature_method=settings.temperature_method, library_path=settings.library_xml_path
    )
    if extra_tallies:
        model, tallies = extra_tallies(model, cells_ids)
        extra_keys = list(tallies.keys())
    else:
        tallies = {}
        extra_keys = None
    model.settings.batches = settings.cycles
    model.settings.inactive = settings.passive_cycles
    model.settings.particles = settings.histories
    model.settings.photon_transport = settings.photon_transport
    model.settings.run_mode = settings.run_mode
    for attribute, value in settings.extra_settings.items():
        setattr(model.settings, attribute, value)
    if settings.source is not None:
        model.settings.source = openmc_source(settings.source, core)
    if settings.library_xml_path:
        model.materials.cross_sections = settings.library_xml_path
    if settings.surface_source_path:
        model.settings.surf_source_read = dict(path=str(settings.surface_source_path))
    if not settings.create_tallies_txt_file:
        model.settings.output = dict(tallies=False)
    q_tallies = {}
    surface_file_location = None
    for query in queries:
        if isinstance(query, KQuery):
            continue
        elif isinstance(query, VolumeQuery):
            t = openmc_query_tally(query, cells_ids)
            tallies[query] = t.id
            q_tallies[query] = t.id
            model.tallies.append(t)
        elif isinstance(query, MeshQuery):
            t = meshtally_from_meshquery(query)
            tallies[query] = t.id
            q_tallies[query] = t.id
            model.tallies.append(t)
        elif isinstance(query, SurfaceTracksQuery):
            surface_ids = []
            for _, comp in core.named_components:
                for index, comp_surface in enumerate(comp.geometry.surfaces):
                    for query_surface in query.surfaces:
                        if query_surface.isclose(comp_surface) or query_surface.isclose(-comp_surface):
                            surface_index = abs(surface_cache.find_surface(comp_surface, None)[0])
                            surface_ids.append(surface_index)
            model.settings.surf_source_write = dict(
                surface_ids=surface_ids, max_particles=query.maximal_number_of_particles or 1000
            )
            surface_file_location = query.path
        elif isinstance(query, SurfaceCurrentQuery):
            t = surface_current_tally(query, cells_ids, dict(core.named_components), surface_cache)
            tallies[query] = t.id
            q_tallies[query] = t.id
            model.tallies.append(t)
        elif isinstance(query, HeatingRateQuery):
            t = power_tally(query)
            tallies[query] = t.id
            q_tallies[query] = t.id
            model.tallies.append(t)
    return model, tallies, q_tallies, cells_ids, extra_keys, surface_file_location


def _find_statepoint(workspace: Path) -> openmc.StatePoint:
    # Dealing with the case of multiple statepoints
    statepoint_paths = list(filter(lambda x: "statepoint" in x.name, workspace.iterdir()))
    max_statepoint_path = max(statepoint_paths, key=lambda x: int(x.name.split(".")[1]))
    return openmc.StatePoint(max_statepoint_path)


def _load_tallies(
    workspace: Path,
    sp: Optional[openmc.StatePoint] = None,
    named_components: Optional[dict[PurePath, Component]] = None,
):
    if not sp:
        sp = _find_statepoint(workspace)
    with open(workspace / "tallies_queries.pkl", "rb") as file:
        queries, query_tallies, q_tallies, cells_ids, surface_file_location = pickle.load(file)
    tallies_by_query = {query: sp.tallies[tally_id] for query, tally_id in q_tallies.items()}
    tallies_by_query["k"] = sp.keff, sp.k_generation
    results = {}
    if surface_file_location:
        default_surface_path = workspace / "surface_source.h5"
        os.system(f"cp {default_surface_path} {surface_file_location}")
    if not named_components:
        with (workspace / "state.pkl").open("rb") as file:
            named_components = dict(pickle.load(file).core.named_components)
    for query in queries:
        results[query] = get_result_from_statepoint(tallies_by_query, cells_ids, query, named_components)
    return results


def run_openmc(
    state: OperationalState,
    *queries: Query,
    settings: Settings,
    tasks: Optional[int] = None,
    directory: Optional[Path] = None,
    prefix: str = "",
    boundary_condition: str = "vacuum",
    cleanup=False,
    extra_tallies: Optional[Callable[[openmc.model.Model, dict], tuple[openmc.model.Model, dict]]] = None,
) -> OracleResult:
    """
    This function performs an openmc calculation, saves the results and returns
    results to the queries.
    Parameters
    ----------
    state - the state on which the calculation is preformed
    queries - the queries to which the oracle has to give a result
    settings - the settings used for the run (particles,active/inactive batches)
    tasks - the number of cpu's to use in the run
    directory - the directory where to save the results
    prefix - Prefix string to use at the beginning of the unique directory.
    cleanup - Flag for whether to clean up the working directory.
    boundary_condition - the boundary conditions around the core. could be one
                        of "reflective","vacuum" and "periodic"
    extra_tallies - function that gets a model, adds tallies to it and returns it
                    together with a dictionary whose values are those tallies ids.
                    This is used for preforming computation which are not answering
                    querying, for example tallying reaction rates for a homogenization
                    process.

    """
    try:
        worker = get_worker()
        default_local_directory = Path(worker.local_directory)
    except ValueError:
        default_local_directory = Path()
    else:
        if not tasks:
            tasks = settings.threads
    if directory:
        directory.mkdir(exist_ok=True)
    else:
        directory = default_local_directory
    core = state.core
    named_components = dict(core.named_components)
    model, tallies, q_tallies, cells_ids, extra_keys, surface_file_location = _model(
        core, queries, settings, boundary_condition, extra_tallies
    )
    with TemporaryDirectory(prefix=prefix, suffix="_openmc", dir=directory, clean_dir=cleanup) as tmpdir:
        workspace: Path = Path(tmpdir)
        model.export_to_xml(directory=str(workspace))
        with open(workspace / "tallies_queries.pkl", "wb") as file:
            pickle.dump((queries, tallies, q_tallies, cells_ids, surface_file_location), file)
        with open(workspace / "state.pkl", "wb") as file:
            # The copying of the state is due to the addition of burnup
            # isotopes which change the core object.
            pickle.dump(state.copy(core=core), file)
        with (workspace / "out").open("w", buffering=1) as f:
            with redirect_stdout(f):
                model.run(threads=tasks, cwd=workspace, export_model_xml=False)

        sp = openmc.StatePoint(workspace / f"statepoint.{settings.cycles}.h5")
        results = _load_tallies(workspace, sp=sp, named_components=named_components)
        if extra_keys:
            for tally_name in extra_keys:
                tally_id = tallies[tally_name]
                if isinstance(tally_id, int):
                    openmc.Tally()
                    results[tally_name] = sp.tallies[tally_id].get_pandas_dataframe()
    return results
