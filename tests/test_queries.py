import os
import pickle
from operator import itemgetter
from pathlib import PurePath, Path
from tempfile import TemporaryDirectory

import numpy as np
import openmc
import pytest
import reactions
from coremaker.example import lattice_limits, hafnium_block_size, \
    hafnium_block_height, heavy_water
from coremaker.mesh import CartesianMesh
from coremaker.surfaces import Plane
from coreoperator.example import example_state
from isotopes import U235, Xe135, Xe136, Xe137, O18, H, O
from macroxs.macroxs import tabulated_neutron_cross_section, tabulated_photon_cross_section
from more_itertools.more import first
from corecompute.query import VolumeQuery, KQuery, Score, ReactionScore, TabulatedScore
from corecompute.query.meshquery import MeshQuery
from corecompute.query.powerquery import HeatingRateQuery
from corecompute.query.surfacequery import SurfaceCurrentQuery
from reactions import Neutron, Photon
from toolz import valmap
from uncertainties import ufloat

from openmcadapter.openmc_oracle import OpenMCOracle, Settings
from openmcadapter.tally_adapter.burnup_tallies import openmc_particle


def test_openmc_particle():
    assert 'neutron' == openmc_particle(Neutron)


@pytest.fixture(scope='module')
def oracle():
    return OpenMCOracle(settings=Settings(100, 20, 1))


@pytest.fixture(scope='module')
def direct(oracle):
    def _direct(state, *queries, **kwargs):
        with TemporaryDirectory(prefix='openmcadapter_test') as tempdir:
            os.chdir(tempdir)
            return oracle.direct(state, *queries,
                                 boundary_condition='reflective', **kwargs)

    return _direct


def test_oracle_does_not_crash_without_queries(direct):
    queries = [KQuery()]
    direct(example_state, *queries)


def test_oracle_does_not_crash_with_save_workspace(direct):
    queries = [KQuery()]
    with TemporaryDirectory() as tmpdir:
        direct(example_state, *queries, directory=Path(tmpdir) / 'tempdir')


@pytest.fixture(scope='module')
def uranium_paths():
    return tuple(PurePath(f'{site}/coolant/aluminum_block/uranium_block')
                 for site in example_state.core.grid.keys())


@pytest.fixture(scope='module')
def scores():
    return (Score('flux', volume_specific=False),
            ReactionScore(
                reaction=reactions.ProtoReaction(U235, reactions.Typus.NFission, branching={U235: 1}),
                volume_specific=False,
                density_specific=False))


@pytest.fixture(scope='module')
def queries(uranium_paths, scores):
    return {'k': KQuery(),
            'entire_energy_flux': VolumeQuery(names=uranium_paths,
                                              scores=scores[:1]),
            'binned_energy_flux': VolumeQuery(names=uranium_paths,
                                              energies=(0., 2.0, 30., 30e6),
                                              scores=scores[:1]),
            'entire_energy_reaction': VolumeQuery(names=uranium_paths,
                                                  scores=scores[1:]),
            'binned_energy_reaction': VolumeQuery(names=uranium_paths,
                                                  energies=(0., 2.0, 30., 30e6),
                                                  scores=scores[1:])
            }


@pytest.fixture(scope='module', name='results')
def test_oracle_does_not_crash_with_queries(direct, queries):
    results = direct(example_state, *queries.values())
    return results


def test_results_are_sensible(results, uranium_paths, queries, scores,
                              ndarrays_regression):
    assert type(results) == dict
    assert set(results.keys()) == set(queries.values())
    for path in uranium_paths:
        for name, score in zip(['flux', 'reaction'], scores):
            entire_energy_result = first(result['value']
                                         for result in results[queries[f'entire_energy_{name}']]
                                         if result['score'] == score \
                                         and result['component'] == path)
            split_energy_result = sum([result['value']
                                       for result in results[queries[f'binned_energy_{name}']]
                                       if result['score'] == score
                                       and result['component'] == path])
            assert np.isclose(entire_energy_result, split_energy_result)
    tallies_results = {attr: np.array(list(map(itemgetter(attr), results[queries[f'binned_energy_{name}']])),
                                      dtype=typ)
                       for attr, typ in zip(['component', 'value', 'error',
                                             'particle', 'lower_energy',
                                             'upper_energy'],
                                            (str, float, float, str, float,
                                             float))
                       for name in ['flux', 'reaction']}
    ndarrays_regression.check(tallies_results)


def test_xenon_reaction_rate_is_sensible_where_there_are_no_xenon(direct):
    score = ReactionScore(reaction=reactions.Reaction(
        reactions.ProtoReaction(Xe135, reactions.Typus.NGamma, branching={Xe136: 1}), Xe136),
        volume_specific=True,
        density_specific=True)
    query = VolumeQuery(scores=tuple([score]),
                        names=tuple([path for path, comp in example_state.core.named_components if
                                     U235 in comp.mixture]))
    result = direct(example_state, query)[query]
    for rate in result:
        assert 0.1 < rate['value'] < 100


def test_surface_currents_from_opposite_symmetric_sides_are_close_by_example(direct):
    y_translation = lattice_limits[1] * 3 / 4 - hafnium_block_size / 2
    surface_up = Plane(0, 1, 0, y_translation)
    surface_down = Plane(0, 1, 0, -y_translation)
    query1 = SurfaceCurrentQuery(surface_down, to_component=PurePath("CoreTree/pool/south_hafnium_block"))
    query2 = SurfaceCurrentQuery(surface_up, to_component=PurePath("CoreTree/pool/north_hafnium_block"))
    query3 = SurfaceCurrentQuery(surface_down, from_component=PurePath("CoreTree/pool/south_hafnium_block"),
                                 to_component=PurePath("CoreTree/pool"))
    result = direct(example_state, query1, query2, query3)
    assert np.abs(abs(result[query1][0]['value']) - abs(result[query2][0]['value'])) < 2 * result[query1][0][
        'error']
    assert np.abs(result[query1][0]['value'] + result[query3][0]['value']) > 2 * result[query1][0]['error']


def test_surface_currents_equal_mesh_surface_currents_by_example(direct):
    surface_down = Plane(0, 1, 0, -(lattice_limits[1] * 3 / 4
                                    - hafnium_block_size / 2))
    query1 = SurfaceCurrentQuery(surface_down, to_component=PurePath("CoreTree/pool/south_hafnium_block"))

    def mesh_tally(model, cells):
        mesh = openmc.RegularMesh()
        bottom_hafnium_block_center = np.array([0., -lattice_limits[1] * 3 / 4,
                                                0.])
        hafnium_block_dimensions = np.array([hafnium_block_size,
                                             hafnium_block_size,
                                             hafnium_block_height])
        mesh.lower_left = bottom_hafnium_block_center \
                          - hafnium_block_dimensions / 2
        mesh.upper_right = bottom_hafnium_block_center \
                           + hafnium_block_dimensions / 2
        mesh.dimension = [1, 1, 1]
        tally = openmc.Tally()
        tally.filters = [openmc.MeshSurfaceFilter(mesh)]
        tally.scores = ['current']
        model.tallies.append(tally)
        return model, {'mesh_surface_current': tally.id}

    result = direct(example_state, query1, extra_tallies=mesh_tally)
    assert result['mesh_surface_current'].to_numpy()[-5][-2] == abs(result[query1][0]['value'])


def test_compute_power_by_heating_local_doesnt_crash(direct):
    query = HeatingRateQuery('score heating-local')
    result = direct(example_state, query)
    print(result)


def test_cartesian_mesh_volume_division(direct):
    d = 2
    mesh = CartesianMesh(x=(-d / 2, d / 2, 3 * d / 2), y=(-d / 2, d / 2), z=(-d / 2, d / 2))
    integral_flux_score = Score('flux', volume_specific=False)
    average_flux_score = Score('flux', volume_specific=True)
    integral_query = MeshQuery(mesh, scores=(integral_flux_score, ), energies=(0., 10., 20.e6))
    average_query = MeshQuery(mesh, scores=(average_flux_score,), energies=(0., 10., 20.e6))
    result = direct(example_state, integral_query, average_query)
    integral_result = result[integral_query][0]
    average_result = result[average_query][0]
    assert np.allclose((integral_result['mean'] / average_result['mean']).to_series()[lambda x: x.notna()],
                       d ** 3, equal_nan=True)
    assert np.allclose((integral_result['std'] / average_result['std']).to_series()[lambda x: x.notna()],
                       d ** 3, equal_nan=True)


def test_compare_tabulated_to_reaction_scores(direct):
    score = ReactionScore(reaction=reactions.Reaction(
        reactions.ProtoReaction(Xe136, reactions.Typus.NGamma, branching={Xe137: 1}), Xe137),
        volume_specific=True,
        density_specific=True)
    reaction_query = VolumeQuery(scores=tuple([score]),
                                 names=tuple([path for path, comp in example_state.core.named_components if
                                              U235 in comp.mixture]))
    tabulated = tabulated_neutron_cross_section(Xe136, 102)
    tabulated_query = VolumeQuery(names=tuple([path for path, comp in example_state.core.named_components if
                                               U235 in comp.mixture]),
                                  scores=(TabulatedScore(tuple(tabulated.x), tuple(tabulated.y), True),))
    results = direct(example_state, reaction_query, tabulated_query)
    for r1, r2 in zip(results[reaction_query], results[tabulated_query]):
        assert np.isclose(r1['value'], r2['value'])


def test_compare_tabulated_heating_to_heating_tally_of_water():
    """
    This test compares heating computed with the heating score and heating computed using the tabulated
    cross-sections of the 'heating' reaction, i.e. MT=301. k-normalization is applied to compare both values.
    """
    heating_neutron_query = VolumeQuery(scores=tuple([Score('heating', volume_specific=True)]),
                                        names=tuple([PurePath("CoreTree/pool")]))
    heating_photon_query = VolumeQuery(scores=tuple([Score('heating', volume_specific=True)]),
                                       names=tuple([PurePath("CoreTree/pool")]),
                                       particle=Photon)
    expanded_water = heavy_water.expand(heavy_water)
    expanded_water.isotopes.pop(O18)

    neutron_tabulated = {iso: tabulated_neutron_cross_section(iso, 301) for iso in expanded_water}
    neutron_tabulated_queries = valmap(
        lambda tab: VolumeQuery(names=tuple([PurePath("CoreTree/pool")]),
                                scores=(
                                    TabulatedScore(tuple(tab.x), tuple(tab.y),
                                                   True),)), neutron_tabulated)
    photon_tabulated = {iso: tabulated_photon_cross_section(iso, 525) for iso in [H, O]}
    photon_tabulated_queries = valmap(
        lambda tab: VolumeQuery(names=tuple([PurePath("CoreTree/pool")]),
                                scores=(
                                    TabulatedScore(tuple(tab.x), tuple(tab.y),
                                                   True),), particle=Photon),
        photon_tabulated)

    with TemporaryDirectory(prefix='openmcadapter_test') as tempdir:
        os.chdir(tempdir)
        oracle = OpenMCOracle(settings=Settings(1000, 100, 20, photon_transport=True))
        tabulated_queries = list(neutron_tabulated_queries.values()) + list(photon_tabulated_queries.values())
        kq = KQuery()
        results = oracle(example_state, kq, heating_neutron_query, heating_photon_query, *tabulated_queries)
    s = ufloat(0, 0)
    for iso, q in neutron_tabulated_queries.items():
        s += ufloat(results[q][0]['value'] * expanded_water[iso],
                    results[q][0]['error'] * expanded_water[iso])
    assert np.abs(s.nominal_value * results[kq][0].k - results[heating_neutron_query][0]['value']) < \
           results[heating_neutron_query][0]['error']
