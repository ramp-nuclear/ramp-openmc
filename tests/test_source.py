import isotopes
import numpy as np
import openmc.stats
import pytest
from coremaker.example import example_core, hafnium_block_names

from openmcadapter.source import ComponentDependentSource, ComponentSource, OpenMCSource, openmc_source


@pytest.fixture()
def energy_distribution():
    return openmc.stats.Discrete([1.0], [1.0])


def equal_spatial_distribution(s1: OpenMCSource, s2: OpenMCSource):
    t = (s1.space.lower_left == s2.space.lower_left).all()
    t &= (s1.space.upper_right == s2.space.upper_right).all()
    return t


def test_component_source_has_correct_center_in_core(energy_distribution):
    hf_block = list(hafnium_block_names.values())[0]
    compname = [name for name, c in example_core.named_components if name.name == hf_block.name][0]
    source = ComponentSource(compname, energy_distribution)
    c_source = openmc_source(source, example_core)
    space = c_source.space
    center = (space.lower_left + space.upper_right) / 2
    assert np.all(center == (0, -67.5, 0))


def test_component_dependent_source(energy_distribution):
    cs = [name for name, c in example_core.named_components if isotopes.U235 in c.mixture]
    _c_source = ComponentSource(cs[0], energy_distribution)
    c_source = openmc_source(ComponentSource(cs[0], energy_distribution), example_core)
    n = len(cs)
    cs_source = openmc_source(ComponentDependentSource({c: 1.0 / n for c in cs}, energy_distribution), example_core)
    assert equal_spatial_distribution(cs_source[0], c_source)
    assert all(s.strength == 1.0 / n for s in cs_source)
