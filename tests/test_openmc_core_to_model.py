from copy import deepcopy

from openmcadapter import openmc_core_to_model
from coremaker.example import example_core


def test_ids_are_integers():
    core, ids,_ = openmc_core_to_model(deepcopy(example_core), 'vacuum')
    assert all(type(i) == int for i in ids.values())
