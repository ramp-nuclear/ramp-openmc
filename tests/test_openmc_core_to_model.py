from copy import deepcopy

from coremaker.example import example_core

from openmcadapter import openmc_core_to_model


def test_ids_are_integers():
    core, ids, _ = openmc_core_to_model(deepcopy(example_core), "vacuum")
    assert all(type(i) is int for i in ids.values())
