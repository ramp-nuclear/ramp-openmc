import pytest
from coremaker.materials.absorbers import hafnium
from coremaker.materials.aluminium import al1050
from coremaker.materials.water import make_heavy_water

from openmcadapter.mixture_adapter import openmc_material


@pytest.mark.regression
@pytest.mark.parametrize(
    "mix",
    [
        make_heavy_water(20.0),
        hafnium,
        al1050,
    ],
)
def test_openmc_material(num_regression, mix):
    material = openmc_material(mix)
    num_regression.check(material.get_nuclide_atom_densities())
