import pytest
from coremaker.materials.absorbers import aic, b4c, hafnium
from coremaker.materials.aluminium import al1050, al6061
from coremaker.materials.reflectors import beryllium, graphite
from coremaker.materials.steel import steel_304L, steel_316L
from coremaker.materials.water import make_light_water
from coremaker.materials.zirconium import zircalloy_2, zircalloy_4

from openmcadapter.mixture_adapter import openmc_material


@pytest.mark.regression
@pytest.mark.parametrize(
    "mix",
    [
        graphite,
        make_light_water(20.0),
        steel_316L,
        steel_304L,
        b4c,
        aic,
        hafnium,
        al6061,
        al1050,
        beryllium,
        zircalloy_4,
        zircalloy_2,
    ],
)
def test_openmc_material(num_regression, mix):
    material = openmc_material(mix)
    num_regression.check(material.get_nuclide_atom_densities())
