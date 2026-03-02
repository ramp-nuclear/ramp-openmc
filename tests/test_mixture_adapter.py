import pytest
from coremaker.materials.absorbers import hafnium, b4c, aic
from coremaker.materials.aluminium import al6061, al1050
from coremaker.materials.reflectors import graphite, beryllium
from coremaker.materials.steel import steel_304L, steel_316L
from coremaker.materials.water import make_light_water
from coremaker.materials.zirconium import zircalloy_2, zircalloy_4

from openmcadapter.mixture_adapter import openmc_material


@pytest.mark.parametrize('mix', [graphite, make_light_water(20.),
                                 steel_316L, steel_304L, b4c, aic,
                                 hafnium, al6061, al1050,
                                 beryllium, zircalloy_4, zircalloy_2])
def test_openmc_material(num_regression, mix):
    material = openmc_material(mix)
    num_regression.check(material.get_nuclide_atom_densities())

