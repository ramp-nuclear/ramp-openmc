import logging
import os
from functools import lru_cache
from pathlib import Path

import openmc
from coremaker.materials.mixture import Chemical
from coremaker.materials.mixture import Mixture
from isotopes import (Th232, Pa233, Ge72, Ge73, Ge74, Ge76, As75, Se76, Se77,
                      Se78, Se80, Se82, Br79, Br81, Kr80, Kr82, Kr83, Kr84,
                      Kr86, Rb85, Rb87, Sr86, Sr87, Sr88, Zr90, Zr91, Zr92,
                      Y89,
                      Zr93, Zr94, Zr96, Nb94, Mo95, Mo96, Mo97, Tc99, Ru99,
                      Ru101, Ru102, Ru103, Ru104, Ru106, Rh105, Pd107, Pd108,
                      Pd110, Ag109, Cd111, Cd112, Cd113, Cd114, Cd116, In113,
                      In115, Sn115, Sn117, Sn118, Sn119, Sn126, Sb121, Sb123,
                      Sb125, Te122, Te124, Te125, Te126, Te127m1, Te128, Te130,
                      I127, I129, I135, Xe131, Xe132, Xe134, Xe135, Xe136,
                      Cs133, Cs134, Cs135, Cs137, Ba136, Ba137, Ba138, Ce140,
                      Ce142, Pr141, Nd143, Nd144, Nd145, Nd146, Nd148, Nd150,
                      Pm147, Pm149, Sm150, Sm151, Sm152, Sm154, Eu153, Eu155,
                      Gd156, Gd157, Gd158, Gd160, Tb159, Dy161, Dy162, Dy163,
                      Dy164, Ho165, Er166, Er167, U232, U233, Ru100, Pd106,
                      Xe128, Xe130, Ba134, Ba135, Nd142, Pm148, Pm148m1, Sm147,
                      Sm148, Sm149, Eu151, Eu152, Eu154, Gd152, Gd154, Gd155,
                      Tb160, Dy160, Rh103, Pa231, U234, U235, U236, Te123,
                      Pd104, U237, U238, Np239, Pd105, Np237, Pu238, Pu240,
                      Pu239, Pu241, Pu242, Am243, Am241, Am242, Am242m1, ZAID,
                      Cm242, Cm243, Cm244, Isotope)
from isotopes._format import _isotope_symbols
from openmc.data import DataLibrary

logger = logging.getLogger(__name__)

"""
Openmc can't tally microscopic cross sections, therefore all isotopes generated
during burnup should be present from the start in order to perform the burnup
calculation.
"""

uranium_products = (Th232, Pa233, Ge72, Ge73, Ge74, Ge76, As75, Se76, Se77,
                    Se78, Se80, Se82, Br79, Br81, Kr80, Kr82, Kr83, Kr84,
                    Kr86, Rb85, Rb87, Sr86, Sr87, Sr88, Zr90, Zr91, Zr92, Y89,
                    Zr93, Zr94, Zr96, Nb94, Mo95, Mo96, Mo97, Tc99, Ru99,
                    Ru101, Ru102, Ru103, Ru104, Ru106, Rh105, Pd107, Pd108,
                    Pd110, Ag109, Cd111, Cd112, Cd113, Cd114, Cd116, In113,
                    In115, Sn115, Sn117, Sn118, Sn119, Sn126, Sb121, Sb123,
                    Sb125, Te122, Te124, Te125, Te126, Te127m1, Te128, Te130,
                    I127, I129, I135, Xe131, Xe132, Xe134, Xe135, Xe136,
                    Cs133, Cs134, Cs135, Cs137, Ba136, Ba137, Ba138, Ce140,
                    Ce142, Pr141, Nd143, Nd144, Nd145, Nd146, Nd148, Nd150,
                    Pm147, Pm149, Sm150, Sm151, Sm152, Sm154, Eu153, Eu155,
                    Gd156, Gd157, Gd158, Gd160, Tb159, Dy161, Dy162, Dy163,
                    Dy164, Ho165, Er166, Er167, U232, U233, Ru100, Pd106,
                    Xe128, Xe130, Ba134, Ba135, Nd142, Pm148, Pm148m1, Sm147,
                    Sm148, Sm149, Eu151, Eu152, Eu154, Gd152, Gd154, Gd155,
                    Tb160, Dy160, Rh103, Pa231, U234, U235, U236, Te123,
                    Pd104, U237, U238, Np239, Pd105, Np237, Pu238, Pu240,
                    Pu239, Pu241, Pu242, Am243, Am241, Am242, Am242m1,
                    Cm242, Cm243, Cm244)


@lru_cache()
def available_library_data(library_path: Path = None) -> set:
    library_path = library_path or os.environ.get('OPENMC_CROSS_SECTIONS')
    library = DataLibrary.from_xml(library_path)
    available_data = set()
    for lib in library.libraries:
        if lib['type'] == 'neutron':
            available_data.add(lib['materials'][0])
    return available_data


_isomer_symbols = {0: "",
                   1: "_m1"}

_sab_symbols = {Chemical.LightWater: 'c_H_in_H2O', Chemical.HeavyWater: 'c_D_in_D2O',
                Chemical.Be: "c_Be", Chemical.Be_in_BeO: "c_Be_in_BeO",
                Chemical.Graphite: "c_Graphite", Chemical.O_in_BeO: "c_O_in_BeO",
                Chemical.Polyethylene: "c_H_in_CH2", Chemical.Benzene: "c_C6H6"}


def openmc_name(nuclide: ZAID) -> str:
    return f'{_isotope_symbols[nuclide.Z]}{nuclide.A or ""}{_isomer_symbols[nuclide.m]}'


@lru_cache()
def add_burning_isotopes(mixture: Mixture, amount=1e-20):
    """
    Add the isotopes created from burning to the mixture in very small amounts
    The function edits the mixture
    """
    if U235 in mixture:
        for iso in uranium_products:
            if iso not in mixture:
                mixture.isotopes[iso] = amount


def _is_elem(nuclide: ZAID) -> bool:
    return isinstance(nuclide, Isotope) and nuclide.A == 0


@lru_cache()
def openmc_material(mixture: Mixture, add_sab=True, library_path: Path = None) -> openmc.Material:
    """
    This function transforms a RAMP mixture to an openmc material
    """
    available_data = available_library_data(library_path)
    material = openmc.Material(temperature=mixture.temperature + 274)
    density = 0
    elements = list(filter(lambda n: _is_elem(n) and (f'{openmc_name(n)}0' not in available_data),
                      mixture.keys()))
    mixture = mixture.expand(mixture, elements)
    for nuclide, nuclide_density in mixture.items():
        nuclide_name = openmc_name(nuclide)
        if nuclide_name in available_data:
            material.add_nuclide(nuclide_name,
                                 nuclide_density)
            density += nuclide_density
        elif f'{nuclide_name}0' in available_data:
            material.add_nuclide(f'{nuclide_name}0',
                                 nuclide_density)
            density += nuclide_density
        else:
            try:
                isotope_name = openmc_name(nuclide)
                if isotope_name in available_data:
                    material.add_nuclide(isotope_name, nuclide_density)
                    density += nuclide_density
                else:
                    logger.debug(f"{nuclide} doesn't have data in "
                                 f"the library so it is removed from the mixture")
                    continue
            except ValueError:
                logger.debug(f"{nuclide} doesn't have data in "
                             f"the library so it is removed from the mixture")
    material.set_density("atom/b-cm", density)
    if add_sab:
        for s_ab in mixture.sab:
            material.add_s_alpha_beta(_sab_symbols[s_ab])
    return material


def test_mixture_adapter():
    mixture = Mixture({U235: 0.1}, 293, )
    x = openmc_material(mixture)
    y = openmc_material(mixture)
    assert x == y
