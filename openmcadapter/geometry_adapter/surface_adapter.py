"""
This module contains function that transform RAMP surfaces to openmc surfaces
"""
from functools import lru_cache

import openmc
from coremaker.surfaces.cylinder import Cylinder as Cylinder
from coremaker.surfaces.plane import Plane
from coremaker.surfaces.sphere import Sphere
from multipledispatch import dispatch


@dispatch(Plane,object)
@lru_cache(maxsize=None)
def openmc_halfspace(surface: Plane,id:int=None) -> openmc.Halfspace:
    """
    creates an openmc halfspace form a RAMP halfplane
    """
    return +openmc.Plane(*surface.a,surface.b,surface_id=id)


@dispatch(Sphere,object)
@lru_cache(maxsize=None)
def openmc_halfspace(surface: Sphere,id:int=None) -> openmc.Halfspace:
    """
    creates an openmc halfspace form a RAMP sphere
    """
    x0, y0, z0 = surface.center
    sphere = openmc.Sphere(x0=x0, y0=y0,
                           z0=z0, r=surface.radius,surface_id=id)
    if surface.inside:
        return -sphere
    return +sphere


@dispatch(Cylinder,object)
@lru_cache(maxsize=None)
def openmc_halfspace(surface: Cylinder,id:int=None) -> openmc.Halfspace:
    """
    creates an openmc halfspace form a RAMP cylinder
    """
    cylinder = openmc.Cylinder(surface.center[0], surface.center[1],
                               surface.center[2], surface.radius,
                               surface.axis[0], surface.axis[1],
                               surface.axis[2],surface_id=id)
    if surface.inside:
        return -cylinder
    return +cylinder


@dispatch(object,object)
def openmc_halfspace(surface,id=None):
    raise NotImplementedError
