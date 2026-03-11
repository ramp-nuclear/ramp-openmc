"""
This package is used to create openmc input from RAMP objects
"""

from .openmc_oracle import OpenMCOracle as OpenMCOracle, Settings as Settings
from .geometry_adapter.core_adapter import openmc_core_to_model as openmc_core_to_model
