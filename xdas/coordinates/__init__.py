"""
Coordinate types that describe how array axes map to physical values.

Exports :class:`Coordinates` (container) and all concrete coordinate classes:
:class:`Coordinate` (factory/base), :class:`DefaultCoordinate`,
:class:`DenseCoordinate`, :class:`InterpCoordinate`,
:class:`SampledCoordinate`, :class:`ScalarCoordinate`.
"""

__all__ = [
    "Coordinate",
    "Coordinates",
    "DefaultCoordinate",
    "DenseCoordinate",
    "InterpCoordinate",
    "SampledCoordinate",
    "ScalarCoordinate",
    "get_sampling_interval",
]

from .core import Coordinate, Coordinates, get_sampling_interval
from .default import DefaultCoordinate
from .dense import DenseCoordinate
from .interp import InterpCoordinate
from .sampled import SampledCoordinate
from .scalar import ScalarCoordinate
