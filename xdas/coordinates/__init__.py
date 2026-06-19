"""
Coordinate types that describe how array axes map to physical values.

Exports :class:`Coordinates` (container) and all concrete coordinate classes:
:class:`Coordinate` (factory/base), :class:`DenseCoordinate`,
:class:`InterpCoordinate`, :class:`SampledCoordinate`, :class:`ScalarCoordinate`.
"""

__all__ = [
    "Coordinate",
    "Coordinates",
    "DenseCoordinate",
    "FixedInterpCoordinate",
    "InterpCoordinate",
    "SampledCoordinate",
    "ScalarCoordinate",
    "get_sampling_interval",
]

from .core import Coordinate, Coordinates, get_sampling_interval
from .dense import DenseCoordinate
from .interp import FixedInterpCoordinate, InterpCoordinate
from .sampled import SampledCoordinate
from .scalar import ScalarCoordinate
