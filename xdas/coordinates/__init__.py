"""
Coordinate types that describe how array axes map to physical values.

Exports :class:`Coordinates` (container) and all concrete coordinate classes:
:class:`Coordinate` (factory/base), :class:`DenseCoordinate`,
:class:`InterpCoordinate`, :class:`SampledCoordinate`, :class:`ScalarCoordinate`.
"""

__all__ = [
    "AxisCoordinate",
    "Coordinate",
    "Coordinates",
    "DenseCoordinate",
    "InterpCoordinate",
    "PiecewiseMixin",
    "SampledCoordinate",
    "ScalarCoordinate",
    "get_sampling_interval",
]

from .core import (
    AxisCoordinate,
    Coordinate,
    Coordinates,
    PiecewiseMixin,
    get_sampling_interval,
)
from .dense import DenseCoordinate
from .interp import InterpCoordinate
from .sampled import SampledCoordinate
from .scalar import ScalarCoordinate
