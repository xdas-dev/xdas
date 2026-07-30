"""
Core data types for xdas.

Includes :class:`DataArray`, :class:`DataCollection`, and supporting routines,
methods, and NumPy dispatch.
"""

__all__ = [
    "DataArray",
    "DataCollection",
    "DataMapping",
    "DataSequence",
    "align",
    "asdataarray",
    "broadcast_coords",
    "broadcast_to",
    "combine_by_coords",
    "combine_by_field",
    "concat",
    "concat_coords",
    "concatenate",
    "open",
    "open_dataarray",
    "open_datacollection",
    "open_mfdataarray",
    "open_mfdatacollection",
    "open_mfdatatree",
    "plot_availability",
    "split",
]

from .dataarray import DataArray
from .datacollection import DataCollection, DataMapping, DataSequence
from .routines import (
    align,
    asdataarray,
    broadcast_coords,
    broadcast_to,
    combine_by_coords,
    combine_by_field,
    concat,
    concat_coords,
    concatenate,
    open,
    open_dataarray,
    open_datacollection,
    open_mfdataarray,
    open_mfdatacollection,
    open_mfdatatree,
    plot_availability,
    split,
)
