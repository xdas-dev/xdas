"""
Xdas — labeled N-dimensional array library for Distributed Acoustic Sensing data.

Provides :class:`DataArray` with lazy multi-file HDF5/NetCDF4 access, coordinate
types for physical axes, multi-threaded signal processing pipelines, and I/O engines
for common DAS instrument formats.
"""

__version__ = "0.2.7"

__all__ = [
    # submodules
    "atoms",
    "config",
    "coordinates",
    "dataarray",
    "datacollection",
    "fft",
    "io",
    "methods",
    "numpy",
    "parallel",
    "processing",
    "routines",
    "signal",
    "synthetics",
    "virtual",
    # classes
    "Coordinate",
    "Coordinates",
    "DataArray",
    "DataCollection",
    "DataMapping",
    "DataSequence",
    "DefaultCoordinate",
    "DenseCoordinate",
    "InterpCoordinate",
    "SampledCoordinate",
    "ScalarCoordinate",
    # functions
    "align",
    "asdataarray",
    "broadcast_coords",
    "broadcast_to",
    "combine_by_coords",
    "combine_by_field",
    "concat",
    "concat_coords",
    "concatenate",
    "get_sampling_interval",
    "open",
    "open_dataarray",
    "open_datacollection",
    "open_mfdataarray",
    "open_mfdatacollection",
    "open_mfdatatree",
    "plot_availability",
    "split",
]

from . import (
    atoms,
    config,
    coordinates,
    fft,
    io,
    parallel,
    processing,
    signal,
    synthetics,
    virtual,
)
from .coordinates import (
    Coordinate,
    Coordinates,
    DefaultCoordinate,
    DenseCoordinate,
    InterpCoordinate,
    SampledCoordinate,
    ScalarCoordinate,
    get_sampling_interval,
)
from .core import dataarray, datacollection, methods, numpy, routines
from .core.dataarray import DataArray
from .core.datacollection import DataCollection, DataMapping, DataSequence
from .core.methods import *  # noqa: F403
from .core.routines import (
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
