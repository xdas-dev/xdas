"""
Xdas — labeled N-dimensional array library for Distributed Acoustic Sensing data.

Provides :class:`DataArray` with lazy multi-file HDF5/NetCDF4 access, coordinate
types for physical axes, multi-threaded signal processing pipelines, and I/O engines
for common DAS instrument formats.
"""

__version__ = "0.2.9"

__all__ = [  # noqa: RUF022 - grouped by kind, not alphabetically
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
    "testing",
    "virtual",
    # classes
    "Coordinate",
    "Coordinates",
    "DataArray",
    "DataCollection",
    "DataMapping",
    "DataSequence",
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
    "sortby",
    "split",
    "stack",
    "trim_overlaps",
    # task atoms (function forms)
    "annotate",
    "detrend",
    "differentiate",
    "filter",
    "hilbert",
    "integrate",
    "medfilt",
    "mlpicker",
    "pick",
    "rechunk",
    "resample",
    "sliding_mean_removal",
    "stalta",
    "stft",
    "taper",
    "trigger",
    # streaming
    "watch",
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
    testing,
    virtual,
)

# The compat module first, so that a later `import xdas.trigger` finds it in
# `sys.modules` and does not rebind the attribute: the lowercase twin below
# stays `xdas.trigger` for everyone.
from . import trigger as _trigger_module  # noqa: F401  isort: skip
from .atoms.detect import stalta, trigger
from .atoms.kernel import rechunk
from .atoms.ml import annotate, mlpicker, pick
from .atoms.operations import (
    detrend,
    differentiate,
    filter,
    hilbert,
    integrate,
    medfilt,
    resample,
    sliding_mean_removal,
    stft,
    taper,
)
from .coordinates import (
    Coordinate,
    Coordinates,
    DenseCoordinate,
    InterpCoordinate,
    SampledCoordinate,
    ScalarCoordinate,
    get_sampling_interval,
)
from .core import (
    DataArray,
    DataCollection,
    DataMapping,
    DataSequence,
    align,
    asdataarray,
    broadcast_coords,
    broadcast_to,
    combine_by_coords,
    combine_by_field,
    concat,
    concat_coords,
    concatenate,
    dataarray,
    datacollection,
    methods,
    numpy,
    open,
    open_dataarray,
    open_datacollection,
    open_mfdataarray,
    open_mfdatacollection,
    open_mfdatatree,
    plot_availability,
    routines,
    sortby,
    split,
    stack,
    trim_overlaps,
)
from .core.methods import *
from .processing.core import watch
