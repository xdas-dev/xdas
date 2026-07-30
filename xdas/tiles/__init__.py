"""
Lazy tile-backed virtual arrays (ported from the 0.3 line).

:class:`TileArray` exposes a rectilinear grid of file-backed tiles as
one numpy-like lazy array; :class:`Engine` is the per-format tile
reader plugin socket. This backend replaces the serialized-dask-graph
fallback used by the formats that HDF5 virtual datasets cannot serve
(Silixa TDMS, MiniSEED).
"""

from .registry import ENGINES, Engine
from .tilearray import TileArray, extract_array

__all__ = [
    "ENGINES",
    "Engine",
    "TileArray",
    "extract_array",
]
