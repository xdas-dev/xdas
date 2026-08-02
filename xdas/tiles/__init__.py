"""
Lazy tile-backed virtual arrays (ported from the 0.3 line).

:class:`TileArray` exposes a rectilinear grid of file-backed tiles as
one numpy-like lazy array. Tiles are decoded by the ``load_tile`` half
of the :class:`xdas.io.Engine` format plugins, resolved by manifest
engine name on that registry (``Engine[name]``). This backend replaces
the serialized-dask-graph fallback used by the formats that HDF5
virtual datasets cannot serve (Silixa TDMS, MiniSEED).
"""

from .tilearray import TileArray

__all__ = [
    "TileArray",
]
