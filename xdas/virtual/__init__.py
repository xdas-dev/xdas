"""
Virtual (lazy) array backends over on-disk sources.

Two backends coexist, selected by the ``vtype`` of the open functions:
:mod:`xdas.virtual.hdf5` exposes HDF5 virtual datasets
(:class:`VirtualSource`, :class:`VirtualStack`, :class:`VirtualLayout`
under the :class:`VirtualArray` base) and :mod:`xdas.virtual.tiles`
exposes tile manifests (:class:`TileArray`).
"""

__all__ = [
    "Selection",
    "Selectors",
    "SingleSelector",
    "SliceSelector",
    "TileArray",
    "VirtualArray",
    "VirtualLayout",
    "VirtualSource",
    "VirtualStack",
]

from .hdf5 import (
    Selection,
    Selectors,
    SingleSelector,
    SliceSelector,
    VirtualArray,
    VirtualLayout,
    VirtualSource,
    VirtualStack,
)
from .tiles import TileArray
