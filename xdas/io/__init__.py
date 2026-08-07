"""
I/O subsystem: plugin-based :class:`Engine` registry and concrete engines.

Supports xdas native, ASN, APSensing, Febus, ProdML, Silixa and Terra15
formats, plus everything ObsPy reads (MiniSEED, SAC, GSE2, ...). The legacy
`"miniseed"` engine is kept for stored views written by it.
"""

__all__ = [
    "AutoEngine",
    "Engine",
    "apsensing",
    "asn",
    "febus",
    "get_free_port",
    "miniseed",
    "obspy",
    "prodml",
    "silixa",
    "terra15",
    "xdas",
]

from . import apsensing, asn, febus, prodml, silixa, terra15, xdas
from .core import AutoEngine, Engine, get_free_port

# isort: split
# The two obspy-based engines are imported last, and `obspy` before
# `miniseed`. `AutoEngine` tries engines in registration order: these two open
# far more than the format-specific ones, so they must come after them — a
# promiscuous engine placed early would shadow the others — and the new engine
# must be reached before the legacy one it replaces.
from . import obspy

# isort: split
from . import miniseed
