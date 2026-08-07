"""
I/O subsystem: plugin-based :class:`Engine` registry and concrete engines.

Supports xdas native, ASN, APSensing, Febus, ProdML, Silixa and Terra15
formats, plus everything ObsPy reads (MiniSEED, SAC, GSE2, ...).
"""

__all__ = [
    "AutoEngine",
    "Engine",
    "apsensing",
    "asn",
    "febus",
    "get_free_port",
    "obspy",
    "prodml",
    "silixa",
    "terra15",
    "xdas",
]

from . import apsensing, asn, febus, prodml, silixa, terra15, xdas
from .core import AutoEngine, Engine, get_free_port

# isort: split
# `obspy` is imported last: it opens far more than the format-specific engines
# do, so `AutoEngine`, which tries them in registration order, must reach it
# only after them — a promiscuous engine placed early would shadow the others.
from . import obspy
