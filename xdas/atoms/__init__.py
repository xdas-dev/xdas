"""
Stateful processing units (atoms) for building chunked data pipelines.

Exports :class:`Atom`, :class:`State`, :class:`Sequential`, :class:`Partial`,
:func:`atomized`, :func:`as_function`, :func:`compose`, signal-processing
atoms, and the ML-based :class:`MLPicker`.
"""

__all__ = [
    "Atom",
    "DownSample",
    "FIRFilter",
    "IIRFilter",
    "LFilter",
    "MLPicker",
    "Partial",
    "ResamplePoly",
    "SOSFilter",
    "Sequential",
    "State",
    "Trigger",
    "UpSample",
    "as_function",
    "atomized",
    "compose",
]

from ..trigger import Trigger
from .core import Atom, Partial, Sequential, State, as_function, atomized, compose
from .ml import MLPicker
from .signal import (
    DownSample,
    FIRFilter,
    IIRFilter,
    LFilter,
    ResamplePoly,
    SOSFilter,
    UpSample,
)
