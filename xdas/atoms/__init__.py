"""
Stateful processing units (atoms) for building chunked data pipelines.

Exports :class:`Atom`, :class:`State`, :class:`Sequential`, :class:`Partial`,
:func:`atomized`, signal-processing atoms, and the ML-based :class:`MLPicker`.
"""

from ..trigger import Trigger
from .core import Atom, Partial, Sequential, State, atomized
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
