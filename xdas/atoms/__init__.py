"""
Stateful processing units (atoms) for building chunked data pipelines.

Two layers so far:

- :mod:`xdas.atoms.core`: the machinery — :class:`Atom`, :class:`State`,
  :class:`Sequential`, :class:`Partial`, :func:`atomized`,
  :func:`as_function`, :func:`compose`.
- :mod:`xdas.atoms.kernel`: the expert layer — exact stateful primitives with
  machine parameters (:class:`LFilter`, :class:`SOSFilter`,
  :class:`DownSample`, :class:`UpSample`, :class:`Polyphase`).

Plus the signal-processing atoms of :mod:`xdas.atoms.signal` and the ML-based
:class:`MLPicker`.
"""

__all__ = [
    "Atom",
    "DownSample",
    "FIRFilter",
    "IIRFilter",
    "LFilter",
    "MLPicker",
    "Partial",
    "Polyphase",
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
from .kernel import DownSample, LFilter, Polyphase, SOSFilter, UpSample
from .ml import MLPicker
from .signal import FIRFilter, IIRFilter, ResamplePoly
