"""
Stateful processing units (atoms) for building chunked data pipelines.

Three layers:

- :mod:`xdas.atoms.core`: the machinery — :class:`Atom`, :class:`State`,
  :class:`Sequential`, :class:`Partial`, :func:`atomized`,
  :func:`as_function`, :func:`compose`.
- :mod:`xdas.atoms.kernel`: the expert layer — exact stateful primitives with
  machine parameters (:class:`LFilter`, :class:`SOSFilter`,
  :class:`DownSample`, :class:`UpSample`, :class:`Polyphase`).
- :mod:`xdas.atoms.operations`: the public layer — operation atoms with
  physical parameters only (:class:`Filter`, :class:`Resample`, ...), each
  with a function form exported at the top level of :mod:`xdas`.

Plus the detection atoms of :mod:`xdas.atoms.detect` and the ML-based
:class:`Annotate`.
"""

__all__ = [
    "STFT",
    "Annotate",
    "Atom",
    "Differentiate",
    "DownSample",
    "FIRFilter",
    "Filter",
    "IIRFilter",
    "Integrate",
    "LFilter",
    "MLPicker",
    "Partial",
    "Picker",
    "Polyphase",
    "Rechunk",
    "Resample",
    "ResamplePoly",
    "SOSFilter",
    "Sequential",
    "State",
    "Trigger",
    "UpSample",
    "as_function",
    "atomized",
    "compose",
    "trigger",
]

from .core import Atom, Partial, Sequential, State, as_function, atomized, compose
from .detect import Trigger, trigger
from .kernel import DownSample, LFilter, Polyphase, Rechunk, SOSFilter, UpSample
from .ml import Annotate, MLPicker, Picker
from .operations import (
    STFT,
    Differentiate,
    Filter,
    FIRFilter,
    IIRFilter,
    Integrate,
    Resample,
    ResamplePoly,
)
