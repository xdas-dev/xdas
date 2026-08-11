"""
Stateful processing units (atoms) for building chunked data pipelines.

Three layers:

- :mod:`xdas.atoms.core`: the machinery — :class:`Atom`, :class:`State`,
  :class:`Sequential`, :class:`Partial`, :func:`atomized`,
  :func:`as_function`, :func:`compose`.
- :mod:`xdas.atoms.kernel`: the expert layer — exact stateful primitives with
  machine parameters (:class:`LFilter`, :class:`SOSFilter`,
  :class:`DownSample`, :class:`UpSample`, :class:`Polyphase`).
- :mod:`xdas.atoms.tasks`: the public layer — task atoms with physical
  parameters only (:class:`Filter`, :class:`Decimate`, :class:`Resample`,
  ...), each with a function form exported at the top level of :mod:`xdas`.

Plus the signal-processing atoms of :mod:`xdas.atoms.signal` and the ML-based
:class:`Annotate`.
"""

__all__ = [
    "STFT",
    "Annotate",
    "Atom",
    "Decimate",
    "Differentiate",
    "DownSample",
    "FIRFilter",
    "Filter",
    "IIRFilter",
    "Integrate",
    "LFilter",
    "MLPicker",
    "Partial",
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
from .ml import Annotate, MLPicker
from .signal import FIRFilter, IIRFilter, ResamplePoly
from .tasks import STFT, Decimate, Differentiate, Filter, Integrate, Resample
