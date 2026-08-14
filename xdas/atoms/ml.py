"""
Machine-learning atom: :class:`Annotate` wraps SeisBench models as pipeline atoms.

Torch and SeisBench are loaded lazily so they remain optional dependencies.
"""

import importlib
import warnings
from collections import deque
from fnmatch import fnmatch

import numpy as np

from ..core import DataArray, concat, stack
from .core import Atom, Sequential, State, atomized
from .detect import Trigger
from .tasks import Filter, Resample


class LazyModule:
    """Defer importing *name* until the first attribute access."""

    def __init__(self, name):
        self._name = name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = importlib.import_module(self._name)
            except ImportError:
                raise ImportError(
                    f"{self._name} is not installed by default, "
                    f"please install it manually"
                )
        return getattr(self._module, name)


torch = LazyModule("torch")

#: Horizontal component letters SeisBench treats as interchangeable when its
#: ``flexible_horizontal_components`` argument is set (which is its default),
#: so that ``Z12H`` weights accept ``ZNE`` data and vice versa.
FLEXIBLE_HORIZONTALS = {"1": "N", "N": "1", "2": "E", "E": "2"}

#: Fallbacks for the annotate arguments these atoms read. SeisBench's own
#: ``_argdict_get_with_default`` falls back to the model's ``_annotate_args``,
#: but ``blinding`` is not in the ``WaveformModel`` base and a model need not
#: declare one at all, so the accessor needs a default of its own rather than a
#: subscript of ``None``.
ANNOTATE_DEFAULTS = {
    "overlap": 0,
    "stacking": "avg",
    "flexible_horizontal_components": True,
}

#: Number of corners ``obspy``'s ``Stream.filter`` uses when the weight set
#: does not say, and the order :class:`~xdas.atoms.Filter` defaults to. The two
#: agree, which is what makes a model-declared filter reproducible exactly.
OBSPY_CORNERS = 4

#: Collection level names :meth:`Annotate.gather` will consider collapsing into
#: a component dimension. A *default*, not a rule: ``components="whatever"``
#: replaces the list, exactly as SeisBench's ``guess_channel_coord_name`` walks
#: a candidate list on the input side. Our engines name the level ``channel``,
#: but nothing in the implementation may assume the literal.
COMPONENT_LEVELS = ("channel", "component")

#: Lengths a collection key may have and still be read as a channel code: three
#: for a SEED code (band, instrument, orientation) and one for a bare
#: orientation letter. This is what keeps the gather off a station level, since
#: the last-letter matcher alone resolves ``STA1``/``STA2`` cleanly against
#: ``ZNE`` weights — see :func:`component_keys`.
CHANNEL_CODE_LENGTHS = (1, 3)

#: Label of the noise class, which is never picked. SeisBench identifies it by
#: this exact name — ``classify_aggregate`` skips ``phase == "N"`` — and so do
#: we, rather than by position: the label *order* is a property of the weight
#: set and flips between them (``NPS`` for PhaseNet's ``original``, ``PSN`` for
#: most others), so a positional rule would silently pick noise.
NOISE_LABEL = "N"

#: Detection threshold used when neither the weight set's ``default_args`` nor
#: the model's ``_annotate_args`` declares one. SeisBench's ``PhaseNet``
#: documents the same value under the ``*_threshold`` key.
DEFAULT_THRESHOLD = 0.3


def annotate_arg(model, argdict, key):
    """
    Read one annotate argument, SeisBench-style but crash-free.

    Mirrors ``WaveformModel._argdict_get_with_default``: what the call site
    passes wins, else what the weight set declares, else the model's
    ``_annotate_args`` documentation default. The last step is where SeisBench
    raises on a key its base class does not declare, so this falls back to
    :data:`ANNOTATE_DEFAULTS` instead. The two SeisBench internals this module
    relies on are isolated here and nowhere else.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        The model whose ``_annotate_args`` documents the fallbacks.
    argdict : dict
        The weight set's ``default_args`` merged with the call's kwargs.
    key : str
        Name of the annotate argument.

    Returns
    -------
    value : object
        The value of *key*, from the argdict, the model or the fallbacks.
    """
    if key in argdict:
        return argdict[key]
    entry = getattr(model, "_annotate_args", {}).get(key)
    if entry is None:
        return ANNOTATE_DEFAULTS[key]
    return entry[1]


def model_phases(model):
    """
    Return the labels a model's output classes carry, in order.

    ``labels`` is per-weight-set and SeisBench lets a weight set leave it unset
    or make it a callable, resolving both only when it builds its output
    stream. ``None`` falls back to positional labels ``0, 1, ... classes - 1``,
    exactly as ``WaveformModel._predictions_to_stream`` does. A callable is
    refused: SeisBench feeds it the identity of the trace being annotated,
    which these atoms do not carry, and a silently wrong ``phase`` coordinate
    would key :class:`~xdas.atoms.Trigger`'s thresholds onto the wrong class.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        The model whose weight set is read.

    Returns
    -------
    list
        One label per output class.
    """
    labels = model.labels
    if labels is None:
        classes = getattr(model, "classes", None)
        if classes is None:
            raise ValueError(
                "the model declares neither `labels` nor `classes`, so "
                "neither the number nor the names of its outputs can be "
                "known: set `model.labels` on the weight set"
            )
        return list(range(classes))
    if callable(labels):
        raise TypeError(
            "the model's `labels` is a callable, which SeisBench resolves "
            "from the identity of the trace it annotates; this atom "
            "annotates arrays, so set `model.labels` to the list of "
            "output names instead"
        )
    return list(labels)


def model_pick_labels(model):
    """
    List the labels of *model* that are picked, as SeisBench picks them.

    A model may declare which of its outputs are phases in a ``phases``
    attribute — beware the name, SeisBench's ``model.phases`` is this *subset*
    while :attr:`Annotate.phases` is the full label list. ``classify_aggregate``
    walks exactly that attribute, so ``EQTransformer``'s detection trace and
    ``EQTP``'s polarities are left out of the picking the way noise is: still
    emitted in the characteristic function, never triggered on. A model
    declaring no subset gets all its labels but :data:`NOISE_LABEL`.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        The model whose weight set is read.

    Returns
    -------
    list
        One label per picked class.

    Examples
    --------
    >>> from xdas.atoms.ml import model_pick_labels

    >>> class PhaseNet:
    ...     labels = "NPS"
    >>> model_pick_labels(PhaseNet())
    ['P', 'S']

    >>> class EQTransformer(PhaseNet):
    ...     labels, phases = ("Detection", "P", "S"), "PS"
    >>> model_pick_labels(EQTransformer())
    ['P', 'S']
    """
    phases = getattr(model, "phases", None)
    if phases is None:
        return [label for label in model_phases(model) if label != NOISE_LABEL]
    return list(phases)


def _model_thresholds(model, **annotate_kwargs):
    """
    Build the per-phase detection thresholds a weight set declares.

    One entry per picked label, as :func:`model_pick_labels` resolves them —
    SeisBench's ``classify_aggregate`` picks no others, and leaving them out of
    the mapping is what stops :class:`~xdas.atoms.Trigger` triggering on them
    while :class:`Annotate` keeps emitting them. Each threshold is looked up as
    SeisBench does: what the call passes wins, else what the weight set
    declares in ``default_args[f"{label}_threshold"]``, else the model's own
    documented default for that key, else the ``*_threshold`` catch-all, else
    :data:`DEFAULT_THRESHOLD`.

    Values are passed through faithfully, including thresholds above one —
    PhaseNet's ``iquique`` declares ``P_threshold = 1.12``, which simply never
    fires. That is the weight set's own metadata, not ours to clamp.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        The model whose weight set is read.
    **annotate_kwargs
        SeisBench annotate arguments overriding the weight set's, including
        the ``f"{label}_threshold"`` keys themselves.

    Returns
    -------
    dict
        One threshold per picked label, keyed on the label.

    Examples
    --------
    >>> from xdas.atoms.ml import _model_thresholds

    A weight set declaring nothing falls back to 0.3 per phase, and the noise
    class gets no entry however the labels are ordered:

    >>> class Plain:
    ...     labels, default_args = "NPS", {}
    >>> _model_thresholds(Plain())
    {'P': 0.3, 'S': 0.3}

    What the weight set declares wins, and what the call passes wins over that:

    >>> class Geofon(Plain):
    ...     labels = "PSN"
    ...     default_args = {"P_threshold": 0.57, "S_threshold": 0.073}
    >>> _model_thresholds(Geofon())
    {'P': 0.57, 'S': 0.073}
    >>> _model_thresholds(Geofon(), S_threshold=0.5)
    {'P': 0.57, 'S': 0.5}

    A detection trace is no more picked than noise is:

    >>> class EQTransformer(Plain):
    ...     labels, phases = ("Detection", "P", "S"), "PS"
    >>> _model_thresholds(EQTransformer())
    {'P': 0.3, 'S': 0.3}
    """
    argdict = dict(model.default_args) | annotate_kwargs
    annotate_args = getattr(model, "_annotate_args", {})
    fallback = annotate_args.get("*_threshold", (None, DEFAULT_THRESHOLD))[1]
    thresholds = {}
    for label in model_pick_labels(model):
        key = f"{label}_threshold"
        if key in argdict:
            value = argdict[key]
        elif key in annotate_args:
            value = annotate_args[key][1]
        else:
            value = fallback
        thresholds[label] = float(value)
    return thresholds


def resolve_sample_dim(da, dim):
    """
    Resolve *dim* against *da*, accepting the ``first``/``last`` aliases.

    Parameters
    ----------
    da : DataArray
        The input whose dimensions *dim* is resolved against.
    dim : str
        A dimension name, or ``"first"`` or ``"last"``.

    Returns
    -------
    str
        The name of the resolved dimension.
    """
    if dim == "first":
        return da.dims[0]
    if dim == "last":
        return da.dims[-1]
    if dim not in da.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input (got {da.dims})")
    return dim


def component_labels(da, dim):
    """
    Return the string labels of *dim*, or ``None`` if it carries none.

    Labels are read as text only: a numeric coordinate is excluded by dtype
    rather than by failing to match, without which an integer identifier axis
    of ``[1, 2]`` would resolve cleanly against ``Z12H`` weights.

    Parameters
    ----------
    da : DataArray
        The input carrying the coordinate.
    dim : str
        Name of the dimension to read.

    Returns
    -------
    list of str or None
        The labels, or ``None`` when *dim* has no string coordinate.
    """
    if dim not in da.coords:
        return None
    coord = da.coords[dim]
    if coord.dtype.kind not in "SU":
        return None
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in coord.values
    ]


def match_components(labels, order, flexible=True):
    """
    Map *labels* onto model input slots, or ``None`` if they are not components.

    Parameters
    ----------
    labels : list of str
        The labels to match, each ending with a component letter.
    order : str
        The model's ``component_order``, e.g. ``"ENZ"`` or ``"Z12H"``.
    flexible : bool, optional
        Whether ``1``/``N`` and ``2``/``E`` are interchangeable, as SeisBench's
        ``flexible_horizontal_components`` makes them by default.

    Returns
    -------
    list of int or None
        One slot index per label, or ``None`` if any label names no component.
    """
    order = list(order)
    slots = []
    for label in labels:
        letter = label[-1:]
        if letter not in order and flexible:
            letter = FLEXIBLE_HORIZONTALS.get(letter, letter)
        if letter not in order:
            return None
        slots.append(order.index(letter))
    return slots


def component_slots(da, dim, order, flexible=True):
    """
    Resolve the labels of *dim* into distinct model slots, or ``None``.

    Parameters
    ----------
    da : DataArray
        The input carrying the coordinate.
    dim : str
        Name of the candidate component dimension.
    order : str
        The model's ``component_order``.
    flexible : bool, optional
        Whether horizontal components are matched flexibly.

    Returns
    -------
    list of int or None
        One slot index per label, or ``None`` when *dim* is not a component
        dimension.
    """
    labels = component_labels(da, dim)
    if labels is None:
        return None
    slots = match_components(labels, order, flexible)
    if slots is None:
        return None
    if len(set(slots)) != len(slots):
        raise ValueError(
            f"the {dim!r} dimension repeats component orientations "
            f"({labels}): it mixes several instruments, split them first"
        )
    return slots


def is_channel_code(key):
    """
    Whether *key* has the shape of a channel code.

    True for a three-character SEED code (band, instrument, orientation) and
    for a bare orientation letter, false for anything longer, for a purely
    numeric key, and for a non-string one.

    Parameters
    ----------
    key : object
        A key of a collection level.

    Returns
    -------
    bool
        Whether *key* may be read as a channel code.

    Examples
    --------
    >>> from xdas.atoms.ml import is_channel_code
    >>> [is_channel_code(key) for key in ("SHZ", "Z", "STA1", "S-Z", "2", 2)]
    [True, True, False, False, False, False]
    """
    return (
        isinstance(key, str)
        and len(key) in CHANNEL_CODE_LENGTHS
        and key.isalnum()
        and not key.isdigit()
    )


def component_keys(keys, order, flexible=True, level=None):
    """
    Resolve the keys of a collection level into distinct model input slots.

    The key rule of :meth:`Annotate.gather`, and deliberately stricter than
    :func:`match_components`, which is what recognises a component
    *dimension*. The flexible horizontal matching makes any label ending in
    ``1`` or ``2`` a horizontal, so a station level keyed ``STA1``/``STA2``
    resolves cleanly against ``ZNE`` weights. On a dimension that is harmless,
    since the duplicate and count checks catch it; on a tree *level* it would
    silently fold two stations into one instrument. So a key must look like a
    whole channel code — :func:`is_channel_code` — and the keys must agree on
    their length and on their band code, which is the character before the
    instrument code. Only the instrument code may vary, because it does: an
    OBS station is ``BHZ``/``BH1``/``BH2``/``BDH``, whose hydrophone is a
    pressure instrument and whose stem is therefore *not* common.

    Three outcomes, and the distinction between the last two is the point:

    - every key resolves to a distinct slot: the level is the component
      dimension, and the slots are returned;
    - no key resolves: the level is not a component level despite its name,
      and ``None`` is returned so the caller walks its leaves — several DAS
      formats call their spatial axis ``channel``;
    - some keys resolve: someone meant components and the data disagrees, so
      this raises, naming the conflict.

    Parameters
    ----------
    keys : iterable
        The keys of the collection level.
    order : str
        The model's ``component_order``, e.g. ``"ENZ"`` or ``"Z12H"``.
    flexible : bool, optional
        Whether ``1``/``N`` and ``2``/``E`` are interchangeable, as SeisBench's
        ``flexible_horizontal_components`` makes them by default.
    level : str, optional
        Name of the level, used in the error messages only.

    Returns
    -------
    list of int or None
        One slot index per key, or ``None`` when no key names a component.

    Raises
    ------
    ValueError
        If only some of the keys name a component, if two of them name the
        same one, or if they do not agree on their length and band code.

    Examples
    --------
    >>> from xdas.atoms.ml import component_keys

    A station's three components resolve, in the model's own order:

    >>> component_keys(["SHZ", "SHN", "SHE"], "ENZ")
    [2, 1, 0]

    A DAS cable whose spatial level happens to be called ``channel`` does not,
    which is what sends the caller back to walking the leaves:

    >>> component_keys(["0", "1", "2"], "ZNE") is None
    True
    """
    keys = list(keys)
    resolved = {}
    unresolved = []
    for key in keys:
        slots = (
            match_components([key], order, flexible) if is_channel_code(key) else None
        )
        if slots is None:
            unresolved.append(key)
        else:
            resolved[key] = slots[0]
    where = "the level" if level is None else f"the {level!r} level"
    if not resolved:
        return None
    if unresolved:
        raise ValueError(
            f"{where} names components ({sorted(resolved)}) and other things "
            f"({unresolved}) at once: it cannot be collapsed into a component "
            f"dimension of {order!r}, split it or rename it"
        )
    slots = list(resolved.values())
    if len(set(slots)) != len(slots):
        raise ValueError(
            f"{where} repeats component orientations ({keys}): it holds "
            "several instruments, split them first"
        )
    if len({len(key) for key in keys}) > 1 or len({key[:-1][:1] for key in keys}) > 1:
        raise ValueError(
            f"{where} does not name one instrument ({keys}): its keys differ "
            "by more than their orientation, split them first"
        )
    return slots


def resolve_component_dim(da, sample_dim, order, components=None, flexible=True):
    """
    Find the component dimension of *da* and the slots its labels name.

    Detection is by labels, never by name: the component dimension is the one
    whose labels each end with a distinct letter of *order*.

    Parameters
    ----------
    da : DataArray
        The input to inspect.
    sample_dim : str
        The already-resolved sample dimension, never a component candidate.
    order : str
        The model's ``component_order``.
    components : str, False or None, optional
        Name of the component dimension, ``False`` to disable detection, or
        ``None`` (default) to detect it.
    flexible : bool, optional
        Whether horizontal components are matched flexibly.

    Returns
    -------
    dim : str or None
        The component dimension, or ``None`` when the data has none.
    slots : list of int or None
        The model input slots its labels name, or ``None`` with *dim*.
    """
    if components is False:
        return None, None
    if components is not None:
        if components not in da.dims:
            raise ValueError(
                f"{components!r} is not a dimension of the input (got {da.dims})"
            )
        slots = component_slots(da, components, order, flexible)
        if slots is None:
            raise ValueError(
                f"the {components!r} dimension is not labelled by components: "
                f"expected labels ending with distinct letters of {order!r}"
            )
        return components, slots
    found = {}
    for dim in da.dims:
        if dim == sample_dim:
            continue
        slots = component_slots(da, dim, order, flexible)
        if slots is not None:
            found[dim] = slots
    if not found:
        return None, None
    if len(found) > 1:
        raise ValueError(
            f"several dimensions could be the component dimension "
            f"({sorted(found)}): name it explicitly with `components=`"
        )
    return found.popitem()


class Annotate(Atom):
    """
    Wraps a SeisBench model as a streaming :class:`Atom`.

    Slides the model over the data with the overlap the weight set declares,
    stitches the per-window outputs back into one continuous characteristic
    function and appends the model's labels as a ``phase`` dimension. Every
    parameter the model can decide — the window overlap, the stacking rule, the
    normalisation, the blinding — is read off the model instance rather than
    assumed, since in SeisBench all of them belong to the *weight set* and not
    to the architecture.

    Input dimension names are never assumed: *dim* names the sample dimension
    (the repository's ``"first"`` and ``"last"`` aliases resolve too) and the
    component dimension is found by its labels, not by its name. The output
    keeps the input's order among the remaining dimensions but is laid out
    sample-last, ``(..., "phase", dim)``: the characteristic function of one
    phase of one channel is then contiguous, which is the layout its consumers
    reduce along.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        A SeisBench model in evaluation mode (will be moved to *device*).
    dim : str, optional
        Dimension the model slides along. Defaults to ``"time"``; ``"first"``
        and ``"last"`` resolve against the input.
    components : str or False, optional
        Name of the component dimension. ``None`` (default) detects it by its
        labels: the dimension whose labels each end with a distinct letter of
        the model's ``component_order``. ``False`` disables detection, which is
        the escape hatch when an identifier axis carries labels that could
        collide with component letters.

        On a collection, this also names the *level* to gather (see
        :meth:`gather`), overriding the :data:`COMPONENT_LEVELS` candidates;
        ``False`` disables the gather along with the detection.
    component_strategy : str, optional
        How the model's input slots are filled from the data:

        ``"auto"`` (default)
            Reproduces SeisBench: ``"clone"`` when the data has no component
            dimension (its DAS wrapper's default), ``"pad"`` when it has a
            partial one (its ``strict=False`` default).
        ``"clone"``
            Replicate the single available signal into every slot.
        ``"pad"``
            Single signal in the first slot, zeros in the rest.
        ``"E"``, ``"N"``, ``"Z"``, ...
            Single signal in that named slot of ``component_order``, zeros in
            the rest.
        ``"strict"``
            Any missing component is an error (SeisBench's ``strict=True``).

        With a component dimension, present components always go to the slot
        their label names and the remaining slots are zeroed.
    device : str or torch.device, optional
        Torch device. Defaults to CUDA if available, else CPU.
    tolerance : scalar, None or False, optional
        Grid-snapping budget forwarded to :func:`xdas.stack` when a component
        level is gathered (see :meth:`gather`). ``None`` (default) spends a
        hundredth of a sample, which is what lets three components whose start
        times were rounded a nanosecond apart stack; ``False`` restores strict
        equality.
    max_buffers : int, optional
        Depth of the in-flight output queue on a CUDA device, the same
        bounded-staging pattern :class:`~xdas.processing.DataArrayLoader`
        uses. Each completed window's device-to-host transfer is issued
        asynchronously (pinned, non-blocking) and :meth:`call` emits only
        the outputs whose transfer has completed — the 0..n contract already
        allows late emission — so the CPU keeps feeding the model while
        results cross back; :meth:`flush` drains whatever is still in
        flight. At most `max_buffers` transfers are left pending (default
        2); ``0`` restores fully synchronous emission. On the CPU the
        transfers complete immediately and the queue changes nothing.
    **annotate_kwargs
        SeisBench annotate arguments (``overlap``, ``stacking``, ``blinding``,
        ...) overriding what the weight set declares in ``default_args``.

    Warnings
    --------
    ``component_strategy="pad"`` is *positional*: like SeisBench it fills slot
    0 of ``component_order``, which is ``Z`` for the many ``ZNE`` weight sets
    but ``E`` for ``ENZ`` ones such as PhaseNet's ``original``. Pass the letter
    itself (``"Z"``) to name a slot rather than count to it.

    Examples
    --------
    >>> import torch
    >>> import xdas as xd
    >>> from xdas.atoms import Annotate

    Any SeisBench model works; here is a stand-in small enough to inline, with
    a four-sample window, two labels and a 50 % overlap declared by its weights:

    >>> class Model(torch.nn.Module):
    ...     in_samples, in_channels, classes = 4, 3, 2
    ...     labels, component_order = "PS", "ZNE"
    ...     default_args = {"overlap": 0.5}
    ...     def annotate_batch_pre(self, batch, argdict):
    ...         return batch
    ...     def annotate_batch_post(self, batch, piggyback, argdict):
    ...         return torch.transpose(batch, -1, -2)
    ...     def forward(self, batch):
    ...         return batch[:, : self.classes]

    >>> da = xd.testing.dummy(dims=("time", "distance"), shape=(16, 3))
    >>> atom = Annotate(Model(), dim="time", device="cpu")
    >>> result = atom(da)

    The labels become a ``phase`` dimension and the samples end up last:

    >>> result.dims
    ('distance', 'phase', 'time')
    >>> result.coords["phase"].values
    array(['P', 'S'], dtype='<U1')
    >>> result.sizes["time"] == da.sizes["time"]
    True

    Given a collection whose ``channel`` level holds one component per key,
    the atom takes that level as its component dimension rather than
    annotating each component on its own (see :meth:`gather`):

    >>> traces = {code: da.isel(distance=0) for code in ("SHZ", "SHN", "SHE")}
    >>> atom(xd.DataCollection(traces, "channel")).dims
    ('phase', 'time')
    """

    def __init__(
        self,
        model,
        dim="time",
        components=None,
        component_strategy="auto",
        device=None,
        tolerance=None,
        max_buffers=2,
        **annotate_kwargs,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        valid = {"auto", "clone", "pad", "strict", *model.component_order}
        if component_strategy not in valid:
            raise ValueError(
                f"component_strategy must be one of {sorted(valid)}, "
                f"got {component_strategy!r}"
            )
        self.device = device
        self.model = model.eval().to(self.device)
        self.dim = dim
        self.components = components
        self.component_strategy = component_strategy
        self.tolerance = tolerance
        self.max_buffers = max_buffers
        self.argdict = dict(model.default_args) | annotate_kwargs
        if self.stacking not in ("avg", "max"):
            raise ValueError(f"stacking must be 'avg' or 'max', got {self.stacking!r}")
        if not 0 <= self.noverlap < self.nperseg:
            raise ValueError(
                f"overlap must be shorter than one model window "
                f"({self.noverlap} of {self.nperseg} samples)"
            )
        self.sample_dim = State(...)
        self.component_dim = State(...)
        self.batch_dims = State(...)
        self.out_dims = State(...)
        self.slots = State(...)
        self.started = State(...)
        self.buffer = State(...)
        self.circular_input = State(...)
        self.model_input = State(...)
        self.circular_output = State(...)
        self.circular_counts = State(...)
        self.inflight = State(...)

    def _annotate_arg(self, key):
        """Read one annotate argument through :func:`annotate_arg`."""
        return annotate_arg(self.model, self.argdict, key)

    @property
    def nperseg(self):
        """Number of samples per segment (= model input length)."""
        return self.model.in_samples

    @property
    def noverlap(self):
        """Overlap between consecutive segments, in samples, as the model declares it."""
        overlap = self._annotate_arg("overlap")
        return int(overlap * self.nperseg) if overlap < 1 else int(overlap)

    @property
    def step(self):
        """Stride between the start of consecutive segments."""
        return self.nperseg - self.noverlap

    @property
    def stacking(self):
        """How overlapping windows are combined: ``"avg"`` or ``"max"``."""
        return self._annotate_arg("stacking")

    @property
    def phases(self):
        """
        List of the phase labels produced by the model.

        Read off the weight set through :func:`model_phases`, which is also
        what :func:`_model_thresholds` keys on, so the ``phase`` coordinate and
        the thresholds of a pipeline agree by construction.
        """
        return model_phases(self.model)

    @property
    def in_channels(self):
        """Number of input channels the model expects."""
        return getattr(self.model, "in_channels", len(self.model.component_order))

    @property
    def classes(self):
        """
        Number of output classes the model produces.

        Read off the labels rather than ``model.classes``, which counts only
        what the architecture's *picking* head emits: ``EQTransformer`` sets it
        to 2 while labelling three outputs, the third being its detection
        trace.
        """
        return len(self.phases)

    @property
    def fill(self):
        """Neutral element of the stacking rule, used to reset freed samples."""
        return 0.0 if self.stacking == "avg" else -np.inf

    def gather(self, mapping):
        """
        Collapse a component level of a collection into a component dimension.

        So that annotating a collection needs no prior :func:`xdas.stack`: an
        ObsPy-style tree keeps one leaf per channel, but three channels of one
        instrument are one input to the model, not three. What counts as a
        component is a property of the *model* — ``Z12H`` and ``ENZ`` disagree
        about which channels group together — which is why the reader cannot
        do this and this atom can.

        The gather is deliberately conservative, because folding a station
        level into a component axis would silently destroy the distinction
        between stations. It fires only when **both** the level's name is one
        that *components* accepts (:data:`COMPONENT_LEVELS` by default) and
        its keys resolve to distinct components under
        :func:`component_keys`. A level whose keys resolve but whose name is
        not a candidate is left alone; a level whose name is a candidate but
        whose keys resolve to nothing is left alone too, which is what lets a
        DAS collection whose spatial level is called ``channel`` walk
        straight past. Only a level that is clearly component-ish and
        malformed raises.

        This is the one place a *name* is consulted: detection on the array
        side is purely label-based. The asymmetry is deliberate — a
        mis-detected dimension is caught at once by the count and letter
        checks, a mis-collapsed level is not.

        Parameters
        ----------
        mapping : DataMapping
            The collection level about to be walked.

        Returns
        -------
        DataArray, DataCollection or None
            The level collapsed onto a dimension named after it, or ``None``
            to walk its leaves.
        """
        if self.components is False:
            return None
        explicit = isinstance(self.components, str)
        candidates = (self.components,) if explicit else COMPONENT_LEVELS
        level = getattr(mapping, "name", None)
        if level not in candidates:
            return None
        keys = list(mapping)
        slots = component_keys(
            keys,
            self.model.component_order,
            self._annotate_arg("flexible_horizontal_components"),
            level,
        )
        if slots is None:
            if explicit:
                raise ValueError(
                    f"could not identify the component level of the collection: "
                    f"tried {list(candidates)}, and the keys of {level!r} "
                    f"({keys}) name no component of "
                    f"{self.model.component_order!r}. Please provide it "
                    "explicitly with `components=`, or pass `components=False` "
                    "to walk the collection leaf by leaf"
                )
            return None
        return stack(mapping, level, tolerance=self.tolerance)

    def initialize(self, da, chunk_dim=None, **flags):
        """Resolve the dimensions and allocate the sliding-window buffers."""
        dim = resolve_sample_dim(da, self.dim)
        component_dim, slots = resolve_component_dim(
            da,
            dim,
            self.model.component_order,
            self.components,
            self._annotate_arg("flexible_horizontal_components"),
        )
        self.sample_dim = State(dim)
        self.component_dim = State(component_dim)
        self.slots = State(self._resolve_slots(slots))
        self.batch_dims = State(
            tuple(other for other in da.dims if other not in (dim, component_dim))
        )
        self.out_dims = State(self.batch_dims + ("phase", dim))
        batch = int(np.prod([da.sizes[other] for other in self.batch_dims], dtype=int))
        ncomp = 1 if component_dim is None else da.sizes[component_dim]
        self.circular_input = State(self._zeros(batch, ncomp, self.nperseg))
        self.model_input = State(self._zeros(batch, self.in_channels, self.nperseg))
        self.circular_output = State(
            torch.full(
                (batch, self.classes, self.nperseg + self.step),
                self.fill,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.circular_counts = State(
            torch.zeros(
                batch,
                self.classes,
                self.nperseg + self.step,
                dtype=torch.int32,
                device=self.device,
            )
        )
        self.started = State(False)
        self.buffer = State(da.isel({dim: slice(0, 0)}) if chunk_dim == dim else None)
        self.inflight = State(deque())

    def _zeros(self, *shape):
        """Allocate a zeroed float32 tensor on the atom's device."""
        return torch.zeros(*shape, dtype=torch.float32, device=self.device)

    def call(self, da, **flags):
        """
        Run the model over *da*, managing a carry-over buffer for chunked input.

        A window is annotated as soon as it is complete, but its samples are
        only emitted once the *following* window has been annotated: the
        end-aligned last window of a record may reach back into them, and
        :meth:`flush` is where that is settled. A chunk that completes no new
        window therefore produces no output at all. On a CUDA device the
        emission is one step later still: each output's device-to-host
        transfer is issued asynchronously and only the completed ones are
        returned (see `max_buffers`), the rest following on later calls or
        at :meth:`flush`.

        Chunked along a dimension other than `dim`, that carry-over would be
        a leak: the next chunk holds *other* lanes, so nothing of this one —
        neither the tail buffer nor the circular window — applies to it. Such
        a chunk is a whole record on its own and is run from a fresh state
        and settled here.
        """
        chunk_dim = flags.get("chunk_dim")
        if chunk_dim is not None and chunk_dim != self.sample_dim:
            return self._call_independent(da, **flags)
        dim = self.sample_dim
        if self.buffer is None:
            if da.sizes[dim] < self.nperseg:
                raise ValueError(
                    f"the record is shorter along {dim!r} "
                    f"({da.sizes[dim]} samples) than one model window "
                    f"({self.nperseg} samples)"
                )
            self._process(da)
        else:
            da = concat([self.buffer, da], dim)
            if da.sizes[dim] < self.nperseg:
                self.buffer = State(da)
            else:
                self._process(da)
        return self._harvest() or None

    def _call_independent(self, da, **flags):
        """Annotate one whole record, settled on the spot, leaving no state behind."""
        dim = self.sample_dim
        if da.sizes[dim] < self.nperseg:
            raise ValueError(
                f"the record is shorter along {dim!r} "
                f"({da.sizes[dim]} samples) than one model window "
                f"({self.nperseg} samples)"
            )
        # Reallocate: the lanes of this chunk are not the lanes the buffers
        # were sized and filled for, and the last chunk may hold fewer.
        self.initialize(da, **flags)
        if flags.get("chunk_dim") == self.component_dim:
            # Not a lane axis: the model reads every component of a window at
            # once, so a chunk holding a subset of them is not a record.
            raise ValueError(
                f"{self.component_dim!r} is the component dimension of the "
                "model and cannot be chunked: the components of one window "
                "are read together"
            )
        self._process(da)
        chunks = self._harvest() + self.flush()
        # One record in, one record out: the pieces are consecutive along
        # `dim`, and downstream is joining along the *chunked* dimension.
        return concat(chunks, dim) if len(chunks) > 1 else chunks[0]

    def flush(self):
        """
        Emit the end-aligned final window and everything it completes.

        SeisBench appends one last window ending on the record's last sample
        whenever the stride leaves a remainder, so the output spans the input.
        Firing once per run, this stays chunk-invariant. The in-flight
        output queue is drained here, so nothing survives the end of a run.
        """
        dim = self.sample_dim
        if self.started is not True:
            buffered = self.buffer
            if isinstance(buffered, DataArray) and buffered.sizes[dim] > 0:
                # The whole stream was shorter than one window: say so, as
                # the eager call does, rather than answering with nothing.
                raise ValueError(
                    f"the record is shorter along {dim!r} "
                    f"({buffered.sizes[dim]} samples) than one model window "
                    f"({self.nperseg} samples)"
                )
            return self._harvest(block=True)
        buffer = self.buffer
        remainder = buffer.sizes[dim] - self.nperseg
        if remainder > 0:
            self._advance(buffer.isel({dim: slice(-remainder, None)}), remainder)
        self._emit(buffer, 0, self.step - remainder, self.nperseg + remainder)
        self.buffer = State(buffer.isel({dim: slice(0, 0)}))
        self.started = State(False)
        self.circular_output.fill_(self.fill)
        self.circular_counts.fill_(0)
        return self._harvest(block=True)

    def _process(self, da):
        """Annotate every window of *da* that the buffered state can complete."""
        dim = self.sample_dim
        nperseg, step = self.nperseg, self.step
        if self.started:
            first = step  # the window at 0 was annotated by the previous call
        else:
            self._prime(da)
            first = 0
        last = None
        for idx in range(first, da.sizes[dim] - nperseg + 1, step):
            tail = da.isel({dim: slice(idx + nperseg - step, idx + nperseg)})
            self._advance(tail, step)
            if idx > 0:  # the first window of a run completes nothing yet
                self._emit(da, idx - step, 0, step)
            last = idx
        if last is None:
            self.buffer = State(da)
            return
        self.started = State(True)
        self.buffer = State(da.isel({dim: slice(last, None)}))

    def _prime(self, da):
        """Stage the head of the first window so that the first slide completes it."""
        head = da.isel({self.sample_dim: slice(0, self.noverlap)})
        self.circular_input.narrow(-1, self.step, self.noverlap).copy_(
            self._to_device(head)
        )

    def _advance(self, tail, shift):
        """Slide the window by *shift* samples, run the model and stack its output."""
        self._slide(self.circular_input, shift, -1).copy_(self._to_device(tail))
        self._slide(self.circular_output, shift, -1).fill_(self.fill)
        self._slide(self.circular_counts, shift, -1).fill_(0)
        self._fill_model_input()
        with torch.inference_mode():
            # `annotate_batch_post` writes into its input, which is an inference
            # tensor: outside this block that raises.
            batch = self.model.annotate_batch_pre(self.model_input, self.argdict)
            piggyback = None
            if isinstance(batch, tuple):
                batch, piggyback = batch
            out = self.model(batch)
            out = self.model.annotate_batch_post(
                out, piggyback=piggyback, argdict=self.argdict
            )
        self._accumulate(out)

    def _slide(self, buffer, shift, axis):
        """Slide *buffer* left by *shift* samples, returning the freed tail view."""
        size = buffer.shape[axis]
        kept = buffer.narrow(axis, shift, size - shift).clone()
        buffer.narrow(axis, 0, size - shift).copy_(kept)
        return buffer.narrow(axis, size - shift, shift)

    def _fill_model_input(self):
        """Permute the staged components into the model's input slots."""
        if self.slots is None:
            self.model_input[:] = self.circular_input
        else:
            self.model_input[:, self.slots] = self.circular_input

    def _accumulate(self, out):
        """
        Stack one window of ``(batch, samples, classes)`` predictions.

        Transposed on arrival into the ``(batch, classes, samples)`` the buffers
        hold, which costs nothing — it is a view, and the reduction runs along
        the axis the model's own output is contiguous in either way.

        The model blinds its own output by writing NaN into it, so the sum
        ignores NaN and the count tracks what was finite: dividing the two
        reproduces SeisBench's ``nanmean`` over covering windows exactly, and
        ``stacking="max"`` its ``nanmax``.

        Raises
        ------
        ValueError
            If the post-processed batch is not ``(..., in_samples, classes)``.
            Shipped SeisBench models do land here — ``CRED`` on both counts,
            since it keeps the ``WaveformModel`` default
            ``annotate_batch_post``, which leaves the batch as
            ``(batch, classes, samples)`` rather than transposing it as
            ``PhaseNet`` does, *and* predicts 19 samples for a 3000-sample
            window. Without this check either mistake surfaces as a bare
            ``RuntimeError`` from the broadcast.
        """
        expected = (self.nperseg, self.classes)
        if tuple(out.shape[-2:]) != expected:
            raise ValueError(
                f"{type(self.model).__name__}.annotate_batch_post returned a "
                f"batch ending in {tuple(out.shape[-2:])}, not "
                f"{expected} = (in_samples, classes): this atom adopts "
                "SeisBench's stacking contract, `(batch, samples, classes)`. "
                "A model leaving the batch as `(batch, classes, samples)` — "
                "the `WaveformModel` default, which `PhaseNet` overrides — "
                "must transpose it; a model whose window prediction is not "
                "one sample per input sample cannot be stacked this way."
            )
        out = torch.transpose(out, -1, -2)
        window = self.circular_output.narrow(-1, self.step, self.nperseg)
        counts = self.circular_counts.narrow(-1, self.step, self.nperseg)
        finite = torch.isfinite(out)
        values = torch.nan_to_num(out, nan=self.fill)
        if self.stacking == "max":
            torch.maximum(window, values, out=window)
        else:
            window += values
        counts += finite.to(counts.dtype)

    def _pull(self, start, length):
        """
        Reduce *length* stacked samples from *start* into a fresh tensor.

        Fresh so that the device-to-host transfer of the result can stay in
        flight while the circular buffers slide on: the reduction allocates,
        it never views the stack.
        """
        values = self.circular_output.narrow(-1, start, length)
        counts = self.circular_counts.narrow(-1, start, length)
        if self.stacking == "max":
            return torch.where(counts > 0, values, float("nan"))
        return values / counts  # samples no window covered come out NaN

    def _to_device(self, chunk):
        """Stage *chunk* as a float32 tensor on the device, async on CUDA."""
        dims = self.batch_dims
        if self.component_dim is not None:
            dims += (self.component_dim,)
        chunk = chunk.transpose(*dims, self.sample_dim)
        values = np.ascontiguousarray(chunk.values, dtype=np.float32)
        batch, ncomp = self.circular_input.shape[:2]
        values = values.reshape(batch, ncomp, chunk.sizes[self.sample_dim])
        data = torch.from_numpy(values)
        if self.device.type == "cuda":  # pragma: no cover
            # A pinned staging copy lets the transfer overlap with compute.
            return data.pin_memory().to(self.device, non_blocking=True)
        return data

    def _emit(self, da, offset, start, length):
        """Queue the output chunk of *length* samples found at *start* in the stack."""
        dim = self.sample_dim
        data = self._pull(start, length)
        # Slice the whole chunk, not just the dimension coordinate: a label
        # attached to the samples (a tag, a pick id) has to follow them, or
        # it keeps the input length and is silently dropped on assembly.
        coords = da.isel({dim: slice(offset, offset + length)}).coords.copy()
        if self.component_dim is not None:
            coords = coords.drop_dims(self.component_dim)
        coords["phase"] = self.phases
        shape = tuple(da.sizes[other] for other in self.batch_dims)
        shape = (*shape, self.classes, length)
        self._submit(data, (shape, coords, self.out_dims, da.name, da.attrs))

    def _submit(self, data, meta):
        """
        Queue one emitted output, its device-to-host transfer in flight.

        On CUDA the transfer is issued into a pinned staging buffer,
        non-blocking, with an event marking its completion; the source
        tensor rides along in the queue so it outlives the copy. At most
        `max_buffers` transfers are left pending — the older ones are
        synchronized — which is what bounds the staging memory. On the CPU
        there is nothing to wait for and the item is complete on arrival.
        """
        if self.device.type == "cuda":  # pragma: no cover
            values = torch.empty(data.shape, dtype=data.dtype, pin_memory=True)
            values.copy_(data, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            values, event = data, None
        self.inflight.append((event, values, data, meta))
        excess = len(self.inflight) - self.max_buffers
        if excess > 0:
            event = self.inflight[excess - 1][0]
            if event is not None:  # pragma: no cover
                event.synchronize()

    def _harvest(self, block=False):
        """
        Emit the queued outputs whose transfer has completed, in order.

        The queue is strictly ordered, so the harvest stops at the first
        transfer still in flight; *block* waits every transfer out instead,
        which is what :meth:`flush` does to drain the run.
        """
        if self.inflight is ...:
            return []
        chunks = []
        while self.inflight:
            event = self.inflight[0][0]
            if event is not None and not block and not event.query():
                break  # pragma: no cover
            chunks.append(self._realize(self.inflight.popleft()))
        return chunks

    def _realize(self, item):
        """Build the output chunk of one completed transfer."""
        event, values, _, (shape, coords, dims, name, attrs) = item
        if event is not None:  # pragma: no cover
            event.synchronize()
        return DataArray(values.numpy().reshape(shape), coords, dims, name, attrs)

    def _resolve_slots(self, slots):
        """Turn ``component_strategy`` into the input slots the data fills."""
        order = list(self.model.component_order)
        strategy = self.component_strategy
        if slots is None:
            if strategy in ("auto", "clone"):
                return None  # SeisBench's DAS default: clone into every slot
            if strategy == "strict":
                raise ValueError(
                    "component_strategy='strict' needs a component dimension, "
                    "and none was found: name it with `components=`"
                )
            return [0 if strategy == "pad" else order.index(strategy)]
        if strategy == "strict" and len(slots) < self.in_channels:
            raise ValueError(
                f"the data fills {len(slots)} of the model's {self.in_channels} "
                "input slots and component_strategy is 'strict'"
            )
        if strategy == "clone" or strategy in order:
            if len(slots) > 1:
                raise ValueError(
                    f"component_strategy={strategy!r} needs a single component "
                    f"per lane, but the data has {len(slots)}"
                )
            return None if strategy == "clone" else [order.index(strategy)]
        return slots


class _ChannelFilter(Filter):
    """
    Filter only the channels whose labels match a glob pattern.

    The subset form of :class:`~xdas.atoms.Filter`, whose parameters it takes,
    for the per-channel preprocessing filters some SeisBench weight sets ship:
    ``obs`` declares a 0.5 Hz highpass on ``"??H"``, i.e. on its hydrophone
    alone. The channels the pattern does not match pass through untouched, and a
    pattern matching none of them makes the whole atom a no-op — which is what
    SeisBench's ``stream.select(channel=...)`` does with a pattern no trace
    answers.

    The channel dimension is found exactly as :class:`Annotate` finds it: by
    its labels, each ending with a distinct letter of the model's
    ``component_order``, never by its name.

    Parameters
    ----------
    pattern : str
        Channel glob, matched against the whole label with :mod:`fnmatch`.
    freq : tuple of float or None
        Corner frequencies ``(low, high)`` in Hz, as :class:`~xdas.atoms.Filter`
        takes them: ``(0.5, None)`` is a highpass.
    components : str, False or None, optional
        Name of the channel dimension. ``None`` (default) detects it by its
        labels, ``False`` disables detection, which makes the atom a no-op.
    component_order : str, optional
        The model's ``component_order``, against which the channel dimension is
        recognised. Defaults to ``"ZNE"``.
    flexible : bool, optional
        Whether ``1``/``N`` and ``2``/``E`` are interchangeable while
        recognising the channel dimension, as SeisBench's
        ``flexible_horizontal_components`` makes them by default.
    **kwargs
        Passed to :class:`~xdas.atoms.Filter`: ``order``, ``zerophase`` and
        ``dim``. The default order of 4 is also ``obspy``'s ``corners``
        default, which is what makes a model-declared filter reproducible
        exactly. ``ftype="fir"`` is refused: the FIR form compensates its group
        delay by shifting the coordinate, which would leave the channels this
        atom does *not* filter on somebody else's samples.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd
    >>> from xdas.atoms.ml import _ChannelFilter

    A four-channel OBS station, with the hydrophone last:

    >>> da = xd.testing.dummy(dims=("time", "channel"), shape=(1000, 4))
    >>> da["channel"] = np.array(["BHZ", "BH1", "BH2", "BDH"])
    >>> atom = _ChannelFilter("??H", (0.5, None), component_order="Z12H")
    >>> result = atom(da)

    The three seismometer channels are untouched, the hydrophone is not:

    >>> np.allclose(result.isel(channel=slice(0, 3)).values, da.values[:, :3])
    True
    >>> np.allclose(result.isel(channel=3).values, da.values[:, 3])
    False
    """

    def __init__(
        self,
        pattern,
        freq,
        components=None,
        component_order="ZNE",
        flexible=True,
        **kwargs,
    ):
        super().__init__(freq, **kwargs)
        if self.ftype == "fir":
            raise ValueError(
                "`ftype='fir'` cannot filter a subset of the channels: it "
                "compensates its group delay by shifting the coordinate, and "
                "the untouched channels would then sit on the wrong samples"
            )
        self.pattern = pattern
        self.components = components
        self.component_order = component_order
        self.flexible = flexible
        self.mask = State(...)

    def initialize(self, da, **flags):
        """Find the channel dimension and build the mask the pattern selects."""
        super().initialize(da, **flags)
        channel_dim, _ = resolve_component_dim(
            da,
            resolve_sample_dim(da, self.dim),
            self.component_order,
            self.components,
            self.flexible,
        )
        labels = () if channel_dim is None else component_labels(da, channel_dim)
        selected = [fnmatch(label, self.pattern) for label in labels]
        shape = [-1 if dim == channel_dim else 1 for dim in da.dims]
        self.mask = State(np.reshape(selected, shape) if any(selected) else None)

    def call(self, da, **flags):
        """Filter every channel, then restore the ones the pattern left out."""
        if self.mask is None:
            return da
        # Filtering all of them and dropping what is not wanted keeps the
        # inherited filter state one shape and costs a handful of channels.
        filtered = super().call(da, **flags)
        values = np.where(self.mask, filtered.values, da.values)
        return DataArray(values, da.coords, da.dims, da.name, da.attrs)


def _model_filter(model, dim="time", components=None, **annotate_kwargs):
    """
    Build the preprocessing filter stage a weight set declares, if any.

    SeisBench's ``annotate_stream_pre`` filters the waveforms *before*
    resampling them, with whatever the weight set puts in ``filter_args`` and
    ``filter_kwargs``; skipping it feeds the network something it was not
    trained on. This is not a user-facing bandpass — compose
    :class:`~xdas.atoms.Filter` yourself for that — and most weight sets
    declare none, which is why the stage is optional.

    Two declarations are understood, both spelled as ``obspy``'s
    ``Stream.filter`` arguments:

    - *flat*, e.g. ``filter_args=("highpass",)`` with
      ``filter_kwargs={"freq": 1}``, applied to everything;
    - *per channel*, a dict from channel glob to arguments with
      ``filter_kwargs`` keyed identically, e.g. ``{"??H": ("highpass",)}``,
      which becomes one :class:`_ChannelFilter` per pattern, applied in
      declaration order. SeisBench's own DAS wrapper refuses this form; here it
      works, through the label matching :class:`Annotate` already does.

    The translation is exact rather than approximate: ``obspy`` filters with
    ``corners=4``, ``zerophase=False`` and a Butterworth in second-order
    sections, which is what :class:`~xdas.atoms.Filter` defaults to. Two
    concessions are copied from SeisBench's ``_get_filter_args``: a zero-phase
    declaration warns and doubles the order instead, since exact zero-phase IIR
    filtering has no causal streaming form, and no corner is allowed above half
    the Nyquist of the model's own sampling rate.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        The model whose weight set is read.
    dim : str, optional
        Dimension the filter runs along. Defaults to ``"time"``.
    components : str, False or None, optional
        Name of the channel dimension, for the per-channel form. ``None``
        (default) detects it by its labels.
    **annotate_kwargs
        SeisBench annotate arguments overriding the weight set's, of which only
        ``flexible_horizontal_components`` is read here.

    Returns
    -------
    Atom or None
        The stage to run before resampling, or ``None`` when the weight set
        declares no filter — in which case the pipeline is one stage shorter.

    Examples
    --------
    >>> from xdas.atoms.ml import _model_filter

    A weight set declaring nothing gets no stage:

    >>> class Plain:
    ...     component_order, sampling_rate = "ZNE", 100
    ...     default_args, filter_args, filter_kwargs = {}, None, None
    >>> _model_filter(Plain()) is None
    True

    The per-channel form of the ``obs`` weight set:

    >>> class OBS(Plain):
    ...     component_order = "Z12H"
    ...     filter_args = {"??H": ("highpass",)}
    ...     filter_kwargs = {"??H": {"freq": 0.5}}
    >>> stage = _model_filter(OBS())
    >>> stage.pattern, stage.freq, stage.order
    ('??H', (0.5, None), 4)
    """
    filter_args = getattr(model, "filter_args", None)
    if filter_args is None:
        return None
    filter_kwargs = getattr(model, "filter_kwargs", None)
    sampling_rate = getattr(model, "sampling_rate", None)
    if not isinstance(filter_args, dict):
        freq, order = _translate_filter(filter_args, filter_kwargs, sampling_rate)
        return Filter(freq, order=order, dim=dim)
    argdict = dict(model.default_args) | annotate_kwargs
    flexible = annotate_arg(model, argdict, "flexible_horizontal_components")
    stages = []
    for pattern, args in filter_args.items():
        if not isinstance(filter_kwargs, dict) or pattern not in filter_kwargs:
            raise ValueError(
                f"the weight set declares a filter for the channels matching "
                f"{pattern!r} in `filter_args` but not in `filter_kwargs`"
            )
        freq, order = _translate_filter(args, filter_kwargs[pattern], sampling_rate)
        stages.append(
            _ChannelFilter(
                pattern,
                freq,
                order=order,
                dim=dim,
                components=components,
                component_order=model.component_order,
                flexible=flexible,
            )
        )
    if not stages:
        return None
    return stages[0] if len(stages) == 1 else Sequential(stages)


def _translate_filter(args, kwargs, sampling_rate=None):
    """
    Translate one ``obspy`` filter declaration into `Filter` parameters.

    Returns the ``(low, high)`` corner pair and the filter order, applying
    SeisBench's two concessions: zero-phase becomes a doubled order, and no
    corner sits above half the Nyquist of *sampling_rate*.
    """
    args = tuple(args)
    if len(args) != 1:
        raise ValueError(
            "a weight set's filter declaration must name exactly one obspy "
            f"filter type, got {args!r}"
        )
    name = args[0]
    kwargs = dict(kwargs or {})
    order = kwargs.get("corners", OBSPY_CORNERS)
    if kwargs.get("zerophase", False):
        warnings.warn(
            f"the weight set declares a zero-phase {name} filter, which has no "
            f"causal streaming form: filtering forward only with the order "
            f"doubled ({order} -> {2 * order}), as SeisBench does",
            UserWarning,
            stacklevel=3,
        )
        order *= 2
    # As SeisBench does: no corner frequency may sit above half the Nyquist,
    # so that the filter stays valid on data at a legal but lower rate.
    top = np.inf if not sampling_rate else 0.999999 * 0.25 * sampling_rate
    match name:
        case "highpass":
            return (min(kwargs["freq"], top), None), order
        case "lowpass":
            return (None, min(kwargs["freq"], top)), order
        case "bandpass":
            return (kwargs["freqmin"], min(kwargs["freqmax"], top)), order
        case _:
            raise ValueError(
                f"the weight set declares a {name!r} filter, which has no "
                "`Filter` equivalent: pass the stage yourself, or drop it"
            )


class Picker(Sequential):
    """
    Pick phases with a SeisBench model: waveforms in, one pick table out.

    The whole pipeline SeisBench's ``model.classify(stream)`` runs, assembled
    from the weight set and nothing else::

        model-declared filter  ->  Resample  ->  Annotate  ->  Trigger

    Every stage is configured from the *weight set*, so two pickers built on
    one model class can differ in stage count, sampling rate and thresholds::

        Picker(PhaseNet("original"))       Picker(PhaseNet("obs"))
          Resample(100.0)                    _ChannelFilter('??H', 0.5 Hz)
          Annotate                           Resample(100.0)
          Trigger({'P': 0.3, 'S': 0.3})      Annotate
                                             Trigger({'P': 0.2, 'S': 0.1})

    Being a :class:`Sequential` rather than a factory function, a picker keeps
    everything a pipeline can do: ``>>`` composes it, ``repr`` shows its
    stages, it pickles, and ``picker.process(source, out=...)`` streams it. It
    also inherits the two collection hooks from the stages that define them —
    :meth:`Annotate.gather` collapses a ``channel`` level into the component
    dimension before the first stage runs, and :meth:`Trigger.merge` folds the
    per-leaf tables — so picking a whole network is one call answering with
    one table.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        The model to pick with, weights loaded.
    thresh : float, mapping or None, optional
        Trigger-on thresholds, as :class:`~xdas.atoms.Trigger` takes them.
        ``None`` (default) reads them off the weight set: one per non-noise
        label, so the noise class is never picked.
    resample : bool, optional
        Whether to resample to the model's own ``sampling_rate`` — which is
        not always 100 Hz, PhaseNet's ``diting`` running at 50. ``True`` by
        default; the stage is a no-op on data already at that rate, so it
        costs nothing to leave in. ``False`` drops it, which is what to pass
        when the data is already there or when the polyphase resampling is
        not wanted. It is the one stage that does not match SeisBench, and by
        a difference of passband rather than of accuracy: SeisBench resamples
        with obspy's ``Trace.resample``, whose default ``window="hann"`` is
        applied in the frequency domain and halves the amplitude at half the
        input Nyquist, where the polyphase filter used here is flat.
    filter : bool, optional
        Whether to apply the preprocessing filter the weight set ships, if it
        ships one. ``True`` by default. Of the 17 cached PhaseNet weight sets
        only ``obs`` declares one, which is why most pipelines are three
        stages.
    dim : str, optional
        The sample dimension. Defaults to ``"time"``. Unlike
        :class:`Annotate`, the ``"first"``/``"last"`` aliases are refused: the
        pick table names its columns after coordinates, so the dimension has
        to be nameable before any data is seen.
    components : str, False or None, optional
        Name of the component dimension, and of the collection level to
        gather. ``None`` (default) detects both, ``False`` disables both. See
        :class:`Annotate`.
    component_strategy : str, optional
        How the model's input slots are filled, see :class:`Annotate`.
    device : str or torch.device, optional
        Torch device. Defaults to CUDA if available, else CPU.
    tolerance : scalar, None or False, optional
        Grid-snapping budget forwarded to :func:`xdas.stack` when a component
        level is gathered, see :meth:`Annotate.gather`.
    coords : sequence of str, "auto" or None, optional
        The coordinates annotating the picks, as
        :class:`~xdas.atoms.Trigger` takes them. Defaults to ``"auto"``: the
        scalar coordinates lead, so a pick carries the identity its array
        carries.
    **annotate_kwargs
        SeisBench annotate arguments (``overlap``, ``stacking``, ``blinding``,
        ``P_threshold``, ...) overriding what the weight set declares.

    Warnings
    --------
    A weight set whose filter is declared *per channel* — ``obs``'s 0.5 Hz
    highpass on ``??H`` — can only select the channels it names once they are
    labelled. On unlabelled data (a DAS section, a bare trace) the glob
    matches nothing and the stage is a silent no-op, exactly as
    ``stream.select`` is, while ``component_strategy="clone"`` still clones
    the signal into the hydrophone slot. Label the channels, or do not run OBS
    weights on unlabelled data.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> import xdas as xd
    >>> from xdas.atoms import Picker

    A stand-in for a real weight set, small enough to inline: an eight-sample
    window, three components, three classes and one declared threshold.

    >>> class Model(torch.nn.Module):
    ...     in_samples, in_channels, classes = 8, 3, 3
    ...     labels, component_order = "PSN", "ZNE"
    ...     sampling_rate = 100.0
    ...     default_args = {"overlap": 0, "P_threshold": 0.5}
    ...     def annotate_batch_pre(self, batch, argdict):
    ...         return batch
    ...     def annotate_batch_post(self, batch, piggyback, argdict):
    ...         return torch.transpose(batch, -1, -2)
    ...     def forward(self, batch):
    ...         return batch

    The stages come from the weight set. This one declares no filter, so the
    pipeline is three stages, and only the phases it can pick get a threshold
    — the noise class never does:

    >>> picker = Picker(Model(), device="cpu")
    >>> [type(stage).__name__ for stage in picker]
    ['Resample', 'Annotate', 'Trigger']
    >>> picker[-1].thresh
    {'P': 0.5, 'S': 0.3}

    Picking a three-component record. The model here returns its input, so
    the vertical channel is the ``P`` class:

    >>> values = np.zeros((64, 3))
    >>> values[20, 0] = 0.9
    >>> da = xd.DataArray(
    ...     values,
    ...     {
    ...         "time": {
    ...             "tie_indices": [0, 63],
    ...             "tie_values": [0.0, 0.63],
    ...             "sampling_interval": 0.01,
    ...         },
    ...         "channel": ["SHZ", "SHN", "SHE"],
    ...     },
    ...     ("time", "channel"),
    ... )
    >>> da = da.assign_coords(network="IA", station="DBNFM")
    >>> xd.pick(da, Model(), device="cpu")
      network station phase  time  value
    0      IA   DBNFM     P   0.2    0.9

    """

    def __init__(
        self,
        model,
        thresh=None,
        resample=True,
        filter=True,
        dim="time",
        components=None,
        component_strategy="auto",
        device=None,
        tolerance=None,
        coords="auto",
        **annotate_kwargs,
    ):
        if dim in ("first", "last"):
            raise ValueError(
                f"a picker needs its sample dimension by name, not as {dim!r}: "
                "the picks are annotated with coordinates, which are named"
            )
        stages = []
        if filter:
            stage = _model_filter(
                model, dim=dim, components=components, **annotate_kwargs
            )
            if stage is not None:
                stages.append(stage)
        if resample:
            rate = getattr(model, "sampling_rate", None)
            if not rate:
                raise ValueError(
                    "the weight set declares no `sampling_rate`, so there is "
                    "nothing to resample to: pass `resample=False` to pick at "
                    "the data's own rate"
                )
            stages.append(Resample(rate, dim=dim))
        stages.append(
            Annotate(
                model,
                dim=dim,
                components=components,
                component_strategy=component_strategy,
                device=device,
                tolerance=tolerance,
                **annotate_kwargs,
            )
        )
        if thresh is None:
            thresh = _model_thresholds(model, **annotate_kwargs)
        stages.append(Trigger(thresh, dim=dim, coords=coords))
        # named so that composing a picker nests it rather than flattening it
        super().__init__(stages, name="Picker")


class MLPicker(Annotate):
    """
    Deprecated alias of :class:`Annotate`, removed in 0.4.

    The old signature is kept, positional arguments included, but the results
    move: the output is laid out sample-last, ``(..., "phase", dim)``, rather
    than leading with the sample dimension; the component dimension is found
    by its labels rather than assumed; the window overlap is read off the
    weight set rather than fixed at half a window; and the end-aligned final
    window SeisBench appends is emitted at :meth:`flush`.
    """

    def __init__(
        self, model, dim="time", device=None, component_strategy="clone", **kwargs
    ):
        warnings.warn(
            "MLPicker is deprecated and will be removed in 0.4, use Annotate instead",
            DeprecationWarning,
            stacklevel=3,
        )
        super().__init__(
            model,
            dim=dim,
            device=device,
            component_strategy=component_strategy,
            **kwargs,
        )


annotate = atomized(Annotate)
pick = atomized(Picker)
mlpicker = atomized(MLPicker)
