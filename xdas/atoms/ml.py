"""
Machine-learning atom: :class:`Annotate` wraps SeisBench models as pipeline atoms.

Torch and SeisBench are loaded lazily so they remain optional dependencies.
"""

import importlib
import warnings
from typing import ClassVar

import numpy as np

from ..core import DataArray, concat
from .core import Atom, State, atomized


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
                    f"please install is manually"
                )
        return getattr(self._module, name)


torch = LazyModule("torch")

#: Horizontal component letters SeisBench treats as interchangeable when its
#: ``flexible_horizontal_components`` argument is set (which is its default),
#: so that ``Z12H`` weights accept ``ZNE`` data and vice versa.
FLEXIBLE_HORIZONTALS = {"1": "N", "N": "1", "2": "E", "E": "2"}


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
    """

    #: Fallbacks for the annotate arguments this atom reads. SeisBench's own
    #: ``_argdict_get_with_default`` falls back to the model's
    #: ``_annotate_args``, but ``blinding`` is not in the ``WaveformModel``
    #: base and a model need not declare one at all, so the accessor needs a
    #: default of its own rather than a subscript of ``None``.
    _annotate_defaults: ClassVar[dict] = {
        "overlap": 0,
        "stacking": "avg",
        "flexible_horizontal_components": True,
    }

    def __init__(
        self,
        model,
        dim="time",
        components=None,
        component_strategy="auto",
        device=None,
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

    def _annotate_arg(self, key):
        """
        Read one annotate argument, SeisBench-style but crash-free.

        Mirrors ``WaveformModel._argdict_get_with_default``: what the call site
        passes wins, else what the weight set declares, else the model's
        ``_annotate_args`` documentation default. The last step is where
        SeisBench raises on a key its base class does not declare, so this
        falls back to :attr:`_annotate_defaults` instead. The two SeisBench
        internals this atom relies on are isolated here and nowhere else.
        """
        if key in self.argdict:
            return self.argdict[key]
        entry = getattr(self.model, "_annotate_args", {}).get(key)
        if entry is None:
            return self._annotate_defaults[key]
        return entry[1]

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
        """List of phase label strings produced by the model."""
        return list(self.model.labels)

    @property
    def in_channels(self):
        """Number of input channels the model expects."""
        return getattr(self.model, "in_channels", len(self.model.component_order))

    @property
    def classes(self):
        """Number of output classes (phases) the model produces."""
        return getattr(self.model, "classes", len(self.phases))

    @property
    def fill(self):
        """Neutral element of the stacking rule, used to reset freed samples."""
        return 0.0 if self.stacking == "avg" else -np.inf

    def initialize(self, da, chunk_dim=None, **flags):
        """Resolve the dimensions and allocate the sliding-window buffers."""
        dim = self._resolve_sample_dim(da)
        component_dim, slots = self._resolve_component_dim(da, dim)
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
        window therefore produces no output at all.

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
        else:
            da = concat([self.buffer, da], dim)
            if da.sizes[dim] < self.nperseg:
                self.buffer = State(da)
                return None
        return self._process(da)

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
        chunk = self._process(da)
        chunks = ([] if chunk is None else [chunk]) + self.flush()
        # One record in, one record out: the pieces are consecutive along
        # `dim`, and downstream is joining along the *chunked* dimension.
        return concat(chunks, dim) if len(chunks) > 1 else chunks[0]

    def flush(self):
        """
        Emit the end-aligned final window and everything it completes.

        SeisBench appends one last window ending on the record's last sample
        whenever the stride leaves a remainder, so the output spans the input.
        Firing once per run, this stays chunk-invariant.
        """
        if self.started is not True:
            return []
        dim = self.sample_dim
        buffer = self.buffer
        remainder = buffer.sizes[dim] - self.nperseg
        if remainder > 0:
            self._advance(buffer.isel({dim: slice(-remainder, None)}), remainder)
        chunk = self._emit(buffer, 0, self.step - remainder, self.nperseg + remainder)
        self.buffer = State(buffer.isel({dim: slice(0, 0)}))
        self.started = State(False)
        self.circular_output.fill_(self.fill)
        self.circular_counts.fill_(0)
        return [chunk]

    def _process(self, da):
        """Annotate every window of *da* that the buffered state can complete."""
        dim = self.sample_dim
        nperseg, step = self.nperseg, self.step
        if self.started:
            first = step  # the window at 0 was annotated by the previous call
        else:
            self._prime(da)
            first = 0
        chunks = []
        last = None
        for idx in range(first, da.sizes[dim] - nperseg + 1, step):
            tail = da.isel({dim: slice(idx + nperseg - step, idx + nperseg)})
            self._advance(tail, step)
            if idx > 0:  # the first window of a run completes nothing yet
                chunks.append(self._emit(da, idx - step, 0, step))
            last = idx
        if last is None:
            self.buffer = State(da)
            return None
        self.started = State(True)
        self.buffer = State(da.isel({dim: slice(last, None)}))
        return concat(chunks, dim) if chunks else None

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
        """Reduce *length* stacked samples from *start* into a numpy array."""
        values = self.circular_output.narrow(-1, start, length)
        counts = self.circular_counts.narrow(-1, start, length)
        if self.stacking == "max":
            data = torch.where(counts > 0, values, float("nan"))
        else:
            data = values / counts  # samples no window covered come out NaN
        return data.cpu().numpy()

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
        """Build the output chunk of *length* samples found at *start* in the stack."""
        dim = self.sample_dim
        data = self._pull(start, length)
        coords = da.coords.copy()
        if self.component_dim is not None:
            coords = coords.drop_dims(self.component_dim)
        coords[dim] = coords[dim][offset : offset + length]
        coords["phase"] = self.phases
        shape = tuple(da.sizes[other] for other in self.batch_dims)
        return DataArray(
            data.reshape(*shape, self.classes, length),
            coords,
            self.out_dims,
            da.name,
            da.attrs,
        )

    def _resolve_sample_dim(self, da):
        """Resolve *dim* against the input, accepting the ``first``/``last`` aliases."""
        if self.dim == "first":
            return da.dims[0]
        if self.dim == "last":
            return da.dims[-1]
        if self.dim not in da.dims:
            raise ValueError(
                f"{self.dim!r} is not a dimension of the input (got {da.dims})"
            )
        return self.dim

    def _labels(self, da, dim):
        """Return the string labels of *dim*, or ``None`` if it carries none."""
        if dim not in da.coords:
            return None
        coord = da.coords[dim]
        if coord.dtype.kind not in "SU":
            return None
        return [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in coord.values
        ]

    def _match_components(self, labels):
        """Map *labels* onto model input slots, or ``None`` if they are not components."""
        order = list(self.model.component_order)
        flexible = self._annotate_arg("flexible_horizontal_components")
        slots = []
        for label in labels:
            letter = label[-1:]
            if letter not in order and flexible:
                letter = FLEXIBLE_HORIZONTALS.get(letter, letter)
            if letter not in order:
                return None
            slots.append(order.index(letter))
        return slots

    def _component_slots(self, da, dim):
        """Resolve the labels of *dim* into distinct model slots, or ``None``."""
        labels = self._labels(da, dim)
        if labels is None:
            return None
        slots = self._match_components(labels)
        if slots is None:
            return None
        if len(set(slots)) != len(slots):
            raise ValueError(
                f"the {dim!r} dimension repeats component orientations "
                f"({labels}): it mixes several instruments, split them first"
            )
        return slots

    def _resolve_component_dim(self, da, sample_dim):
        """Find the component dimension of *da* and the slots its labels name."""
        if self.components is False:
            return None, None
        if self.components is not None:
            if self.components not in da.dims:
                raise ValueError(
                    f"{self.components!r} is not a dimension of the input "
                    f"(got {da.dims})"
                )
            slots = self._component_slots(da, self.components)
            if slots is None:
                raise ValueError(
                    f"the {self.components!r} dimension is not labelled by "
                    f"components: expected labels ending with distinct letters "
                    f"of {self.model.component_order!r}"
                )
            return self.components, slots
        found = {}
        for dim in da.dims:
            if dim == sample_dim:
                continue
            slots = self._component_slots(da, dim)
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


class MLPicker(Annotate):
    """
    Deprecated alias of :class:`Annotate`, removed in 0.4.

    Beyond the name, the output of this atom is now laid out sample-last,
    ``(..., "phase", dim)``, rather than leading with the sample dimension.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MLPicker is deprecated and will be removed in 0.4, use Annotate instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


annotate = atomized(Annotate)
mlpicker = atomized(MLPicker)
