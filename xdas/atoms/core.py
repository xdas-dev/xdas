"""
Base classes for stateful processing atoms.

Includes :class:`Atom`, :class:`State`, :class:`Sequential`, :class:`Partial`,
the :func:`atomized` decorator, the :func:`as_function` generator of the
function form of an atom class, and the :func:`compose` primitive behind the
``>>`` operator.
"""

import importlib
import inspect
import re
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..coordinates import AxisCoordinate
from ..coordinates.core import parse_scalar_delta
from ..core import (
    DataArray,
    DataCollection,
    DataMapping,
    DataSequence,
    concat,
    open_datacollection,
    split,
)
from ..virtual import VirtualBackend


def _announce_splits(x, dim, count):
    """
    Warn that a source will be processed as several runs.

    Splitting on discontinuities must not be silent: each reset restarts the
    warm-up of every stateful stage, so the user should know how often it
    happens. The message names the source by its start so a collection walk
    reports every leaf rather than being deduplicated to the first.
    """
    start = x.coords[dim].start if dim in getattr(x, "coords", {}) else "?"
    plural = "discontinuities" if count > 1 else "discontinuity"
    warnings.warn(
        f"source starting at {start} has {count} {plural} along {dim!r}; "
        "state is flushed and reset at each",
        UserWarning,
        stacklevel=3,
    )


def _aschunks(value):
    """
    Normalize an atom output into a list of chunks.

    Atoms follow the transducer contract: ``call()`` maps one input chunk to
    zero or more output chunks. A bare object is one chunk, ``None`` is zero
    chunks, and a list or :class:`DataSequence` is taken chunk by chunk. Empty
    chunks are dropped: they carry no information and their degenerate
    coordinates would poison the initialization of downstream atoms.
    """
    if value is None:
        return []
    if not isinstance(value, (list, DataSequence)):
        value = [value]
    return [
        chunk for chunk in value if not (isinstance(chunk, DataArray) and chunk.empty)
    ]


def _join_chunks(chunks, dim=None):
    """
    Join heterogeneous output chunks into a single result.

    DataArrays are concatenated along *dim* with the gaps kept in the
    coordinates, falling back to a :class:`DataSequence` when concatenation
    cannot represent the result in one array; DataFrames are concatenated
    with a fresh index. Anything else is returned as a plain list. Zero
    chunks give ``None`` and a single chunk is returned bare.
    """
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    if all(isinstance(chunk, DataArray) for chunk in chunks):
        if dim is not None:
            try:
                return concat(chunks, dim)
            except (TypeError, ValueError, KeyError):
                pass
        return DataCollection(chunks)
    if all(isinstance(chunk, pd.DataFrame) for chunk in chunks):
        return pd.concat(chunks, ignore_index=True)
    return list(chunks)


def _extend_path(path, name, key):
    """Extend the tree path with the *key* a *name*d collection level was entered by."""
    return path if name is None else path | {name: key}


def _annotate_path(result, path):
    """
    Tag the tables in *result* with the tree path they were produced under.

    A leaf reached through ``IA / DBNFM / --`` gets a ``network``, a
    ``station`` and a ``location`` column, filled with those keys, *as its
    result is produced* rather than reconstructed afterwards — which is what
    lets a streaming walk hand a leaf straight to a sink and still carry its
    identity. Only tables are annotated: an atom returning arrays sees
    nothing change, and its tree is rebuilt as before.

    The columns lead, in tree order, so the identity of a pick comes first
    whatever the atom put in the table. A column the table already carries —
    a `network` scalar coordinate on the leaf, say, which the ObsPy engine
    attaches — is *the same column*, not a second one: it is moved into its
    leading position and the tree path wins, warning if the two disagree.

    *result* is what one leaf produced, so it is a table, an array, or one of
    the containers a leaf can answer with: a :class:`DataSequence` of chunks
    the atom could not join, or a plain list of chunks. It is never a mapping
    level — those the walk recurses into itself.
    """
    if not path:
        return result
    if isinstance(result, pd.DataFrame):
        return _annotate_frame(result, path)
    if isinstance(result, DataSequence):
        return DataCollection(
            [_annotate_path(value, path) for value in result], result.name
        )
    if isinstance(result, list):
        return [_annotate_path(value, path) for value in result]
    return result


def _annotate_frame(frame, path):
    """Prepend the *path* identity columns to the table *frame*."""
    frame = frame.copy()
    for name, key in path.items():
        if name in frame.columns and not (frame[name] == key).all():
            warnings.warn(
                f"the {name!r} column of a table disagrees with the tree path "
                f"the leaf was reached by ({key!r}): the tree path wins. The "
                "column comes from a coordinate of the leaf; rename it or "
                "drop it from `coords` to keep both.",
                UserWarning,
                stacklevel=2,
            )
        frame[name] = key
    rest = [name for name in frame.columns if name not in path]
    return frame[list(path) + rest]


def _iter_results(tree):
    """Yield the leaf results of a walked collection, in walk order."""
    if isinstance(tree, DataMapping):
        for value in tree.values():
            yield from _iter_results(value)
    elif isinstance(tree, DataSequence):
        for value in tree:
            yield from _iter_results(value)
    else:
        yield tree


def _flush_through(atoms, **flags):
    """
    Codec-drain a linear chain of atoms.

    Flush the first atom and push its tail through the remaining atoms, then
    flush the second one, and so on. Tails flow downstream as ordinary data:
    each downstream atom folds them into its own state before being flushed
    itself.
    """
    atoms = list(atoms)
    chunks = []
    for index, atom in enumerate(atoms):
        tail = atom.flush()
        for downstream in atoms[index + 1 :]:
            tail = [
                chunk
                for out in (downstream(x, **flags) for x in tail)
                for chunk in _aschunks(out)
            ]
        chunks.extend(tail)
    return chunks


class State:
    """
    A class to declare a new state or to update a preexising one into an Atom object.

    Parameters
    ----------
    state: Any
        The state to be passed to the Atom object.

    Examples
    --------
    In practice the State object is used when implementing new Atom objects. Bellow a
    dummy example without any class declaration.

    >>> from xdas.atoms import Atom, State

    Let's create an empty Atom object:

    >>> self = Atom()
    >>> self.state
    {}

    We can declare and initialize a new state:

    >>> self.count = State(0)
    >>> self.state
    {'count': 0}

    To update the state, we must use the State object again. The state is updated by:

    >>> self.count = State(self.count + 1)
    >>> self.state
    {'count': 1}

    """

    def __init__(self, state):
        self.state = state


class Atom(np.lib.mixins.NDArrayOperatorsMixin):
    """
    The base class for atoms. Used to implement new Atom objects.

    Atoms are the building blocks to perform massive computations. They represent
    processing units that can be combined to create complex data processing pipelines.
    Each Atom object is a callable that takes a unique input data object and returns a
    unique processed data object. Atoms can be stateful meaning that they can store
    some memory between calls to ensure the continuity of the processing across chunks.
    The memory of the Atom is stored in the state attribute. The state is a dictionary
    that is updated during each execution of the Atom. The Atom object is initialized
    with an initial state at the first call. In subsequent calls, the state is updated
    **if and only if the `chunk_dim` flag is provided** along with the dimension along
    wich chunking was performed. If this flag is not provided, the state is reset
    between calls. The Atom can be reset manually to its initial state by calling the
    `reset` method.

    When implementing a new Atom object, the user must sublcass the Atom class, and
    at minima define the `initialize` and the `call` methods. The `initialize` method
    is called at the first call to the Atom and is used to initialize the Atom with the
    input data. The `call` method is called at each subsequent call to the Atom and is
    used to perform the main processing logic. The Atom class handles when an how those
    two methods are called.

    To reduce the size of the state that need to be stored, a good practive is to also
    define the `initialize_from_state` method. This method is called in the
    `initialize` as soon as the minimal set of states is initialized. The other states
    that are usefull for the processing but that can be recomputed from the minimal set
    are initialized in the `initialize_from_state` method.

    Atoms compose into pipelines with the ``>>`` operator (see :func:`compose`)
    and trace ordinary numpy expressions: applying a ufunc to an atom appends
    the operation to the pipeline instead of computing it.

    Attributes
    ----------
        state: dict
            Returns the current state of the atom recursively including
            the state of nested atoms.
        initialized: bool
            Wether the atom has been initialized or not.
        on_discontinuity: str
            Seam policy for chunked processing: ``"reset"`` (default) flushes
            and starts a new run at every gap or rate change, ``"raise"``
            refuses discontinuous input. Overlaps always raise.
        merge: callable or None
            The optional hook folding the per-leaf results of a collection
            walk into one object. ``None`` (the default) means the atom has
            none and the tree is rebuilt as it always was. See
            :meth:`~xdas.atoms.Trigger.merge` for an implementation.

    Methods
    -------
        gather(mapping)
            Optionally collapse a mapping level into an array before the
            walk descends into it. See :meth:`gather`.
        initialize(x, **flags)
            Initializes the atom with the given input.
        initialize_from_state()
            Initializes the atom from its minimal state.
        call(x, **flags)
            Performs the main processing logic of the atom. May return zero
            or more output chunks (the transducer contract).
        flush()
            Drains buffered samples at the end of a run.
        reset()
            Resets the atom to its initial state.
        merge(results)
            Optional. Folds the leaf results of a collection walk into one
            object; undefined by default.
        fresh()
            Returns a stateless clone sharing the configuration.
        iter_chunks(source)
            Streams a chunk source through the atom, seams and flush included.

    """

    on_discontinuity = "reset"
    #: Undefined by default: an atom that does not fold its collection
    #: results leaves the walked tree as it is.
    merge = None

    def __init__(self):
        object.__setattr__(self, "_config", {})
        object.__setattr__(self, "_state", {})
        object.__setattr__(self, "_atoms", {})
        object.__setattr__(self, "_seam", None)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    __hash__ = object.__hash__

    def __repr__(self):
        name = self.__class__.__name__
        sig = ", ".join(
            f"{key}={value}" for key, value in self._config.items() if value is not None
        )
        s = f"{name}({sig})"
        for name, filter in self._atoms.items():
            s += "\n" + "\n".join(f"  {e}" for e in repr(filter).split("\n"))
        return s

    def __setattr__(self, name, value):
        match value:
            case State(state=state):
                self._state[name] = state
                object.__setattr__(self, name, state)
            case Atom():
                self._atoms[name] = value
                object.__setattr__(self, name, value)
            case other:
                self._config[name] = value
                object.__setattr__(self, name, other)

    @property
    def state(self):
        """Dict of the current state, including nested atom states."""
        return self._state | {
            name: filter.state for name, filter in self._atoms.items() if filter.state
        }

    @property
    def initialized(self):
        """``True`` if every state key, nested atoms included, is initialised."""
        return all(value is not ... for value in self._state.values()) and all(
            atom.initialized for atom in self._atoms.values()
        )

    def initialize(self, x, **flags):
        """Initialise the atom from a first chunks of data."""
        return NotImplemented

    def initialize_from_state(self):
        """Initialise the atom from its current state."""
        return NotImplemented

    def call(self, x, **flags):
        """Process a chunk of data."""
        return NotImplemented

    def gather(self, mapping):
        """
        Return *mapping* collapsed to an array, or ``None`` to map over it.

        Consulted on every mapping level of a collection *before* the walk
        descends into it, so an atom that knows a level is really an axis of
        its input can take it as one. :class:`~xdas.atoms.Annotate` is the
        implementation: it knows the model's ``component_order``, so it knows
        that a ``channel`` level keyed ``SHZ``/``SHN``/``SHE`` is the
        component dimension of one instrument rather than three
        independent leaves, and collapses it with :func:`xdas.stack`.

        Returning ``None`` — the default, and what every other atom does —
        leaves the walk exactly as it was.

        Parameters
        ----------
        mapping : DataMapping
            The collection level about to be walked.

        Returns
        -------
        DataArray, DataCollection or None
            The collapsed input, which the walk continues on in place of the
            level, or ``None`` to map over the level's leaves.
        """
        return

    def __call__(self, x, **flags):
        """
        Process input data, returning zero or more output chunks.

        Eager calls (no ``chunk_dim`` flag) auto-split gappy input into runs,
        process each run with a fresh state, flush the tails and re-join the
        outputs into a single object with gap-aware coordinates when possible
        (falling back to a :class:`DataSequence`). Chunked calls (``chunk_dim``
        given) carry state across continuous chunks and handle seams: on a gap
        or a rate change the atom flushes, resets and starts a new run (see
        `on_discontinuity`); on an overlap it raises. Sequence collections are
        folded element by element through the same seam-aware machinery, so
        resets emerge from the coordinates; mapping collections map over their
        leaves.

        Walking a collection, each leaf's result is annotated with the tree
        path it was produced under — one column per level, leading, filled
        with the key the level was entered by — and the annotated results are
        then folded by the atom's `merge` hook if it declares one, so a
        table-valued atom answers a whole collection with one table rather
        than with a tree of them. ``merge=False`` opts out and returns the
        walked tree, annotations included. Mapping levels are first offered
        to `gather`, which may collapse a level into an axis of the input
        (see :meth:`gather`).

        A single output chunk is returned bare; otherwise a
        :class:`DataSequence` of chunks is returned.
        """
        merge = flags.pop("merge", True)
        chunk_dim = flags.get("chunk_dim", None)
        self._check_chunk_dim(x, chunk_dim)
        if (
            chunk_dim is None
            and isinstance(x, DataArray)
            and isinstance(x.data, VirtualBackend)
            and x.nbytes > config.get("memory_limit")
        ):
            raise ValueError(
                f"this eager call would load the full virtual array "
                f"(~{x.nbytes / 2**30:.1f} GiB, above the 'memory_limit' "
                "configuration entry) in memory: stream it chunk by chunk "
                "with `.process(da, out=...)` instead, or raise the limit "
                "with `xdas.config.set('memory_limit', ...)`"
            )
        if isinstance(x, (DataMapping, DataSequence)):
            if isinstance(x, DataMapping):
                result = self._walk(x, flags, {})
            else:
                result = self._fold(x, flags, {})
            if merge and self.merge is not None:
                return self.merge(list(_iter_results(result)))
            return result
        if chunk_dim is None:
            dim = self._resolve_dim(x)
            runs = self._split_runs(x, dim)
            if len(runs) > 1:
                _announce_splits(x, dim, len(runs) - 1)
            chunks = []
            for run in runs:
                self.initialize(run, **flags)
                chunks += _aschunks(self.call(run, **flags))
                chunks += self.flush()
            self.reset()
            return self._join(chunks, dim)
        else:
            chunks = []
            for run in self._split_runs(x, chunk_dim):
                chunks += self._call_run(run, flags)
            return self._join(chunks, None)

    def _call_run(self, x, flags):
        """Seam-aware chunked call on one internally-regular chunk."""
        if isinstance(x, DataArray) and x.empty:
            return []
        chunk_dim = flags["chunk_dim"]
        chunks = []
        stateful = self._live_state()
        if stateful:
            info = self._seam_info(x, chunk_dim)
            verdict = self._judge_seam(info)
            if verdict in ("gap", "rate"):
                match self.on_discontinuity:
                    case "reset":
                        chunks += self.flush()
                        self.reset()
                    case "raise":
                        raise ValueError(
                            f"the incoming chunk is discontinuous with the "
                            f"stream processed so far ({verdict} detected "
                            f"along {chunk_dim!r}) and `on_discontinuity` is "
                            "set to 'raise'"
                        )
                    case other:
                        raise ValueError(
                            "`on_discontinuity` must be 'reset' or 'raise', "
                            f"got {other!r}"
                        )
            elif verdict == "overlap":
                raise ValueError(
                    f"the incoming chunk overlaps the stream processed so far "
                    f"(the {chunk_dim!r} coordinate goes backward across the "
                    "seam); sort or deduplicate the input, or call `reset()` "
                    "to explicitly start a new run"
                )
        if not self.initialized:
            self.initialize(x, **flags)
        chunks += _aschunks(self.call(x, **flags))
        if stateful and info is not None:
            if info["delta"] is None and verdict == "continuous":
                info["delta"] = self._seam["delta"]
            object.__setattr__(self, "_seam", info)
        return chunks

    def _live_state(self):
        """Return ``True`` if any state entry (own or nested) holds a live value."""

        def live(state):
            return any(
                live(value) if isinstance(value, dict) else value is not None
                for value in state.values()
            )

        return live(self.state)

    def _resolve_dim(self, x):
        """Resolve the dimension this atom operates along on *x*, or ``None``."""
        dim = getattr(self, "dim", None)
        if not isinstance(x, DataArray) or not isinstance(dim, str):
            return None
        if dim == "first":
            dim = x.dims[0]
        elif dim == "last":
            dim = x.dims[-1]
        return dim if dim in x.coords else None

    def _split_runs(self, x, dim):
        """Split *x* at the discontinuities of its *dim* coordinate."""
        if not isinstance(x, DataArray) or dim not in getattr(x, "coords", {}):
            return [x]
        coord = x.coords[dim]
        if not isinstance(coord, AxisCoordinate) or not coord.isregular():
            return [x]
        indices = coord.get_split_indices(
            "discontinuities", getattr(coord, "tolerance", None)
        )
        if not indices.size:
            return [x]
        return list(split(x, indices, dim))

    def _seam_info(self, x, chunk_dim):
        """Extract the seam-judgment metadata of a chunk, or ``None``."""
        if not isinstance(x, DataArray) or chunk_dim not in x.coords:
            return None
        coord = x.coords[chunk_dim]
        if not isinstance(coord, AxisCoordinate) or coord.empty:
            return None
        return {
            "chunk_dim": chunk_dim,
            "start": coord.start,
            "end": coord.end,
            "delta": coord.get_sampling_interval(cast=False),
            "tolerance": parse_scalar_delta(
                getattr(coord, "tolerance", None), coord.dtype, default_zero=True
            ),
            "size": len(coord),
        }

    def _judge_seam(self, info):
        """
        Compare an incoming chunk with the expected continuation of the stream.

        Returns ``None`` when there is nothing to judge against (first chunk,
        non-array chunk), else one of ``"continuous"``, ``"gap"``, ``"rate"``
        or ``"overlap"``. Both O(1) checks of the regularity contract happen
        here: the sampling interval must match within tolerance, and the chunk
        must start one interval after the previous end within the jitter
        budget.
        """
        seam = self._seam
        if info is None or seam is None or seam["chunk_dim"] != info["chunk_dim"]:
            return None
        for entry in (seam, info):
            if entry["delta"] is None and entry["size"] > 1:
                dim = info["chunk_dim"]
                raise ValueError(
                    f"chunked processing along {dim!r} requires a regular "
                    "coordinate (one that declares its `sampling_interval`); "
                    "regularize it first, e.g. `da[dim] = da[dim].to_regular()` "
                    "or open the files with a tolerance"
                )
        if seam["delta"] is None:
            return None
        tolerance = max(seam["tolerance"], info["tolerance"])
        if info["delta"] is not None and np.abs(info["delta"] - seam["delta"]) > (
            tolerance
        ):
            return "rate"
        jump = info["start"] - (seam["end"] + seam["delta"])
        if np.abs(jump) <= tolerance:
            return "continuous"
        return "gap" if jump > 0 else "overlap"

    def _walk(self, x, flags, path):
        """
        Walk a collection leaf by leaf, carrying the tree path down.

        Mapping levels are first offered to `gather`, which may take the whole
        level as an axis of the input rather than as leaves to map over; the
        level is then consumed and contributes no path column. Otherwise
        mapping levels recurse under their key, sequence levels fold (see
        `_fold`), and every leaf result is annotated with the path it was
        produced under before it goes anywhere else. Carrying the path *down*
        rather than rebuilding it on the way up is what a streaming walk
        needs: a leaf is complete the moment it is produced.

        One atom instance walks the leaves sequentially — the eager path
        already resets it at the end of each run — because an atom holding a
        model either saturates the CPU or holds a lot of device memory, so
        only one should be live per node.
        """
        if isinstance(x, DataMapping):
            gathered = self.gather(x)
            if gathered is not None:
                return self._walk(gathered, flags, path)
            if flags.get("chunk_dim", None) is not None:
                raise NotImplementedError(
                    "chunked processing of mapping collections is not supported: "
                    "process each leaf with its own atom instance"
                )
            name = getattr(x, "name", None)
            return DataCollection(
                {
                    key: self._walk(value, flags, _extend_path(path, name, key))
                    for key, value in x.items()
                },
                name,
            )
        if isinstance(x, DataSequence):
            return self._fold(x, flags, path)
        return _annotate_path(self(x, **flags), path)

    def _fold(self, x, flags, path=None):
        """
        Fold a sequence collection through the same seam-aware call.

        A collection is multiple chunks delivered at once: each element goes
        through the chunked path along the atom's dimension, so state carries
        across continuous elements and resets emerge from the coordinates.

        The level contributes its positional keys as a column, each output
        chunk taking the index of the element that produced it. The flushed
        tail is attributed to the last element, which is where it came out;
        no finer answer exists, since a folded element's buffered samples are
        released by the element that follows it.
        """
        name = getattr(x, "name", None)
        path = {} if path is None else path
        chunk_dim = flags.get("chunk_dim", None)
        if chunk_dim is None:
            first = next((el for el in x if isinstance(el, DataArray)), None)
            dim = self._resolve_dim(first)
            if dim is None:
                return DataCollection(
                    [
                        self._walk(el, flags, _extend_path(path, name, index))
                        for index, el in enumerate(x)
                    ],
                    name,
                )
            flags = flags | {"chunk_dim": dim}
            chunks = []
            for index, el in enumerate(x):
                chunks += _annotate_path(
                    _aschunks(self(el, **flags)), _extend_path(path, name, index)
                )
            chunks += _annotate_path(
                self.flush(), _extend_path(path, name, max(len(x) - 1, 0))
            )
            self.reset()
            return DataCollection(chunks, name)
        chunks = []
        for index, el in enumerate(x):
            chunks += _annotate_path(
                _aschunks(self(el, **flags)), _extend_path(path, name, index)
            )
        return DataCollection(chunks, name)

    def _join(self, chunks, dim):
        """Re-join output chunks: one chunk bare, else gap-aware concat or sequence."""
        if len(chunks) == 1:
            return chunks[0]
        if dim is not None and chunks and all(isinstance(c, DataArray) for c in chunks):
            try:
                return concat(chunks, dim)
            except (TypeError, ValueError):
                return DataCollection(chunks)
        if chunks and all(isinstance(c, pd.DataFrame) for c in chunks):
            # A pick table is a chunk type of its own: an atom emitting one per
            # run, plus one at flush, still answers with a single table.
            return pd.concat(chunks, ignore_index=True)
        return DataCollection(chunks)

    def flush(self):
        """
        Drain buffered samples, returning zero or more output chunks.

        Stateful atoms that hold samples back waiting for the next chunk
        override this to emit what remains computable at the end of a run.
        Called at the end of the stream, at every seam, and at the end of
        every eager call. Default is a no-op.
        """
        return []

    def iter_chunks(self, source, chunk_dim=None):
        """
        Iterate over the output chunks of this atom applied to a chunk source.

        The manual chunk-loop surface: wraps seam handling, buffering and the
        final flush into a plain generator, so
        ``for out in atom.iter_chunks(source): ...`` is a complete streaming
        loop. The serial executor is literally this generator plus a writer.

        Parameters
        ----------
        source : iterable of DataArray
            The chunks to process. Any iterable works; a loader exposing a
            ``chunk_dim`` attribute provides the chunked dimension.
        chunk_dim : str, optional
            The dimension along which chunks follow each other. Defaults to
            the source's ``chunk_dim`` attribute, else ``"time"``.

        Yields
        ------
        Zero or more output chunks per input chunk, then the flushed tail.
        """
        if chunk_dim is None:
            chunk_dim = getattr(source, "chunk_dim", "time")
        for chunk in source:
            yield from _aschunks(self(chunk, chunk_dim=chunk_dim))
        yield from self.flush()
        self.reset()

    def process(self, source, out=None, chunks=None, until=None, merge=True):
        """
        Process any chunk source through this atom, writing to any sink.

        The one-call form of chunked execution: the input is resolved into a
        chunk source and the output into a writer automatically (see
        :func:`xdas.processing.process`, which this method binds to the
        atom). The same pipeline that runs eagerly with ``pipeline(da)``
        streams a massive archive with ``pipeline.process(da, out=...)``.

        A :class:`~xdas.DataCollection` is walked exactly as ``atom(dc)``
        walks it — `gather` first, then mapping levels recursed and sequence
        levels folded, each leaf streamed in turn — so that
        ``atom.process(dc, out=None) == atom(dc)``. The streaming form is
        not second-class, which matters most where a leaf is too large to
        call eagerly at all.

        Parameters
        ----------
        source : DataArray, DataCollection, str, Path, iterable or loader
            What to process: an in-memory or virtual :class:`DataArray`, a
            :class:`~xdas.DataCollection` to walk leaf by leaf, a file path,
            directory or glob pattern, a ``"tcp://..."`` address,
            :func:`xdas.watch` for realtime, or any iterable of chunks.
        out : str, Path, writer or None, optional
            Where to write the output: ``None`` accumulates in memory and
            returns the joined result (size-guarded); a path is matched with
            the first output chunk (directory for DataArray or Stream
            chunks, ``*.csv`` for DataFrames, ``"tcp://..."`` to publish); a
            writer instance passes through. Walking a collection, a file, a
            URL or a writer instance is shared by every leaf while a
            directory fans out into one subdirectory per leaf.
        chunks : dict or "auto", optional
            Chunk sizes for DataArray sources, e.g. ``{"time": 1000}``.
            Virtual arrays default to ``"auto"``: chunk boundaries aligned
            to the storage tiling.
        until : str, datetime64 or float, optional
            Stop at this coordinate value along the chunked dimension; the
            clean way to bound an unbounded source.
        merge : bool, optional
            Whether to fold the per-leaf results of a collection walk
            through the `merge` hook, as ``atom(dc)`` does. Walk-level, not
            stage-level, and only meaningful for ``out=None``.

        Returns
        -------
        result : object
            The writer result: the joined output for ``out=None``, whatever
            the resolved writer returns otherwise, or ``None`` when the
            pipeline emitted no output.

        Examples
        --------
        >>> import numpy as np
        >>> import xdas as xd
        >>> pipeline = xd.decimate(..., target=50.0) >> np.square
        >>> pipeline.process(da_virtual, out="results/")  # doctest: +SKIP
        >>> pipeline.process("archive/*.h5", out="results/")  # doctest: +SKIP
        >>> pipeline.process(xd.watch("/incoming"), out="sds/")  # doctest: +SKIP
        """
        from ..processing.core import process

        return process(self, source, out=out, chunks=chunks, until=until, merge=merge)

    def _check_chunk_dim(self, x, chunk_dim):
        """Raise if this atom cannot process *x* chunked along *chunk_dim*."""

    def _refuse_chunked_along(self, dim, chunk_dim, x=None):
        """
        Raise if a whole-record operation is being chunked along its own dim.

        The guard for atoms that need the whole record along the dimension
        they work on: call it from :meth:`initialize` (or a
        :meth:`_check_chunk_dim` override) with the dimension the atom works
        along and the dimension the stream is chunked along. *dim* may also
        be a mapping whose keys are the working dimensions (a kernel dict).
        ``"first"`` and ``"last"`` aliases are resolved against *x* when
        given, so the comparison is never made on an unresolved alias.
        """
        if chunk_dim is None:
            return
        dims = list(dim.keys()) if isinstance(dim, dict) else [dim]
        if x is not None and hasattr(x, "dims"):
            dims = [
                x.dims[0] if d == "first" else x.dims[-1] if d == "last" else d
                for d in dims
            ]
        if any(d is None or d in ("first", "last") or d == chunk_dim for d in dims):
            name = (
                getattr(self, "name", None)
                or getattr(getattr(self, "func", None), "__name__", None)
                or type(self).__name__
            )
            raise ValueError(
                f"{name} needs the whole record along {dim!r} and cannot "
                f"process data chunked along {chunk_dim!r}: process the "
                f"stream unchunked, or chunk along another dimension"
            )

    def __rshift__(self, other):
        """Compose with *other* into a new pipeline: ``atom >> atom``."""
        if isinstance(other, Atom) or callable(other):
            return compose(self, other)
        return NotImplemented

    def __rrshift__(self, other):
        """Prepend a bare callable, or apply the pipeline: ``da >> atom``."""
        if callable(other):
            return compose(other, self)
        return self(other)

    __irshift__ = __rshift__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Append the ufunc to the pipeline instead of computing it (tracing)."""
        if ufunc is np.right_shift and method == "__call__" and len(inputs) == 2:
            # ``da >> atom``: the DataArray operand dispatched through numpy.
            left, right = inputs
            if right is self and not isinstance(left, Atom):
                return self.__rrshift__(left)
        if method != "__call__":
            return NotImplemented
        if "out" in kwargs:
            # In-place operators rebind the target name with their return
            # value, so tracing them keeps value semantics: drop the `out`
            # and append the out-of-place operation.
            if kwargs["out"] != (self,):
                return NotImplemented
            kwargs = {key: value for key, value in kwargs.items() if key != "out"}
        if sum(input is self for input in inputs) != 1 or any(
            isinstance(input, Atom) and input is not self for input in inputs
        ):
            # Never bail silently into computation: fan-in is a design
            # boundary, and the error must say so at the line that traced it.
            raise TypeError(
                "traced fan-in is not supported: a pipeline is a single "
                "chain, so a ufunc may involve one atom exactly once "
                "(combine the branches eagerly, or wrap the whole "
                "expression in a function and Partial it)"
            )
        args = tuple(... if input is self else input for input in inputs)
        return compose(self, Partial(ufunc, *args, **kwargs))

    def reset(self):
        """Reset all state entries to ``...`` (uninitialised sentinel)."""
        object.__setattr__(self, "_seam", None)
        for key in self._state:
            setattr(self, key, State(...))
        for filter in self._atoms.values():
            filter.reset()

    def fresh(self):
        """
        Return a stateless clone of this atom: same config, no state.

        Config is shared *by reference* (a model is never deep-copied),
        nested atoms are recursed, and every state entry comes back
        uninitialised — where :meth:`reset` wipes this instance, ``fresh``
        leaves it untouched, so one configured atom can serve several
        independent runs.
        """
        clone = type(self).__new__(type(self))
        Atom.__init__(clone)
        for name, value in vars(self).items():
            if name in ("_config", "_state", "_atoms", "_seam"):
                continue
            if name in self._atoms:
                setattr(clone, name, self._atoms[name].fresh())
            elif name in self._state:
                setattr(clone, name, State(...))
            elif name in self._config:
                setattr(clone, name, value)
            else:
                # Attributes opted out of the registries with
                # `object.__setattr__` (pure helpers) travel as they are.
                object.__setattr__(clone, name, value)
        return clone

    def save_state(self, path):
        """Serialise the current state to a NetCDF4 file at *path*."""
        DataCollection(self.state).to_netcdf(path)

    def set_state(self, state):
        """
        Restore the atom state from a previously saved state dict.

        Parameters
        ----------
        state : dict
            Mapping of state key → value as returned by :attr:`state`.
        """
        for key, value in state.items():
            if isinstance(value, DataArray):
                setattr(
                    self, key, State(value.__array__())
                )  # TODO: shouldn't need __array__
                self.initialize_from_state()
            else:
                filter = getattr(self, key)
                filter.set_state(value)

    def load_state(self, path):
        """Load the atom state from the NetCDF4 file at *path*."""
        state = open_datacollection(path).load()
        self.set_state(state)


class Sequential(Atom, list):
    """
    A class to handle a sequence of operations.

    Each operation is represented by an Atom class object, which contains the
    function and its arguments. Sequence inherits from list, and therefore
    behaves as it.

    Parameters
    ----------
    atoms: list
        The sequence of operations. Each element must either be an Atom, a Sequence, or
        an unitary callable.
    name: str
        A label given to this sequence.

    Examples
    --------
    >>> from xdas.atoms import Partial, Sequential
    >>> import xdas.signal as xs
    >>> import numpy as np

    Basic usage:

    >>> seq = Sequential(
    ...     [
    ...         Partial(xs.taper, dim="time"),
    ...         Partial(xs.lfilter, [1.0], [0.5], ..., dim="time", zi=...),
    ...         Partial(np.square),
    ...     ],
    ...     name="Low frequency energy",
    ... )
    >>> seq
    Low frequency energy:
      0: taper(..., dim=time)
      1: lfilter([1.0], [0.5], ..., dim=time)  [stateful]
      2: square(...)

    Nested sequences:

    >>> seq = Sequential(
    ...     [
    ...         Partial(xs.decimate, 16, dim="distance"),
    ...         seq,
    ...     ]
    ... )
    >>> seq
    Sequence:
      0: decimate(..., 16, dim=distance)
      1:
        Low frequency energy:
          0: taper(..., dim=time)
          1: lfilter([1.0], [0.5], ..., dim=time)  [stateful]
          2: square(...)

    Applying the sequence to data:

    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()
    >>> seq(da)
    <xdas.DataArray (time: 300, distance: 26)>
    [[0.000000e+00 0.000000e+00 0.000000e+00 ... 0.000000e+00 0.000000e+00
      0.000000e+00]
     [5.925923e-10 3.640952e-11 1.315744e-11 ... 4.024388e-14 3.245748e-12
      7.679807e-12]
     [3.497487e-09 1.191342e-10 4.021543e-11 ... 7.463132e-12 9.458801e-11
      2.833292e-10]
     ...
     [2.331440e-08 5.785383e-10 4.336722e-11 ... 1.827452e-11 5.099163e-10
      1.320907e-09]
     [1.826236e-09 5.470673e-11 1.045146e-11 ... 1.561169e-13 3.063598e-14
      7.832009e-16]
     [0.000000e+00 0.000000e+00 0.000000e+00 ... 0.000000e+00 0.000000e+00
      0.000000e+00]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """

    def __init__(self, atoms: Any, name: str | None = None) -> None:
        super().__init__()
        for key, atom in enumerate(atoms):
            if not isinstance(atom, Atom):
                atom = Partial(atom)
            self.append(atom)
            self._atoms[key] = atom
        self.name = name

    def call(self, x: Any, **flags) -> Any:
        """
        Pass *x* through each atom in order and return the output chunks.

        Each stage may emit zero or more chunks per input chunk (seam tails,
        rechunking, reductions); the streams are folded stage by stage, so
        cadence mismatches are absorbed inside the pipeline.
        """
        chunks = [x]
        for atom in self:
            chunks = [
                chunk
                for out in (atom(x, **flags) for x in chunks)
                for chunk in _aschunks(out)
            ]
        return chunks

    def flush(self):
        """
        Cascade-flush the pipeline, codec-drain style.

        Flush the first stage and push its tail through the following stages,
        then flush the second stage, and so on. Returns the drained chunks.
        """
        flags = {"chunk_dim": self._seam["chunk_dim"]} if self._seam else {}
        return _flush_through(self, **flags)

    def _resolve_dim(self, x):
        """Resolve the operating dimension from the first stage that has one."""
        for atom in self:
            dim = atom._resolve_dim(x)
            if dim is not None:
                return dim
        return None

    def gather(self, mapping):
        """
        Collapse *mapping* through the first stage that claims it.

        The gather happens once, before the *first* stage runs, whichever
        stage claimed it — a pipeline is one transformation of one input, and
        a level a later stage needs as an axis has to be an axis by the time
        the input enters the pipeline. That is what a picking pipeline relies
        on: the component level is recognised by the stage that knows the
        model, and the earlier filter and resampling stages see the
        components as a dimension, which is the only form in which a
        per-channel filter can select one of them.

        First claim wins. Claiming is a structural statement about the level
        — that it is one axis of the input — so two stages that both claim it
        agree on what it is, and can differ only in how they would collapse
        it; the first stage in the pipeline is then as good an arbiter as any,
        and the only one that keeps the answer independent of the stages
        downstream.
        """
        for atom in self:
            gathered = atom.gather(mapping)
            if gathered is not None:
                return gathered
        return None

    @property
    def merge(self):
        """
        The `merge` hook of the last stage declaring one, else ``None``.

        The last stage is the one whose output leaves the pipeline, so it is
        the one that knows what folding its results means — a pipeline
        ending on a :class:`~xdas.atoms.Trigger` merges pick tables without
        having to say so.
        """
        for atom in reversed(self):
            if atom.merge is not None:
                return atom.merge
        return None

    def fresh(self):
        """Return a stateless clone: each stage cloned, config shared."""
        return type(self)([atom.fresh() for atom in self], name=self.name)

    def __repr__(self) -> str:
        width = len(str(len(self)))
        name = self.name if self.name is not None else "sequence"
        s = f"{name.capitalize()}:\n"
        for idx, value in enumerate(self):
            label = f"  {idx:{width}}: "
            if isinstance(value, Partial):
                s += label + repr(value) + "\n"
            else:
                s += label + "\n"
                s += "\n".join(f"    {e}" for e in repr(value).split("\n")[:-1]) + "\n"
        return s


class Partial(Atom):
    """
    Wraps a function into an Atom.

    It works similarly to `functools.partial` but with additional features. If the
    input is not the first argument, an `Ellipsis` (`...`) can be used to indicate the
    position of the input. A `name` argument can be used to better identify the
    resulting atom.

    Some level of state passing can be achieved by passing `...` for one or several
    keyword arguments. In that case, the function is expected to accept `...` as
    keyword arguments, to properly initialize the corresponding states and to return as
    many additional outputs as there are stateful arguments.

    Partial uses several reserved keyword arguments that cannot by passed to `func`:
    'func', 'name' and 'state'.

    Parameters
    ----------
    func : Callable
        The function that is called. One of its argument is used as the input of the
        resulting Atom object while other parameters are fixed. The position of the
        input is by default the first argument. Otherwise, an Ellipsis (`...`) must be
        provided in the `*args` parameters to indicate the position of the input.
        The function must return a unique output except if the function is stateful. In
        that case, the function must return the processed data as first output and the
        updated state as additional outputs.
    *args : Any
        Positional arguments to pass to `func`. If the data to process is passed as the
        nth argument, the nth element of `args` must contain an Ellipsis (`...`).
    name : str
        Name to identify the function.
    **kwargs : Any
        Keyword arguments to pass to `func`. If one of the keyword arguments is `...`,
        it will be treated as a passing state and initialized or updated at each call.


    Examples
    --------
    >>> import numpy as np
    >>> import scipy.signal as sp
    >>> import xdas.signal as xs
    >>> from xdas.atoms import Partial

    Examples of a stateless atom:

    >>> Partial(xs.decimate, 2, dim="time")
    decimate(..., 2, dim=time)

    >>> Partial(np.square)
    square(...)

    Examples of a stateful atom with input data as second argument:

    >>> sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
    >>> Partial(xs.sosfilt, sos, ..., dim="time", zi=...)
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

    """

    def __init__(
        self, func: Callable, *args: Any, name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__()
        if not callable(func):
            raise TypeError("`func` should be callable")
        if not any(arg is ... for arg in args):
            args = (...,) + args
        if sum(arg is ... for arg in args) > 1:
            raise ValueError("`*args` must contain at most one Ellipsis")
        self.func = func
        self.args = args
        self.kwargs = {}
        self.name = name
        for key, value in kwargs.items():
            if value is ...:
                setattr(self, key, State(...))
            elif isinstance(value, State):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        # The operating dimension is resolved from the call arguments so the
        # whole-record guard can compare it with the chunked one and so eager
        # calls split gappy input into runs along it. A `_whole_record`-marked
        # function may name a different argument (a kernel dict, say) as the
        # one carrying its working dimensions.
        dim_arg = getattr(func, "_whole_record_dim_arg", None)
        try:
            bound = inspect.signature(func).bind_partial(*self.args, **self.kwargs)
            bound.apply_defaults()
            dim = bound.arguments.get("dim")
            if isinstance(dim, dict) and len(dim) == 1:
                # {input_dim: output_dim} mapping (e.g. the fft functions):
                # the operating dimension is the input one.
                ((dim, _),) = dim.items()
            self.dim = dim
            refuse_dim = bound.arguments.get(dim_arg) if dim_arg else None
        except (TypeError, ValueError):
            self.dim = None
            refuse_dim = None
        object.__setattr__(self, "_refuse_dim", refuse_dim)

    def _check_chunk_dim(self, x, chunk_dim):
        """Refuse chunking along the working dim of a whole-record function."""
        if getattr(self.func, "_whole_record_dim_arg", None) is not None:
            self._refuse_chunked_along(self._refuse_dim, chunk_dim, x)

    @property
    def stateful(self):
        """``True`` if any keyword argument is being passed as state."""
        return bool(self._state)

    def call(self, x: Any, **flags) -> Any:
        """Call the wrapped function with *x* substituted at the ``...`` position."""
        args = tuple(x if arg is ... else arg for arg in self.args)
        kwargs = self.kwargs | self._state
        if self.stateful:
            x, *state = self.func(*args, **kwargs)
            for key, value in zip(self._state, state):
                setattr(self, key, State(value))
            return x
        else:
            return self.func(*args, **self.kwargs)

    def __repr__(self) -> str:
        func = getattr(self.func, "__name__", "<function>")
        args = []
        for value in self.args:
            if value is ...:
                args.append("...")
            elif len(str(value)) > 10:
                args.append(f"<{type(value).__name__}>")
            else:
                args.append(str(value))
        kwargs = []
        for key, value in self.kwargs.items():
            if len(str(value)) > 10:
                value = f"<{type(value).__name__}>"
            kwargs.append(f"{key}={value}")
        params = ", ".join(args + kwargs)
        return f"{func}({params})" + ("  [stateful]" if self.stateful else "")

    def __reduce__(self):
        return self.from_state, (self.get_state(),)

    @classmethod
    def from_state(cls, state):
        """Reconstruct a :class:`Partial` from a serialised *state* dict."""
        func = getattr(
            importlib.import_module(state["func"]["module"]), state["func"]["name"]
        )
        return cls(func, *state["args"], name=state["name"], **state["kwargs"])

    def get_state(self):
        """Return a JSON-serialisable dict describing the wrapped function and args."""
        return {
            "func": {"module": self.func.__module__, "name": self.func.__name__},
            "args": self.args,
            "kwargs": self.kwargs,
            "name": self.name,
        }


def compose(input, output):
    """
    Chain *input* then *output* into a new Sequential with value semantics.

    Composition never mutates its operands: each call returns a fresh
    Sequential, so intermediate pipelines stay usable on their own. Bare
    callables are wrapped into Partial atoms. Unnamed Sequentials are
    flattened; a named input Sequential keeps its name, a named output
    Sequential stays nested.

    This is the primitive behind the ``>>`` operator and operator tracing.

    Parameters
    ----------
    input : Atom or callable
        The upstream atom or pipeline.
    output : Atom or callable
        The atom or pipeline to append.

    Returns
    -------
    Sequential
        A new pipeline running *input* then *output*.
    """
    if not isinstance(input, Atom):
        input = Partial(input)
    if not isinstance(output, Atom):
        output = Partial(output)
    head = list(input) if isinstance(input, Sequential) else [input]
    tail = (
        list(output)
        if isinstance(output, Sequential) and output.name is None
        else [output]
    )
    name = input.name if isinstance(input, Sequential) else None
    return Sequential(head + tail, name=name)


def atomized(func):
    """
    Make the function return an Atom if `...` or an atom is passed as argument.

    In case `...` is passed as a positional argument, the function is wrapped into a
    Partial object. If an Atom object is passed as a positional argument, a new
    Sequential is returned that chains that atom with the atomized function (the
    input atom is never mutated). Otherwise, the function is called as is.

    Applied to an Atom subclass, `atomized` instead generates its function
    form (see :func:`as_function`): a function taking the data as first
    argument followed by the class parameters, with the same `...`/atom
    dispatch as above.

    Parameters
    ----------
    func: callable or type
        The function to wrap as a Partial atom if any `...` or input atom is passed.
        It must handle the `...` argument as a placeholder for the input data and for
        the passing states. It must return a unique output except if the function is
        stateful. In that case, the function must return the processed data as first
        output and the updated state as additional outputs. If an Atom subclass is
        given, its function form is returned instead.

    Returns
    -------
    output or atom: Any or (Partial or Sequential)
        if no `...` or Atom object is passed as a positional argument, returns the
        output of the function. If an Atom object is passed as a positional argument,
        returns a new Sequential chaining the Atom object and the atomized function.
        If `...` is passed as a positional argument, returns a Partial object containing
        the atomized function. This latter has the same documentation and names than the
        original function.

    Examples
    --------
    >>> import numpy as np
    >>> from xdas.atoms import atomized

    Basic usage:

    >>> @atomized
    ... def square(x):
    ...     return x ** 2
    >>> square(2)
    4
    >>> square(...)
    square(...)

    Passing an Atom object as input:

    >>> square(square(...))
    Sequence:
      0: square(...)
      1: square(...)

    Passing a stateful function:

    >>> @atomized
    ... def cumsum(x, cum=None):
    ...     return_state = cum is not None
    ...     if cum is None or cum is ...:
    ...         cum = 0.0
    ...     out = np.cumsum(x) + cum
    ...     cum += out[-1]
    ...     if return_state:
    ...         return out, cum
    ...     else:
    ...         return out
    >>> cumsum(..., cum=...)
    cumsum(...)  [stateful]

    """
    if isinstance(func, type) and issubclass(func, Atom):
        return as_function(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Dispatch to Partial/Sequential when ``...`` or an Atom is passed, else call directly."""
        if any(arg is ... for arg in args):
            return Partial(func, *args, **kwargs)
        elif objs := tuple(arg for arg in args if isinstance(arg, Atom)):
            if len(objs) == 1:
                input = objs[0]
            else:
                raise ValueError("Only one Atom object can be passed as function input")
            args = tuple(... if isinstance(arg, Atom) else arg for arg in args)
            return compose(input, Partial(func, *args, **kwargs))
        else:
            return func(*args, **kwargs)

    return wrapper


def _whole_record(dim_arg="dim"):
    """
    Mark a function as needing the whole record along its working dimension.

    The decorator to apply at the definition site of a whole-record function,
    under :func:`atomized`: the resulting atoms refuse chunked execution
    along the dimension named by the *dim_arg* argument (resolved from the
    call arguments, aliases included), via
    :meth:`Atom._refuse_chunked_along`.
    """

    def decorator(func):
        func._whole_record_dim_arg = dim_arg
        return func

    return decorator


def as_function(cls):
    """
    Generate the function form of an Atom subclass.

    A lowercase function taking the data as first argument, then the class
    parameters: called with data it builds the atom and applies it eagerly,
    called with ``...`` in the data slot it returns the configured atom, and
    called with an atom or a pipeline it returns a new pipeline extended
    with this atom.

    Parameters
    ----------
    cls : type
        The :class:`Atom` subclass to generate the function form of.

    Returns
    -------
    callable
        The function form, named after the class in snake case.
    """
    parameters = [
        parameter
        for key, parameter in inspect.signature(cls.__init__).parameters.items()
        if key != "self"
    ]

    def wrapper(da, *args, **kwargs):
        atom = cls(*args, **kwargs)
        if da is ...:
            return atom
        if isinstance(da, Atom):
            return compose(da, atom)
        return atom(da)

    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()
    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__module__ = cls.__module__
    wrapper.__signature__ = inspect.Signature(
        [inspect.Parameter("da", inspect.Parameter.POSITIONAL_OR_KEYWORD), *parameters]
    )
    wrapper.__doc__ = (
        f"Apply a :class:`{cls.__name__}` atom to `da`.\n\n"
        "Passing ``...`` as `da` returns the atom itself; passing an atom or a\n"
        "pipeline returns a new pipeline extended with this atom. The other\n"
        f"parameters are those of :class:`{cls.__name__}`, documented below.\n\n"
    ) + (inspect.getdoc(cls) or "")
    return wrapper
