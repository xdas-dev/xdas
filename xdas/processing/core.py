"""
Core processing infrastructure for chunked pipeline execution.

Includes :class:`DataArrayLoader`, :class:`DataArrayWriter`,
:class:`DataFrameWriter`, :class:`StreamWriter`, :class:`ZMQPublisher`,
:class:`ZMQSubscriber` with the :class:`SubscriptionTracker` shared with the
ASN publisher, :class:`RealTimeLoader`, :func:`watch`, and the :func:`process`
dispatch boundary with its :func:`get_source` / :func:`get_writer` resolution
machinery.
"""

import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory

import numpy as np
import obspy
import pandas as pd
import zmq
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .. import config
from ..atoms.core import (
    _annotate_path,
    _announce_splits,
    _aschunks,
    _extend_path,
    _iter_results,
    _join_chunks,
)
from ..coordinates import AxisCoordinate
from ..coordinates.core import parse_scalar_delta
from ..core import (
    DataArray,
    DataCollection,
    DataMapping,
    DataSequence,
    concat,
    open_dataarray,
    open_mfdataarray,
)
from ..virtual import TileArray, VirtualBackend, VirtualStack
from .monitor import Monitor
from .pools import ProcessPool

AUTO_CHUNK_NBYTES = 256 * 2**20
"""Target in-memory chunk size (in bytes) for ``chunks="auto"``."""

WELCOME = b"xdas"
"""
Greeting a :class:`ZMQPublisher` sends to each subscriber as it registers it.

Never a packet — those are netCDF binaries — so subscribers skip it. Receiving
it is how a subscriber knows the publisher has applied its subscription.
"""


POOLS = {
    "threads": lambda max_workers, max_buffers, slot_nbytes: ThreadPoolExecutor(
        max_workers
    ),
    "processes": ProcessPool,
}
"""Worker pools available for chunk ingress and egress, name → factory."""


def get_pool(pool, max_workers, max_buffers=1, slot_nbytes=None):
    """
    Build the worker pool used to load or write chunks.

    Threads are enough when the work releases the GIL, but compressed HDF5
    does not: reading a chunk goes through one virtual layout, hence one h5py
    call holding the global HDF5 lock, so decompression of several chunks
    cannot overlap in-process and extra threads only contend. Worker
    processes each hold their own lock. What crosses to a worker is the
    *manifest* of the chunk (a sliced virtual array, kilobytes), not data,
    and the loaded chunk crosses *back* through shared memory: the worker
    writes it once into an arena slot, the parent maps the same pages
    (read-only). See :class:`~xdas.processing.pools.ProcessPool`.

    Parameters
    ----------
    pool : str
        Pool kind, ``"threads"`` (default everywhere) or ``"processes"``.
    max_workers : int
        Number of workers.
    max_buffers : int, optional
        Chunks the caller keeps in flight, which sizes the arena. Default 1.
    slot_nbytes : int, optional
        Largest chunk the arena can carry. Defaults to the ``chunks="auto"``
        budget; pass the real chunk size when it is known, so that no chunk
        has to fall back to the pickle path.

    Returns
    -------
    executor
        A :class:`~concurrent.futures.Executor`-like pool.

    Examples
    --------
    >>> from xdas.processing.core import get_pool
    >>> with get_pool("threads", 2) as pool:
    ...     pool.submit(abs, -1).result()
    1
    """
    if pool not in POOLS:
        raise ValueError(f"no worker pool named {pool!r}; available: {sorted(POOLS)}")
    if slot_nbytes is None:
        slot_nbytes = AUTO_CHUNK_NBYTES
    return POOLS[pool](max_workers, max_buffers, slot_nbytes)


def _load(da):
    """Load a (virtual) DataArray. The unit of work shipped to ingress workers."""
    return da.load()


def _dump(chunk, path, encoding):
    """Write *chunk* to *path* and return it virtually. Egress unit of work."""
    chunk.to_netcdf(path, encoding=encoding)
    return open_dataarray(path)


def process(atom, source, out=None, chunks=None, until=None, merge=True):
    """
    Execute a processing pipeline over any chunk source, into any sink.

    The dispatch boundary of chunked processing: the input is resolved into a
    chunk source with :func:`get_source` and the output into a writer with
    :func:`get_writer` (deferred to the first output chunk, so the writer can
    match what the pipeline actually emits). Explicit loader and writer
    instances pass through untouched, so the historical
    ``process(atom, data_loader, data_writer)`` form keeps working.

    Parameters
    ----------
    atom : Atom or callable
        The operation to execute on each chunk of data. It may emit zero or
        more output chunks per input chunk (seam tails, rechunking,
        reductions); at the end of the stream it is flushed so buffering
        atoms emit their remainder. :meth:`Atom.process` is this function
        with the atom bound.
    source : DataArray, DataCollection, str, Path, iterable or loader
        What to process. An in-memory :class:`DataArray` is processed in one
        eager call (or chunk by chunk if `chunks` is given); a virtual
        DataArray streams through a :class:`DataArrayLoader`; a
        :class:`~xdas.DataCollection` is walked leaf by leaf; a path,
        directory or glob pattern is opened with :func:`open_mfdataarray`
        first; a ``"tcp://..."`` address subscribes to a
        :class:`ZMQSubscriber`; any iterable of chunks (including
        :func:`watch` and generators) is consumed as is.
    out : str, Path, writer or None, optional
        Where to write the output. ``None`` (default) accumulates the output
        chunks in memory and returns the joined result, guarded by the
        ``"memory_limit"`` configuration entry. A path is matched with the
        first output chunk: a directory for DataArray (netcdf chunks) or
        Stream (SDS) chunks, a ``*.csv`` file for DataFrame chunks, a
        ``"tcp://..."`` address publishes DataArrays. Non-inferable
        configuration (miniseed data quality, encodings) is passed as a
        ready writer instance. Walking a collection, a single file, a URL
        or a ready writer is *shared* by every leaf — one table for the
        whole collection, the tree-path columns keeping the rows apart —
        while a directory fans out, one subdirectory per leaf mirroring the
        tree path (see :class:`_CollectionSink`).
    chunks : dict or "auto", optional
        Chunk sizes for DataArray sources, e.g. ``{"time": 1000}``.
        ``"auto"`` (the default for virtual sources) aligns chunk boundaries
        to the storage tiling, merged up to ``AUTO_CHUNK_NBYTES``.
    until : str, datetime64 or float, optional
        Stop processing at this coordinate value along the chunked dimension.
        The chunk containing it is truncated; the pipeline is then flushed
        normally. This is the clean way to bound an unbounded source.
    merge : bool, optional
        Whether to fold the per-leaf results of a collection walk through
        the atom's :attr:`~xdas.atoms.Atom.merge` hook, as ``atom(dc)``
        does. Walk-level, not stage-level, and only meaningful for
        ``out=None``: with any other sink the leaves were written as they
        were produced and there is nothing left to fold. Default ``True``.

    Returns
    -------
    result : object
        The writer result: the joined output for ``out=None``, whatever the
        resolved writer's ``result()`` returns otherwise, or ``None`` when
        the pipeline emitted no output (no empty outputs are created).
        Walking a collection, the result of a shared sink is its single
        result and everything else answers with the tree, so
        ``atom.process(dc, out=None)`` equals ``atom(dc)``.

    Notes
    -----
    Unbounded sources (:func:`watch`, ZMQ subscriptions — anything exposing
    ``unbounded = True``) get streaming semantics: no byte total on the
    progress monitor, and a clean :exc:`KeyboardInterrupt` stops the loop,
    flushes the pipeline and returns the writer result instead of raising.

    A :class:`~xdas.DataCollection` source is *walked*, exactly as
    ``atom(dc)`` walks it (see :meth:`~xdas.atoms.Atom.__call__`): each
    mapping level is first offered to the atom's
    :meth:`~xdas.atoms.Atom.gather` hook — before anything is chunked, so a
    channel level becomes a component dimension and the stacked array
    streams as one thing — then recursed into, sequence levels are folded
    element by element, and every leaf is streamed through the single-source
    path with `chunks` and `until` applying per leaf. One atom instance
    walks the leaves sequentially, reset between them: an atom holding a
    model either saturates the CPU or holds a lot of device memory, so only
    one should be live per node. Output chunks carry the tree path of the
    leaf that produced them as they are produced, so a streaming walk hands
    a leaf straight to a sink and still knows whose it was.

    A :class:`~xdas.DataSequence` handed over as a collection is walked like
    any other, so it folds rather than streaming as one concatenated result.
    A glob or a directory that *opens* to a sequence is still a single
    source, chained by :func:`get_source`; passing ``get_source(sequence)``
    explicitly asks for the same of a collection in hand.
    """
    if isinstance(source, (DataMapping, DataSequence)):
        return _process_collection(atom, source, out, chunks, until, merge)
    return _process_source(atom, source, out, chunks, until)


def _process_source(atom, source, out, chunks, until, path=None):
    """
    Stream one source through *atom* into one sink (see :func:`process`).

    The single-source path, and the only place chunks are actually driven.
    *path* is the tree path of the leaf being streamed, when this runs as
    one step of a collection walk: every output chunk is labelled with it
    on its way to the sink, as it is produced.
    """
    path = {} if path is None else path
    source = get_source(source, chunks)
    if isinstance(source, DataArray):
        # In-memory, unchunked: direct eager call, then sink dispatch on the
        # result so `process(da, out=...)` and `pipeline(da)` stay twins.
        if until is not None:
            # `until` truncates the source rather than the stream here, which
            # is the same cut: inclusive, as `sel` slices.
            dim = getattr(atom, "_resolve_dim", lambda _: None)(source) or "time"
            source = source.sel({dim: slice(None, until)})
        result = _annotate_path(atom(source), path)
        if out is None:
            return result
        outputs = _aschunks(result)
        if not outputs:
            return None
        # Nothing was chunked, so there is no chunk dimension to name: the
        # chunks one eager call returns are joined the way `concat` defaults to.
        writer = get_writer(out, outputs[0], "first")
        for chunk in outputs:
            writer.write(chunk)
        result = writer.result()
        _close_if_owned(writer, out)
        return result
    if hasattr(atom, "reset"):
        atom.reset()
    chunk_dim = getattr(source, "chunk_dim", "time")
    unbounded = bool(getattr(source, "unbounded", False))
    total = None if unbounded else getattr(source, "nbytes", None)
    if isinstance(until, str):
        until = np.datetime64(until)
    # Writer instances pass through upfront; inferred writers are deferred to
    # the first output chunk (the correct writer depends on what the pipeline
    # emits, and deferral avoids creating empty outputs).
    writer = out if hasattr(out, "write") and hasattr(out, "result") else None

    def write(chunk):
        nonlocal writer
        chunk = _annotate_path(chunk, path)
        if writer is None:
            writer = get_writer(out, chunk, chunk_dim)
        writer.write(chunk)

    if not unbounded and isinstance(getattr(source, "da", None), DataArray):
        # Free to know upfront: the source coordinate says where the runs
        # split, before any data is read.
        coord = source.da.coords.get(chunk_dim, None)
        if isinstance(coord, AxisCoordinate) and coord.isregular():
            indices = coord.get_split_indices(
                "discontinuities", getattr(coord, "tolerance", None)
            )
            if indices.size:
                _announce_splits(source.da, chunk_dim, int(indices.size))
    previous = None
    monitor = Monitor(total=total)
    monitor.tic("read")
    try:
        for chunk in source:
            last = False
            if until is not None and isinstance(chunk, DataArray):
                coord = chunk.coords.get(chunk_dim, None)
                if isinstance(coord, AxisCoordinate) and not coord.empty:
                    if coord.start > until:
                        break
                    if coord.end >= until:
                        # `until` is inclusive, like `sel(slice(None, until))`.
                        chunk = chunk.sel({chunk_dim: slice(None, until)})
                        last = True
            if unbounded and isinstance(chunk, DataArray):
                # A realtime source cannot be inspected upfront: announce
                # each seam as it arrives instead.
                previous = _announce_realtime_seam(previous, chunk, chunk_dim)
            monitor.tic("proc")
            result = atom(chunk, chunk_dim=chunk_dim)
            monitor.tic("write")
            for chunk_out in _aschunks(result):
                write(chunk_out)
            monitor.toc(getattr(chunk, "nbytes", 0))
            monitor.tic("read")
            if last:
                break
    except KeyboardInterrupt:
        if not unbounded:
            raise
    finally:
        if unbounded and hasattr(source, "stop"):
            source.stop()
    if hasattr(atom, "flush"):
        for chunk_out in atom.flush():
            write(chunk_out)
    monitor.close()
    if writer is None:
        return None
    result = writer.result()
    _close_if_owned(writer, out)
    return result


def _close_if_owned(writer, out):
    """
    Release a sink :func:`process` opened itself.

    A publisher built from a ``"tcp://..."`` spec holds a socket that nothing
    else will ever close, whereas one the caller passed in stays theirs to
    reuse and to close.
    """
    if writer is not out and isinstance(writer, ZMQEndpoint):
        writer.close()


def _process_collection(atom, dc, out, chunks, until, merge):
    """
    Walk a collection leaf by leaf, streaming each leaf into the sink.

    The streaming twin of :meth:`~xdas.atoms.Atom._walk`, and it mirrors it
    step for step so the two cannot drift apart: ``atom.process(dc,
    out=None)`` returns what ``atom(dc)`` returns.
    """
    walk = _Walk(atom, _CollectionSink(out), chunks, until)
    result = walk.sink.result(walk.level(dc, {}, ()))
    if out is None and merge and getattr(atom, "merge", None) is not None:
        return atom.merge(list(_iter_results(result)))
    return result


class _Walk:
    """
    One walk of a collection: what runs, where it writes, how it streams.

    Holds everything a walk keeps constant — the atom, the resolved sink,
    and the `chunks` and `until` that apply per leaf — so the recursion
    carries only what changes: the level, and the tree path it was reached
    by.

    Parameters
    ----------
    atom : Atom or callable
        The operation, one instance for the whole walk.
    sink : _CollectionSink
        The out spec, resolved per leaf.
    chunks, until
        As :func:`process`, applied per leaf.
    """

    def __init__(self, atom, sink, chunks, until):
        self.atom = atom
        self.sink = sink
        self.chunks = chunks
        self.until = until

    def level(self, x, path, where):
        """Walk one collection level, or stream *x* if it is a leaf."""
        if isinstance(x, DataMapping):
            gathered = self.atom.gather(x) if hasattr(self.atom, "gather") else None
            if gathered is not None:
                # Consulted before anything is chunked: the level becomes an
                # axis of the input and the stacked array streams as one
                # thing. Being consumed, it contributes no path column.
                return self.level(gathered, path, where)
            name = getattr(x, "name", None)
            return DataCollection(
                {
                    key: self.level(value, _extend_path(path, name, key), (*where, key))
                    for key, value in x.items()
                },
                name,
            )
        if isinstance(x, DataSequence):
            return self.fold(x, path, where)
        if hasattr(self.atom, "reset"):
            self.atom.reset()  # one atom instance, the leaves taken one by one
        return _asleaf(self.stream(self.atom, x, self.sink.spec(where), path))

    def fold(self, x, path, where):
        """
        Fold a sequence level: one stream delivered in pieces.

        State carries from element to element — the seams are judged from
        the coordinates, as everywhere else — so the atom is flushed once,
        after the last element, and its tail is attributed to that last
        element. The whole sequence shares one sink: it is one stream, so
        its chunks belong in one directory and concatenate into one result.
        Only the tree-path column tells the elements apart, and only
        approximately, since a buffering atom releases element *i-1*'s
        samples while element *i* is being fed.
        """
        name = getattr(x, "name", None)
        if hasattr(self.atom, "reset"):
            self.atom.reset()
        first = next((el for el in x if isinstance(el, DataArray)), None)
        resolve = getattr(self.atom, "_resolve_dim", None)
        dim = resolve(first) if resolve is not None else None
        if dim is None:
            # Nothing to fold along: each element is a leaf of its own.
            return DataCollection(
                [
                    self.level(
                        element, _extend_path(path, name, index), (*where, index)
                    )
                    for index, element in enumerate(x)
                ],
                name,
            )
        spec = self.sink.spec(where)
        writer = _SharedWriter(ResultWriter(None) if spec is None else spec)
        for index, element in enumerate(x):
            if not isinstance(element, DataArray):
                raise NotImplementedError(
                    "chunked processing of mapping collections is not supported: "
                    "process each leaf with its own atom instance"
                )
            source = get_source(element, self.chunks)
            if isinstance(source, DataArray):
                source = _Element(source, dim)
            self.stream(
                _Held(self.atom), source, writer, _extend_path(path, name, index)
            )
        # Past the `dim` resolution the operation is an atom, so it flushes
        # and resets; the tail is what its last element left buffered.
        for chunk in self.atom.flush():
            writer.write(
                _annotate_path(chunk, _extend_path(path, name, max(len(x) - 1, 0)))
            )
        self.atom.reset()
        result = writer.close()
        if spec is None:
            # As `Atom._fold`: the level answers with the chunks of its stream.
            return DataCollection(_aschunks(result), name)
        return _asleaf(result)

    def stream(self, atom, source, out, path):
        """Stream one leaf, or one element of a folded sequence, into *out*."""
        return _process_source(atom, source, out, self.chunks, self.until, path)


def _asleaf(result):
    """Normalize a leaf result: nothing written is an empty collection."""
    return DataCollection([]) if result is None else result


class _Held:
    """
    An atom facade whose flush is held back.

    A sequence level is one stream delivered in pieces, so its elements must
    not each end the stream: the state carries across and the tail is
    flushed once, by the walk, after the last element. Holding the flush is
    also what keeps :func:`_process_source` from resetting the atom between
    elements — the facade declares no ``reset``.
    """

    def __init__(self, atom):
        self.atom = atom

    def __call__(self, chunk, **flags):
        """Process one chunk through the held atom."""
        return self.atom(chunk, **flags)

    def flush(self):
        """Emit nothing: the stream is not over yet."""
        return []


class _Element:
    """One in-memory element of a folded sequence, as a single-chunk source."""

    def __init__(self, da, chunk_dim):
        self.da = da
        self.chunk_dim = chunk_dim

    @property
    def nbytes(self):
        """Size of the element, in bytes."""
        return self.da.nbytes

    def __iter__(self):
        yield self.da


class _SharedWriter:
    """
    One writer shared by several streams of a collection walk.

    A ``*.csv`` sink is one table for the whole collection — the tree-path
    columns keep the leaves apart — so the writer must outlive the leaf that
    created it and must not be closed by it. Resolution stays deferred to
    the first output chunk (:func:`get_writer`), and :meth:`result` answers
    nothing until the walk closes the sink itself.

    Parameters
    ----------
    out : str, Path or writer
        The out spec to resolve on the first chunk written.
    """

    def __init__(self, out):
        self.out = out
        self.writer = None

    def write(self, chunk):
        """Write one chunk, resolving the underlying writer on the first."""
        if self.writer is None:
            self.writer = get_writer(self.out, chunk)
        self.writer.write(chunk)

    def result(self):
        """Answer nothing: a shared sink is closed by the walk, not by a leaf."""
        return

    def close(self):
        """Close the underlying writer and return its result."""
        if self.writer is None:
            return None
        result = self.writer.result()
        _close_if_owned(self.writer, self.out)
        return result


class _CollectionSink:
    """
    The `out` spec of a collection walk, resolved per leaf.

    Three rules, one per kind of destination:

    - ``None`` accumulates each leaf in memory, so the walk answers with the
      tree the eager call returns;
    - a single file, a URL or a ready writer instance is **shared** by every
      leaf — one table, one archive, one socket for the whole collection,
      the tree-path columns each chunk already carries keeping the rows
      apart — and the walk answers with that one result;
    - a directory **fans out**: one subdirectory per leaf, mirroring the
      tree path, since a directory of netcdf chunks describes one stream.

    Parameters
    ----------
    out : str, Path, writer or None
        The out spec given to :func:`process`.
    """

    def __init__(self, out):
        self.out = out
        self.shared = None
        if out is None:
            pass
        elif hasattr(out, "write") and hasattr(out, "result"):
            self.shared = _SharedWriter(out)
        elif isinstance(out, (str, Path)):
            spec = str(out)
            if re.match(r"[a-z0-9+.-]+://", spec) or Path(spec).suffix:
                self.shared = _SharedWriter(out)
        else:
            raise TypeError(f"cannot infer a writer from `out` of type {type(out)}")

    def spec(self, where):
        """Return the out spec of the leaf reached by the keys *where*."""
        if self.shared is not None:
            return self.shared
        if self.out is None:
            return None
        # keyed on the keys themselves, not on the annotation path: an
        # unnamed level contributes no column but still needs its own
        # directory, or its leaves would overwrite one another
        return os.path.join(str(self.out), *(str(key) for key in where))

    def result(self, tree):
        """Return the walk's answer, given its walked *tree* of leaf results."""
        if self.shared is not None:
            return self.shared.close()
        return tree


def _announce_realtime_seam(previous, chunk, chunk_dim):
    """
    Warn when a realtime chunk arrives discontinuous with the previous one.

    Returns the seam information of *chunk*, to pass back on the next call.
    """
    coord = chunk.coords.get(chunk_dim, None)
    if not isinstance(coord, AxisCoordinate) or coord.empty:
        return previous
    info = {
        "start": coord.start,
        "end": coord.end,
        "delta": coord.get_sampling_interval(cast=False),
        "tolerance": parse_scalar_delta(
            getattr(coord, "tolerance", None), coord.dtype, default_zero=True
        ),
    }
    if previous is not None and previous["delta"] is not None:
        tolerance = max(previous["tolerance"], info["tolerance"])
        jump = info["start"] - (previous["end"] + previous["delta"])
        if jump > tolerance:
            warnings.warn(
                f"realtime source has a discontinuity along {chunk_dim!r} at "
                f"{info['start']}; state is flushed and reset",
                UserWarning,
                stacklevel=2,
            )
    if info["delta"] is None and previous is not None:
        info["delta"] = previous["delta"]
    return info


def watch(path, engine="xdas"):
    """
    Watch a directory for new files, as an unbounded chunk source.

    Sugar over :class:`RealTimeLoader`: every file closed under `path` is
    opened with `engine`, loaded, and yielded as a chunk. Realtime is always
    *named* — a bare directory path passed to :func:`process` means "process
    what is there", never "block forever".

    Parameters
    ----------
    path : str or Path
        Directory to watch.
    engine : str or Engine, optional
        Engine used to open arriving files. Defaults to ``"xdas"``.

    Returns
    -------
    RealTimeLoader
        An unbounded source for :func:`process` / :meth:`Atom.process`.

    Examples
    --------
    >>> pipeline.process(xd.watch("/incoming"), out="sds/")  # doctest: +SKIP
    """
    return RealTimeLoader(path, engine)


def get_source(source, chunks=None):
    """
    Resolve a :func:`process` input into a chunk source.

    The source contract is a duck-typed protocol: an iterable yielding
    chunks, optionally exposing ``chunk_dim``, ``nbytes`` and ``unbounded``.
    Anything already satisfying it (loaders, generators, collections) passes
    through. An in-memory :class:`DataArray` with no `chunks` is returned as
    is, meaning "process eagerly in one call".

    Parameters
    ----------
    source : DataArray, str, Path or iterable
        See :func:`process`.
    chunks : dict or "auto", optional
        Chunk sizes for DataArray sources. Defaults to ``"auto"`` for
        virtual DataArrays.

    Returns
    -------
    DataArray or iterable
        The chunk source, or a bare in-memory DataArray for the eager path.
    """
    if isinstance(source, (str, Path)):
        spec = str(source)
        match = re.match(r"(?P<scheme>[a-z0-9+.-]+)://", spec)
        if match:
            scheme = match["scheme"]
            if scheme not in SOURCE_SCHEMES:
                raise ValueError(
                    f"no source registered for the {scheme!r} URL scheme; "
                    f"available: {sorted(SOURCE_SCHEMES)}"
                )
            return SOURCE_SCHEMES[scheme](spec)
        if os.path.isdir(spec):
            spec = os.path.join(spec, "*")
        source = open_mfdataarray(spec)
    if isinstance(source, DataSequence) and all(
        isinstance(element, DataArray) and isinstance(element.data, VirtualBackend)
        for element in source
    ):
        # A virtual multi-acquisition collection: stream each run through its
        # own loader instead of materializing whole runs as single chunks.
        return _ChainSource(source, chunks)
    if isinstance(source, DataArray):
        if isinstance(chunks, dict):
            # A chunk cannot be larger than what it cuts: the last acquisition
            # of a sequence is routinely shorter than the size asked for, and
            # one chunk is what that means.
            chunks = {dim: min(size, source.sizes[dim]) for dim, size in chunks.items()}
        if isinstance(source.data, VirtualBackend):
            return DataArrayLoader(source, "auto" if chunks is None else chunks)
        if chunks is None:
            return source
        return DataArrayLoader(source, chunks)
    if hasattr(source, "__iter__") or hasattr(source, "__next__"):
        return source
    raise TypeError(
        f"cannot use a {type(source).__name__} object as a source: expected a "
        "DataArray, a path, directory or glob, a URL, or an iterable of chunks"
    )


def get_writer(out, chunk, chunk_dim="time"):
    """
    Resolve a :func:`process` output spec into a writer.

    Dispatch happens on *(out spec × first-chunk type)*: the same directory
    path means a netcdf chunk store for DataArray chunks and an SDS archive
    for Stream chunks. Writer instances (anything with ``write`` and
    ``result``) pass through.

    Parameters
    ----------
    out : str, Path, dict, writer or None
        See :func:`process`.
    chunk : object
        The first output chunk of the pipeline.
    chunk_dim : str, optional
        Dimension along which DataArray chunks follow each other; used to
        join them, in memory when `out` is ``None`` and virtually when it is
        a directory. It is the dimension the *source* was chunked along,
        which need not lead the output.

    Returns
    -------
    writer
        An object with ``write(chunk)`` and ``result()``.
    """
    if hasattr(out, "write") and hasattr(out, "result"):
        return out
    if out is None:
        return ResultWriter(chunk_dim)
    if not isinstance(out, (str, Path)):
        raise TypeError(f"cannot infer a writer from `out` of type {type(out)}")
    spec = str(out)
    match = re.match(r"(?P<scheme>[a-z0-9+.-]+)://", spec)
    if match:
        scheme = match["scheme"]
        if scheme not in SINK_SCHEMES:
            raise ValueError(
                f"no writer registered for the {scheme!r} URL scheme; "
                f"available: {sorted(SINK_SCHEMES)}"
            )
        return SINK_SCHEMES[scheme](spec)
    if isinstance(chunk, DataArray):
        if Path(spec).suffix:
            raise ValueError(
                f"cannot write DataArray chunks to {spec!r}: pass a directory "
                "(chunks are stored as netcdf files and virtually "
                "concatenated), or a configured writer instance"
            )
        return DataArrayWriter(spec, create_dirs=True, dim=chunk_dim)
    if isinstance(chunk, pd.DataFrame):
        if not spec.endswith(".csv"):
            raise ValueError(
                f"cannot write DataFrame chunks to {spec!r}: pass a `*.csv` "
                "path or a configured writer instance"
            )
        return DataFrameWriter(spec, create_dirs=True)
    if isinstance(chunk, obspy.Stream):
        return StreamWriter(spec, "D")
    raise TypeError(
        f"no writer known for output chunks of type {type(chunk).__name__}; "
        "pass a configured writer instance as `out`"
    )


class ResultWriter:
    """
    Accumulate output chunks in memory and join them at the end.

    The writer behind ``out=None``: chunks are collected as they arrive and
    ``result()`` returns them joined (gap-aware concatenation for DataArrays,
    :func:`pandas.concat` for DataFrames, merged :class:`obspy.Stream`).
    Accumulation is guarded by the ``"memory_limit"`` configuration entry.

    Parameters
    ----------
    chunk_dim : str, optional
        Dimension along which DataArray chunks are concatenated.
    """

    def __init__(self, chunk_dim="time"):
        self.chunk_dim = chunk_dim
        self.chunks = []
        self.nbytes = 0

    def write(self, chunk):
        """Accumulate one chunk, enforcing the in-memory size guard."""
        self.chunks.append(chunk)
        self.nbytes += _sizeof(chunk)
        limit = config.get("memory_limit")
        if self.nbytes > limit:
            raise ValueError(
                f"the accumulated in-memory result exceeds {_to_human(limit)} "
                "(the 'memory_limit' configuration entry): write the output "
                "to disk with `out=...`, or raise the limit with "
                "`xdas.config.set('memory_limit', ...)`"
            )

    def result(self):
        """Return the joined result, or ``None`` if nothing was written."""
        if self.chunks and all(isinstance(c, obspy.Stream) for c in self.chunks):
            out = obspy.Stream()
            for st in self.chunks:
                out += st
            return out
        return _join_chunks(self.chunks, self.chunk_dim)


class _ChainSource:
    """Chain per-run loaders over a virtual multi-acquisition collection."""

    def __init__(self, collection, chunks):
        self.loaders = []
        for element in collection:
            if isinstance(chunks, dict):
                ((dim, size),) = chunks.items()
                element_chunks = {dim: min(size, element.sizes[dim])}
            else:
                element_chunks = "auto" if chunks is None else chunks
            self.loaders.append(DataArrayLoader(element, element_chunks))

    @property
    def chunk_dim(self):
        """Chunked dimension, taken from the first per-run loader."""
        return self.loaders[0].chunk_dim

    @property
    def nbytes(self):
        """Total bytes over all runs."""
        return sum(loader.nbytes for loader in self.loaders)

    def __iter__(self):
        for loader in self.loaders:
            yield from loader


def _sizeof(chunk):
    """
    Return the in-memory size of an output chunk, in bytes.

    A pipeline emits arrays, tables and streams, and only the first knows
    `nbytes`: taking the others as weightless would silence the accumulation
    guard exactly where it matters, on the unbounded walk of a collection
    into one pick table.
    """
    if isinstance(chunk, pd.DataFrame):
        return int(chunk.memory_usage(deep=True).sum())
    if hasattr(chunk, "traces"):
        return sum(trace.data.nbytes for trace in chunk.traces)
    return getattr(chunk, "nbytes", 0)


def _to_human(nbytes):
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _auto_chunks(da):
    """
    Derive tile-aligned chunk boundaries from the storage blocking of *da*.

    Returns the chunked dimension and the list of chunk boundaries along it.
    The storage blocking is authoritative when there is one (tile extents for
    the tiles vtype, per-source extents for stacked HDF5 virtual datasets);
    consecutive blocks are merged up to the ``AUTO_CHUNK_NBYTES`` budget.
    Sources with no blocking (single files, in-memory arrays) fall back to
    fixed-size chunks from the same byte budget.
    """
    data = da.data
    if isinstance(data, TileArray):
        tiling = data.chunks
        axis = int(np.argmax([len(extents) for extents in tiling]))
        extents = list(tiling[axis])
    elif isinstance(data, VirtualStack):
        axis = data.axis
        extents = [source.shape[axis] for source in data.sources]
    else:
        axis, extents = 0, None
    dim = da.dims[axis]
    size = da.sizes[dim]
    nbytes_per_slice = max(da.nbytes // max(size, 1), 1)
    target = max(AUTO_CHUNK_NBYTES // nbytes_per_slice, 1)
    if extents is None:
        step = int(min(target, size))
        divs = list(range(0, size, step)) + [size]
        return dim, divs
    divs = [0]
    accumulated = 0
    for extent in extents:
        if accumulated and accumulated + extent > target:
            divs.append(divs[-1] + accumulated)
            accumulated = 0
        accumulated += extent
    # The loop always leaves the last block(s) in the accumulator.
    divs.append(divs[-1] + accumulated)
    return dim, divs


class DataArrayLoader:
    """
    A class to handle data chunked data ingestion.

    To optimize I/O latencies, chunks are loaded before they are used asynchronously
    in a buffer as soon as the iterator is created.

    Parameters
    ----------
    da : ``DataArray``
        The (virtual) DataArray that contains the data to be chunked
    chunks : dict or "auto"
        The sizes of the chunks along each dimension. Needs to be of the form:
        ``{"dim": int}``. The key correspond with the dimension (usually "time"),
        and the value is an integer indicating the size of the chunk (in samples)
        along that dimension. ``"auto"`` aligns chunk boundaries to the storage
        blocking of the array (tile extents, per-file extents), merged up to
        the ``AUTO_CHUNK_NBYTES`` byte budget.
    max_buffers : int, default=1
        The maximum number of chunks to load into memory at the same time.
    max_workers : int, default=1
        The maximum number of workers used to load the chunks.
    pool : {"threads", "processes"}, default="threads"
        The kind of workers to load with. Compressed HDF5 decodes under the
        global HDF5 lock, so several threads do not decode concurrently and
        ``max_workers`` above one only pays off with ``"processes"``: each
        worker receives the manifest of its chunk (kilobytes), reads its own
        files, and returns the loaded chunk through a shared-memory object
        store — zero-copy for the parent, with chunk data arriving
        read-only. Requires the optional ``ray`` dependency. See
        :func:`get_pool`.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.processing import DataArrayLoader
    >>> da = xd.open_dataarray(...)  # doctest: +SKIP

    Create chunks along the time dimension

    >>> chunks = {"time": 1000}
    >>> dl = DataArrayLoader(da, chunks)  # doctest: +SKIP

    Iterate over the chunks

    >>> for chunk in dl:
    ...     process(chunk)  # doctest: +SKIP

    Decode four chunks at a time, one worker process each

    >>> dl = DataArrayLoader(da, chunks, 4, 4, pool="processes")  # doctest: +SKIP

    """

    def __init__(self, da, chunks, max_buffers=1, max_workers=1, pool="threads"):
        if not isinstance(da, DataArray):
            raise TypeError(f"`da` must by a DataArray object, not a {type(da)}")
        if isinstance(chunks, str) and chunks == "auto":
            chunk_dim, divs = _auto_chunks(da)
            chunk_size = None
        elif isinstance(chunks, dict) and len(chunks) == 1:
            ((chunk_dim, chunk_size),) = chunks.items()
            chunk_dim = str(chunk_dim)
            chunk_size = int(chunk_size)
            if chunk_dim not in da.dims:
                raise ValueError(
                    f"chunking dimension {chunk_dim} not found in `da` "
                    f"dimensions {da.dims}"
                )
            if chunk_size > da.sizes[chunk_dim]:
                raise ValueError(
                    f"chunking size {chunk_size} is greater than `da` "
                    f"size {da.sizes[chunk_dim]} along dim {chunk_dim}"
                )
            size = da.sizes[chunk_dim]
            divs = list(range(0, size, chunk_size)) + [size]
        else:
            raise TypeError(
                "`chunks` must be a dict that maps a unique "
                "dimension to a unique size ({'dim': int}) or 'auto'"
            )
        self.da = da
        self.chunk_dim = chunk_dim
        self.chunk_size = chunk_size
        self._divs = divs
        self.max_buffers = max_buffers
        self.max_workers = max_workers
        self.pool = pool

    def __len__(self):
        return len(self._divs) - 1

    def _select(self, idx):
        """Return chunk *idx* as a lazy selection: the manifest, not the data."""
        start = self._divs[idx]
        end = self._divs[idx + 1]
        query = {
            dim: slice(start, end) if dim == self.chunk_dim else slice(None)
            for dim in self.da.dims
        }
        return self.da[query]

    def __iter__(self):
        # The chunk sizes are known here, so the arena can be cut to fit the
        # biggest of them and no chunk ever has to take the pickle path.
        largest = int(np.argmax(np.diff(self._divs)))
        with get_pool(
            self.pool, self.max_workers, self.max_buffers, self._select(largest).nbytes
        ) as executor:
            it = iter(range(len(self)))

            def submit(idx):
                # The task is the sliced virtual array, so a process worker
                # receives kilobytes and reads its own files.
                return executor.submit(_load, self._select(idx))

            futures = []
            try:
                for _ in range(self.max_buffers):
                    futures.append(submit(next(it)))
            except StopIteration:
                pass

            while futures:
                future = futures.pop(0)
                result = future.result()

                try:
                    futures.append(submit(next(it)))
                except StopIteration:
                    pass

                yield result

    @property
    def nbytes(self):
        """Total bytes of the underlying :class:`DataArray`."""
        return self.da.nbytes


class RealTimeLoader(Observer):
    """
    Real-time :class:`DataArray` loader that watches a directory for new files.

    Parameters
    ----------
    path : str or Path
        Directory to watch.
    engine : str or Engine, optional
        Engine used to open arriving files, given by name or as a configured
        :class:`~xdas.io.Engine` instance.  Defaults to ``"xdas"``.
    """

    chunk_dim = "time"
    unbounded = True

    def __init__(self, path, engine="xdas"):
        super().__init__()
        self.path = str(path) if isinstance(path, Path) else path
        self.queue = Queue()
        self.handler = Handler(self.queue, engine)
        self.schedule(self.handler, self.path, recursive=True)
        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.queue.get()
        if chunk is None:
            raise StopIteration
        else:
            return chunk


class Handler(FileSystemEventHandler):
    """Watchdog event handler that loads closed files into a queue."""

    def __init__(self, queue, engine):
        self.engine = engine
        self.queue = queue

    def on_closed(self, event):
        """Load the newly-closed file and place it in the queue."""
        da = open_dataarray(event.src_path, engine=self.engine)
        self.queue.put(da.load())


class DataArrayWriter:
    """
    A class to handle chunked data egress.

    Parameters
    ----------
    dirpath : str or Path
        The directory to store the output of a processing pipeline. The
        directory needs to exist.
    encoding : dict
        The encoding to use when dumping the DataArrays to bytes.
    max_buffers : int, default=1
        The maximum number of chunks to load into memory at the same time.
    max_workers : int, default=1
        The maximum number of thread used to load the chunks.
    create_dirs : bool, optional
        Whether to create parent directories if they do not exist. Default is False.
    dim : str, optional
        Dimension the chunks follow each other along, used to join them at
        :meth:`result`. Defaults to ``"first"``, which is only right when the
        chunked dimension leads the output: a pipeline emitting it elsewhere
        must name it.
    pool : {"threads", "processes"}, default="threads"
        The kind of workers to write with. Compression happens under the
        global HDF5 lock, so as on the read side several threads do not
        compress concurrently; ``"processes"`` sends each chunk to a worker
        through the shared-memory object store — one memcpy at memory
        bandwidth — in exchange for parallel compression. Requires the
        optional ``ray`` dependency. See :func:`get_pool`.
    mode : {"overwrite", "append"}, default="overwrite"
        With ``"overwrite"``, chunk files a previous writer left in
        `dirpath` are cleared before this one writes, so rerunning a
        pipeline at the same `dirpath` reflects only the new run. With
        ``"append"``, existing chunk files are kept and new ones continue
        the numbering, so a rerun accumulates onto the previous output.

    Examples
    --------
    >>> import xdas as xd
    >>> import xdas.processing as xp

    >>> expected = xd.DataArray(np.random.rand(1000, 100), dims=("time", "distance"))

    >>> dw = DataArrayWriter("some_path")  # doctest: +SKIP
    >>> for chunk in chunks:
    ...     dw.submit(chunk)  # doctest: +SKIP
    >>> result = dw.result  # doctest: +SKIP

    >>> assert result.equals(expected)  # doctest: +SKIP

    """

    _CHUNK_NAME_RE = re.compile(r"^\d{9}$")

    def __init__(
        self,
        dirpath,
        encoding=None,
        max_buffers=1,
        max_workers=1,
        create_dirs=False,
        dim="first",
        pool="threads",
        mode="overwrite",
    ):
        dirpath = str(dirpath) if isinstance(dirpath, Path) else dirpath
        if create_dirs:
            os.makedirs(dirpath, exist_ok=True)
        if not os.path.exists(dirpath):
            raise OSError(f"no directory {dirpath}")
        if mode not in ("overwrite", "append"):
            raise ValueError(f"`mode` must be 'overwrite' or 'append', got {mode!r}")
        existing = sorted(
            name for name in os.listdir(dirpath) if self._CHUNK_NAME_RE.match(name)
        )
        if mode == "overwrite":
            for name in existing:
                os.remove(os.path.join(dirpath, name))
            count = 0
        else:
            count = int(existing[-1]) + 1 if existing else 0
        self.dirpath = dirpath
        self.dim = dim
        self.encoding = encoding
        self.max_buffers = max_buffers
        self.max_workers = max_workers
        self.pool = pool
        self.mode = mode
        self._executor = get_pool(pool, self.max_workers, self.max_buffers)
        self._futures = []
        self._results = []
        self._count = count

    def submit(self, chunk):
        """
        Asynchronously write *chunk* to disk and register the path for later concat.

        Parameters
        ----------
        chunk : DataArray
            Processed data chunk to persist. Empty chunks are silently
            dropped (many flushes produce nothing).
        """
        if not isinstance(chunk, DataArray):
            raise TypeError(f"`chunk` must by a DataArray object, not a {type(chunk)}")
        if chunk.empty:
            return
        if not len(self._futures) < self.max_buffers:
            future = self._futures.pop(0)
            result = future.result()
            self._results.append(result)
        path = os.path.join(self.dirpath, f"{self._count:09d}")
        self._futures.append(self._executor.submit(_dump, chunk, path, self.encoding))
        self._count += 1

    def write(self, chunk):
        """Alias for :meth:`submit`."""
        return self.submit(chunk)

    def shutdown(self):
        """Shut down the internal thread pool."""
        self._executor.shutdown()

    def result(self):
        """Flush all pending writes and return the concatenated :class:`DataArray`."""
        while self._futures:
            future = self._futures.pop(0)
            result = future.result()
            self._results.append(result)
        self.shutdown()
        return concat(self._results, self.dim)


class DataFrameWriter:
    """
    A class for writing pandas DataFrames to a CSV file asynchronously.

    Parameters
    ----------
    path : str
        The path to the csv file.
    create_dirs : bool, optional
        Whether to create parent directories if they do not exist. Default is False.
    mode : {"overwrite", "append"}, default="overwrite"
        With ``"overwrite"``, an existing file at `path` is replaced by this
        run's output, so rerunning a pipeline at the same `path` reflects
        only the new run. With ``"append"``, rows are added to whatever the
        file already contains, letting a restarted acquisition keep filling
        the same table.

    Examples
    --------
    >>> import pandas as pd
    >>> import xdas.processing as xp

    >>> dw = xp.DataFrameWriter("output.csv")
    >>> for df in dfs:
    ...     dw.submit(dfs). # doctest: +SKIP
    >>> result = dw.result()  # doctest: +SKIP

    >>> expected = pd.concat(dfs, ignore_index=True)  # doctest: +SKIP
    >>> assert result.equals(expected)  # doctest: +SKIP

    """

    def __init__(self, path, create_dirs=False, mode="overwrite"):
        dirpath = os.path.dirname(path)
        if create_dirs and dirpath:
            os.makedirs(dirpath, exist_ok=True)
        if dirpath and not os.path.exists(dirpath):
            raise OSError(f"no directory {dirpath}")
        if mode not in ("overwrite", "append"):
            raise ValueError(f"`mode` must be 'overwrite' or 'append', got {mode!r}")
        self.path = str(path) if isinstance(path, Path) else path
        self.mode = mode
        # Clear eagerly, not on the first write: a run that submits nothing --
        # or only empty frames, which `submit` drops -- must still leave the
        # previous run's rows behind, as `DataArrayWriter` does.
        if mode == "overwrite" and os.path.exists(self.path):
            os.remove(self.path)
        self._started = False
        self._datetime_columns = []
        self._executor = ThreadPoolExecutor(1)
        self._future = None

    def submit(self, df):
        """
        Asynchronously append *df* to the CSV file.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame chunk to write. Empty frames are silently dropped.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` must by a DataFrame object, not a {type(df)}")
        if df.empty:
            return
        if self._future is not None:
            self._future.result()
        self._future = self._executor.submit(self._write, df)

    def write(self, df):
        """Alias for :meth:`submit`."""
        return self.submit(df)

    def _write(self, df):
        for column in df.columns:
            if (
                column not in self._datetime_columns
                and pd.api.types.is_datetime64_any_dtype(df[column])
            ):
                self._datetime_columns.append(column)
        # Only the first write of a run may need to start a fresh table;
        # every later write in the same run appends to what it just wrote.
        if not self._started and not (
            self.mode == "append" and os.path.exists(self.path)
        ):
            df.to_csv(self.path, mode="w", header=True, index=False)
        else:
            df.to_csv(self.path, mode="a", header=False, index=False)
        self._started = True

    def shutdown(self):
        """Shut down the internal thread pool."""
        self._executor.shutdown()

    def result(self):
        """Flush pending writes and return the full CSV as a :class:`pandas.DataFrame`."""
        if self._future is not None:
            self._future.result()
        self.shutdown()
        try:
            return pd.read_csv(self.path, parse_dates=self._datetime_columns)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()


class StreamWriter:
    """
    A class for writing obspy Streams to miniseed files asynchronously.

    Parameters
    ----------
    path : str
        The path of the miniseed file or the folder name where the miniseed files will
        be written.
    dataquality : str
        Data quality of the waveforms.
    kw_merge : dict
        Keyword arguments for merging the Streams, following the arguments of the
        obspy.core.stream.Stream.merge function.
    kw_write : dict
        Keyword arguments for writing the Streams, following the arguments of the
        obspy.core.stream.Stream.write function.
    output_format : str
        The output format of the miniseed files. Can be "flat" or "SDS".
        If "flat", the miniseed files will be written in a single file.
        If "SDS", the miniseed files will be written in the SDS file structure.
        For more information about SDS see:
        https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html
    mode : {"overwrite", "append"}, default="overwrite"
        With ``"overwrite"``, an existing destination file (the flat file,
        or an SDS day file this run touches) is replaced by this run's
        output. With ``"append"``, this run's data is merged with whatever
        the destination already holds before it is rewritten.

    Examples
    --------
    >>> import obspy
    >>> import numpy as np
    >>> import xdas as xd
    >>> import xdas.processing as xp

    Generate some DataArray:

    >>> data = np.random.randint(
    ...     low=-1000, high=1000, size=(1000, 10), dtype=np.int32
    ... )
    >>> starttime = np.datetime64("2023-01-01T00:00:00")
    >>> endtime = starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1)
    >>> distance = 5.0 * np.arange(data.shape[1])
    >>> da = xd.DataArray(
    ...     data=data,
    ...     coords={
    ...         "time": {
    ...             "tie_indices": [0, data.shape[0] - 1],
    ...             "tie_values": [starttime, endtime],
    ...             "sampling_interval": np.timedelta64(10, "ms"),
    ...         },
    ...         "distance": distance,
    ...     },
    ... )

    StreamWriter works great with the `DataArray.to_stream` method that can be used as
    an atom like this:

    >>> atom = lambda da, **kwargs: da.to_stream(
    ...     network="NT",
    ...     station="ST{:03}",
    ...     channel="HN1",
    ...     location="00",
    ...     dim={"distance": "time"},
    ... )
    >>> data_loader = xp.DataArrayLoader(da, chunks={"time": 100})

    This is how a StreamWriter can be used to write the data to a miniseed file:

    >>> kw_merge = {"method": 1}
    >>> kw_write = {"reclen": 4096}
    >>> data_writer = xp.StreamWriter(
    ...     "some_directory", "M", kw_merge, kw_write, output_format="SDS"
    ... )
    >>> result = xp.process(atom, data_loader, data_writer)

    The data will be written to the SDS file structure in the specified directory.

    >>> st = obspy.read("some_directory/2023/NT/*/HN1.D/NT.*.00.HN1.D.2023.001")

    Clean up:

    >>> import shutil
    >>> shutil.rmtree("some_directory")

    """

    def __init__(
        self,
        path,
        dataquality,
        kw_merge=None,
        kw_write=None,
        output_format="SDS",
        mode="overwrite",
    ):
        path = str(path) if isinstance(path, Path) else path
        if output_format == "SDS":
            os.makedirs(path, exist_ok=True)
            self.dirpath = path
            self.fname = None
        elif output_format == "flat":
            head, tail = os.path.split(path)
            if not os.path.exists(head):
                raise OSError(f"The directory {head} does not exist.")
            self.dirpath = head
            self.fname = tail
        else:
            raise ValueError(
                "output_format must be either 'SDS' or 'flat'. "
                f"Got {output_format} instead."
            )
        if mode not in ("overwrite", "append"):
            raise ValueError(f"`mode` must be 'overwrite' or 'append', got {mode!r}")
        self.dataquality = dataquality
        self.kw_merge = kw_merge if kw_merge is not None else {}
        self.kw_write = kw_write if kw_write is not None else {}
        self.output_format = output_format
        self.mode = mode
        self._executor = ThreadPoolExecutor(1)
        self._future = None

    @staticmethod
    def _split_and_fill(st):
        """Break any masked (gapped) trace into filled, contiguous pieces."""
        result = obspy.Stream()
        for tr in st:
            for piece in tr.split():
                if isinstance(piece.data, np.ma.masked_array):  # pragma: no cover
                    piece.data = piece.data.filled()
                result += piece
        return result

    def _to_SDS(self, st):
        for tr in st:
            new_st = obspy.Stream()
            new_st += tr
            new_st = new_st[0].split()
            for new_tr in new_st:
                if isinstance(new_tr.data, np.ma.masked_array):  # pragma: no cover
                    new_tr.data = new_tr.data.filled()
                new_tr.stats.mseed["dataquality"] = self.dataquality
            year = new_st[0].stats.starttime.year
            network = new_st[0].stats.network
            station = new_st[0].stats.station
            channel = new_st[0].stats.channel
            location = new_st[0].stats.location
            julday = new_st[0].stats.starttime.julday
            dirpath = os.path.join(
                self.dirpath, str(year), network, station, channel + ".D"
            )
            os.makedirs(dirpath, exist_ok=True)
            fname = f"{network}.{station}.{location}.{channel}.D.{year}.{julday:03d}"
            sds_path = os.path.join(dirpath, fname)
            if self.mode == "append" and os.path.exists(sds_path):
                merged = (obspy.read(sds_path) + new_st).merge(**self.kw_merge)
                new_st = self._split_and_fill(merged)
            new_st.write(sds_path, format="MSEED", **self.kw_write)

    def _to_flat(self, st):
        new_st = obspy.Stream()
        for tr in st:
            tmp_st = obspy.Stream()
            tmp_st += tr
            tmp_st = tmp_st[0].split()
            for new_tr in tmp_st:
                if isinstance(new_tr.data, np.ma.masked_array):  # pragma: no cover
                    new_tr.data = new_tr.data.filled()
                new_st += new_tr
        path = os.path.join(self.dirpath, self.fname)
        if self.mode == "append" and os.path.exists(path):
            merged = (obspy.read(path) + new_st).merge(**self.kw_merge)
            new_st = self._split_and_fill(merged)
        new_st.write(path, **self.kw_write)

    def submit(self, st):
        """
        Asynchronously write *st* to a temporary MiniSEED file.

        Parameters
        ----------
        st : obspy.Stream
            Stream chunk to persist.
        """
        if not isinstance(st, obspy.Stream):
            raise TypeError(f"`st` must be a Stream object, not a {type(st)}")
        if self._future is not None:
            self._future.result()
        self._future = self._executor.submit(self._write, st)

    def write(self, st):
        """Alias for :meth:`submit`."""
        return self.submit(st)

    def _write(self, st):
        st.write(f"{self.dirpath}/{st[0].stats.starttime}_tmp.mseed", **self.kw_write)

    def shutdown(self):
        """Shut down the internal thread pool."""
        self._executor.shutdown()

    def result(self):
        """Merge all temporary MiniSEED files and write the final output."""
        if self._future is None:
            # A pipeline that emitted nothing leaves nothing to merge; a
            # writer passed in by the caller still gets asked for its result.
            self.shutdown()
            return obspy.Stream()
        self._future.result()
        self.shutdown()
        pattern = f"{self.dirpath}/*_tmp.mseed"
        out = obspy.read(pattern)
        out = out.merge(**self.kw_merge)
        if self.output_format == "flat":
            self._to_flat(out)
        elif self.output_format == "SDS":  # pragma: no branch
            self._to_SDS(out)
        files_to_remove = glob(pattern)
        for file in files_to_remove:
            os.remove(file)
        return out


class ZMQEndpoint:
    """
    Ownership of the ZeroMQ context and socket an endpoint speaks through.

    A socket nobody closes keeps its file descriptor, and its context keeps an
    I/O thread, until the process ends. Releasing them is therefore part of the
    interface rather than an afterthought: as a context manager where the
    endpoint has a scope, with :meth:`close` where it does not, and on garbage
    collection for one that is merely dropped — the last of which is a safety
    net, not the way to write it, since when it runs is not for the caller
    to know.

    Endpoints inheriting this must expose their socket as ``_socket`` and the
    context it came from as ``_context``.
    """

    _socket = None
    _context = None

    def close(self):
        """Close the socket and terminate the context. Closing twice is a no-op."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SubscriptionTracker(ZMQEndpoint):
    """
    Subscriber bookkeeping for publishers that own a ``zmq.XPUB`` socket.

    A ZeroMQ publisher silently drops whatever it sends before a subscriber's
    subscription has travelled to it — the "slow joiner" problem. Unlike
    ``PUB``, an ``XPUB`` socket hands each subscription to the application, and
    only once it has been applied to the socket's routing table. Reading one is
    therefore proof that the peer is connected and that everything sent from
    then on reaches it.

    A real-time publisher never waits: an instrument streams what it measures
    whether or not anyone listens, and a subscriber that joins late is meant to
    pick the stream up from wherever it lands. So the count is there to be
    *observed* — and to be waited on by the one publisher that is not
    real-time, the one replaying a recording, where the head of a finite stream
    would otherwise be lost to whoever was not connected yet.

    A mixin rather than part of :class:`ZMQPublisher` because the ASN publisher
    (:class:`xdas.io.asn.ZMQPublisher`), which speaks the protocol of the
    instrument rather than this one, needs exactly the same bookkeeping.

    Publishers mixing this in must expose their bound ``zmq.XPUB`` socket as
    ``_socket`` with ``zmq.XPUB_VERBOSE`` set — so that peers subscribing to an
    already-subscribed topic are announced too — the context it came from as
    ``_context``, their address as ``address``, and initialize
    ``_nsubscribers`` to zero. Closing comes with :class:`ZMQEndpoint`.
    """

    @property
    def nsubscribers(self):
        """The number of currently subscribed peers."""
        self._read_subscriptions(0.0)
        return self._nsubscribers

    def wait_for_subscribers(self, count=1, timeout=60.0):
        """
        Block until at least *count* peers are subscribed.

        Whatever is published once this returns is guaranteed to reach those
        peers. This is for replaying a finite recording, where losing the first
        packets to a subscriber that had not connected yet would lose them for
        good; a real-time publisher has nothing to wait for and never calls it.

        Parameters
        ----------
        count : int, optional
            The number of subscribers to wait for. Defaults to one.
        timeout : float or None, optional
            How many seconds to wait at most. None waits forever.

        Returns
        -------
        int
            The number of subscribers connected once the wait is over, which
            can exceed *count* if several peers joined at once.

        Raises
        ------
        TimeoutError
            If fewer than *count* subscribers showed up in time.

        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.nsubscribers < count:
            remaining = None if deadline is None else deadline - time.monotonic()
            if not self._read_subscriptions(remaining):
                raise TimeoutError(
                    f"got {self._nsubscribers} of the {count} subscriber(s) "
                    f"expected on {self.address} after {timeout} seconds"
                )
        return self._nsubscribers

    def _read_subscriptions(self, timeout):
        """
        Fold pending subscription events into the subscriber count.

        Waits up to *timeout* seconds (None waits forever) for a first event,
        then folds in whatever else is already queued. Returns whether any
        event was read.
        """
        wait = None if timeout is None else max(0, round(1000 * timeout))
        received = False
        while self._socket.poll(wait, zmq.POLLIN):
            # XPUB only ever delivers subscriptions (\x01) and their
            # cancellations (\x00), both followed by the topic.
            if self._socket.recv().startswith(b"\x01"):
                self._nsubscribers += 1
            else:
                self._nsubscribers -= 1
            received = True
            wait = 0
        return received


class ZMQPublisher(SubscriptionTracker):
    """
    A class for publishing DataArray chunks over ZeroMQ.

    Parameters
    ----------
    address : str
        The address to bind the publisher to.
    encoding : dict
        The encoding to use when dumping the DataArrays to bytes.

    Attributes
    ----------
    nsubscribers : int
        The number of currently subscribed peers.

    Methods
    -------
    submit(da)
        Send a DataArray over ZeroMQ.
    wait_for_subscribers(count, timeout)
        Blocks until *count* peers are subscribed, so that nothing published
        afterwards is dropped.
    close()
        Release the socket and its context.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.processing import ZMQPublisher, ZMQSubscriber

    First we generate some data and split it into packets

    >>> packets = xd.split(xd.testing.dummy(), 10)

    We initialize the publisher at a given address. Used as a context manager,
    it releases its socket on the way out.

    >>> address = f"tcp://localhost:{xd.io.get_free_port()}"

    We can then publish the packets

    >>> with ZMQPublisher(address) as publisher:
    ...     for da in packets:
    ...         publisher.submit(da)

    To reduce the size of the packets, we can also specify an encoding

    >>> import hdf5plugin

    >>> address = f"tcp://localhost:{xd.io.get_free_port()}"
    >>> encoding = {"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=1e-6)}
    >>> with ZMQPublisher(address, encoding) as publisher:
    ...     for da in packets:
    ...         publisher.submit(da)

    """

    def __init__(self, address, encoding=None):
        self.address = address
        self.encoding = encoding
        self._nsubscribers = 0
        self._context = zmq.Context()
        # XPUB publishes exactly like PUB but also reports who subscribes.
        self._socket = self._context.socket(zmq.XPUB)
        self._socket.setsockopt(zmq.XPUB_VERBOSE, True)
        # The greeting is a socket option, not a rendez-vous: the publisher
        # waits for nobody, and hands it to each new peer in passing, on the
        # next `submit`. A subscriber that receives it knows it is registered.
        self._socket.setsockopt(zmq.XPUB_WELCOME_MSG, WELCOME)
        self._socket.bind(self.address)

    def submit(self, da):
        """
        Send a DataArray over ZeroMQ.

        Parameters
        ----------
        da : DataArray
            The DataArray to be sent.

        """
        # Taking the subscriptions the socket has queued is what greets the
        # peers behind them — ZeroMQ holds a welcome message back until the
        # application reads the subscription it answers — and what keeps the
        # subscriber count current. Neither costs the publisher any waiting.
        self._read_subscriptions(0.0)
        self._socket.send(tobytes(da, self.encoding))

    def write(self, da):
        """Alias for :meth:`submit`."""
        self.submit(da)

    def result(self):
        """Return ``None`` — ZMQPublisher has no aggregated result."""
        return


class ZMQSubscriber(ZMQEndpoint):
    """
    A class for subscribing to DataArray chunks over ZeroMQ.

    Parameters
    ----------
    address : str
        The address to connect the subscriber to.
    timeout : float or None, optional
        How many seconds to wait at most for each packet. None, the default,
        waits forever.

    Methods
    -------
    wait_until_subscribed()
        Block until the publisher has registered this subscription.
    close()
        Release the socket and its context.

    Examples
    --------
    >>> import threading

    >>> import xdas as xd
    >>> from xdas.processing import ZMQSubscriber

    First we generate some data and split it into packets

    >>> da = xd.testing.dummy()
    >>> packets = xd.split(da, 10)

    >>> address = f"tcp://localhost:{xd.io.get_free_port()}"
    >>> publisher = ZMQPublisher(address)

    A publisher drops what it sends to subscribers it does not know about yet.
    Here the packets come from a recording and we want every one of them, so
    the replay waits for its audience — a real-time publisher does not, and its
    subscribers pick the stream up wherever they land, waiting instead with
    :meth:`ZMQSubscriber.wait_until_subscribed`.

    >>> def publish():
    ...     publisher.wait_for_subscribers()
    ...     for packet in packets:
    ...         publisher.submit(packet)

    >>> thread = threading.Thread(target=publish)
    >>> thread.start()

    Now let's receive the packets. The subscriber is an infinite iterator, so
    we stop it once the whole stream has been received.

    >>> subscriber = ZMQSubscriber(address)
    >>> received = []
    >>> for packet in subscriber:
    ...     received.append(packet)
    ...     if len(received) == len(packets):
    ...         break
    >>> assert xd.concat(received).equals(da)

    Both ends hold a socket until they are closed. Where the two cannot be
    written as one ``with`` block, as here, closing them by hand does the same.

    >>> thread.join()
    >>> subscriber.close()
    >>> publisher.close()
    """

    chunk_dim = "time"
    unbounded = True

    def __init__(self, address, timeout=None):
        self.address = address
        self.timeout = timeout
        self._subscribed = False
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            message = self._recv()
            if message == WELCOME:
                # Sent again whenever the socket reconnects, e.g. to a
                # publisher that restarted.
                self._subscribed = True
            else:
                return frombuffer(message)

    def wait_until_subscribed(self):
        """
        Block until the publisher has registered this subscription.

        A publisher drops what it sends to a peer it does not know about yet,
        and a subscriber cannot tell from its own side whether its
        subscription has arrived — being connected is not being subscribed.
        The publisher answers it with a greeting, in passing as it streams, and
        receiving that greeting is proof that nothing published from then on
        will be missed. Nothing is asked of the publisher in return, and a
        real-time stream is never delayed by anyone.

        It follows that only a publisher that publishes can acknowledge
        anybody: waiting on one that has gone quiet — or on an address nothing
        is bound to — raises :exc:`TimeoutError` once the subscriber's
        ``timeout`` has passed, and waits forever without one. To be sure of
        receiving a *recording* whole, it is the publisher that must wait, with
        :meth:`ZMQPublisher.wait_for_subscribers`, since nothing a subscriber
        does can hold back a replay that has already started.

        Returns immediately once the publisher has greeted this subscriber.
        """
        while not self._subscribed:
            self._subscribed = self._recv() == WELCOME

    def _recv(self):
        if self.timeout is not None and not self._socket.poll(
            round(1000 * self.timeout)
        ):
            raise TimeoutError(
                f"no packet received from {self.address} after {self.timeout} seconds"
            )
        return self._socket.recv()


def tobytes(da, encoding=None):
    """
    Serialise *da* to raw NetCDF4 bytes via a temporary file.

    Parameters
    ----------
    da : DataArray
        DataArray to serialise.
    encoding : dict, optional
        HDF5/NetCDF4 encoding options forwarded to :meth:`DataArray.to_netcdf`.

    Returns
    -------
    bytes
    """
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tmp.nc")
        da.to_netcdf(path, virtual=False, encoding=encoding)
        with open(path, "rb") as file:
            return file.read()


def frombuffer(da):
    """
    Deserialise raw NetCDF4 *da* bytes into a loaded :class:`DataArray`.

    Parameters
    ----------
    da : bytes
        Raw bytes as produced by :func:`tobytes`.

    Returns
    -------
    DataArray
    """
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tmp.nc")
        with open(path, "wb") as file:
            file.write(da)
        return open_dataarray(path).load()


SOURCE_SCHEMES = {"tcp": ZMQSubscriber}
"""Registry of URL schemes accepted as sources, scheme → source factory."""

SINK_SCHEMES = {"tcp": ZMQPublisher}
"""Registry of URL schemes accepted as sinks, scheme → writer factory."""
