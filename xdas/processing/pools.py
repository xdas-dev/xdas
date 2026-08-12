"""
The worker pool behind ``pool="processes"``.

A process pool is the right answer for compressed HDF5 — decompression happens
under the global HDF5 lock, so only separate processes overlap it — but the
loaded chunk crossing back would be pickled through a pipe, which caps out
well below memory bandwidth and spends parent CPU doing it. This module
replaces that pipe with an arena of shared memory: the worker writes the chunk
once into a slot, the parent maps the same pages, and no bytes are serialized
in either direction.

The arena is a single file in ``/dev/shm``, sliced into fixed-size slots and
reused for the whole run. Reuse is the point: allocating a fresh block per
chunk makes the kernel zero and fault in every page again, which costs more
than it saves. Slots are handed out by the parent alone, so there is no
cross-process allocator and no lock in the data path.

What crosses the boundary is a :class:`ShmRef` — a shape/dtype placeholder
standing in for the array, exactly the duck type
:class:`~xdas.virtual.VirtualArray` already presents to
:class:`~xdas.core.DataArray`. Anything that is not a plain numeric array (a
virtual array, a DataFrame, an oversized chunk, an arena with no free slot)
simply travels the ordinary pickle path: the transport is an optimization, and
falling back to it is never wrong, only slower.

Releasing the memory is the other half of the design. Pages die with the last
mapping that holds them, so anything that gets to run Python on the way out —
an interrupt included — frees them at once, and the arena's *name* is unlinked
when the pool shuts down, again by a finalizer at interpreter exit, and swept
at the start of the next run if its owner is gone. Nothing in ``/dev/shm``
survives two runs.

A run killed outright is the exception, and not because of the arena: loky's
workers do not notice that their parent has died, so they stay, and what they
still hold mapped stays with them until they are killed too. That is worth
knowing when a scheduler or an out-of-memory killer ends a job, and it is the
same with or without shared memory in the picture.

All of that rests on POSIX semantics — unlinking a file that is still mapped,
and signal zero as a liveness probe — so the arena is built on POSIX only.
Elsewhere the pool is a plain process pool that pickles its chunks, slower but
identical in what it computes.
"""

import mmap
import os
import re
import shutil
import tempfile
import threading
import weakref
from uuid import uuid4

import numpy as np
from loky import ProcessPoolExecutor, process_executor

from ..core import DataArray

SLOT_NBYTES = 256 * 2**20
"""Default slot size, matching the target chunk size of ``chunks="auto"``."""

SHM_FRACTION = 0.25
"""Share of the free shared-memory filesystem a single arena may claim."""

PREFIX = "xdas-shm-"
"""Arena filename prefix, carrying the owner pid so stale ones can be swept."""

_ARENAS = {}
"""Mappings held by this process, path → mmap. Populated by :func:`attach`."""


def _directory():
    """Where arenas live: RAM-backed if the platform offers it, else temp."""
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


def _unlink(path):
    """Drop an arena's name. The pages go when the last mapping does."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _discard(path):
    """
    Let go of an arena entirely: its name, and this process's handle on it.

    Dropping the handle is what actually frees the pages, and it has to be
    done for a pool that was never shut down as much as for one that was, or a
    process that opens pools in a loop would hold every arena it ever made.
    The mapping itself is not closed: chunks still in the caller's hands are
    views on it, and they keep it alive on their own until they are collected.
    """
    _ARENAS.pop(path, None)
    _unlink(path)


def sweep():
    """
    Unlink arenas whose owner process is gone.

    A pool unlinks its own arena on shutdown and at interpreter exit, so this
    only ever finds arenas orphaned by a kill that ran no Python at all.
    """
    if os.name != "posix":  # pragma: no cover - signal zero kills on Windows
        return
    directory = _directory()
    try:
        names = os.listdir(directory)
    except OSError:  # pragma: no cover - unreadable shm directory
        return
    for name in names:
        match = re.fullmatch(rf"{PREFIX}(\d+)-[0-9a-f]+", name)
        if match is None:
            continue
        try:
            os.kill(int(match.group(1)), 0)
        except ProcessLookupError:
            _unlink(os.path.join(directory, name))
        except PermissionError:  # pragma: no cover - alive, another user
            pass


def attach(path, size):
    """
    Map an existing arena into this process.

    Runs in the parent when the arena is created, and in every worker through
    :func:`_init_worker`, so a worker is mapped before it is handed any task.

    Parameters
    ----------
    path : str
        Filesystem path of the arena.
    size : int
        Its size in bytes, as passed to :meth:`Arena.__init__`.
    """
    if path not in _ARENAS:
        fd = os.open(path, os.O_RDWR)
        try:
            _ARENAS[path] = mmap.mmap(fd, size)
        finally:
            os.close(fd)


def _init_worker(path, size):
    """
    Prepare a worker: map the arena, and stop loky reading it as a leak.

    A worker filling a slot sees its own resident memory grow by the size of
    the chunk it wrote, because shared pages are counted like any other. Loky
    watches that figure and recycles a worker that grows by more than 300 MB,
    so chunks of any real size had workers quitting and respawning mid-run.
    The arena is bounded and reused, so it is not what that check is for; it
    is discounted here, leaving the original allowance on top of it to catch
    leaks that are real.

    Parameters
    ----------
    path : str
        Filesystem path of the arena.
    size : int
        Its size in bytes.
    """
    if hasattr(process_executor, "_MAX_MEMORY_LEAK_SIZE"):  # pragma: no branch
        process_executor._MAX_MEMORY_LEAK_SIZE += size
    attach(path, size)


class ShmRef:
    """
    A placeholder for an array parked in an arena slot.

    Presents the ``shape``/``dtype``/``ndim`` surface
    :class:`~xdas.core.DataArray` asks of its data, so a chunk can carry one
    across the process boundary in place of its array and be rebuilt on the
    far side by whoever has the arena mapped.

    Parameters
    ----------
    path : str
        Arena the slot belongs to.
    offset : int
        Byte offset of the slot within the arena.
    shape : tuple of int
        Shape of the parked array.
    dtype : str
        Its dtype, as a ``numpy`` type string.
    """

    def __init__(self, path, offset, shape, dtype):
        self.path = path
        self.offset = offset
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        """Number of dimensions of the parked array."""
        return len(self.shape)

    @property
    def nbytes(self):
        """Size of the parked array in bytes."""
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    def __array__(self, dtype=None, copy=None):
        """Resolve to the shared pages. Requires this process to be mapped."""
        out = view(self)
        return out.astype(dtype) if dtype is not None else out

    def __repr__(self):
        return f"ShmRef(shape={self.shape}, dtype={self.dtype})"


def view(ref):
    """
    Return the array a :class:`ShmRef` points at, without copying.

    Parameters
    ----------
    ref : ShmRef
        Reference to resolve.

    Returns
    -------
    numpy.ndarray
        A view on the shared pages.
    """
    arena = _ARENAS.get(ref.path)
    if arena is None:
        raise RuntimeError(f"arena {ref.path} is not mapped in process {os.getpid()}")
    return np.ndarray(ref.shape, np.dtype(ref.dtype), arena, ref.offset)


def _offloadable(value, capacity):
    """Whether *value* is a chunk whose array can be parked in a slot."""
    if not isinstance(value, DataArray):
        return False
    data = value.data
    return (
        isinstance(data, np.ndarray)
        and not data.dtype.hasobject
        and 0 < data.nbytes <= capacity
    )


def _park(value, path, offset):
    """Copy *value*'s array into a slot and return it carrying a reference."""
    ref = ShmRef(path, offset, value.shape, value.data.dtype.str)
    np.copyto(view(ref), value.data)
    return value.copy(data=ref)


def _resolve(value):
    """Rebuild a chunk whose array was parked by the other side."""
    if isinstance(value, DataArray) and isinstance(value.data, ShmRef):
        return value.copy(data=view(value.data))
    return value


def _run(fn, outbox, args, kwargs):
    """
    Run one task in a worker, translating chunks in and out of the arena.

    Arguments arrive as references and are resolved to views; the result is
    parked in *outbox* when there is a slot for it and it is worth parking.
    """
    args = tuple(_resolve(arg) for arg in args)
    result = fn(*args, **kwargs)
    if outbox is not None:
        path, offset, capacity = outbox
        if _offloadable(result, capacity):
            result = _park(result, path, offset)
    return result


class Arena:
    """
    A file of shared memory, sliced into reusable fixed-size slots.

    Slots are reserved and released by the parent only, so the free list needs
    no cross-process synchronization — just a lock, because releases can come
    from a finalizer running on whichever thread the garbage collector was on.

    Parameters
    ----------
    nslots : int
        Number of slots.
    slot_nbytes : int
        Size of each slot, and so the largest array that can be parked.
    """

    def __init__(self, nslots, slot_nbytes):
        sweep()
        self.nslots = nslots
        self.slot_nbytes = slot_nbytes
        self.size = nslots * slot_nbytes
        self.path = os.path.join(
            _directory(), f"{PREFIX}{os.getpid()}-{uuid4().hex[:8]}"
        )
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, self.size)
        finally:
            os.close(fd)
        # Unlink on garbage collection and at interpreter exit, whichever
        # comes first; the sweep covers the case where neither runs.
        self._finalizer = weakref.finalize(self, _discard, self.path)
        try:
            attach(self.path, self.size)
        except OSError:  # pragma: no cover - mapping refused
            self._finalizer()
            raise
        self._free = list(range(nslots))
        self._lock = threading.Lock()

    @classmethod
    def create(cls, nslots, slot_nbytes):
        """
        Build an arena, or return ``None`` if shared memory cannot host it.

        Sizing is capped at :data:`SHM_FRACTION` of what the filesystem has
        free: the file is sparse, so an oversized arena costs nothing until it
        is used, but writing into pages the filesystem cannot back would take
        the worker down with a bus error rather than an exception.

        Parameters
        ----------
        nslots : int
            Desired number of slots; reduced to fit if need be.
        slot_nbytes : int
            Size of each slot.

        Returns
        -------
        Arena or None
            ``None`` when fewer than two slots fit, when the platform is not
            POSIX, or when the filesystem cannot be read.
        """
        if os.name != "posix":  # pragma: no cover - no unlink-while-mapped
            return None
        try:
            free = shutil.disk_usage(_directory()).free
        except OSError:  # pragma: no cover - unreadable directory
            return None
        nslots = min(nslots, int(free * SHM_FRACTION // slot_nbytes))
        if nslots < 2:
            return None
        try:
            return cls(nslots, slot_nbytes)
        except OSError:  # pragma: no cover - creation refused
            return None

    def reserve(self):
        """
        Take a free slot, or ``None`` if they are all out on loan.

        Returns
        -------
        tuple or None
            The ``(path, offset, capacity)`` an outbox is described by.
        """
        with self._lock:
            if not self._free:
                return None
            index = self._free.pop()
        return self.path, index * self.slot_nbytes, self.slot_nbytes

    def release(self, offset):
        """
        Hand a slot back to the free list.

        Parameters
        ----------
        offset : int
            Byte offset of the slot being released.
        """
        with self._lock:
            self._free.append(offset // self.slot_nbytes)

    def close(self):
        """
        Unlink the arena. Mappings already handed out stay valid.

        The pages are freed by the kernel once the last mapping goes, which is
        when the last chunk still pointing into them is collected.
        """
        self._finalizer()


class ProcessFuture:
    """
    The handle :class:`ProcessPool` hands out for a submitted task.

    Carries the same ``result()`` surface the loader and writer use, adding
    the step where a parked result is turned back into an array — a view on
    the shared pages, not a copy.

    Parameters
    ----------
    pool : ProcessPool
        Pool that submitted the task.
    inner : concurrent.futures.Future
        The underlying worker future.
    outbox : tuple or None
        Slot reserved for the result, if any.
    """

    def __init__(self, pool, inner, outbox):
        self._pool = pool
        self._inner = inner
        self._outbox = outbox
        self._value = None
        self._collected = False

    def result(self, timeout=None):
        """
        Wait for the task and return its result.

        Returns
        -------
        object
            Whatever the task returned. A chunk that travelled through the
            arena comes back read-only: its data is the shared page range
            itself, which the pool recycles once nothing refers to it any
            more.
        """
        if self._collected:
            return self._value
        try:
            value = self._inner.result(timeout)
        except BaseException:
            # Only a task that is over has stopped writing into its slot; one
            # that merely outran `timeout` still owns those pages.
            if self._inner.done():
                self._release()
            raise
        self._value = self._pool._collect(value, self._outbox)
        self._collected = True
        self._outbox = None
        return self._value

    def cancel(self):
        """Try to cancel the task, releasing its slot if that worked."""
        cancelled = self._inner.cancel()
        if cancelled:
            self._release()
        return cancelled

    def done(self):
        """Whether the task has finished, one way or another."""
        return self._inner.done()

    def _release(self):
        """Give back the result slot of a task that produced nothing."""
        if self._outbox is not None and self._pool._arena is not None:
            self._pool._arena.release(self._outbox[1])
        self._outbox = None


class ProcessPool:
    """
    A process pool whose chunks cross through shared memory.

    Quacks like a :class:`~concurrent.futures.Executor` as far as the loader
    and the writer need — ``submit``/``shutdown``/context manager — and keeps
    the per-process HDF5 locks a process pool is chosen for. What it adds is
    the data path: a chunk returned by a worker is written once into an arena
    slot and mapped by the parent, and a chunk sent to a worker is staged the
    same way. Neither direction serializes the array.

    The price of mapping rather than copying is immutability: chunk data
    arrives read-only, since the pages are the pool's to recycle. Atoms honor
    this already by allocating their outputs.

    Whatever cannot be parked — a chunk larger than a slot, an arena with no
    free slot, a result that is not an array — travels the ordinary pickle
    path instead, which is slower but always correct. Should shared memory be
    unavailable altogether, the pool degrades to exactly that.

    Parameters
    ----------
    max_workers : int
        Number of worker processes.
    max_buffers : int, optional
        Chunks the caller keeps in flight, which is what the arena is sized
        from. Default is 1.
    slot_nbytes : int, optional
        Largest chunk the arena can carry. Defaults to :data:`SLOT_NBYTES`.
    """

    def __init__(self, max_workers, max_buffers=1, slot_nbytes=None):
        if slot_nbytes is None:
            slot_nbytes = SLOT_NBYTES
        # Sized well past what streaming needs -- one slot per chunk in
        # flight, plus the one being consumed and the one whose release is
        # still pending. Spare slots are sparse pages that cost nothing until
        # they are written to, whereas running out quietly costs a pickle.
        nslots = 2 * max_buffers + max_workers + 4
        self._arena = Arena.create(nslots, slot_nbytes)
        if self._arena is None:
            self._executor = ProcessPoolExecutor(max_workers)
        else:
            self._executor = ProcessPoolExecutor(
                max_workers,
                initializer=_init_worker,
                initargs=(self._arena.path, self._arena.size),
            )

    def submit(self, fn, /, *args, **kwargs):
        """
        Schedule ``fn(*args, **kwargs)`` on a worker.

        Parameters
        ----------
        fn : callable
            The unit of work. Chunk arguments are staged into the arena and
            resolved back to arrays inside the worker.

        Returns
        -------
        ProcessFuture
            A ``result()``-able handle.
        """
        outbox = None
        staged = []
        if self._arena is not None:
            args = tuple(self._stage(arg, staged) for arg in args)
            outbox = self._arena.reserve()
        inner = self._executor.submit(_run, fn, outbox, args, kwargs)
        if staged:
            # The worker reads a staged argument while it runs, so the slot
            # comes back only once the task is over -- whatever its outcome.
            inner.add_done_callback(lambda _: self._release(staged))
        return ProcessFuture(self, inner, outbox)

    def shutdown(self, wait=True):
        """
        Shut the workers down and unlink the arena.

        Chunks still held by the caller keep their pages: the mapping outlives
        the name, and the kernel frees it when the last one is dropped.

        Parameters
        ----------
        wait : bool, optional
            Whether to block until running tasks complete.
        """
        self._executor.shutdown(wait=wait)
        if self._arena is not None:
            self._arena.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.shutdown()

    def _stage(self, value, staged):
        """Park an outgoing chunk in a slot, if it is one and one is free."""
        if not _offloadable(value, self._arena.slot_nbytes):
            return value
        slot = self._arena.reserve()
        if slot is None:
            return value
        path, offset, _ = slot
        staged.append(offset)
        return _park(value, path, offset)

    def _release(self, offsets):
        """Return staged argument slots to the free list."""
        for offset in offsets:
            self._arena.release(offset)

    def _collect(self, value, outbox):
        """Turn a parked result into a read-only view that recycles itself."""
        if outbox is None:
            return value
        offset = outbox[1]
        if not (isinstance(value, DataArray) and isinstance(value.data, ShmRef)):
            self._arena.release(offset)
            return value
        data = view(value.data)
        data.flags.writeable = False
        # The slot belongs to the chunk now: it goes back to the free list
        # when the array is collected, which is exactly when nothing can read
        # those pages any more.
        weakref.finalize(data, self._arena.release, offset)
        return value.copy(data=data)
