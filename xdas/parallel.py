"""
How xdas spends the cores it is given.

Two kinds of work, sized by the same rule — never wider than the work is
worth. :func:`parallelize` splits one array across threads, which is enough
where the work releases the GIL. Scanning file metadata does not: it goes
through h5py, which holds one global HDF5 lock, so it has to cross to the
processes of :func:`get_scan_pool`.
"""

import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import numpy as np
from loky import ProcessPoolExecutor

from . import config

BYTES_PER_WORKER = 8 * 2**20
"""Array bytes a thread must be given before it is worth starting.

A thread costs a few milliseconds to start, hand a slice to and join, twice
per call. Below this much work per thread that overhead is the whole runtime:
splitting a small array across every core is slower than not splitting it.
"""

SCAN_THRESHOLD = 100
"""Files below which a scan stays in the calling process.

Measured break-even: under a hundred files, starting the pool costs more than
the scan it is meant to accelerate.
"""

SCAN_TIMEOUT = 300.0
"""Seconds an idle scan worker is kept alive.

Long enough to sit through the pauses of an interactive session, short enough
that workers orphaned by a killed parent do not outlive it for long.
"""


def parallelize(split_axis=0, concat_axis=0, parallel=None):
    """
    Split array positional arguments across threads.

    Parameters
    ----------
    split_axis : int or tuple of int, optional
        Axis (or axes) along which to split positional array arguments.
        Use ``None`` for arguments that should not be split.
    concat_axis : int or tuple of int, optional
        Axis (or axes) along which to concatenate the per-worker outputs.
    parallel : int, bool, or None, optional
        Worker count override.  Forwarded to :func:`get_workers_count`.

    Returns
    -------
    decorator : callable
        A function decorator.
    """

    def decorator(func):
        """Return a thread-parallelised wrapper for *func*."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Split inputs, dispatch to a thread pool, then concatenate outputs."""
            split_axes = split_axis if isinstance(split_axis, tuple) else (split_axis,)
            split_axes += (None,) * (len(args) - len(split_axes))
            inputs = tuple(
                value for value, axis in zip(args, split_axes) if axis is not None
            )
            input_axes = tuple(axis for axis in split_axes if axis is not None)
            args = tuple(value for value, axis in zip(args, split_axes) if axis is None)

            def fn(_inputs, tuplize=True):
                """Call *func* on one chunk; optionally wrap scalar output in a tuple."""
                _inputs = iter(_inputs)
                _args = iter(args)
                _args = tuple(
                    next(_inputs) if axis is not None else next(_args)
                    for axis in split_axes
                )
                _outputs = func(*_args, **kwargs)
                if tuplize and not isinstance(_outputs, tuple):
                    return (_outputs,)
                else:
                    return _outputs

            if all(value.ndim <= axis for value, axis in zip(inputs, input_axes)):
                return fn(inputs, tuplize=False)

            n_jobs = inputs[0].shape[input_axes[0]]
            nbytes = sum(value.nbytes for value in inputs)
            n_cores = get_workers_count(parallel, nbytes)
            n_workers = min(n_jobs, n_cores)
            if n_workers == 1:
                return fn(inputs, tuplize=False)

            if not all(
                value.shape[axis] == inputs[0].shape[input_axes[0]]
                for value, axis in zip(inputs, input_axes)
            ):
                raise ValueError(
                    "mismatch in size along parallelization axis between inputs"
                )
            inputs = list(
                zip(
                    *tuple(
                        np.array_split(value, n_workers, axis)
                        for axis, value in zip(input_axes, inputs)
                    )
                )
            )
            with ThreadPoolExecutor(n_workers) as executor:
                outputs = tuple(zip(*list(executor.map(fn, inputs))))
            concat_axes = (
                concat_axis if isinstance(concat_axis, tuple) else (concat_axis,)
            )
            concat_axes += (None,) * (len(outputs) - len(concat_axes))
            output = tuple(
                (
                    concatenate(value, axis, n_workers=n_workers)
                    if axis is not None
                    else value[0]
                )
                for axis, value in zip(concat_axes, outputs)
            )
            if len(output) == 1:
                return output[0]
            else:
                return output

        return wrapper

    return decorator


def concatenate(arrays, axis=0, out=None, dtype=None, n_workers=None):
    """
    Multithreaded version of numpy.concatenate.

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays: sequence of array_like
        The arrays must have the same shape, except in the dimension corresponding to
        `axis` (the first, by default).
    axis: int, optional
        The axis along which the arrays will be joined. Default is 0.
    out: ndarray, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what concatenate would have returned if no out argument were
        specified.
    dtype: str or numpy.dtype
        If provided, the destination array will have this dtype. Cannot be provided
        together with out.
    n_workers : int or None, optional
        Number of threads to use for writing chunks. None defers to the global
        xdas configuration. Default is None.

    Returns
    -------
    ndarray:
        The concatenated array.

    """
    arrays = [np.asarray(array, dtype) for array in arrays]

    ndim = {array.ndim for array in arrays}
    if len(ndim) == 1:
        (ndim,) = ndim
    else:
        raise ValueError("arrays must have the same number of dimensions.")

    dtype = {array.dtype for array in arrays}
    if len(dtype) == 1:
        (dtype,) = dtype
    else:
        raise ValueError("arrays must have the same dtype.")

    shapes = [list(array.shape) for array in arrays]
    section_sizes = [shape.pop(axis) for shape in shapes]
    subshape = {tuple(shape) for shape in shapes}
    if len(subshape) == 1:
        (subshape,) = subshape
    else:
        raise ValueError("arrays must have the same shape on axes other than `axis`.")
    shape = list(subshape)
    shape.insert(axis, sum(section_sizes))
    shape = tuple(shape)

    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if not (out.ndim == ndim and out.dtype == dtype and out.shape == shape):
            raise ValueError("`out` does not match with provided arrays.")

    div_points = np.cumsum([0] + section_sizes, dtype=int)

    with ThreadPoolExecutor(n_workers) as executor:
        for idx, array in enumerate(arrays):
            start = div_points[idx]
            end = div_points[idx + 1]
            slices = tuple(
                slice(start, end) if n == axis else slice(None) for n in range(ndim)
            )
            executor.submit(out.__setitem__, slices, array)

    return out


def get_workers_count(parallel, nbytes=None):
    """
    Get the number of cores to use for multithreaded operations.

    Parameters
    ----------
    parallel: int or bool, optional
        if `parallel` is an integer, that number of cores will be used. if `parallel`
        is a bool either single threading (False) will be used or all cores (True). If
        `parallel` is not given (None) the default value taken from the global xdas
        configuration will be used. You can see and update this value with
        `xdas.config.get("n_workers")` and `xdas.config.set("n_workers", <your_value>)`
    nbytes: int, optional
        Size of the work to be split. When given, and only when `parallel` is not
        specified, the count is scaled to it so that a small array is not split
        across more threads than it can keep busy. The configured value stays the
        ceiling.

    Returns
    -------
    n_workers: int
        The number of cores to use.

    Examples
    --------
    >>> import xdas
    >>> from xdas.parallel import get_workers_count

    An explicit request is honoured whatever the size of the work:

    >>> get_workers_count(4)
    4

    Left to itself the count follows the work, so that a small array is not split
    across more threads than it can keep busy:

    >>> previous = xdas.config.get("n_workers")
    >>> xdas.config.set("n_workers", 16)
    >>> get_workers_count(None, nbytes=2**20)
    1
    >>> get_workers_count(None, nbytes=2**30)
    16
    >>> xdas.config.set("n_workers", previous)

    """
    if parallel is None:
        n_workers = config.get("n_workers")
        if nbytes is not None:
            n_workers = min(n_workers, max(1, nbytes // BYTES_PER_WORKER))
        return n_workers
    elif isinstance(parallel, bool):
        if parallel:
            return os.cpu_count()
        else:
            return 1
    elif isinstance(parallel, int):
        return parallel
    else:
        raise TypeError("`parallel` must be either None, bool or int.")


_pool = None
"""The live scan pool, or None when there is none."""

_pool_workers = None
"""Worker count :data:`_pool` was built for."""


def _warm():
    """Import xdas in a worker, so that workers warm in parallel."""
    import xdas  # noqa: F401


def get_scan_workers(parallel, n_files):
    """
    Return the number of workers to scan *n_files* files with.

    Parameters
    ----------
    parallel : int, bool or None
        Worker count override, as taken by the ``open_mf*`` functions. An
        explicit request is always honoured, however few the files.
    n_files : int
        Number of files to be scanned.

    Returns
    -------
    int
        Workers to use; 1 means scanning in the calling process.

    Examples
    --------
    >>> from xdas.parallel import get_scan_workers

    Small scans stay at home, whatever the machine:

    >>> get_scan_workers(None, 10)
    1

    But an explicit request is still obeyed:

    >>> get_scan_workers(2, 10)
    2

    """
    if parallel is None and n_files < SCAN_THRESHOLD:
        return 1
    if parallel is None:
        return config.get("scan_workers")
    return get_workers_count(parallel)


def get_scan_pool(max_workers):
    """
    Return the shared scan pool, building it if it is not up yet.

    The pool is xdas's own rather than loky's shared ``get_reusable_executor``.
    That singleton is process-global: any caller anywhere asking for a different
    worker count silently replaces it, and replacing it waits for every worker
    of the old one to exit. Owning the pool also means owning how long it lives,
    which is what keeps a session responsive — workers stay warm across
    :data:`SCAN_TIMEOUT`, so the second scan of a session costs nothing.

    That timeout is deliberately long and deliberately finite. Loky does not
    notice a parent that dies to ``SIGKILL``, and neither does :mod:`atexit`, so
    the idle timeout is the only thing that ever reaps workers orphaned by a
    hard kill. It also bounds how long a worker can serve code its parent has
    since edited.

    Parameters
    ----------
    max_workers : int
        Number of worker processes wanted.

    Returns
    -------
    loky.ProcessPoolExecutor
        The pool. It is xdas's, and shared across scans, so it must not be shut
        down by its callers; use :func:`shutdown_scan_pool` for that.
    """
    global _pool, _pool_workers
    if _pool is not None and _pool_workers != max_workers:
        shutdown_scan_pool()
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers, timeout=SCAN_TIMEOUT, initializer=_warm
        )
        _pool_workers = max_workers
    return _pool


@atexit.register
def shutdown_scan_pool():
    """
    Shut the scan pool down, if one is up.

    Registered to run at interpreter exit, which covers every way of leaving a
    session but a hard kill; :data:`SCAN_TIMEOUT` covers that one.
    """
    global _pool, _pool_workers
    if _pool is not None:
        _pool.shutdown(kill_workers=True)
        _pool = None
        _pool_workers = None
