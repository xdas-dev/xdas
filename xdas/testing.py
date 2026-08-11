"""Test utilities for xdas."""

import warnings

import numpy as np
import pandas as pd

from .coordinates import Coordinate
from .core import DataArray, DataCollection, concat, split


def dummy(
    dims=("time", "distance"),
    shape=(100, 10),
    dtype=float,
    step=(0.01, 10.0),
    ctype="interpolated",
    datetime=True,
):
    """
    Return a minimal :class:`DataArray` for quick testing.

    Parameters
    ----------
    dims : tuple of str, optional
        Dimension names. Length must match ``shape``. Defaults to
        ``("time", "distance")``.
    shape : tuple of int, optional
        Size along each dimension. Defaults to ``(100, 10)``.
    dtype : dtype-like, optional
        Data type for the array values. Defaults to ``float``.
    step : scalar or tuple, optional
        Step size for each dimension. A single value is applied to all
        dimensions; a tuple must have the same length as ``dims``. Defaults
        to ``(0.01, 10.0)`` (100 Hz, 10 m spacing → 1 s × 100 m total).
        When ``datetime=True``, a float step for the first dimension is
        interpreted as seconds and converted to :class:`numpy.timedelta64`.
    ctype : {"interpolated", "sampled", "dense"}, optional
        Coordinate type for all dimensions. Defaults to ``"interpolated"``.
    datetime : bool, optional
        If ``True`` (default), the first dimension uses
        :class:`numpy.datetime64` coordinates starting at 2024-05-21.
        All other dimensions use float coordinates starting at 0.0.

    Returns
    -------
    DataArray
        Array filled with sequential integers (via :func:`numpy.arange`)
        reshaped to ``shape`` and cast to ``dtype``.

    Examples
    --------
    >>> import xdas as xd
    >>> da = xd.testing.dummy()
    >>> da.shape
    (100, 10)
    >>> da = xd.testing.dummy(dims=("x",), shape=(50,), datetime=False, step=1.0)
    >>> da.shape
    (50,)
    >>> da = xd.testing.dummy(dims=("x",), shape=(10,), datetime=False, step=2.0)
    >>> float(da.coords["x"].sampling_interval)
    2.0

    """
    if len(dims) != len(shape):
        raise ValueError(f"len(dims)={len(dims)} must equal len(shape)={len(shape)}")
    if isinstance(step, (tuple, list)) and len(step) != len(dims):
        raise ValueError(f"len(step)={len(step)} must equal len(dims)={len(dims)}")

    data = np.arange(int(np.prod(shape))).reshape(shape).astype(dtype)

    coords = {}
    for i, (dim, size) in enumerate(zip(dims, shape)):
        s = step[i] if isinstance(step, (tuple, list)) else step
        if datetime and i == 0:
            start = np.datetime64("2024-05-21T00:00:00.000000000")
            if isinstance(s, (int, float)):
                s = np.timedelta64(int(s * 1e9), "ns")
        else:
            start = 0.0
        coords[dim] = Coordinate[ctype].from_block(start, size, s, dim=dim)

    return DataArray(data=data, coords=coords)


def assert_chunk_invariant(
    pipeline, da, chunks, rtol=1e-7, atol=0.0, coord_atol=0, gaps=None, cuts=1
):
    """
    Assert that *pipeline* answers the same however the stream is cut.

    Chunk-safety is a claim an atom makes about itself; this is the evidence.
    The pipeline is run once eagerly on the whole array and once streamed
    chunk by chunk with the given `chunks`, and the two results are required
    to match — values, coordinates and all. On top of that single split, the
    invariant is quantified over *cuts*: the same stream is re-chunked at
    other sizes (whose boundaries fall elsewhere, including across any gap)
    and every cutting must answer the same. With `gaps`, discontinuities are
    injected into the input first, so seam resets are exercised at chunk
    boundaries that do not line up with them.

    Parameters
    ----------
    pipeline : Atom
        The pipeline to check. It has to be an atom: a bare callable cannot
        be streamed, since the chunked path hands each chunk a ``chunk_dim``.
        Wrap one with :class:`xdas.atoms.Partial`.
    da : DataArray
        The input to run it on.
    chunks : dict
        Chunk sizes for the streamed run, e.g. ``{"time": 100}``.
    rtol, atol : float, optional
        Tolerances for the value comparison, as in
        :func:`numpy.testing.assert_allclose`.
    coord_atol : int or float, optional
        Tolerance on the dimension coordinates, in their own units
        (nanoseconds for datetime axes). Zero — the default — demands an
        exact match. Rational resampling to a rate that is not an exact
        number of nanoseconds reconstructs its output grid segment by
        segment, so eager and chunked coordinates may differ by a nanosecond
        with bit-identical values; that is what this admits, explicitly.
    gaps : int or sequence of int, optional
        Inject gaps into `da` along the chunked dimension before comparing:
        an int places that many evenly spaced gaps, a sequence gives the
        sample indices where each gap starts. Each gap drops a twentieth of
        the record (at least one sample), which is well beyond any jitter
        tolerance, so the seams judge them as real discontinuities.
    cuts : int or sequence of dict, optional
        How many extra cuttings to check beyond `chunks` (cut-invariance):
        the result must be a function of *which samples were processed*,
        never of where the stream was cut. An int derives that many
        alternative chunk sizes from `chunks` (each smaller and coprime-ish,
        so the boundaries land elsewhere); a sequence gives explicit
        ``chunks``-style dicts. ``0`` restores the single-split check.

    Raises
    ------
    AssertionError
        If any two runs disagree, in shape, in coordinates or in values.

    Notes
    -----
    Pick tables are compared as *sets* of rows: eager processing walks the
    whole record lane by lane while chunked processing walks chunk by chunk,
    so the rows come out in a different order with the same content, and it
    is the content that the invariant is about.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd
    >>> from xdas.atoms import Partial
    >>> da = xd.testing.dummy()
    >>> xd.testing.assert_chunk_invariant(Partial(np.square), da, {"time": 25})

    Injecting gaps exercises the seam resets as well:

    >>> xd.testing.assert_chunk_invariant(
    ...     xd.filter(..., (None, 10.0), dim="time"), da, {"time": 25}, gaps=2
    ... )

    A pipeline that is *not* chunk-invariant says so rather than passing:

    >>> def normalize(da):
    ...     return da / np.std(da.values)
    >>> xd.testing.assert_chunk_invariant(Partial(normalize), da, {"time": 25})
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    AssertionError

    """
    ((dim, size),) = chunks.items()
    with warnings.catch_warnings():
        if gaps is not None:
            # The gaps are this function's own doing: the split announcement
            # would only tell the test suite what it already decided.
            warnings.filterwarnings("ignore", message="source starting at")
            da = inject_gaps(da, dim, gaps)
            size = min(size, da.sizes[dim])
            chunks = {dim: size}
        pipeline.reset()
        eager = pipeline(da)
        pipeline.reset()
        chunked = _stream(pipeline, da, chunks)
        _assert_same(eager, chunked, rtol, atol, coord_atol, "result")
        for cut in _cuttings(cuts, dim, size):
            pipeline.reset()
            recut = _stream(pipeline, da, cut)
            _assert_same(eager, recut, rtol, atol, coord_atol, f"result cut as {cut}")


def _stream(pipeline, da, chunks):
    """Run *pipeline* chunk by chunk over *da* and join the outputs."""
    ((dim, size),) = chunks.items()
    indices = list(range(size, da.sizes[dim], size))
    pieces = split(da, indices, dim) if indices else [da]
    outs = list(pipeline.iter_chunks(pieces, chunk_dim=dim))
    return pipeline._join(outs, dim)


def _cuttings(cuts, dim, size):
    """Derive the extra chunk sizes the cut-invariance pass runs with."""
    if not isinstance(cuts, int):
        return list(cuts)
    sizes = []
    current = size
    for _ in range(cuts):
        # Not a divisor of the previous size, so the boundaries move.
        current = current // 2 + 1 if current > 1 else current + 1
        if current == size or current < 1:
            break
        sizes.append(current)
    return [{dim: value} for value in sizes]


def inject_gaps(da, dim, gaps):
    """
    Return *da* with gaps injected along *dim*, for seam testing.

    Each gap drops a twentieth of the record (at least one sample), which is
    well beyond any jitter tolerance, so downstream seam judgment sees a real
    discontinuity in the coordinate.

    Parameters
    ----------
    da : DataArray
        The array to make gappy.
    dim : str
        The dimension along which to drop samples.
    gaps : int or sequence of int
        An int places that many evenly spaced gaps; a sequence gives the
        sample indices where each gap starts.

    Returns
    -------
    DataArray
        The gappy array: same values minus the dropped spans, with the gaps
        kept in the coordinates.

    Examples
    --------
    >>> import xdas as xd
    >>> da = xd.testing.dummy()
    >>> gappy = xd.testing.inject_gaps(da, "time", 2)
    >>> gappy.sizes["time"]
    90
    """
    size = da.sizes[dim]
    if isinstance(gaps, int):
        starts = [round((index + 1) * size / (gaps + 1)) for index in range(gaps)]
    else:
        starts = sorted(int(start) for start in gaps)
    width = max(1, size // 20)
    pieces = []
    previous = 0
    for start in starts:
        pieces.append(da.isel({dim: slice(previous, start)}))
        previous = min(start + width, size)
    pieces.append(da.isel({dim: slice(previous, None)}))
    pieces = [piece for piece in pieces if piece.sizes[dim]]
    if len(pieces) < 2:
        raise ValueError(
            f"cannot inject {gaps!r} gaps into a record of {size} samples "
            f"along {dim!r}: the gaps leave less than two pieces"
        )
    return concat(pieces, dim)


def _assert_same(eager, chunked, rtol, atol, coord_atol, path):
    """Compare two pipeline outputs of any supported chunk type."""
    if isinstance(eager, DataArray):
        if not isinstance(chunked, DataArray):
            raise AssertionError(
                f"{path}: eager gave a DataArray, chunked gave a "
                f"{type(chunked).__name__} — the chunked run did not join into "
                "one array"
            )
        if eager.dims != chunked.dims:
            raise AssertionError(
                f"{path}: dims differ, {eager.dims} eager vs {chunked.dims} chunked"
            )
        if eager.shape != chunked.shape:
            raise AssertionError(
                f"{path}: shape differs, {eager.shape} eager vs {chunked.shape} chunked"
            )
        np.testing.assert_allclose(
            chunked.values, eager.values, rtol=rtol, atol=atol, err_msg=path
        )
        for dim in eager.dims:
            if dim in eager.coords:
                _assert_same_coord(
                    chunked.coords[dim], eager.coords[dim], coord_atol, path, dim
                )
    elif isinstance(eager, pd.DataFrame):
        _assert_same_frame(eager, chunked, rtol, atol, path)
    elif isinstance(eager, (list, DataCollection)):
        if len(eager) != len(chunked):
            raise AssertionError(
                f"{path}: {len(eager)} chunks eager vs {len(chunked)} chunked"
            )
        for index, (left, right) in enumerate(zip(eager, chunked)):
            _assert_same(left, right, rtol, atol, coord_atol, f"{path}[{index}]")
    else:
        np.testing.assert_allclose(chunked, eager, rtol=rtol, atol=atol, err_msg=path)


def _assert_same_coord(chunked, eager, coord_atol, path, dim):
    """Compare one dimension coordinate, within *coord_atol* of its own units."""
    left = np.asarray(chunked.values)
    right = np.asarray(eager.values)
    message = f"{path}: the {dim!r} coordinate differs"
    if not coord_atol:
        np.testing.assert_array_equal(left, right, err_msg=message)
    elif np.issubdtype(right.dtype, np.datetime64):
        # Stay in integers: an epoch nanosecond does not survive float64,
        # whose ULP up there is a few hundred nanoseconds — wider than any
        # tolerance worth expressing.
        difference = np.abs((left - right).astype("int64"))
        np.testing.assert_array_less(difference, coord_atol + 1, err_msg=message)
    else:
        np.testing.assert_allclose(
            left, right, rtol=0, atol=coord_atol, err_msg=message
        )


def _assert_same_frame(eager, chunked, rtol, atol, path):
    """Compare two pick tables as sets of rows (see the Notes of the caller)."""
    if not isinstance(chunked, pd.DataFrame):
        raise AssertionError(
            f"{path}: eager gave a DataFrame, chunked gave a {type(chunked).__name__}"
        )
    if list(eager.columns) != list(chunked.columns):
        raise AssertionError(
            f"{path}: columns differ, {list(eager.columns)} eager vs "
            f"{list(chunked.columns)} chunked"
        )
    columns = list(eager.columns)
    pd.testing.assert_frame_equal(
        chunked.sort_values(columns).reset_index(drop=True),
        eager.sort_values(columns).reset_index(drop=True),
        rtol=rtol,
        atol=atol,
    )
