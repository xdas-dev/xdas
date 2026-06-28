"""Test utilities for xdas."""

import numpy as np

from .coordinates import Coordinate
from .core import DataArray


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
