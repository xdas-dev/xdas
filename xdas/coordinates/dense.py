"""
:class:`DenseCoordinate`: coordinate backed by a full numpy array.
"""

import numpy as np
import pandas as pd

from .core import Coordinate, parse


class DenseCoordinate(Coordinate, name="dense"):
    """
    Coordinate backed by an explicit numpy array.

    Suitable for irregularly-spaced or small axes where every value must be
    stored.  Look-up is performed via a :class:`pandas.Index`.

    Parameters
    ----------
    data : array-like or None, optional
        1-D array of coordinate values.  ``None`` creates an empty coordinate.
    dim : str, optional
        Dimension name.
    dtype : dtype-like, optional
        Cast *data* to this dtype on construction.
    """

    def __init__(self, data=None, dim=None, dtype=None):
        # empty
        if data is None:
            data = []

        # parse data
        data, dim = parse(data, dim)
        if not self.isvalid(data):
            raise TypeError("`data` must be array-like")

        # store data
        self.data = np.asarray(data, dtype=dtype)
        self.dim = dim

    @property
    def index(self):
        """A :class:`pandas.Index` view of the underlying data array."""
        return pd.Index(self.data)

    @staticmethod
    def isvalid(data):
        """Return ``True`` if *data* converts to a 1-D non-object numpy array."""
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 1)

    def isdense(self):
        """Return ``True`` (this is a :class:`DenseCoordinate`)."""
        return True

    def equals(self, other):
        """Return ``True`` if *other* is a :class:`DenseCoordinate` with identical values and dtype."""
        if isinstance(other, self.__class__):
            return (
                np.array_equal(self.data, other.data)
                and self.dim == other.dim
                and self.dtype == other.dtype
            )
        else:
            return False

    def get_indexer(self, value, method=None):
        """
        Return the integer index (or indices) for *value*.

        Parameters
        ----------
        value : scalar or array-like
            Label(s) to look up.
        method : str, optional
            Forwarded to :meth:`pandas.Index.get_indexer` (e.g. ``"ffill"``).

        Returns
        -------
        int or numpy.ndarray

        Raises
        ------
        KeyError
            If any requested label is not found (indexer returns -1).
        """
        if np.isscalar(value):
            out = self.index.get_indexer([value], method).item()
        else:
            out = self.index.get_indexer(value, method)
        if np.any(out == -1):
            raise KeyError("index not found")
        return out

    def slice_indexer(self, start=None, stop=None, step=None, endpoint=True):
        """Return an integer :class:`slice` for label range [*start*, *stop*] via :class:`pandas.Index`."""
        slc = self.index.slice_indexer(start, stop, step)
        if (
            (not endpoint)
            and (stop is not None)
            and (self[slc.stop - 1].values == stop)
        ):
            slc = slice(slc.start, slc.stop - 1, slc.step)
        return slc

    def concat(self, other):
        """Concatenate *other* :class:`DenseCoordinate` values to this one."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot concatenate {type(other)} to {self.__class__}")
        if not self.dim == other.dim:
            raise ValueError("cannot concatenate coordinate with different dimension")
        if self.empty:
            return other
        if other.empty:
            return self
        if not self.dtype == other.dtype:
            raise ValueError("cannot concatenate coordinate with different dtype")
        return self.__class__(np.concatenate([self.data, other.data]), self.dim)

    def get_div_points(self, tolerance=None):
        """Return sorted split-point indices where consecutive differences exceed *tolerance*."""
        deltas = np.diff(self.data)
        if tolerance is not None:
            div_points = np.nonzero(np.abs(deltas) >= tolerance)[0] + 1
        else:
            raise NotImplementedError(
                "get_div_points without tolerance is not implemented for DenseCoordinate"
            )
        div_points = np.concatenate(([0], div_points, [len(self)]))
        return div_points

    def to_dict(self):
        """Serialise to ``{"dim": ..., "data": ..., "dtype": ...}``."""
        if np.issubdtype(self.dtype, np.datetime64):
            data = self.data.astype(str).tolist()
        else:
            data = self.data.tolist()
        return {"dim": self.dim, "data": data, "dtype": str(self.dtype)}

    @classmethod
    def from_dataset(cls, dataset, name):
        """Extract all coordinates from an xarray *dataset* variable *name* as plain arrays."""
        return {
            name: (
                (
                    coord.dims[0],
                    (
                        coord.values.astype("U")
                        if coord.dtype == np.dtype("O")
                        else coord.values
                    ),
                )
                if coord.dims
                else coord.values
            )
            for name, coord in dataset[name].coords.items()
        }

    @classmethod
    def from_block(cls, start, size, step, dim=None, dtype=None):
        """Build a :class:`DenseCoordinate` from ``start + step * arange(size)``."""
        data = start + step * np.arange(size)
        return cls(data, dim=dim, dtype=dtype)
