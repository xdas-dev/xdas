""":class:`DenseCoordinate`: coordinate backed by a full numpy array."""

import numpy as np
import pandas as pd
from typing_extensions import override

from .core import Coordinate, parse


class DenseCoordinate(Coordinate, ctype="dense"):
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

    @override
    def __init__(self, data=None, dim=None, dtype=None):
        # empty
        if data is None:
            data = []

        # parse data
        data, dim = parse(data, dim)
        if not self._isvalid(data):
            raise TypeError("`data` must be array-like")

        # store data
        self.data = np.asarray(data, dtype=dtype)
        self.dim = dim

    @classmethod
    @override
    def from_block(cls, start, size, step, dim=None, dtype=None):
        data = start + step * np.arange(size)
        return cls(data, dim=dim, dtype=dtype)

    @override
    def __len__(self):
        return self.data.__len__()

    @property
    @override
    def dtype(self):
        return self.data.dtype

    @property
    def index(self):
        """A :class:`pandas.Index` view of the underlying data array."""
        return pd.Index(self.data)

    @staticmethod
    @override
    def _isvalid(data):
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 1)

    @override
    def _is_monotonic_increasing(self):
        if np.issubdtype(self.dtype, np.datetime64):
            zero = np.timedelta64(0)
        else:
            zero = 0
        return np.all(np.diff(self.values) > zero)

    @override
    def _get_value(self, index):
        return self.data[index]

    @override
    def _get_indexer(self, value, method=None):
        if np.isscalar(value):
            out = self.index.get_indexer([value], method).item()
        else:
            out = self.index.get_indexer(value, method)
        if np.any(out == -1):
            raise KeyError("index not found")
        return out

    @override
    def _slice(self, slc):
        return self.__class__(self.data[slc], self.dim)

    @override
    def _concat(self, other):
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

    @override
    def _to_dataset(self, dataset, attrs):
        if self.name is None:
            raise ValueError("cannot serialize a coordinate with no name")
        dataset = dataset.assign_coords(
            {self.name: (self.dim, self.values) if self.dim else self.values}
        )
        return dataset, attrs

    @classmethod
    @override
    def _collect_from_dataset(cls, dataset, name):
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

    @override
    def __repr__(self):
        return np.array2string(self.data, threshold=0, edgeitems=1)

    def __add__(self, other):
        return self.__class__(self.data + other, self.dim)

    def __sub__(self, other):
        return self.__class__(self.data - other, self.dim)

    def get_sampling_interval(self, cast=True):
        """
        Return the average sample spacing (end-to-end distance divided by N-1).

        Parameters
        ----------
        cast : bool, optional
            If ``True`` (default), cast timedelta64 results to seconds (float).

        Returns
        -------
        float or None
            ``None`` if the coordinate has fewer than two elements.
        """
        if len(self) < 2:
            return None
        delta = (self[-1].values - self[0].values) / (len(self) - 1)
        delta = np.asarray(
            delta
        )  # plain Python floats have no .dtype; np.asarray adds it
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

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
