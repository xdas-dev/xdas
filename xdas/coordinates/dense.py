""":class:`DenseCoordinate`: coordinate backed by a full numpy array."""

import numpy as np
import pandas as pd
from typing_extensions import override

from .core import AxisCoordinate, parse_data_dim, parse_scalar_delta
from .interp import InterpCoordinate


class DenseCoordinate(AxisCoordinate, ctype="dense"):
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
        data, dim = parse_data_dim(data, dim)
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

    @override
    def get_sampling_interval(self, cast=True):
        """
        Return ``None``: a dense coordinate never carries a nominal spacing.

        The raw values may happen to be evenly spaced, but regularity is an
        explicit declaration; convert with :meth:`to_regular` to obtain a
        regular :class:`InterpCoordinate`.
        """
        return

    @override
    def to_regular(self, sampling_interval=None, tolerance=None):
        """Convert to a regular :class:`InterpCoordinate` (single continuous ramp).

        The spacing defaults to the end-to-end slope, and every value must lie
        within *tolerance* of the regular grid anchored at the first value;
        otherwise a :exc:`ValueError` is raised. See
        :meth:`AxisCoordinate.to_regular` for the parameter contract.
        """
        if len(self) < 2:
            raise ValueError(
                "cannot make a regular coordinate from fewer than two values"
            )
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
        if sampling_interval is None:
            sampling_interval = (self.data[-1] - self.data[0]) / (len(self) - 1)
        else:
            sampling_interval = parse_scalar_delta(sampling_interval, self.dtype)
        grid = self.data[0] + sampling_interval * np.arange(len(self))
        if not np.all(np.abs(self.data - grid) <= tolerance):
            raise ValueError(
                "values are not evenly spaced by `sampling_interval` within `tolerance`"
            )
        data = {
            "tie_indices": [0, len(self) - 1],
            "tie_values": [self.data[0], self.data[-1]],
            "sampling_interval": sampling_interval,
            "tolerance": tolerance,
        }
        return InterpCoordinate(data, self.dim)

    @override
    def _split_candidates(self):
        steps = np.diff(self.data)
        positions = np.arange(1, len(self))
        if steps.size == 0:
            return positions, steps
        reference = np.median(steps)
        deltas = np.empty(steps.shape, dtype=np.asarray(steps[0] - reference).dtype)
        for i in range(steps.size):
            deltas[i] = steps[i] - reference
            if i + 1 < steps.size and abs(steps[i + 1] - steps[i]) < abs(
                steps[i + 1] - reference
            ):
                reference = steps[i]
        return positions, deltas

    @override
    def simplify(self, tolerance=None, *, reduce=True, regularize=False):
        # a dense coordinate stores every value explicitly; there is nothing to
        # drop and no spacing to promote, so both stages are no-ops.
        return self.copy()
