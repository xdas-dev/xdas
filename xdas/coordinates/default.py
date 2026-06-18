"""
:class:`DefaultCoordinate`: integer-range coordinate.

Used when no coordinate is explicitly provided for an axis.
"""

import numpy as np
from typing_extensions import override

from .core import Coordinate, isscalar, parse


class DefaultCoordinate(Coordinate, name="default"):
    """
    Integer-range coordinate, equivalent to ``np.arange(size)``.

    Used automatically when no explicit coordinate is provided for an axis.
    Internally stored as ``{"size": int}`` rather than a full array to avoid
    memory allocation until values are actually needed.

    Parameters
    ----------
    data : {"size": int} or None, optional
        Mapping with a single ``"size"`` key.  ``None`` creates an empty coordinate.
    dim : str, optional
        Dimension name.
    dtype : ignored
        Not supported; raises :exc:`ValueError` if provided.
    """

    @override
    def __init__(self, data=None, dim=None, dtype=None):
        # empty
        if data is None:
            data = {"size": 0}

        # parse data
        data, dim = parse(data, dim)
        if not self._isvalid(data):
            raise TypeError("`data` must be a mapping {'size': <int>}")

        # check dtype
        if dtype is not None:
            raise ValueError("`dtype` is not supported for DefaultCoordinate")

        # store data
        self.data = data
        self.dim = dim

    @property
    @override
    def empty(self):
        """``True`` if the coordinate has size zero."""
        return self.data["size"] == 0

    @property
    @override
    def dtype(self):
        """Always ``numpy.int64``."""
        return np.int64

    @staticmethod
    @override
    def _isvalid(data):
        """Return ``True`` if *data* is ``{"size": int}``."""
        match data:
            case {"size": None | int(_)}:
                return True
            case _:
                return False

    @override
    def __len__(self):
        if self.data["size"] is None:
            return 0
        else:
            return self.data["size"]

    @override
    def _get_value(self, index):
        return index

    @override
    def _slice(self, slc):
        return Coordinate(self.__array__()[slc], self.dim)

    @override
    def _to_dataset(self, dataset, attrs):
        return dataset, attrs

    def __repr__(self):
        if self.empty:
            return "empty coordinate"
        return f"0 to {len(self) - 1}"

    @override
    def __getitem__(self, item):
        data = self.__array__()[item]
        dim = None if isscalar(data) else self.dim
        return Coordinate(data, dim)

    @override
    def __array__(self, dtype=None, copy=None):
        return np.arange(self.data["size"], dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError

    def get_sampling_interval(self, cast=True):
        """Return the sample spacing, always 1 for integer-range coordinates."""
        return 1

    @override
    def _is_monotonic_increasing(self):
        """Return ``True`` — integer-range coordinates are always increasing."""
        return True

    def _get_indexer(self, value, method=None):
        """Return *value* directly (integer index equals label for range coordinates)."""
        return value

    @override
    def _concat(self, other):
        """Return a new :class:`DefaultCoordinate` whose size is the sum of both sizes."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot concatenate {type(other)} to {self.__class__}")
        if not self.dim == other.dim:
            raise ValueError("cannot concatenate coordinate with different dimension")
        return self.__class__({"size": len(self) + len(other)}, self.dim)

    @classmethod
    @override
    def _collect_from_dataset(cls, dataset, name):
        """Default coordinates are not stored in a dataset; return an empty mapping."""
        return {}

    @classmethod
    @override
    def from_block(cls, start, size, step, dim=None, dtype=None):
        """Build a :class:`DefaultCoordinate` of *size* elements (start and step are ignored)."""
        return cls({"size": size}, dim=dim)
