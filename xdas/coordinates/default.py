"""
:class:`DefaultCoordinate`: integer-range coordinate used when no coordinate
is explicitly provided for an axis.
"""

import numpy as np

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

    def __init__(self, data=None, dim=None, dtype=None):
        # empty
        if data is None:
            data = {"size": 0}

        # parse data
        data, dim = parse(data, dim)
        if not self.isvalid(data):
            raise TypeError("`data` must be a mapping {'size': <int>}")

        # check dtype
        if dtype is not None:
            raise ValueError("`dtype` is not supported for DefaultCoordinate")

        # store data
        self.data = data
        self.dim = dim

    @property
    def empty(self):
        """``True`` if the coordinate has size zero."""
        return self.data["size"] == 0

    @property
    def dtype(self):
        """Always ``numpy.int64``."""
        return np.int64

    @property
    def ndim(self):
        """Always 1."""
        return 1

    @property
    def shape(self):
        """Shape tuple ``(size,)``."""
        return (len(self),)

    @staticmethod
    def isvalid(data):
        """Return ``True`` if *data* is ``{"size": int}``."""
        match data:
            case {"size": None | int(_)}:
                return True
            case _:
                return False

    def __len__(self):
        if self.data["size"] is None:
            return 0
        else:
            return self.data["size"]

    def __getitem__(self, item):
        data = self.__array__()[item]
        dim = None if isscalar(data) else self.dim
        return Coordinate(data, dim)

    def __array__(self, dtype=None):
        return np.arange(self.data["size"], dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError

    def isdefault(self):
        """Return ``True`` (this is a :class:`DefaultCoordinate`)."""
        return True

    def get_sampling_interval(self, cast=True):
        """Return the sample spacing, always 1 for integer-range coordinates."""
        return 1

    def equals(self, other):
        """Return ``True`` if *other* is a :class:`DefaultCoordinate` of the same size."""
        if isinstance(other, self.__class__):
            return self.data["size"] == other.data["size"]

    def get_indexer(self, value, method=None):
        """Return *value* directly (integer index equals label for range coordinates)."""
        return value

    def slice_indexer(self, start=None, stop=None, step=None, endpoint=True):
        """Return a :class:`slice` with *start*, *stop*, *step* unchanged."""
        return slice(start, stop, step)

    def concat(self, other):
        """Return a new :class:`DefaultCoordinate` whose size is the sum of both sizes."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot concatenate {type(other)} to {self.__class__}")
        if not self.dim == other.dim:
            raise ValueError("cannot concatenate coordinate with different dimension")
        return self.__class__({"size": len(self) + len(other)}, self.dim)

    def to_dict(self):
        """Serialise to ``{"dim": ..., "data": ...}``."""
        return {"dim": self.dim, "data": self.data}
