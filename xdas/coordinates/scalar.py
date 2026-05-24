"""
:class:`ScalarCoordinate`: non-dimensional (scalar) coordinate that carries a
single value without being tied to an array axis.
"""

import numpy as np

from .core import Coordinate, parse


class ScalarCoordinate(Coordinate, name="scalar"):
    """
    Non-dimensional coordinate that carries a single scalar value.

    Unlike dimensional coordinates, a :class:`ScalarCoordinate` is not tied
    to an array axis and has no length.  Typical use: metadata attached to a
    :class:`DataArray` (e.g. an instrument identifier or a shot time).

    Parameters
    ----------
    data : scalar-like
        The scalar value.  Cannot be ``None``.
    dim : must be ``None``
        Passing a non-``None`` value raises :exc:`ValueError`.
    dtype : dtype-like, optional
        Cast *data* to this dtype.
    """

    def __init__(self, data=None, dim=None, dtype=None):
        if data is None:
            raise TypeError("scalar coordinate cannot be empty, please provide a value")
        data, dim = parse(data, dim)
        if dim is not None:
            raise ValueError("a scalar coordinate cannot be a dim")
        if not self.__class__.isvalid(data):
            raise TypeError("`data` must be scalar-like")
        self.data = np.asarray(data, dtype=dtype)

    @property
    def dim(self):
        """Always ``None`` — scalar coordinates have no associated dimension."""
        return None

    @dim.setter
    def dim(self, value):
        """Not supported — raises :exc:`ValueError` if *value* is not ``None``."""
        if value is not None:
            raise ValueError("A scalar coordinate cannot have a `dim` other that None")

    @staticmethod
    def isvalid(data):
        """Return ``True`` if *data* converts to a 0-d non-object numpy array."""
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 0)

    def isscalar(self):
        """Return ``True`` (this is a :class:`ScalarCoordinate`)."""
        return True

    def get_sampling_interval(self, cast=True):
        """Return ``None`` — scalar coordinates have no sample spacing."""
        return None

    def equals(self, other):
        """Return ``True`` if *other* is a :class:`ScalarCoordinate` with the same value."""
        if isinstance(other, self.__class__):
            return self.data == other.data
        else:
            return False

    def to_index(self, item, method=None, endpoint=True):
        """Not supported — raises :exc:`NotImplementedError`."""
        raise NotImplementedError("cannot get index of scalar coordinate")

    def to_dict(self):
        """Serialise to ``{"dim": None, "data": ..., "dtype": ...}``."""
        if np.issubdtype(self.dtype, np.datetime64):
            data = self.data.astype(str).item()
        else:
            data = self.data.item()
        return {"dim": self.dim, "data": data, "dtype": str(self.dtype)}
