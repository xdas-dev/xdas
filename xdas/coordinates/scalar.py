"""
:class:`ScalarCoordinate`: non-dimensional (scalar) coordinate.

Carries a single value without being tied to an array axis.
"""

import numpy as np
from typing_extensions import override

from .core import Coordinate, parse_data_dim


class ScalarCoordinate(Coordinate, ctype="scalar"):
    """
    Non-dimensional coordinate that carries a single scalar value.

    Unlike :class:`~xdas.coordinates.AxisCoordinate` subclasses, a
    :class:`ScalarCoordinate` is not tied to an array axis and has no length.
    It therefore implements only the thin :class:`Coordinate` interface.
    Typical use: metadata attached to a :class:`DataArray` (e.g. an instrument
    identifier or a shot time).

    Parameters
    ----------
    data : scalar-like
        The scalar value.  Cannot be ``None``.
    dim : must be ``None``
        Passing a non-``None`` value raises :exc:`ValueError`.
    dtype : dtype-like, optional
        Cast *data* to this dtype.
    """

    @override
    def __init__(self, data=None, dim=None, dtype=None):
        if data is None:
            raise TypeError("scalar coordinate cannot be empty, please provide a value")
        data, dim = parse_data_dim(data, dim)
        if dim is not None:
            raise ValueError("a scalar coordinate cannot be a dim")
        if not self._isvalid(data):
            raise TypeError("`data` must be scalar-like")
        self.data = np.asarray(data, dtype=dtype)

    @staticmethod
    @override
    def _isvalid(data):
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 0)

    @property
    def dim(self):
        """Always ``None`` — scalar coordinates have no associated dimension."""
        return None

    @dim.setter
    def dim(self, value):
        """Not supported — raises :exc:`ValueError` if *value* is not ``None``."""
        if value is not None:
            raise ValueError("A scalar coordinate cannot have a `dim` other that None")

    @property
    @override
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        """Always ``0`` — scalar coordinates have no axis."""
        return 0

    @property
    @override
    def shape(self):
        return ()

    @override
    def __array__(self, dtype=None, copy=None):
        # TODO: drop this workaround once Python 3.10 is no longer supported
        # (EOL Oct 2026). numpy < 2.3 raises when copy=False on a 0-d array;
        # numpy 2.3+ (requires Python 3.11+) handles it correctly.
        if copy:
            return np.array(self.data, dtype=dtype)
        return np.asarray(self.data, dtype=dtype)

    @override
    def __repr__(self):
        return np.array2string(self.data, threshold=0, edgeitems=1)

    def __add__(self, other):
        return self.__class__(self.data + other)

    def __sub__(self, other):
        return self.__class__(self.data - other)

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
        return {}
