"""
:class:`ScalarCoordinate`: non-dimensional (scalar) coordinate.

Carries a single value without being tied to an array axis.
"""

import numpy as np
from typing_extensions import override

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

    @override
    def __init__(self, data=None, dim=None, dtype=None):
        if data is None:
            raise TypeError("scalar coordinate cannot be empty, please provide a value")
        data, dim = parse(data, dim)
        if dim is not None:
            raise ValueError("a scalar coordinate cannot be a dim")
        if not self._isvalid(data):
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

    @property
    @override
    def dtype(self):
        """Dtype of the scalar value."""
        return self.data.dtype

    @property
    @override
    def ndim(self):
        """Always 0 — a scalar coordinate has no axis."""
        return 0

    @property
    @override
    def shape(self):
        """Always the empty tuple ``()``."""
        return ()

    @property
    @override
    def size(self):
        """Always 1."""
        return 1

    @staticmethod
    @override
    def _isvalid(data):
        """Return ``True`` if *data* converts to a 0-d non-object numpy array."""
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 0)

    @override
    def __len__(self):
        raise TypeError("scalar coordinate has no length")

    @override
    def _get_value(self, index):
        raise TypeError("scalar coordinate has no elements to index")

    @override
    def _get_indexer(self, value, method=None):
        raise NotImplementedError("cannot get index of scalar coordinate")

    @override
    def _slice(self, slc):
        raise TypeError("scalar coordinate is not sliceable")

    @override
    def _to_dataset(self, dataset, attrs):
        return super()._to_dataset(dataset, attrs)

    def __repr__(self):
        return np.array2string(self.data, threshold=0, edgeitems=1)

    @override
    def __getitem__(self, item):
        raise TypeError("scalar coordinate is not subscriptable")

    @override
    def __array__(self, dtype=None, copy=None):
        return self.data.__array__(dtype, copy=copy)

    def __array__ufunc__(self, ufunc, method, *inputs, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError

    @override
    def isscalar(self):
        """Return ``True`` (this is a :class:`ScalarCoordinate`)."""
        return True

    def get_sampling_interval(self, cast=True):
        """Return ``None`` — scalar coordinates have no sample spacing."""
        return None

    @override
    def _is_monotonic_increasing(self):
        """Not supported — scalar coordinates have no axis to order."""
        raise TypeError("scalar coordinate has no axis")

    @override
    def to_index(self, item, method=None, endpoint=True):
        """Not supported — raises :exc:`NotImplementedError`."""
        raise NotImplementedError("cannot get index of scalar coordinate")

    @override
    def _concat(self, other):
        """Not supported — scalar coordinates have no axis to concatenate along."""
        raise TypeError("cannot concatenate scalar coordinate")

    @classmethod
    @override
    def _collect_from_dataset(cls, dataset, name):
        """Scalar coordinates are not stored separately in a dataset; return an empty mapping."""
        return {}

    @classmethod
    @override
    def from_block(cls, start, size, step, dim=None, dtype=None):
        """Not supported — scalar coordinates describe no axis block."""
        raise TypeError("cannot build a scalar coordinate from a block")
