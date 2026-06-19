"""
:class:`ScalarCoordinate`: non-dimensional (scalar) coordinate.

Carries a single value without being tied to an array axis.
"""

import numpy as np
from typing_extensions import override

from .core import Coordinate, parse


class ScalarCoordinate(Coordinate, ctype="scalar"):
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

    @classmethod
    @override
    def from_block(cls, start, size, step, dim=None, dtype=None):
        raise TypeError("cannot build a scalar coordinate from a block")

    @override
    def __len__(self):
        return 1

    @override
    def __getitem__(self, item):
        raise TypeError("scalar coordinate is not subscriptable")

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
    @override
    def ndim(self):
        return 0

    @property
    @override
    def shape(self):
        return ()

    @property
    @override
    def indices(self):
        raise TypeError("scalar coordinate has no indices")

    @property
    @override
    def start(self):
        raise TypeError("scalar coordinate has no start")

    @property
    @override
    def end(self):
        raise TypeError("scalar coordinate has no end")

    @staticmethod
    @override
    def _isvalid(data):
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 0)

    @override
    def _is_monotonic_increasing(self):
        raise TypeError("scalar coordinate has no axis")

    @override
    def _get_value(self, index):
        raise TypeError("scalar coordinate has no elements to index")

    @override
    def _get_indexer(self, value, method=None):
        raise TypeError("cannot get index of scalar coordinate")

    @override
    def _slice(self, slc):
        raise TypeError("scalar coordinate is not sliceable")

    @override
    def _concat(self, other):
        raise TypeError("cannot concatenate scalar coordinate")

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

    @override
    def get_sampling_interval(self, cast=True):
        return None

    @override
    def isscalar(self):
        return True

    @override
    def to_index(self, item, method=None, endpoint=True):
        raise NotImplementedError("cannot get index of scalar coordinate")
