import numpy as np

from .core import Coordinate, parse


class ScalarCoordinate(Coordinate):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

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
        return None

    @dim.setter
    def dim(self, value):
        if value is not None:
            raise ValueError("A scalar coordinate cannot have a `dim` other that None")

    def get_sampling_interval(self, cast=True):
        return None

    @staticmethod
    def isvalid(data):
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 0)

    def isscalar(self):
        return True

    def equals(self, other):
        if isinstance(other, self.__class__):
            return self.data == other.data
        else:
            return False

    def to_index(self, item, method=None, endpoint=True):
        raise NotImplementedError("cannot get index of scalar coordinate")

    def to_dict(self):
        if np.issubdtype(self.dtype, np.datetime64):
            data = self.data.astype(str).item()
        else:
            data = self.data.item()
        return {"dim": self.dim, "data": data, "dtype": str(self.dtype)}
