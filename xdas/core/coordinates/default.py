import numpy as np

from .core import Coordinate, isscalar, parse


class DefaultCoordinate(Coordinate):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, data=None, dim=None, dtype=None):
        if data is None:
            data = {"size": 0}
        data, dim = parse(data, dim)
        if not self.isvalid(data):
            raise TypeError("`data` must be a mapping {'size': <int>}")
        if dtype is not None:
            raise ValueError("`dtype` is not supported for DefaultCoordinate")
        self.data = data
        self.dim = dim

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

    @staticmethod
    def isvalid(data):
        match data:
            case {"size": None | int(_)}:
                return True
            case _:
                return False

    def isdefault(self):
        return True

    @property
    def empty(self):
        return bool(self.data["size"])

    @property
    def dtype(self):
        return np.int64

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return (len(self),)

    def get_sampling_interval(self, cast=True):
        return 1

    def equals(self, other):
        if isinstance(other, self.__class__):
            return self.data["size"] == other.data["size"]

    def get_indexer(self, value, method=None):
        return value

    def slice_indexer(self, start=None, stop=None, step=None, endpoint=True):
        return slice(start, stop, step)

    def append(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot append {type(other)} to {self.__class__}")
        if not self.dim == other.dim:
            raise ValueError("cannot append coordinate with different dimension")
        return self.__class__({"size": len(self) + len(other)}, self.dim)

    def to_dict(self):
        return {"dim": self.dim, "data": self.data.tolist(), "dtype": str(self.dtype)}
