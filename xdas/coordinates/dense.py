import numpy as np
import pandas as pd

from .core import Coordinate, parse


class DenseCoordinate(Coordinate):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, data=None, dim=None, dtype=None):
        if data is None:
            data = []
        data, dim = parse(data, dim)
        if not self.isvalid(data):
            raise TypeError("`data` must be array-like")
        self.data = np.asarray(data, dtype=dtype)
        self.dim = dim

    @staticmethod
    def isvalid(data):
        data = np.asarray(data)
        return (data.dtype != np.dtype(object)) and (data.ndim == 1)

    def isdense(self):
        return True

    @property
    def index(self):
        return pd.Index(self.data)

    def equals(self, other):
        if isinstance(other, self.__class__):
            return (
                np.array_equal(self.data, other.data)
                and self.dim == other.dim
                and self.dtype == other.dtype
            )
        else:
            return False

    def get_indexer(self, value, method=None):
        if np.isscalar(value):
            out = self.index.get_indexer([value], method).item()
        else:
            out = self.index.get_indexer(value, method)
        if np.any(out == -1):
            raise KeyError("index not found")
        return out

    def slice_indexer(self, start=None, stop=None, step=None, endpoint=True):
        slc = self.index.slice_indexer(start, stop, step)
        if (
            (not endpoint)
            and (stop is not None)
            and (self[slc.stop - 1].values == stop)
        ):
            slc = slice(slc.start, slc.stop - 1, slc.step)
        return slc

    def append(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot append {type(other)} to {self.__class__}")
        if not self.dim == other.dim:
            raise ValueError("cannot append coordinate with different dimension")
        if self.empty:
            return other
        if other.empty:
            return self
        if not self.dtype == other.dtype:
            raise ValueError("cannot append coordinate with different dtype")
        return self.__class__(np.concatenate([self.data, other.data]), self.dim)

    def to_dict(self):
        if np.issubdtype(self.dtype, np.datetime64):
            data = self.data.astype(str).tolist()
        else:
            data = self.data.tolist()
        return {"dim": self.dim, "data": data, "dtype": str(self.dtype)}

    @classmethod
    def from_dataset(cls, dataset, name):
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
