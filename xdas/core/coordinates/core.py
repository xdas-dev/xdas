import re
from copy import copy, deepcopy
from functools import wraps

import numpy as np
import pandas as pd


def wraps_first_last(func):
    @wraps(func)
    def wrapper(self, dim, *args, **kwargs):
        if dim == "first":
            dim = self._dims[0]
        if dim == "last":
            dim = self._dims[-1]
        return func(self, dim, *args, **kwargs)

    return wrapper


class Coordinates(dict):
    """
    Dictionary like container for coordinates.

    Parameters
    ----------
    coords: dict-like, optional
        Mapping from coordinate names to any of the followings:

        - Coordinate objects
        - tuples (dim, coordinate-like) which can be either dimensional (`dim == name`)
          or non-dimensional (`dim != name` or `dim == None`).
        - coordinate-like objects (that are passed to the Coordinate constructor)
          which are assumed to be a dimensional coordinate with `dim` set to the
          related name.

    dims: squence of str, optional
        An ordered sequence of dimensions. It is meant to match the dimensionality of
        its associated data. If provided, it must at least include all dimensions found
        in `coords` (extras dimensions will be considered as empty coordinates).
        Otherwise, dimensions will be guessed from `coords`.

    Examples
    --------
    >>> import xdas

    >>> coords = {
    ...     "time": {"tie_indices": [0, 999], "tie_values": [0.0, 10.0]},
    ...     "distance": [0, 1, 2],
    ...     "channel": ("distance", ["DAS01", "DAS02", "DAS03"]),
    ...     "interrogator": (None, "SRN"),
    ... }
    >>> xdas.Coordinates(coords)
    Coordinates:
      * time (time): 0.000 to 10.000
      * distance (distance): [0 ... 2]
        channel (distance): ['DAS01' ... 'DAS03']
        interrogator: 'SRN'
    """

    def __init__(self, coords=None, dims=None):
        super().__init__()
        if isinstance(coords, Coordinates):
            if dims is None:
                dims = coords.dims
            coords = dict(coords)
        self._dims = () if dims is None else tuple(dims)
        self._parent = None
        if coords is not None:
            for name in coords:
                self[name] = coords[name]

    @wraps_first_last
    def __getitem__(self, key):
        if key in self.dims and key not in self:
            raise KeyError(f"dimension {key} has no coordinate")
        return super().__getitem__(key)

    @wraps_first_last
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("dimension names must be of type str")
        coord = Coordinate(value)
        if coord.dim is None and not coord.isscalar():
            coord.dim = key
        if self._parent is None:
            if coord.dim is not None and coord.dim not in self.dims:
                self._dims = self.dims + (coord.dim,)
        else:
            if coord.dim is not None:
                if coord.dim not in self.dims:
                    raise KeyError(
                        f"cannot add new dimension {coord.dim} to an existing DataArray"
                    )
                size = self._parent.sizes[coord.dim]
                if not len(coord) == size:
                    raise ValueError(
                        f"conflicting sizes for dimension {coord.dim}: size {len(coord)} "
                        f"in `coords` and size {size} in `data`"
                    )
        coord._parent = self
        return super().__setitem__(key, coord)

    def __repr__(self):
        lines = ["Coordinates:"]
        for name, coord in self.items():
            if self.isdim(name):
                lines.append(f"  * {name} ({coord.dim}): {coord}")
            else:
                if coord.dim is None:
                    lines.append(f"    {name}: {coord}")
                else:
                    lines.append(f"    {name} ({coord.dim}): {coord}")
        return "\n".join(lines)

    def __reduce__(self):
        return self.__class__, (dict(self), self.dims), {"_parent": self._parent}

    @property
    def dims(self):
        return self._dims

    def isdim(self, name):
        return self[name].dim == name

    def get_query(self, item):
        """
        Format a query from one or multiple indexer.

        Parameters
        ----------
        item: indexer-like or sequence or mapping
            Object to be parsed as a query. If item is indexer-like object, it is
            applied on the first dimension. If item is a sequence, positional indexing
            is performed. If item is a mapping, labeled indexing is performed.

        Returns
        -------
        dict:
            A mapping between each dim and a given indexer. If No indexer was found for
            a given dim, slice(None) will be used.
        """
        query = {dim: slice(None) for dim in self.dims}
        if isinstance(item, dict):
            if "first" in item:
                item[self.dims[0]] = item.pop("first")
            if "last" in item:
                item[self.dims[-1]] = item.pop("last")
            query.update(item)
        elif isinstance(item, tuple):
            for k in range(len(item)):
                query[self.dims[k]] = item[k]
        else:
            query[self.dims[0]] = item
        for dim, item in query.items():
            if isinstance(item, tuple):
                msg = f"cannot use tuple {item} to index dim '{dim}'"
                if len(item) == 2:
                    msg += f". Did you mean: {dim}=slice({item[0]}, {item[1]})?"
                raise TypeError(msg)
        return query

    def to_index(self, item, method=None, endpoint=True):
        query = self.get_query(item)
        return {dim: self[dim].to_index(query[dim], method, endpoint) for dim in query}

    def equals(self, other):
        if not isinstance(other, Coordinates):
            return False
        for name in self:
            if not self[name].equals(other[name]):
                return False
        return True

    def to_dict(self):
        """Convert this `Coordinates` object into a pure python dictionnary.

        Examples
        --------

        >>> import xdas

        >>> coords = xdas.Coordinates(
        ...     {
        ...         "time": {"tie_indices": [0, 999], "tie_values": [0.0, 10.0]},
        ...         "distance": [0, 1, 2],
        ...         "channel": ("distance", ["DAS01", "DAS02", "DAS03"]),
        ...         "interrogator": (None, "SRN"),
        ...     }
        ... )
        >>> coords.to_dict()
        {'dims': ('time', 'distance'),
         'coords': {'time': {'dim': 'time',
           'data': {'tie_indices': [0, 999], 'tie_values': [0.0, 10.0]},
           'dtype': 'float64'},
          'distance': {'dim': 'distance', 'data': [0, 1, 2], 'dtype': 'int64'},
          'channel': {'dim': 'distance',
           'data': ['DAS01', 'DAS02', 'DAS03'],
           'dtype': '<U5'},
          'interrogator': {'dim': None, 'data': 'SRN', 'dtype': '<U3'}}}

        """
        return {
            "dims": self.dims,
            "coords": {name: self[name].to_dict() for name in self},
        }

    @classmethod
    def from_dict(cls, dct):
        return cls(
            {key: Coordinate.from_dict(value) for key, value in dct["coords"].items()},
            dct["dims"],
        )

    @classmethod
    def from_dataset(cls, dataset, name):
        return Coordinate.from_dataset(dataset, name)

    def copy(self, deep=True):
        if deep:
            func = deepcopy
        else:
            func = copy
        return self.__class__({key: func(value) for key, value in self.items()})

    @wraps_first_last
    def drop_dims(self, *dims):
        coords = {key: value for key, value in self.items() if value.dim not in dims}
        dims = tuple(value for value in self.dims if value not in dims)
        return self.__class__(coords, dims)

    @wraps_first_last
    def drop_coords(self, *names):
        coords = {key: value for key, value in self.items() if key not in names}
        return self.__class__(coords, self.dims)

    def _assign_parent(self, parent):
        if not len(self.dims) == parent.ndim:
            raise ValueError(
                f"inferred number of dimensions {len(self.dims)} from `coords` does "
                f"not match `data` dimensionality of {parent.ndim}"
            )
        for dim, size in zip(self.dims, parent.shape):
            if (dim in self) and (not len(self[dim]) == size):
                raise ValueError(
                    f"conflicting sizes for dimension {dim}: size {len(self[dim])} "
                    f"in `coords` and size {size} in `data`"
                )
        self._parent = parent


class Coordinate:
    def __new__(cls, data=None, dim=None, dtype=None):
        if data is None:
            raise TypeError("cannot infer coordinate type if no `data` is provided")
        data, dim = parse(data, dim)
        for subcls in cls.__subclasses__():
            if subcls.isvalid(data):
                return object.__new__(subcls)
        raise TypeError("could not parse `data`")

    def __getitem__(self, item):
        data = self.data.__getitem__(item)
        if ScalarCoordinate.isvalid(data):
            return ScalarCoordinate(data)
        else:
            return Coordinate(data, self.dim)

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return np.array2string(self.data, threshold=0, edgeitems=1)

    def __reduce__(self):
        return self.__class__, (self.data, self.dim), {"_parent": self.parent}

    def __add__(self, other):
        return self.__class__(self.data + other, self.dim)

    def __sub__(self, other):
        return self.__class__(self.data - other, self.dim)

    def __array__(self, dtype=None):
        if dtype is None:
            return self.data.__array__()
        else:
            return self.data.__array__(dtype)

    def __array__ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.data.__array__ufunc__(ufunc, method, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return self.data.__array_function__(func, types, args, kwargs)

    @staticmethod
    def isvalid(data):
        raise NotImplementedError

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def values(self):
        return self.__array__()

    @property
    def empty(self):
        return len(self) == 0

    @property
    def parent(self):
        return getattr(self, "_parent", None)

    @property
    def name(self):
        if self.parent is None:
            return self.dim
        return next((name for name in self.parent if self.parent[name] is self), None)

    def isdim(self):
        if self.parent is None or self.name is None:
            return None
        else:
            return self.parent.isdim(self.name)

    def equals(self, other): ...

    def to_index(self, item, method=None, endpoint=True):
        if isinstance(item, slice):
            return self.slice_indexer(item.start, item.stop, item.step, endpoint)
        else:
            return self.get_indexer(item, method)

    def format_index(self, idx, bounds="raise"):
        idx = np.asarray(idx)
        if not np.issubdtype(idx.dtype, np.integer):
            raise IndexError("only integer are valid index")
        idx = idx + (idx < 0) * len(self)
        if bounds == "raise":
            if np.any(idx < 0) or np.any(idx >= len(self)):
                raise IndexError("index is out of bounds")
        elif bounds == "clip":
            idx = np.clip(idx, 0, len(self))
        return idx

    def format_index_slice(self, slc):
        start = slc.start
        stop = slc.stop
        step = slc.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        if step <= 0:
            raise NotImplementedError(
                "negative or zero step when slicing is not supported yet"
            )
        start = self.format_index(start, bounds="clip")
        stop = self.format_index(stop, bounds="clip")
        return slice(start, stop, step)

    def slice_indexer(self, start=None, stop=None, step=None, endpoint=True):
        if start is not None:
            try:
                start_index = self.get_indexer(start, method="bfill")
            except KeyError:
                start_index = len(self)
        else:
            start_index = None
        if stop is not None:
            try:
                end_index = self.get_indexer(stop, method="ffill")
                stop_index = end_index + 1
            except KeyError:
                stop_index = 0
        else:
            stop_index = None
        if step is not None:
            raise NotImplementedError("cannot use step yet")
        if (
            (not endpoint)
            and (stop is not None)
            and (self[stop_index - 1].values == stop)
        ):
            stop_index -= 1
        return slice(start_index, stop_index)

    def isscalar(self):
        return False

    def isdefault(self):
        return False

    def isdense(self):
        return False

    def isinterp(self):
        return False

    def issampled(self):
        return False

    def append(self, other):
        raise NotImplementedError(f"append is not implemented for {self.__class__}")

    def to_dataarray(self):
        from ..dataarray import DataArray  # TODO: avoid defered import?

        if self.name is None:
            raise ValueError("cannot convert unnamed coordinate to DataArray")

        if self.parent is None:
            return DataArray(
                self.values,
                {self.dim: self},
                dims=[self.dim],
                name=self.name,
            )
        else:
            return DataArray(
                self.values,
                {
                    name: coord
                    for name, coord in self.parent.items()
                    if coord.dim == self.dim
                },
                dims=[self.dim],
                name=self.name,
            )

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_dataset(self, dataset, attrs):
        dataset = dataset.assign_coords(
            {self.name: (self.dim, self.values) if self.dim else self.values}
        )
        return dataset, attrs

    @classmethod
    def from_dataset(cls, dataset, name):
        coords = {}
        for subcls in cls.__subclasses__():
            if hasattr(subcls, "from_dataset"):
                coords |= subcls.from_dataset(dataset, name)
        return coords


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
        if ScalarCoordinate.isvalid(data):
            return ScalarCoordinate(data)
        else:
            return Coordinate(data, self.dim)

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


def parse(data, dim=None):
    if isinstance(data, tuple):
        if dim is None:
            dim, data = data
        else:
            _, data = data
    if isinstance(data, Coordinate):
        if dim is None:
            dim = data.dim
        data = data.data
    return data, dim


def get_sampling_interval(da, dim, cast=True):
    """
    Returns the sample spacing along a given dimension.

    Parameters
    ----------
    da : DataArray or DataArray or DataArray
        The data from which extract the sample spacing.
    dim : str
        The dimension along which get the sample spacing.
    cast: bool, optional
        Wether to cast datetime64 to seconds, by default True.

    Returns
    -------
    float
        The sample spacing.
    """
    if da.sizes[dim] < 2:
        raise ValueError(
            "cannot compute sample spacing on a dimension with less than 2 points"
        )
    coord = da[dim]
    if coord.isinterp():
        num = np.diff(coord.tie_values)
        den = np.diff(coord.tie_indices)
        mask = den != 1
        num = num[mask]
        den = den[mask]
        d = np.median(num / den)
    elif coord.issampled():
        d = coord.sampling_interval
    else:
        d = (coord[-1].values - coord[0].values) / (len(coord) - 1)
        d = np.asarray(d)
    if cast and np.issubdtype(d.dtype, np.timedelta64):
        d = d / np.timedelta64(1, "s")
    return d


def is_strictly_increasing(x):
    if np.issubdtype(x.dtype, np.datetime64):
        return np.all(np.diff(x) > np.timedelta64(0, "ns"))
    else:
        return np.all(np.diff(x) > 0)


def format_datetime(x):
    string = str(x)
    if "." in string:
        datetime, digits = string.split(".")
        digits = digits[:3]
        return ".".join([datetime, digits])
    else:
        return string
