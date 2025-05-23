from copy import copy, deepcopy
from functools import wraps

import numpy as np
import pandas as pd
from xinterp import forward, inverse


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
                "infered dimension number from `coords` does not match "
                "`data` dimensionality`"
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
        if ScalarCoordinate.isvalid(data):
            return object.__new__(ScalarCoordinate)
        elif DenseCoordinate.isvalid(data):
            return object.__new__(DenseCoordinate)
        elif InterpCoordinate.isvalid(data):
            return object.__new__(InterpCoordinate)
        else:
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

    def equals(self, other): ...

    def to_index(self, item, method=None, endpoint=True):
        if isinstance(item, slice):
            return self.slice_indexer(item.start, item.stop, item.step, endpoint)
        else:
            return self.get_indexer(item, method)

    def isscalar(self):
        return isinstance(self, ScalarCoordinate)

    def isdense(self):
        return isinstance(self, DenseCoordinate)

    def isinterp(self):
        return isinstance(self, InterpCoordinate)

    def append(self, other):
        raise NotImplementedError(f"append is not implemented for {self.__class__}")

    def to_dataarray(self):
        from .dataarray import DataArray  # TODO: avoid defered import?

        return DataArray(self.values, {self.dim: self}, name=self.dim)

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)


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


class InterpCoordinate(Coordinate):
    """
    Array-like object used to represent piecewise evenly spaced coordinates using the
    CF convention.

    The coordinate ticks are describes by the mean of tie points that are interpolated
    when intermediate values are required. Coordinate objects provides label based
    selections methods.

    Parameters
    ----------
    tie_indices : sequence of integers
        The indices of the tie points. Must include index 0 and be strictly increasing.
    tie_values : sequence of float or datetime64
        The values of the tie points. Must be strictly increasing to enable label-based
        selection. The len of `tie_indices` and `tie_values` sizes must match.
    """

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, data=None, dim=None, dtype=None):
        if data is None:
            data = {"tie_indices": [], "tie_values": []}
        data, dim = parse(data, dim)
        if not self.__class__.isvalid(data):
            raise TypeError("`data` must be dict-like")
        if not set(data) == {"tie_indices", "tie_values"}:
            raise ValueError(
                "both `tie_indices` and `tie_values` key should be provided"
            )
        tie_indices = np.asarray(data["tie_indices"])
        tie_values = np.asarray(data["tie_values"], dtype=dtype)
        if not tie_indices.ndim == 1:
            raise ValueError("`tie_indices` must be 1D")
        if not tie_values.ndim == 1:
            raise ValueError("`tie_values` must be 1D")
        if not len(tie_indices) == len(tie_values):
            raise ValueError("`tie_indices` and `tie_values` must have the same length")
        if not tie_indices.shape == (0,):
            if not np.issubdtype(tie_indices.dtype, np.integer):
                raise ValueError("`tie_indices` must be integer-like")
            if not tie_indices[0] == 0:
                raise ValueError("`tie_indices` must start with a zero")
            if not is_strictly_increasing(tie_indices):
                raise ValueError("`tie_indices` must be strictly increasing")
        if not (
            np.issubdtype(tie_values.dtype, np.number)
            or np.issubdtype(tie_values.dtype, np.datetime64)
        ):
            raise ValueError("`tie_values` must have either numeric or datetime dtype")
        tie_indices = tie_indices.astype(int)
        self.data = dict(tie_indices=tie_indices, tie_values=tie_values)
        self.dim = dim

    @staticmethod
    def isvalid(data):
        match data:
            case {"tie_indices": _, "tie_values": _}:
                return True
            case _:
                return False

    def __len__(self):
        if self.empty:
            return 0
        else:
            return self.tie_indices[-1] - self.tie_indices[0] + 1

    def __repr__(self):
        if len(self) == 0:
            return "empty coordinate"
        elif len(self) == 1:
            return f"{self.tie_values[0]}"
        else:
            if np.issubdtype(self.dtype, np.floating):
                return f"{self.tie_values[0]:.3f} to {self.tie_values[-1]:.3f}"
            elif np.issubdtype(self.dtype, np.datetime64):
                start = format_datetime(self.tie_values[0])
                end = format_datetime(self.tie_values[-1])
                return f"{start} to {end}"
            else:
                return f"{self.tie_values[0]} to {self.tie_values[-1]}"

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice_index(item)
        elif np.isscalar(item):
            return ScalarCoordinate(self.get_value(item), None)
        else:
            return DenseCoordinate(self.get_value(item), self.dim)

    def __add__(self, other):
        return self.__class__(
            {"tie_indices": self.tie_indices, "tie_values": self.tie_values + other},
            self.dim,
        )

    def __sub__(self, other):
        return self.__class__(
            {"tie_indices": self.tie_indices, "tie_values": self.tie_values - other},
            self.dim,
        )

    def __array__(self, dtype=None):
        out = self.values
        if dtype is not None:
            out = out.__array__(dtype)
        return out

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError

    @property
    def tie_indices(self):
        return self.data["tie_indices"]

    @property
    def tie_values(self):
        return self.data["tie_values"]

    @property
    def empty(self):
        return self.tie_indices.shape == (0,)

    @property
    def dtype(self):
        return self.tie_values.dtype

    @property
    def ndim(self):
        return self.tie_values.ndim

    @property
    def shape(self):
        return (len(self),)

    @property
    def indices(self):
        if self.empty:
            return np.array([], dtype="int")
        else:
            return np.arange(self.tie_indices[-1] + 1)

    @property
    def values(self):
        if self.empty:
            return np.array([], dtype=self.dtype)
        else:
            return self.get_value(self.indices)

    def equals(self, other):
        return (
            np.array_equal(self.tie_indices, other.tie_indices)
            and np.array_equal(self.tie_values, other.tie_values)
            and self.dim == other.dim
            and self.dtype == other.dtype
        )

    def get_value(self, index):
        index = self.format_index(index)
        return forward(index, self.tie_indices, self.tie_values)

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

    def slice_index(self, index_slice):
        index_slice = self.format_index_slice(index_slice)
        start_index, stop_index, step_index = (
            index_slice.start,
            index_slice.stop,
            index_slice.step,
        )
        if stop_index - start_index <= 0:
            return self.__class__(dict(tie_indices=[], tie_values=[]))
        elif (stop_index - start_index) <= step_index:
            tie_indices = [0]
            tie_values = [self.get_value(start_index)]
            return self.__class__(dict(tie_indices=tie_indices, tie_values=tie_values))
        else:
            end_index = stop_index - 1
            start_value = self.get_value(start_index)
            end_value = self.get_value(end_index)
            mask = (start_index < self.tie_indices) & (self.tie_indices < end_index)
            tie_indices = np.insert(
                self.tie_indices[mask],
                (0, self.tie_indices[mask].size),
                (start_index, end_index),
            )
            tie_values = np.insert(
                self.tie_values[mask],
                (0, self.tie_values[mask].size),
                (start_value, end_value),
            )
            tie_indices -= tie_indices[0]
            data = {"tie_indices": tie_indices, "tie_values": tie_values}
            coord = self.__class__(data, self.dim)
            if step_index != 1:
                coord = coord.decimate(step_index)
            return coord

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
        start = self.format_index(start, bounds="clip")
        stop = self.format_index(stop, bounds="clip")
        return slice(start, stop, step)

    def get_indexer(self, value, method=None):
        if isinstance(value, str):
            value = np.datetime64(value)
        else:
            value = np.asarray(value)
        try:
            indexer = inverse(value, self.tie_indices, self.tie_values, method)
        except ValueError as e:
            if str(e) == "fp must be strictly increasing":
                raise ValueError(
                    "overlaps were found in the coordinate. If this is due to some "
                    "jitter in the tie values, consider smoothing the coordinate by "
                    "including some tolerance. This can be done by "
                    "`da[dim] = da[dim].simplify(tolerance)`, or by specifying a "
                    "tolerance when opening multiple files."
                )
            else:
                raise e
        return indexer

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
        coord = self.__class__(
            {
                "tie_indices": np.append(
                    self.tie_indices, other.tie_indices + len(self)
                ),
                "tie_values": np.append(self.tie_values, other.tie_values),
            },
            self.dim,
        )
        return coord

    def decimate(self, q):
        tie_indices = (self.tie_indices // q) * q
        for k in range(1, len(tie_indices) - 1):
            if tie_indices[k] == tie_indices[k - 1]:
                tie_indices[k] += q
        tie_values = [self.get_value(idx) for idx in tie_indices]
        tie_indices //= q
        return self.__class__(
            dict(tie_indices=tie_indices, tie_values=tie_values), self.dim
        )

    def simplify(self, tolerance=None):
        if tolerance is None:
            if np.issubdtype(self.dtype, np.datetime64):
                tolerance = np.timedelta64(0, "ns")
            else:
                tolerance = 0.0
        tie_indices, tie_values = douglas_peucker(
            self.tie_indices, self.tie_values, tolerance
        )
        return self.__class__(
            dict(tie_indices=tie_indices, tie_values=tie_values), self.dim
        )

    def get_discontinuities(self):
        """
        Returns a DataFrame containing information about the discontinuities.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:

            - start_index : int
                The index where the discontinuity starts.
            - end_index : int
                The index where the discontinuity ends.
            - start_value : float
                The value at the start of the discontinuity.
            - end_value : float
                The value at the end of the discontinuity.
            - delta : float
                The difference between the end_value and start_value.
            - type : str
                The type of the discontinuity, either "gap" or "overlap".

        """
        (indices,) = np.nonzero(np.diff(self.tie_indices) == 1)
        records = []
        for index in indices:
            start_index = self.tie_indices[index]
            end_index = self.tie_indices[index + 1]
            start_value = self.tie_values[index]
            end_value = self.tie_values[index + 1]
            record = {
                "start_index": start_index,
                "end_index": end_index,
                "start_value": start_value,
                "end_value": end_value,
                "delta": end_value - start_value,
                "type": ("gap" if end_value > start_value else "overlap"),
            }
            records.append(record)
        return pd.DataFrame.from_records(records)

    def get_availabilities(self):
        """
        Returns a DataFrame containing information about the data availability.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:

            - start_index : int
                The index where the discontinuity starts.
            - end_index : int
                The index where the discontinuity ends.
            - start_value : float
                The value at the start of the discontinuity.
            - end_value : float
                The value at the end of the discontinuity.
            - delta : float
                The difference between the end_value and start_value.
            - type : str
                The type of the discontinuity, always "data".

        """
        (indices,) = np.nonzero(np.diff(self.tie_indices) == 1)
        indices = np.insert(indices, [0, len(indices)], [0, len(self.tie_indices) - 1])
        records = []
        for start, end in zip(indices[:-1], indices[1:]):
            start_index = self.tie_indices[start]
            end_index = self.tie_indices[end]
            start_value = self.tie_values[start]
            end_value = self.tie_values[end]
            record = {
                "start_index": start_index,
                "end_index": end_index,
                "start_value": start_value,
                "end_value": end_value,
                "delta": end_value - start_value,
                "type": "data",
            }
            records.append(record)
        return pd.DataFrame.from_records(records)

    @classmethod
    def from_array(cls, arr, dim=None, tolerance=None):
        return cls(
            {"tie_indices": np.arange(len(arr)), "tie_values": arr}, dim
        ).simplify(tolerance)

    def to_dict(self):
        tie_indices = self.data["tie_indices"]
        tie_values = self.data["tie_values"]
        if np.issubdtype(tie_values.dtype, np.datetime64):
            tie_values = tie_values.astype(str)
        data = {
            "tie_indices": tie_indices.tolist(),
            "tie_values": tie_values.tolist(),
        }
        return {"dim": self.dim, "data": data, "dtype": str(self.dtype)}


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
    if isinstance(coord, InterpCoordinate):
        num = np.diff(coord.tie_values)
        den = np.diff(coord.tie_indices)
        mask = den != 1
        num = num[mask]
        den = den[mask]
        d = np.median(num / den)
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


def douglas_peucker(x, y, epsilon):
    mask = np.ones(len(x), dtype=bool)
    stack = [(0, len(x))]
    while stack:
        start, stop = stack.pop()
        ysimple = forward(
            x[start:stop],
            x[[start, stop - 1]],
            y[[start, stop - 1]],
        )
        d = np.abs(y[start:stop] - ysimple)
        index = np.argmax(d)
        dmax = d[index]
        index += start
        if dmax > epsilon:
            stack.append([start, index + 1])
            stack.append([index, stop])
        else:
            mask[start + 1 : stop - 1] = False
    return x[mask], y[mask]


def format_datetime(x):
    string = str(x)
    if "." in string:
        datetime, digits = string.split(".")
        digits = digits[:3]
        return ".".join([datetime, digits])
    else:
        return string
