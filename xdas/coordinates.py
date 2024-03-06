import warnings

import numpy as np
import pandas as pd


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
      * distance (distance): [0 1 2]
        channel (distance): ['DAS01' 'DAS02' 'DAS03']
        interrogator: SRN
    """

    def __init__(self, coords=None, dims=None):
        super().__init__()
        for name in coords:
            if isinstance(coords[name], AbstractCoordinate):
                self[name] = coords[name]
            elif isinstance(coords[name], tuple):
                dim, data = coords[name]
                self[name] = Coordinate(data, dim)
            else:
                self[name] = Coordinate(coords[name], name)
        if dims is None:
            dims = tuple(name for name in self if self.isdim(name))
        self.dims = dims

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
            query.update(item)
        elif isinstance(item, tuple):
            for k in range(len(item)):
                query[self.dims[k]] = item[k]
        else:
            query[self.dims[0]] = item
        return query

    def to_index(self, item):
        query = self.get_query(item)
        return {dim: self[dim].to_index(query[dim]) for dim in query}

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

        >>> coords = {
        ...     "time": {"tie_indices": [0, 999], "tie_values": [0.0, 10.0]},
        ...     "distance": [0, 1, 2],
        ...     "channel": ("distance", ["DAS01", "DAS02", "DAS03"]),
        ...     "interrogator": (None, "SRN"),
        ... }
        >>> xdas.Coordinates(coords).to_dict()
        {'time': {'dim': 'time',
          'data': {'tie_indices': [0, 999], 'tie_values': [0.0, 10.0]}},
         'distance': {'dim': 'distance', 'data': [0, 1, 2]},
         'channel': {'dim': 'distance', 'data': ['DAS01', 'DAS02', 'DAS03']},
         'interrogator': {'dim': None, 'data': 'SRN'}}
        """

        return {name: self[name].to_dict() for name in self}


class Coordinate:
    def __new__(cls, data, dim=None):
        if isinstance(data, AbstractCoordinate):
            if dim is None:
                return data
            else:
                return Coordinate(data.data, dim)
        elif ScalarCoordinate.isvalid(data):
            return ScalarCoordinate(data, dim)
        elif DenseCoordinate.isvalid(data):
            return DenseCoordinate(data, dim)
        elif InterpCoordinate.isvalid(data):
            return InterpCoordinate(data, dim)
        else:
            raise TypeError("could not parse `data`")


class AbstractCoordinate:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __getitem__(self, item):
        data = self.data.__getitem__(item)
        if ScalarCoordinate.isvalid(data):
            return ScalarCoordinate(data)
        else:
            return Coordinate(data, self.dim)

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return self.data.__str__()

    def __add__(self, other):
        return self.__class__(self.data + other, self.dim)

    def __sub__(self, other):
        return self.__class__(self.data - other, self.dim)

    def __array__(self):
        return self.data.__array__()

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

    def equals(self, other):
        return NotImplementedError

    def to_index(self, item):
        if isinstance(item, slice):
            return self.slice_indexer(item.start, item.stop, item.step)
        else:
            return self.get_indexer(item)

    def isscalar(self):
        return isinstance(self, ScalarCoordinate)

    def isdense(self):
        return isinstance(self, DenseCoordinate)

    def isinterp(self):
        return isinstance(self, InterpCoordinate)


class ScalarCoordinate(AbstractCoordinate):
    def __init__(self, data, dim=None):
        if dim is not None:
            raise ValueError("a scalar coordinate cannot be a dim")
        if not self.__class__.isvalid(data):
            raise TypeError("`data` must be scalar-like")
        self.data = np.asarray(data)

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

    def to_index(self, item):
        raise NotImplementedError("cannot get index of scalar coordinate")

    def to_dict(self):
        if np.issubdtype(self.dtype, np.datetime64):
            data = self.data.astype(str).item()
        else:
            data = self.data.item()
        return {"dim": self.dim, "data": data}


class DenseCoordinate(AbstractCoordinate):
    def __init__(self, data, dim=None):
        if not self.isvalid(data):
            raise TypeError("`data` must be array-like")
        self.data = np.asarray(data)
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
            return np.array_equal(self.data, other.data)
        else:
            return False

    def get_indexer(self, value, method=None):
        if np.isscalar(value):
            return self.index.get_indexer([value], method).item()
        else:
            return self.index.get_indexer(value, method)

    def slice_indexer(self, start=None, end=None, step=None):
        return self.index.slice_indexer(start, end, step)

    def to_dict(self):
        if np.issubdtype(self.dtype, np.datetime64):
            data = list(self.data.astype(str))
        else:
            data = list(self.data)
        return {"dim": self.dim, "data": data}


class InterpCoordinate(AbstractCoordinate):
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

    def __init__(self, data, dim=None):
        if not self.__class__.isvalid(data):
            raise TypeError("`data` must be dict-like")
        if not set(data) == {"tie_indices", "tie_values"}:
            raise ValueError(
                "both `tie_indices` and `tie_values` key should be provided"
            )
        tie_indices = np.asarray(data["tie_indices"])
        tie_values = np.asarray(data["tie_values"])
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
        try:
            data = dict(data)
            return True
        except (TypeError, ValueError):
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

    def __array__(self):
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError()

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError()

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
        return np.array_equal(self.tie_indices, other.tie_indices) and np.array_equal(
            self.tie_values, other.tie_values
        )

    def get_value(self, index):
        index = self.format_index(index)
        return linear_interpolate(index, self.tie_indices, self.tie_values)

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
        if method is None:
            index = linear_interpolate(value, self.tie_values, self.tie_indices)
            index = np.rint(index).astype("int")
            index_value = self.get_value(index)
            if np.issubdtype(self.dtype, np.datetime64):
                if not np.all(index_value == value):
                    raise KeyError("value not found in index")
                else:
                    return index
            else:
                if not np.allclose(index_value, value):
                    raise KeyError("value not found in index")
                else:
                    return index
        elif method == "nearest":
            index = linear_interpolate(value, self.tie_values, self.tie_indices)
            return np.rint(index).astype("int")
        elif method == "ffill":
            index = linear_interpolate(
                value, self.tie_values, self.tie_indices, left=np.nan
            )
            if np.any(np.isnan(index)):
                raise KeyError("value not found in index")
            else:
                return np.floor(index).astype("int")
        elif method == "bfill":
            index = linear_interpolate(
                value, self.tie_values, self.tie_indices, right=np.nan
            )
            if np.any(np.isnan(index)):
                raise KeyError("value not found in index")
            else:
                return np.ceil(index).astype("int")
        else:
            raise ValueError("valid methods are: 'nearest', 'before', 'after'")

    def slice_indexer(self, start=None, stop=None, step=None):
        if start is not None:
            try:
                start = self.get_indexer(start, method="bfill")
            except KeyError:
                start = len(self)
        if stop is not None:
            try:
                end = self.get_indexer(stop, method="ffill")
                stop = end + 1
            except KeyError:
                stop = 0
        if step is not None:
            raise NotImplementedError("cannot use step yet")
        return slice(start, stop)

    def decimate(self, q):
        tie_indices = (self.tie_indices // q) * q
        for k in range(1, len(tie_indices) - 1):
            if tie_indices[k] == tie_indices[k - 1]:
                tie_indices[k] += q
        tie_values = [self.get_value(idx) for idx in tie_indices]
        tie_indices //= q
        return self.__class__(dict(tie_indices=tie_indices, tie_values=tie_values))

    def simplify(self, tolerance=None):
        if tolerance is None:
            if np.issubdtype(self.dtype, np.datetime64):
                tolerance = np.timedelta64(0, "us")
            else:
                tolerance = 0.0
        tie_indices, tie_values = douglas_peucker(
            self.tie_indices, self.tie_values, tolerance
        )
        return self.__class__(
            dict(tie_indices=tie_indices, tie_values=tie_values), self.dim
        )

    def get_discontinuities(self):
        (indices,) = np.nonzero(np.diff(self.tie_indices) == 1)
        return [
            {
                self.tie_indices[index]: self.tie_values[index],
                self.tie_indices[index + 1]: self.tie_values[index + 1],
            }
            for index in indices
        ]

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
        data = {"tie_indices": list(tie_indices), "tie_values": list(tie_values)}
        return {"dim": self.dim, "data": data}


class ScaleOffset:
    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def __eq__(self, other):
        try:
            return (self.scale - other.scale == 0) and (self.offset - other.offset == 0)
        except:
            return False

    @classmethod
    def floatize(cls, arr):
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.datetime64):
            unit, count = np.datetime_data(arr.dtype)
            scale = np.timedelta64(count, unit)
            offset = np.min(arr) + (np.max(arr) - np.min(arr)) / 2
        else:
            scale = 1.0
            offset = 0.0
        transform = cls(scale, offset)
        transform.check_resolution(arr)
        return transform

    def direct(self, arr):
        arr = np.asarray(arr)
        self.check_resolution(arr)
        return (arr - self.offset) / self.scale

    def inverse(self, arr):
        arr = np.asarray(arr)
        if np.issubdtype(np.asarray(self.scale).dtype, np.timedelta64):
            arr = np.rint(arr)
        return self.scale * arr + self.offset

    def check_resolution(self, arr):
        arr = np.asarray(arr)
        nmax = 2 ** np.finfo("float").nmant
        if not np.all((arr - self.offset).astype("int") < nmax):
            warnings.warn(
                "float resolution is not sufficient to represent the full integer range"
            )


def linear_interpolate(x, xp, fp, left=None, right=None):
    if not is_strictly_increasing(xp):
        raise ValueError(
            "xp must be strictly increasing. Your coordinate probably has overlaps. "
            "Try to do: db['dim'] = db['dim'].simplify(np.timedelta64(tolerance, 'ms') "
            "with a gradually increasing tolerance until minor overlaps are resolved."
            "Big overlaps needs manual intervention."
        )
    x_transform = ScaleOffset.floatize(xp)
    f_transform = ScaleOffset.floatize(fp)
    x = x_transform.direct(x)
    xp = x_transform.direct(xp)
    fp = f_transform.direct(fp)
    f = np.interp(x, xp, fp, left=left, right=right)
    f = f_transform.inverse(f)
    return f


def is_strictly_increasing(x):
    if np.issubdtype(x.dtype, np.datetime64):
        return np.all(np.diff(x) > np.timedelta64(0, "us"))
    else:
        return np.all(np.diff(x) > 0)


def douglas_peucker(x, y, epsilon):
    mask = np.ones(len(x), dtype=bool)
    stack = [(0, len(x))]
    while stack:
        start, stop = stack.pop()
        ysimple = linear_interpolate(
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
    datetime, digits = str(x).split(".")
    digits = digits[:3]
    return ".".join([datetime, digits])
