import warnings

import numpy as np
import pandas as pd


class Coordinates(dict):
    """
    A dictionary whose keys are dimension names and values are Coordinate objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for dim in self:
            if not isinstance(self[dim], Coordinate):
                self[dim] = pd.Index(self[dim])

    @property
    def dims(self):
        return tuple(self.keys())

    @property
    def ndim(self):
        return len(self)

    def __repr__(self):
        s = "Coordinates:\n"
        for dim, coord in self.items():
            s += f"  * {dim}: "
            s += repr(coord) + "\n"
        return s

    def get_query(self, item):
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
        key = {}
        for dim in query:
            if isinstance(query[dim], slice):
                key[dim] = self[dim].slice_indexer(
                    query[dim].start, query[dim].stop, query[dim].step
                )
            else:
                key[dim] = self[dim].get_indexer(query[dim])
        return key


class Coordinate:
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

    def __init__(self, tie_indices, tie_values):
        tie_indices = np.asarray(tie_indices)
        tie_values = np.asarray(tie_values)
        if not tie_indices.ndim == 1:
            raise ValueError("tie_indices must be 1D.")
        if not tie_values.ndim == 1:
            raise ValueError("tie_values must be 1D.")
        if not len(tie_indices) == len(tie_values):
            raise ValueError("tie_indices and tie_values must have the same length.")
        if not tie_indices.shape == (0,):
            if not tie_indices[0] == 0:
                raise ValueError("tie_indices must start with a zero")
            if not is_strictly_increasing(tie_indices):
                raise ValueError("tie_indices must be strictly increasing")
        self.tie_indices = tie_indices
        self.tie_values = tie_values
        self.kind = "linear"

    def __len__(self):
        if self.empty:
            return 0
        else:
            return self.tie_indices[-1] - self.tie_indices[0] + 1

    def __repr__(self):
        if len(self) == 0:
            return "empty coordinate"
        elif len(self) == 1:
            return f"one point at {self.tie_values[0]}"
        else:
            return (
                f"{len(self.tie_indices)} tie points from {self.tie_values[0]} to "
                f"{self.tie_values[-1]}"
            )

    def __add__(self, other):
        tie_values = self.tie_values + other
        return self.__class__(self.tie_indices, tie_values)

    def __sub__(self, other):
        tie_values = self.tie_values - other
        return self.__class__(self.tie_indices, tie_values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice_index(item)
        else:
            return self.get_value(item)

    def __array__(self):
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError()

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError()

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
        start = self.format_index(start, bounds="clip")
        stop = self.format_index(stop, bounds="clip")
        return slice(start, stop, step)

    def get_value(self, index):
        index = self.format_index(index)
        return linear_interpolate(index, self.tie_indices, self.tie_values)

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
        elif method == "before":
            index = linear_interpolate(
                value, self.tie_values, self.tie_indices, left=np.nan
            )
            if np.any(np.isnan(index)):
                raise KeyError("value not found in index")
            else:
                return np.floor(index).astype("int")
        elif method == "after":
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
                start = self.get_indexer(start, method="after")
            except KeyError:
                start = len(self)
        if stop is not None:
            try:
                end = self.get_indexer(stop, method="before")
                stop = end + 1
            except KeyError:
                stop = 0
        if step is not None:
            raise NotImplementedError("cannot use step yet")
        return slice(start, stop)

    def slice_index(self, index_slice):
        index_slice = self.format_index_slice(index_slice)
        start_index, stop_index, step_index = (
            index_slice.start,
            index_slice.stop,
            index_slice.step,
        )
        if stop_index - start_index <= 0:
            return Coordinate([], [])
        elif (stop_index - start_index) <= step_index:
            tie_indices = [0]
            tie_values = [self.get_value(start_index)]
            return Coordinate(tie_indices, tie_values)
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
            coord = Coordinate(tie_indices, tie_values)
            if step_index != 1:
                coord = coord.decimate(step_index)
            return coord

    def decimate(self, q):
        tie_indices = (self.tie_indices // q) * q
        for k in range(1, len(tie_indices) - 1):
            if tie_indices[k] == tie_indices[k - 1]:
                tie_indices[k] += q
        tie_values = [self.get_value(idx) for idx in tie_indices]
        tie_indices //= q
        return self.__class__(tie_indices, tie_values)

    def simplify(self, tolerance=None):
        if tolerance is None:
            if np.issubdtype(self.dtype, np.datetime64):
                tolerance = np.timedelta64(0, "us")
            else:
                tolerance = 0.0
        tie_indices, tie_values = douglas_peucker(
            self.tie_indices, self.tie_values, tolerance
        )
        return self.__class__(tie_indices, tie_values)

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
    def from_array(cls, arr, tolerance=None):
        return cls(np.arange(len(arr)), arr).simplify(tolerance)


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
        raise ValueError("xp must be strictly increasing")
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
