import re

import numpy as np
import pandas as pd
from xinterp import forward, inverse

from .core import Coordinate, format_datetime, is_strictly_increasing, parse


class InterpCoordinate(Coordinate, name="interpolated"):
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
        # empty
        if data is None:
            data = {"tie_indices": [], "tie_values": []}

        # parse data
        data, dim = parse(data, dim)
        if not self.__class__.isvalid(data):
            raise TypeError("`data` must be dict-like")
        if not set(data) == {"tie_indices", "tie_values"}:
            raise ValueError(
                "both `tie_indices` and `tie_values` key should be provided"
            )
        tie_indices = np.asarray(data["tie_indices"])
        tie_values = np.asarray(data["tie_values"], dtype=dtype)

        # check shapes
        if not tie_indices.ndim == 1:
            raise ValueError("`tie_indices` must be 1D")
        if not tie_values.ndim == 1:
            raise ValueError("`tie_values` must be 1D")
        if not len(tie_indices) == len(tie_values):
            raise ValueError("`tie_indices` and `tie_values` must have the same length")

        # check dtypes
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

        # store data
        tie_indices = tie_indices.astype(int)
        self.data = dict(tie_indices=tie_indices, tie_values=tie_values)
        self.dim = dim

    @property
    def tie_indices(self):
        return self.data["tie_indices"]

    @property
    def tie_values(self):
        return self.data["tie_values"]

    @property
    def dtype(self):
        return self.tie_values.dtype

    @property
    def empty(self):
        return self.tie_indices.shape == (0,)

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
            return Coordinate(self.get_value(item), None)
        else:
            return Coordinate(self.get_value(item), self.dim)

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

    def isinterp(self):
        return True

    def get_sampling_interval(self, cast=True):
        if len(self) < 2:
            return None
        num = np.diff(self.tie_values)
        den = np.diff(self.tie_indices)
        mask = den != 1
        num = num[mask]
        den = den[mask]
        delta = np.median(num / den)
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

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

    def slice_index(self, index_slice):
        start_index, stop_index, step_index = index_slice.indices(len(self))
        if step_index < 0:
            raise NotImplementedError("negative slice step is not implemented")
        if stop_index - start_index <= 0:
            return self.__class__(dict(tie_indices=[], tie_values=[]), dim=self.dim)
        elif (stop_index - start_index) <= step_index:
            tie_indices = [0]
            tie_values = [self.get_value(start_index)]
            return self.__class__(
                dict(tie_indices=tie_indices, tie_values=tie_values), dim=self.dim
            )
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

    def get_split_indices(self, tolerance=None):
        (indices,) = np.nonzero(np.diff(self.tie_indices) == 1)
        indices += 1
        if tolerance is not None:
            deltas = self.tie_values[indices + 1] - self.tie_values[indices]
            indices = indices[np.abs(deltas) >= tolerance]
        return np.array(
            [self.tie_indices[index] for index in indices], dtype=self.tie_indices.dtype
        )

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

    def to_dataset(self, dataset, attrs):
        mapping = f"{self.name}: {self.name}_indices {self.name}_values"
        if "coordinate_interpolation" in attrs:
            attrs["coordinate_interpolation"] += " " + mapping
        else:
            attrs["coordinate_interpolation"] = mapping
        tie_indices = self.tie_indices
        tie_values = (
            self.tie_values.astype("M8[ns]")
            if np.issubdtype(self.tie_values.dtype, np.datetime64)
            else self.tie_values
        )
        interp_attrs = {
            "interpolation_name": "linear",
            "tie_points_mapping": f"{self.name}_points: {self.name}_indices {self.name}_values",
        }
        dataset.update(
            {
                f"{self.name}_interpolation": ((), np.nan, interp_attrs),
                f"{self.name}_indices": (f"{self.name}_points", tie_indices),
                f"{self.name}_values": (f"{self.name}_points", tie_values),
            }
        )
        return dataset, attrs

    @classmethod
    def from_dataset(cls, dataset, name):
        coords = {}
        mapping = dataset[name].attrs.pop("coordinate_interpolation", None)
        if mapping is not None:
            matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
            for match in matches:
                dim, indices, values = match
                data = {"tie_indices": dataset[indices], "tie_values": dataset[values]}
                coords[dim] = Coordinate(data, dim)
        return coords

    @classmethod
    def from_block(cls, start, size, step, dim=None, dtype=None):
        return cls(
            {
                "tie_indices": [0, size - 1],
                "tie_values": [start, start + step * (size - 1)],
            },
            dim=dim,
        )


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
