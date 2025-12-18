import re

import numpy as np
import pandas as pd

from .core import Coordinate, format_datetime, is_strictly_increasing, parse


class SampledCoordinate(Coordinate):
    """
    A coordinate that is sampled at regular intervals.

    Parameters
    ----------
    data : dict-like
        The data of the coordinate.
    dim : str, optional
        The dimension name of the coordinate, by default None.
    dtype : str or numpy.dtype, optional
        The data type of the coordinate, by default None.
    """

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, data=None, dim=None, dtype=None):
        # empty
        if data is None:
            data = {"tie_values": [], "tie_lengths": [], "sampling_interval": None}
            empty = True
        else:
            empty = False

        # parse data
        data, dim = parse(data, dim)
        if not self.__class__.isvalid(data):
            raise TypeError(
                "`data` must be dict-like and contain `tie_values`, `tie_lengths`, and "
                "`sampling_interval`"
            )
        tie_values = np.asarray(data["tie_values"], dtype=dtype)
        tie_lengths = np.asarray(data["tie_lengths"])
        sampling_interval = data["sampling_interval"]

        # check shapes
        if not tie_values.ndim == 1:
            raise ValueError("`tie_values` must be 1D")
        if not tie_lengths.ndim == 1:
            raise ValueError("`tie_lengths` must be 1D")
        if not len(tie_values) == len(tie_lengths):
            raise ValueError("`tie_values` and `tie_lengths` must have the same length")

        # check dtypes
        if not empty:
            if not (
                np.issubdtype(tie_values.dtype, np.number)
                or np.issubdtype(tie_values.dtype, np.datetime64)
            ):
                raise ValueError(
                    "`tie_values` must have either numeric or datetime dtype"
                )
            if not np.issubdtype(tie_lengths.dtype, np.integer):
                raise ValueError("`tie_lengths` must be integer-like")
            if not np.all(tie_lengths > 0):
                raise ValueError("`tie_lengths` must be strictly positive integers")
            if not np.isscalar(sampling_interval):
                raise ValueError("`sampling_interval` must be a scalar value")
            if np.issubdtype(tie_values.dtype, np.datetime64):
                if not np.issubdtype(
                    np.asarray(sampling_interval).dtype, np.timedelta64
                ):
                    raise ValueError(
                        "`sampling_interval` must be timedelta64 for datetime64 `tie_values`"
                    )

        # store data
        self.data = {
            "tie_values": tie_values,
            "tie_lengths": tie_lengths,
            "sampling_interval": sampling_interval,
        }
        self.dim = dim

    @property
    def tie_values(self):
        return self.data["tie_values"]

    @property
    def tie_lengths(self):
        return self.data["tie_lengths"]

    @property
    def sampling_interval(self):
        return self.data["sampling_interval"]

    @property
    def dtype(self):
        return self.tie_values.dtype

    @staticmethod
    def isvalid(data):
        match data:
            case {
                "tie_values": _,
                "tie_lengths": _,
                "sampling_interval": _,
            }:
                return True
            case _:
                return False

    def issampled(self):
        return True

    def get_sampling_interval(self, cast=True):
        return self.sampling_interval

    def __len__(self):
        if self.empty:
            return 0
        else:
            return sum(self.tie_lengths)

    def __repr__(self):
        if self.empty:
            return "empty coordinate"
        elif len(self) == 1:
            return f"{self.tie_values[0]}"
        else:
            if np.issubdtype(self.dtype, np.floating):
                return f"{self.start:.3f} to {self.end:.3f}"
            elif np.issubdtype(self.dtype, np.datetime64):
                start_str = format_datetime(self.start)
                end_str = format_datetime(self.end)
                return f"{start_str} to {end_str}"
            else:
                return f"{self.start} to {self.end}"

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice_index(item)
        elif np.isscalar(item):
            return Coordinate(self.get_value(item), None)
        else:
            return Coordinate(self.get_value(item), self.dim)

    def __add__(self, other):
        return self.__class__(
            {
                "tie_values": self.tie_values + other,
                "tie_lengths": self.tie_lengths,
                "sampling_interval": self.sampling_interval,
            },
            self.dim,
        )

    def __sub__(self, other):
        return self.__class__(
            {
                "tie_values": self.tie_values - other,
                "tie_lengths": self.tie_lengths,
                "sampling_interval": self.sampling_interval,
            },
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
        return np.concatenate(([0], np.cumsum(self.tie_lengths[:-1])))

    @property
    def empty(self):
        return self.tie_values.shape == (0,)

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
            return np.arange(len(self))

    @property
    def values(self):
        if self.empty:
            return np.array([], dtype=self.dtype)
        else:
            return self.get_value(self.indices)

    @property
    def start(self):
        return self.tie_values[0]

    @property
    def end(self):
        return self.tie_values[-1] + self.sampling_interval * self.tie_lengths[-1]

    def equals(self, other):
        return (
            np.array_equal(self.tie_values, other.tie_values)
            and np.array_equal(self.tie_lengths, other.tie_lengths)
            and self.sampling_interval == other.sampling_interval
            and self.dim == other.dim
            and self.dtype == other.dtype
        )

    def get_value(self, index):
        index = self.format_index(index)
        if np.any(index < 0) or np.any(index >= len(self)):
            raise IndexError("index is out of bounds")
        reference = np.searchsorted(self.tie_indices, index, side="right") - 1
        return self.tie_values[reference] + (
            (index - self.tie_indices[reference]) * self.sampling_interval
        )

    def slice_index(self, index_slice):
        index_slice = self.format_index_slice(index_slice)

        # TODO: optimize when start and/or stop are None

        # get indices relative to tie points
        relative_start_index = np.clip(
            index_slice.start - self.tie_indices, 0, self.tie_lengths
        )
        relative_stop_index = np.clip(
            index_slice.stop - self.tie_indices, 0, self.tie_lengths
        )

        # keep segments with data
        mask = relative_start_index < relative_stop_index

        # compute new tie points ane lengths
        tie_values = (
            self.tie_values[mask] + relative_start_index[mask] * self.sampling_interval
        )
        tie_lengths = relative_stop_index[mask] - relative_start_index[mask]

        # adjust for step if needed
        if index_slice.step == 1:
            sampling_interval = self.sampling_interval
        else:
            tie_lengths = (self.tie_lengths + index_slice.step - 1) // index_slice.step
            sampling_interval = self.sampling_interval * index_slice.step

        # build new coordinate
        data = {
            "tie_values": tie_values,
            "tie_lengths": tie_lengths,
            "sampling_interval": sampling_interval,
        }
        return self.__class__(data, self.dim)

    def get_indexer(self, value, method=None):
        if isinstance(value, str):
            value = np.datetime64(value)
        else:
            value = np.asarray(value)
        # Check that value lies within the coordinate value range (vectorized)
        if np.any(value < self.start) or np.any(value >= self.end):
            raise KeyError("index not found")
        if not is_strictly_increasing(self.tie_values):
            raise ValueError("tie_values must be strictly increasing")
        reference = np.searchsorted(self.tie_values, value, side="right") - 1
        offset = (value - self.tie_values[reference]) / self.sampling_interval
        match method:
            case None:
                if np.any(offset % 1 != 0):
                    raise KeyError("index not found")
                offset = offset.astype(int)
            case "nearest":
                offset = np.round(offset).astype(int)
            case "ffill":
                offset = np.floor(offset).astype(int)
            case "bfill":
                offset = np.ceil(offset).astype(int)
            case _:
                raise ValueError(
                    "method must be one of `None`, 'nearest', 'ffill', or 'bfill'"
                )
        return self.tie_indices[reference] + offset

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
        if not self.sampling_interval == other.sampling_interval:
            raise ValueError(
                "cannot append coordinate with different sampling intervals"
            )
        tie_values = np.concatenate([self.tie_values, other.tie_values])
        tie_lengths = np.concatenate([self.tie_lengths, other.tie_lengths])
        return self.__class__(
            {
                "tie_values": tie_values,
                "tie_lengths": tie_lengths,
                "sampling_interval": self.sampling_interval,
            },
            self.dim,
        )

    def decimate(self, q):
        return self[::q]

    def simplify(self, tolerance=None):
        tie_values = [self.tie_values[0]]
        tie_lengths = [self.tie_lengths[0]]
        for value, length in zip(self.tie_values[1:], self.tie_lengths[1:]):
            delta = value - (tie_values[-1] + self.sampling_interval * tie_lengths[-1])
            if np.abs(delta) <= tolerance:
                tie_lengths[-1] += length
            else:
                tie_values.append(value)
                tie_lengths.append(length)
        return self.__class__(
            {
                "tie_values": np.array(tie_values),
                "tie_lengths": np.array(tie_lengths),
                "sampling_interval": self.sampling_interval,
            },
            self.dim,
        )

    def get_discontinuities(self):
        if self.empty:
            return pd.DataFrame(
                columns=[
                    "start_index",
                    "end_index",
                    "start_value",
                    "end_value",
                    "delta",
                    "type",
                ]
            )
        records = []
        for index in self.tie_indices[:-1]:
            start_index = index
            end_index = index + 1
            start_value = self.get_value(index)
            end_value = self.get_value(index + 1)
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
        if self.empty:
            return pd.DataFrame(
                columns=[
                    "start_index",
                    "end_index",
                    "start_value",
                    "end_value",
                    "delta",
                    "type",
                ]
            )
        records = []
        for index, value, length in zip(
            self.tie_indices, self.tie_values, self.tie_indices
        ):
            start_index = index
            end_index = index + length - 1
            start_value = value
            end_value = value + self.sampling_interval * (length - 1)
            records.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_value": start_value,
                    "end_value": end_value,
                    "delta": end_value - start_value,
                    "type": "data",
                }
            )
        return pd.DataFrame.from_records(records)

    @classmethod
    def from_array(cls, arr, dim=None, sampling_interval=None):
        raise NotImplementedError("from_array is not implemented for SampledCoordinate")

    def to_dict(self):
        tie_values = self.data["tie_values"]
        tie_lengths = self.data["tie_lengths"]
        if np.issubdtype(tie_values.dtype, np.datetime64):
            tie_values = tie_values.astype(str)
        data = {
            "tie_values": tie_values.tolist(),
            "tie_lengths": tie_lengths.tolist(),
            "sampling_interval": self.sampling_interval,
        }
        return {"dim": self.dim, "data": data, "dtype": str(self.dtype)}

    def to_dataset(self, dataset, attrs):
        mapping = f"{self.name}: {self.name}_values {self.name}_lengths"
        if "coordinate_sampling" in attrs:
            attrs["coordinate_sampling"] += " " + mapping
        else:
            attrs["coordinate_sampling"] = mapping
        tie_values = (
            self.tie_values.astype("M8[ns]")
            if np.issubdtype(self.tie_values.dtype, np.datetime64)
            else self.tie_values
        )
        tie_lengths = self.tie_lengths
        interp_attrs = {
            "sampling_interval": self.sampling_interval,
            "tie_points_mapping": f"{self.name}_points: {self.name}_values {self.name}_lengths",
        }
        dataset.update(
            {
                f"{self.name}_sampling": ((), np.nan, interp_attrs),
                f"{self.name}_values": (f"{self.name}_points", tie_values),
                f"{self.name}_lengths": (f"{self.name}_points", tie_lengths),
            }
        )
        return dataset, attrs

    @classmethod
    def from_dataset(cls, dataset, name):
        coords = {}
        mapping = dataset[name].attrs.pop("coordinate_sampling", None)
        if mapping is not None:
            matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
            for match in matches:
                dim, values, lengths = match
                sampling_interval = ...
                data = {
                    "tie_values": dataset[values],
                    "tie_lengths": dataset[lengths],
                    "sampling_interval": sampling_interval,
                }
                coords[dim] = Coordinate(data, dim)
        return coords
