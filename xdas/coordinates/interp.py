"""
:class:`InterpCoordinate`: piecewise-linear coordinate.

Defined by tie points, using ``xinterp`` for forward and inverse interpolation.
"""

import re

import numpy as np
from typing_extensions import override
from xinterp import forward, inverse

from .core import (
    Coordinate,
    SampledMixin,
    is_monotonic_increasing,
    parse_data_dim,
    parse_scalar_delta,
)


class InterpCoordinate(SampledMixin, Coordinate, ctype="interpolated"):
    """
    Piecewise-linear coordinate described by tie points (CF convention).

    Values between tie points are recovered by linear interpolation.
    Discontinuities are represented by two consecutive tie points at adjacent
    indices.  Supports label-based selection via :meth:`~Coordinate.to_index`.

    Parameters
    ----------
    data : dict with keys ``tie_indices`` and ``tie_values``
        ``tie_indices`` : sequence of int
            Positions of the tie points.  Must start at 0 and be strictly
            increasing.
        ``tie_values`` : sequence of float or datetime64
            Values at the tie points.  Must be strictly increasing to enable
            label-based selection.  Length must match ``tie_indices``.
    dim : str, optional
        Name of the dimension this coordinate is associated with.
    dtype : dtype-like, optional
        Desired dtype for ``tie_values``.

    Examples
    --------
    >>> import xdas as xd
    >>> coord = xd.Coordinate(
    ...     {"tie_indices": [0, 9, 10, 19], "tie_values": [0.0, 90.0, 200.0, 290.0]}
    ... )
    >>> coord
    0.000 to 290.000
    """

    @override
    def __init__(self, data=None, dim=None, dtype=None):
        # empty
        if data is None:
            data = {"tie_indices": [], "tie_values": []}

        # parse data
        data, dim = parse_data_dim(data, dim)
        if not InterpCoordinate._isvalid(data):
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
            if not is_monotonic_increasing(tie_indices):
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
        """Integer array of tie-point positions (starts at 0, strictly increasing)."""
        return self.data["tie_indices"]

    @property
    def tie_values(self):
        """Array of tie-point values (numeric or datetime64, strictly increasing)."""
        return self.data["tie_values"]

    @property
    @override
    def dtype(self):
        return self.tie_values.dtype

    @classmethod
    @override
    def from_block(cls, start, size, step, dim=None, dtype=None):
        data = {
            "tie_indices": [0, size - 1],
            "tie_values": [start, start + step * (size - 1)],
        }
        return cls(data, dim=dim, dtype=dtype)

    @override
    def __len__(self):
        if len(self.tie_indices) > 0:
            return self.tie_indices[-1] - self.tie_indices[0] + 1
        else:
            return 0

    @staticmethod
    @override
    def _isvalid(data):
        match data:
            case {"tie_indices": _, "tie_values": _}:
                return True
            case _:
                return False

    @override
    def _is_monotonic_increasing(self):
        return not self.get_split_indices(
            "overlaps", tolerance=False
        ).size  # TODO: do not call split_indices

    def is_valid_sampling_interval(self, sampling_interval, tolerance=None):
        if len(self) < 2:
            valid = True
        else:
            num = np.diff(self.tie_values)
            den = np.diff(self.tie_indices)
            mask = den != 1
            num = num[mask]
            den = den[mask]
            dmin = (num - 2 * tolerance) / den
            dmax = (num + 2 * tolerance) / den
            print(dmin, dmax, sampling_interval)
            valid = np.all((dmin <= sampling_interval) & (sampling_interval <= dmax))
            print(sampling_interval <= dmax)
        return valid

    @override
    def _get_value(self, index):
        return forward(index, self.tie_indices, self.tie_values)

    @override
    def _get_indexer(self, value, method=None):
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
            else:  # pragma: no cover
                raise e
        return indexer

    @override
    def _slice(self, index_slice):
        start_index, stop_index, step_index = (
            index_slice.start,
            index_slice.stop,
            index_slice.step,
        )
        if stop_index - start_index <= 0:
            return self.__class__(dict(tie_indices=[], tie_values=[]), dim=self.dim)
        elif (stop_index - start_index) <= step_index:
            tie_indices = [0]
            tie_values = [self._get_value(start_index)]
            return self.__class__(
                dict(tie_indices=tie_indices, tie_values=tie_values), dim=self.dim
            )
        else:
            end_index = stop_index - 1
            start_value = self._get_value(start_index)
            end_value = self._get_value(end_index)
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

            if step_index != 1:
                tie_indices = (tie_indices // step_index) * step_index
                for k in range(1, len(tie_indices) - 1):
                    if tie_indices[k] == tie_indices[k - 1]:
                        tie_indices[k] += step_index
                tie_values = [self._get_value(start_index + idx) for idx in tie_indices]
                tie_indices //= step_index

            data = {"tie_indices": tie_indices, "tie_values": tie_values}
            return self.__class__(data, self.dim)

    @override
    def _concat(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot concatenate {type(other)} to {self.__class__}")
        if not self.dim == other.dim:
            raise ValueError("cannot concatenate coordinate with different dimension")
        if self.empty:
            return other
        if other.empty:
            return self
        if not self.dtype == other.dtype:
            raise ValueError("cannot concatenate coordinate with different dtype")
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

    @override
    def _to_dataset(self, dataset, attrs):
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
    @override
    def _collect_from_dataset(cls, dataset, name):
        coords = {}
        mapping = dataset[name].attrs.pop("coordinate_interpolation", None)
        if mapping is not None:
            matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
            for match in matches:
                dim, indices, values = match
                data = {"tie_indices": dataset[indices], "tie_values": dataset[values]}
                coords[dim] = Coordinate(data, dim)
        return coords

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

    @override
    def get_sampling_interval(self, cast=True):
        if len(self) < 2:
            return None
        num = np.diff(self.tie_values)
        den = np.diff(self.tie_indices)
        mask = den != 1
        num = num[mask]
        den = den[mask]
        if len(num) == 0:
            return None
        delta = np.median(num / den)
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

    @override
    def simplify(self, tolerance=None):
        if tolerance is False:
            return self.copy()
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
        tie_indices, tie_values = _douglas_peucker(
            self.tie_indices, self.tie_values, tolerance
        )
        return self.__class__(
            dict(tie_indices=tie_indices, tie_values=tie_values), self.dim
        )

    @override
    def get_split_indices(self, kind="discontinuities", tolerance=False):
        valid_kinds = {"discontinuities", "gaps", "overlaps"}
        if kind not in valid_kinds:
            raise ValueError(f"`kind` must be one of {valid_kinds}; got {kind!r}")

        (indices,) = np.nonzero(np.diff(self.tie_indices) == 1)
        indices += 1

        # Fast path: no filtering requested
        if kind == "discontinuities" and tolerance is False:
            return self.tie_indices[indices]

        sampling_interval = self.get_sampling_interval(cast=False)
        deltas = (
            self.tie_values[indices] - self.tie_values[indices - 1] - sampling_interval
        )

        if tolerance is False:
            zero = np.timedelta64(0) if np.issubdtype(self.dtype, np.datetime64) else 0

            match kind:
                case "gaps":
                    mask = deltas >= zero
                case "overlaps":  # pragma: no branch
                    mask = deltas < zero

        else:
            tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)

            match kind:
                case "discontinuities":
                    mask = np.abs(deltas) > tolerance
                case "gaps":
                    mask = deltas > tolerance
                case "overlaps":  # pragma: no branch
                    mask = deltas < -tolerance

        return self.tie_indices[indices[mask]]


class FixedInterpCoordinate(InterpCoordinate, ctype="fixinterp"):
    """
    Array-like object used to represent piecewise evenly spaced coordinates using the
    CF convention augmented by a sampling interval proper definition.

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
    sampling_interval : scalar
        The acquisition sampling interval. Slight sampling variations around that
        value are authorized (see below). This parameters is somehow redudent with the
        `tie_indices` and `tie_values` but ensure proper sampling rate definition to
        pass to further signal processing routines.
    tolerance : scalar
        The tolerated jitter defined as the variation in sampling around the ideal
        value. This parameter is used to check the sampling_interval consistency.
    """

    def __init__(self, data=None, dim=None, dtype=None):
        if data is None:
            data = {
                "tie_indices": [],
                "tie_values": [],
                "sampling_interval": None,
                "tolerance": None,
            }

        data, dim = parse_data_dim(data, dim)
        sampling_interval = data["sampling_interval"]
        tolerance = data.get("tolerance", None)
        data = {
            k: v for k, v in data.items() if k not in ("sampling_interval", "tolerance")
        }

        super().__init__(data, dim, dtype)

        self.assign_sampling_interval(sampling_interval, tolerance)

    def assign_sampling_interval(self, sampling_interval, tolerance=None):
        sampling_interval = parse_scalar_delta(sampling_interval, self.dtype)
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)

        if self.is_valid_sampling_interval(sampling_interval, tolerance):
            self.data["sampling_interval"] = sampling_interval
            self.data["tolerance"] = tolerance
        else:
            raise ValueError(
                "`sampling_interval`and `tolerance` are not consistent with "
                "the `tie_indices` and `tie_values`"
            )

    @property
    def sampling_interval(self):
        return self.data["sampling_interval"]

    @property
    def tolerance(self):
        return self.data["tolerance"]

    @staticmethod
    def isvalid(data):
        match data:
            case {
                "tie_indices": _,
                "tie_values": _,
                "sampling_interval": _,
                **rest,
            } if set(rest) <= {"tolerance"}:
                return True
            case _:
                return False

    def get_sampling_interval(self, cast=True):
        delta = self.sampling_interval
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

    def equals(self, other):
        return super().equals(other) and (
            self.sampling_interval == other.sampling_interval
        )

    def append(self, other):
        if not self.sampling_interval == other.sampling_interval:
            raise ValueError(
                "cannot append coordinate with different sampling interval"
            )
        coord = super().append(other)
        coord.data["sampling_interval"] = self.sampling_interval
        coord.data["tolerance"] = self.tolerance
        return coord

    def decimate(self, q):
        coord = super().__init__(q)
        sampling_interval = self.sampling_interval / q  # TODO: what about interger-like
        coord.data["sampling_interval"] = sampling_interval
        coord.data["tolerance"] = self.tolerance
        return coord

    def simplify(self, tolerance=None):  # TODO: shoul ensure that still OK
        return super().__init__(tolerance)

    @classmethod
    def from_array(cls, arr, dim=None, tolerance=None):
        coord = super().__init__(arr, dim, tolerance)
        coord.sampling_rate = coord.get_sampling_rate(cast=False)
        return coord

    def to_dict(self):
        d = super().to_dict()
        d["data"]["sampling_interval"] = self.sampling_interval
        return d

    def to_dataset(self, dataset, attrs):
        dataset, attrs = super().to_dataset(dataset, attrs)
        dataset[f"{self.name}_interpolation"].attrs["sampling_interval"] = (
            self.sampling_interval
        )
        # TODO: what about datetime64 ?
        return dataset, attrs

    @classmethod
    def from_dataset(cls, dataset, name): ...

    # coords = super().from_dataset(dataset, name)
    # for name, coord in coords.items():

    # coords = {}
    # mapping = dataset[name].attrs.pop("coordinate_interpolation", None)
    # if mapping is not None:
    #     matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
    #     for match in matches:
    #         dim, indices, values = match
    #         data = {"tie_indices": dataset[indices], "tie_values": dataset[values]}
    #         coords[dim] = Coordinate(data, dim)
    # return coords

    @classmethod
    def from_block(cls, start, size, step, dim=None, dtype=None):
        coord = super().from_block(start, size, step, dim, dtype)
        coord.data["sampling_interval"] = step
        return coord


def _douglas_peucker(x, y, epsilon):
    """
    Reduce the piecewise-linear curve *(x, y)* using the Douglas-Peucker algorithm.

    Points are dropped when they deviate less than *epsilon* from the simplified
    line connecting their neighbours.

    Parameters
    ----------
    x : numpy.ndarray
        Monotonically increasing sample positions (tie indices).
    y : numpy.ndarray
        Corresponding coordinate values (tie values).
    epsilon : float or numpy.timedelta64
        Maximum allowed deviation to retain a point.

    Returns
    -------
    x_simplified : numpy.ndarray
    y_simplified : numpy.ndarray
    """
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
