"""
:class:`SampledCoordinate`: regularly-sampled coordinate.

Described by tie points and a fixed ``sampling_interval`` between them.
"""

import re

import numpy as np
from typing_extensions import override

from .core import (
    Coordinate,
    SampledMixin,
    is_monotonic_increasing,
    parse_data_dim,
    parse_scalar_delta,
)

CODE_TO_UNITS = {
    "h": "hours",
    "m": "minutes",
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
    "ns": "nanoseconds",
}
UNITS_TO_CODE = {v: k for k, v in CODE_TO_UNITS.items()}


class SampledCoordinate(SampledMixin, Coordinate, ctype="sampled"):
    """
    Coordinate sampled at a fixed interval, with optional gaps between segments.

    More compact and numerically stable than
    :class:`InterpCoordinate` for strictly uniform grids.  Each contiguous
    block is described by its start value and element count; all blocks share
    the same ``sampling_interval``.

    Parameters
    ----------
    data : dict with keys ``tie_values``, ``tie_lengths``, and ``sampling_interval``
        ``tie_values`` : sequence of float or datetime64
            Start value of each segment.
        ``tie_lengths`` : sequence of int
            Number of samples in each segment.  All values must be > 0.
        ``sampling_interval`` : scalar
            Fixed step between consecutive samples, shared across all segments.
            Must be :class:`numpy.timedelta64` when ``tie_values`` are
            :class:`numpy.datetime64`.
    dim : str, optional
        Name of the dimension this coordinate is associated with.
    dtype : dtype-like, optional
        Desired dtype for ``tie_values``.

    Examples
    --------
    >>> import numpy as np
    >>> from xdas.coordinates import SampledCoordinate
    >>> coord = SampledCoordinate(
    ...     {
    ...         "tie_values": [np.datetime64("2024-01-01T00:00:00", "ms")],
    ...         "tie_lengths": [1000],
    ...         "sampling_interval": np.timedelta64(4, "ms"),
    ...     }
    ... )
    >>> coord
    2024-01-01T00:00:00.000 to 2024-01-01T00:00:03.996
    """

    @override
    def __init__(self, data=None, dim=None, dtype=None):
        # empty
        if data is None:
            data = {"tie_values": [], "tie_lengths": [], "sampling_interval": None}
            empty = True
        else:
            empty = False

        # parse data
        data, dim = parse_data_dim(data, dim)
        if not self._isvalid(data):
            raise ValueError(
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

        # check dtypes and values
        if not empty:
            # tie_values
            if not (
                np.issubdtype(tie_values.dtype, np.number)
                or np.issubdtype(tie_values.dtype, np.datetime64)
            ):
                raise ValueError(
                    "`tie_values` must have either numeric or datetime dtype"
                )

            # tie_lengths
            if not np.issubdtype(tie_lengths.dtype, np.integer):
                raise ValueError("`tie_lengths` must be integer-like")
            if not np.all(tie_lengths > 0):
                raise ValueError("`tie_lengths` must be strictly positive integers")

            # sampling_interval
            if not np.ndim(sampling_interval) == 0:
                raise ValueError("`sampling_interval` must be a scalar value")
            sampling_interval = np.asarray(sampling_interval)[()]  # ensure numpy scalar
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
        """Start values of each regularly-sampled segment."""
        return self.data["tie_values"]

    @property
    def tie_lengths(self):
        """Number of samples in each regularly-sampled segment."""
        return self.data["tie_lengths"]

    @property
    def sampling_interval(self):
        """Fixed step between consecutive samples (shared across all segments)."""
        return self.data["sampling_interval"]

    @property
    def tie_indices(self):
        """Start integer index of each segment within the full coordinate array."""
        return np.concatenate(([0], np.cumsum(self.tie_lengths[:-1])))

    @property
    @override
    def dtype(self):
        return self.tie_values.dtype

    @classmethod
    @override
    def from_block(cls, start, size, step, dim=None, dtype=None):
        data = {
            "tie_values": [start],
            "tie_lengths": [size],
            "sampling_interval": step,
        }
        return cls(data, dim=dim, dtype=dtype)

    @override
    def __len__(self):
        return sum(self.tie_lengths)

    @staticmethod
    @override
    def _isvalid(data):
        match data:
            case {
                "tie_values": _,
                "tie_lengths": _,
                "sampling_interval": _,
            }:
                return True
            case _:
                return False

    @override
    def _is_monotonic_increasing(self):
        return not self.get_split_indices(
            "overlaps", tolerance=False
        ).size  # TODO: do not clall split_indices

    @override
    def _get_value(self, index):
        reference = np.searchsorted(self.tie_indices, index, side="right") - 1
        return self.tie_values[reference] + (
            (index - self.tie_indices[reference]) * self.sampling_interval
        )

    @override
    def _get_indexer(self, value, method=None):
        if isinstance(value, str):
            value = np.datetime64(value)
        else:
            value = np.asarray(value)
        if not is_monotonic_increasing(
            self.tie_values
        ):  # TODO: make it work even in this case
            raise ValueError("tie_values must be strictly increasing")

        # find preceeding tie point
        reference = np.searchsorted(self.tie_values, value, side="right") - 1
        reference = np.maximum(reference, 0)

        # overlaps
        before = np.maximum(reference - 1, 0)
        end = (
            self.tie_values[before]
            + (self.tie_lengths[before] - 1) * self.sampling_interval
        )
        if np.any((reference > 0) & (value <= end)):
            raise KeyError("value is in an overlap region")

        # gap
        after = np.minimum(reference + 1, len(self.tie_values) - 1)
        end = (
            self.tie_values[reference]
            + (self.tie_lengths[reference] - 1) * self.sampling_interval
        )
        match method:
            case "nearest":
                mask = (reference < len(self.tie_values) - 1) & (
                    value - end >= self.tie_values[after] - value
                )
                reference = np.where(mask, after, reference)
            case "bfill":
                mask = (reference < len(self.tie_values) - 1) & (value >= end)
                reference = np.where(mask, after, reference)
            case "ffill" | None:
                pass
            case _:
                raise ValueError(
                    "method must be one of `None`, 'nearest', 'ffill', or 'bfill'"
                )

        offset = (value - self.tie_values[reference]) / self.sampling_interval

        match method:  # pragma: no branch
            case None:
                if np.any(
                    (offset % 1 != 0)
                    | (offset < 0)
                    | (offset >= self.tie_lengths[reference])
                ):
                    raise KeyError("index not found")
                offset = offset.astype(int)
            case "nearest":
                offset = np.round(offset).astype(int)
                offset = np.clip(offset, 0, self.tie_lengths[reference] - 1)
            case "ffill":
                offset = np.floor(offset).astype(int)
                if np.any(offset < 0):
                    raise KeyError("index not found")
                offset = np.minimum(offset, self.tie_lengths[reference] - 1)
            case "bfill":  # pragma: no branch
                offset = np.ceil(offset).astype(int)
                if np.any(offset > self.tie_lengths[reference] - 1):
                    raise KeyError("index not found")
                offset = np.maximum(offset, 0)
        return self.tie_indices[reference] + offset

    @override
    def _slice(self, slc):
        start, stop, step = slc.start, slc.stop, slc.step

        # align stop
        stop += (start - stop) % step  # TODO: check for negative step

        # get relative start and stop within each tie
        q, r = np.divmod(start - self.tie_indices, step)
        lo = np.maximum(q, 0) * step + r

        q, r = np.divmod(self.tie_indices + self.tie_lengths - stop, step)
        hi = self.tie_lengths - np.maximum(q, 0) * step + r

        # filter empty segments
        mask = hi > lo
        lo = lo[mask]
        hi = hi[mask]

        # compute new tie values, tie lengths and sampling interval
        tie_values = self.tie_values[mask] + lo * self.sampling_interval
        tie_lengths = (hi - lo) // step
        sampling_interval = self.sampling_interval * step

        # build new coordinate
        data = {
            "tie_values": tie_values,
            "tie_lengths": tie_lengths,
            "sampling_interval": sampling_interval,
        }
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
        if not self.sampling_interval == other.sampling_interval:
            raise ValueError(
                "cannot concatenate coordinate with different sampling intervals"
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

    @override
    def _to_dataset(self, dataset, attrs):
        mapping = f"{self.name}: {self.name}_sampling"
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
            "tie_point_mapping": f"{self.dim}: {self.name}_values {self.name}_lengths",
        }

        # timedelta
        if np.issubdtype(self.sampling_interval.dtype, np.timedelta64):
            code, count = np.datetime_data(self.sampling_interval.dtype)
            interp_attrs["dtype"] = "timedelta64[ns]"
            interp_attrs["units"] = CODE_TO_UNITS[code]
            sampling_interval = count * self.sampling_interval.astype(int)
        else:
            sampling_interval = self.sampling_interval

        dataset.update(
            {
                f"{self.name}_sampling": ((), sampling_interval, interp_attrs),
                f"{self.name}_values": (f"{self.name}_points", tie_values),
                f"{self.name}_lengths": (f"{self.name}_points", tie_lengths),
            }
        )
        return dataset, attrs

    @classmethod
    @override
    def _collect_from_dataset(cls, dataset, name):
        coords = {}
        mapping = dataset[name].attrs.pop("coordinate_sampling", None)
        if mapping is not None:
            matches = re.findall(r"(\w+): (\w+)", mapping)
            for match in matches:
                name, sampling = match
                dim, values, lengths = re.match(
                    r"(\w+): (\w+) (\w+)", dataset[sampling].attrs["tie_point_mapping"]
                ).groups()
                data = {
                    "tie_values": dataset[values].values,
                    "tie_lengths": dataset[lengths].values,
                    "sampling_interval": dataset[sampling].values[()],
                }

                # timedelta
                if (
                    "dtype" in dataset[sampling].attrs
                    and "units" in dataset[sampling].attrs
                ):
                    data["sampling_interval"] = np.timedelta64(
                        data["sampling_interval"],
                        UNITS_TO_CODE[dataset[sampling].attrs.pop("units")],
                    ).astype(dataset[sampling].attrs.pop("dtype"))

                coords[name] = Coordinate(data, dim)
        return coords

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

    @override
    def get_sampling_interval(self, cast=True):
        if len(self) < 2:
            return None
        delta = self.sampling_interval
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

    @override
    def simplify(self, tolerance=None):
        if tolerance is False:
            return self.copy()
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
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

    @override
    def get_split_indices(self, kind="discontinuities", tolerance=False):
        valid_kinds = {"discontinuities", "gaps", "overlaps"}
        if kind not in valid_kinds:
            raise ValueError(f"`kind` must be one of {valid_kinds}; got {kind!r}")

        indices = self.tie_indices[1:]

        # Fast path: no filtering requested
        if kind == "discontinuities" and tolerance is False:
            return indices

        deltas = self.tie_values[1:] - (
            self.tie_values[:-1] + self.sampling_interval * self.tie_lengths[:-1]
        )

        if tolerance is False:
            zero = np.timedelta64(0) if np.issubdtype(self.dtype, np.datetime64) else 0

            match kind:  # pragma: no branch
                case "gaps":
                    mask = deltas >= zero
                case "overlaps":  # pragma: no branch
                    mask = deltas < zero

        else:
            tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)

            match kind:  # pragma: no branch
                case "discontinuities":
                    mask = np.abs(deltas) > tolerance
                case "gaps":
                    mask = deltas > tolerance
                case "overlaps":  # pragma: no branch
                    mask = deltas < -tolerance

        return indices[mask]
