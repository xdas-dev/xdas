"""
:class:`SampledCoordinate`: regularly-sampled coordinate described by tie
points and a fixed ``sampling_interval`` between them.
"""

import re

import numpy as np

from .core import (
    Coordinate,
    format_datetime,
    is_monotonic_increasing,
    parse,
    parse_tolerance,
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


class SampledCoordinate(Coordinate, name="sampled"):
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
    def dtype(self):
        """Dtype of the tie values (and of all materialised coordinate values)."""
        return self.tie_values.dtype

    @property
    def tie_indices(self):
        """Start integer index of each segment within the full coordinate array."""
        return np.concatenate(([0], np.cumsum(self.tie_lengths[:-1])))

    @property
    def empty(self):
        """``True`` if no segments have been set."""
        return self.tie_values.shape == (0,)

    @property
    def ndim(self):
        """Always 1."""
        return self.tie_values.ndim

    @property
    def shape(self):
        """Shape tuple ``(len(self),)``."""
        return (len(self),)

    @property
    def indices(self):
        """Full integer index array from 0 to ``len(self) - 1``."""
        if self.empty:
            return np.array([], dtype="int")
        else:
            return np.arange(len(self))

    @property
    def values(self):
        """Materialised numpy array of all coordinate values."""
        if self.empty:
            return np.array([], dtype=self.dtype)
        else:
            return self.get_value(self.indices)

    @property
    def start(self):
        """Value at index 0 (first tie value)."""
        return self.tie_values[0]

    @property
    def end(self):
        """Value one step past the last sample (exclusive upper bound)."""
        return self.tie_values[-1] + self.sampling_interval * self.tie_lengths[-1]

    @staticmethod
    def isvalid(data):
        """Return ``True`` if *data* has ``tie_values``, ``tie_lengths``, and ``sampling_interval`` keys."""
        match data:
            case {
                "tie_values": _,
                "tie_lengths": _,
                "sampling_interval": _,
            }:
                return True
            case _:
                return False

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
        else:
            return Coordinate(
                self.get_value(item), None if np.isscalar(item) else self.dim
            )

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

    def issampled(self):
        """Return ``True`` (this is a :class:`SampledCoordinate`)."""
        return True

    def get_sampling_interval(self, cast=True):
        """
        Return the sampling interval.

        Parameters
        ----------
        cast : bool, optional
            If ``True`` (default), cast timedelta64 to seconds (float).
        """
        delta = self.sampling_interval
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

    def equals(self, other):
        """Return ``True`` if *other* has identical tie values, lengths, sampling interval, dim, and dtype."""
        return (
            np.array_equal(self.tie_values, other.tie_values)
            and np.array_equal(self.tie_lengths, other.tie_lengths)
            and self.sampling_interval == other.sampling_interval
            and self.dim == other.dim
            and self.dtype == other.dtype
        )

    def get_value(self, index):
        """Compute coordinate value(s) at integer position(s) *index* using the stored segments."""
        index = self.format_index(index, bounds="raise")
        reference = np.searchsorted(self.tie_indices, index, side="right") - 1
        return self.tie_values[reference] + (
            (index - self.tie_indices[reference]) * self.sampling_interval
        )

    def slice_index(self, index_slice):
        """Return a new :class:`SampledCoordinate` for the integer slice *index_slice*."""
        # normalize slice
        start, stop, step = index_slice.indices(len(self))

        if step < 0:
            raise NotImplementedError("negative slice step is not implemented")

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

    def get_indexer(self, value, method=None):
        """
        Return the integer index for label *value* using the segment structure.

        Parameters
        ----------
        value : scalar, str (ISO datetime), or array-like
            Label(s) to locate.
        method : {None, "nearest", "ffill", "bfill"}, optional
            How to handle values that fall in gaps or between samples.

        Returns
        -------
        int or numpy.ndarray

        Raises
        ------
        KeyError
            If *value* falls in an overlap region or is not found (exact mode).
        """
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

    def concat(self, other):
        """Append *other* :class:`SampledCoordinate` segments after this one."""
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

    def decimate(self, q):
        """Return a new coordinate keeping every *q*-th sample (integer decimation)."""
        return self[::q]

    def simplify(self, tolerance=None):
        """
        Merge adjacent segments whose gap is within *tolerance* of the sampling interval.

        Parameters
        ----------
        tolerance : float, timedelta, or None
            Maximum allowed discrepancy between the expected and actual start of the
            next segment.  ``None`` uses zero tolerance.  ``False`` returns ``self`` unchanged.
        """
        if tolerance is False:
            return self  # TODO: copy
        tolerance = parse_tolerance(tolerance, self.dtype)
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

    def get_split_indices(self, kind="discontinuities", tolerance=False):
        """
        Return integer indices of segment boundaries (start of each segment except the first).

        Parameters
        ----------
        kind : {"discontinuities", "gaps", "overlaps"}, optional
            Which boundary type to return. Default ``"discontinuities"``.
        tolerance : float, timedelta, or ``False``
            Minimum magnitude of the discrepancy to report.

        Returns
        -------
        numpy.ndarray
        """
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
            tolerance = parse_tolerance(tolerance, self.dtype)

            match kind:  # pragma: no branch
                case "discontinuities":
                    mask = np.abs(deltas) > tolerance
                case "gaps":
                    mask = deltas > tolerance
                case "overlaps":  # pragma: no branch
                    mask = deltas < -tolerance

        return indices[mask]

    @classmethod
    def from_array(cls, arr, dim=None, sampling_interval=None):
        """Not supported — raises :exc:`NotImplementedError`."""
        raise NotImplementedError("from_array is not implemented for SampledCoordinate")

    def to_dict(self):
        """Serialise to ``{"dim": ..., "data": {"tie_values": ..., "tie_lengths": ..., "sampling_interval": ...}, "dtype": ...}``."""
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
        """Write sampling metadata into an xarray *dataset* using CF tie-point conventions."""
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
    def from_dataset(cls, dataset, name):
        """Read sampled coordinates from *dataset* using the ``coordinate_sampling`` attribute."""
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

    @classmethod
    def from_block(cls, start, size, step, dim=None, dtype=None):
        """Build a single-segment :class:`SampledCoordinate` starting at *start* with *size* samples and step *step*."""
        data = {
            "tie_values": [start],
            "tie_lengths": [size],
            "sampling_interval": step,
        }
        return cls(data, dim=dim, dtype=dtype)
