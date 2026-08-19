"""
:class:`SampledCoordinate`: regularly-sampled coordinate.

Described by tie points and a fixed ``sampling_interval`` between them.
"""

import re
import warnings

import numpy as np
from typing_extensions import override
from xinterp import deviation_step, forward_step, inverse_step, simplify_step

from .core import (
    UNITS_TO_CODE,
    AxisCoordinate,
    Coordinate,
    decode_delta,
    divide_sampling_ratio,
    encode_delta,
    is_monotonic_increasing,
    parse_data_dim,
    parse_scalar_delta,
    reduce_sampling_ratio,
)


class SampledCoordinate(AxisCoordinate, ctype="sampled"):
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
            :class:`numpy.datetime64`. Stored internally as ``sampling_numerator``
            / ``sampling_denominator`` (an exact, gcd-reduced ratio); either
            spelling may be passed instead, provided together.
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
                "`sampling_interval` (or `sampling_numerator` and "
                "`sampling_denominator`)"
            )
        tie_values = np.asarray(data["tie_values"], dtype=dtype)
        tie_lengths = np.asarray(data["tie_lengths"])
        has_ratio = "sampling_numerator" in data or "sampling_denominator" in data
        if has_ratio and "sampling_interval" in data:
            raise ValueError(
                "cannot pass both `sampling_interval` and "
                "`sampling_numerator`/`sampling_denominator`"
            )
        if has_ratio:
            if ("sampling_numerator" in data) != ("sampling_denominator" in data):
                raise ValueError(
                    "`sampling_numerator` and `sampling_denominator` must be "
                    "provided together"
                )
            sampling_interval = data["sampling_numerator"]
            sampling_denominator = data["sampling_denominator"]
        else:
            sampling_interval = data["sampling_interval"]
            sampling_denominator = 1 if sampling_interval is not None else None

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

            # sampling_interval / sampling_numerator
            if not np.ndim(sampling_interval) == 0:
                raise ValueError("`sampling_interval` must be a scalar value")
            sampling_interval = np.asarray(sampling_interval)[()]  # ensure numpy scalar
            if np.issubdtype(tie_values.dtype, np.datetime64) and not np.issubdtype(
                np.asarray(sampling_interval).dtype, np.timedelta64
            ):
                raise ValueError(
                    "`sampling_interval` must be timedelta64 for datetime64 `tie_values`"
                )

            # sampling_denominator
            if not np.ndim(sampling_denominator) == 0:
                raise ValueError("`sampling_denominator` must be a scalar value")
            sampling_denominator = int(sampling_denominator)
            if not sampling_denominator > 0:
                raise ValueError("`sampling_denominator` must be strictly positive")

            sampling_numerator, sampling_denominator = reduce_sampling_ratio(
                sampling_interval, sampling_denominator, tie_values.dtype
            )
        else:
            sampling_numerator, sampling_denominator = None, None

        # store data
        self.data = {
            "tie_values": tie_values,
            "tie_lengths": tie_lengths,
            "sampling_numerator": sampling_numerator,
            "sampling_denominator": sampling_denominator,
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
    def _sampling_ratio(self):
        """``(numerator, denominator)`` pair backing :attr:`sampling_interval`.

        Both are ``None`` together for an empty coordinate. This is the one place
        (besides :func:`~.core.parse_scalar_delta`) that assumes the rate is a
        single scalar rather than a per-segment array; per-segment intervals are
        expected later.
        """
        return self.data["sampling_numerator"], self.data["sampling_denominator"]

    @property
    def sampling_interval(self):
        """Fixed step between consecutive samples (shared across all segments)."""
        numerator, denominator = self._sampling_ratio
        return divide_sampling_ratio(numerator, denominator, self.dtype)

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
            case {"tie_values": _, "tie_lengths": _, **rest} if rest and set(rest) <= {
                "sampling_interval",
                "sampling_numerator",
                "sampling_denominator",
            }:
                return True
            case _:
                return False

    @override
    def _is_monotonic_increasing(self):
        # inside a segment the step is the sampling interval, at a seam it is
        # the jump plus one interval. A segment of a single sample has no
        # interior, so a coordinate made of those alone is ordered by its seams
        # only, whatever the sign of the interval.
        if self.empty:
            return True
        numerator, _ = self._sampling_ratio
        # the sign of the ratio is the sign of the numerator alone (the
        # denominator is always strictly positive), so this avoids the
        # exact-dtype floor division in `sampling_interval` rounding a small
        # positive fraction down to zero and flipping the verdict
        if np.any(self.tie_lengths > 1) and not numerator > numerator - numerator:
            return False
        _, deltas = self._split_candidates()
        zero = self.sampling_interval - self.sampling_interval
        return bool(np.all(deltas + self.sampling_interval > zero))

    def _step_rate(self):
        """``(num, den)`` lists as consumed by the xinterp step kernels.

        Float axes have no exact rate (D2): xinterp's float path takes the
        step itself as ``num`` and requires ``den == 1``, which is always the
        case here (Phase A forces a float denominator to 1). Integer and
        datetime axes pass the exact tick ratio.
        """
        numerator, denominator = self._sampling_ratio
        if np.issubdtype(self.dtype, np.floating):
            return [float(numerator)], [1]
        ticks = (
            int(numerator.astype("i8"))
            if np.issubdtype(self.dtype, np.datetime64)
            else int(numerator)
        )
        return [ticks], [int(denominator)]

    @override
    def _get_value(self, index):
        if np.size(index) == 0:
            # nothing to evaluate: skip straight past forward_step, which
            # (like `self.tie_values[-1]` just below) needs at least one
            # real tie point even when there is nothing to look up
            return np.empty(np.shape(index), dtype=self.dtype)
        # each block is its own, independently anchored piece: extend the
        # tie points with a final boundary at `len(self)` so xinterp's
        # boundary-pair model (piece i spans tie_indices[i]:tie_indices[i+1])
        # lines up with our blocks, one piece per block. The appended value
        # is never evaluated -- valid indices stop one tick short of it.
        tie_indices = np.append(self.tie_indices, len(self))
        tie_values = np.append(self.tie_values, self.tie_values[-1])
        num, den = self._step_rate()
        return forward_step(index, tie_indices, tie_values, num, den)

    def _resolve_offset(self, value, reference, method):
        """Within-block offset for each already-resolved `reference` block.

        `reference` (from the overlap/gap dance above) may point to
        different blocks for different elements of `value`, and each block
        has its own anchor -- so this resolves one block at a time, over the
        elements referring to it, via `inverse_step` on that block's own
        two boundary points (start, end). A single-sample block has no step
        to invert; it is resolved directly against the method's semantics.
        """
        scalar = np.ndim(reference) == 0
        value = np.atleast_1d(value)
        reference = np.atleast_1d(reference)

        numerator, denominator = self._sampling_ratio
        is_float = np.issubdtype(self.dtype, np.floating)
        is_datetime = np.issubdtype(self.dtype, np.datetime64)
        if is_datetime:
            # a query value may carry a finer datetime64 resolution than the
            # axis itself (e.g. a millisecond timestamp against a
            # second-rate axis); `inverse_step` truncates `f` down to
            # `tie_values.dtype` internally, which would silently drop the
            # sub-tick remainder, so widen everything to their common
            # (always exact, never lossy) resolution first
            common_dtype = np.result_type(value.dtype, self.dtype)
            scale = int(
                np.timedelta64(1, np.datetime_data(self.dtype)[0])
                .astype(f"timedelta64[{np.datetime_data(common_dtype)[0]}]")
                .astype("i8")
            )
            value = value.astype(common_dtype)
            num = [int(numerator.astype("i8")) * scale]
            den = [int(denominator)]
        elif is_float:
            num, den = [float(numerator)], [1]
        else:
            num, den = [int(numerator)], [int(denominator)]

        offset = np.empty(reference.shape, dtype="int64")
        for seg in np.unique(reference):
            mask = reference == seg
            length = int(self.tie_lengths[seg])
            start = self.tie_values[seg]
            local_value = value[mask]
            if length == 1:
                match method:
                    case None:
                        bad = local_value != start
                    case "ffill":
                        bad = local_value < start
                    case "bfill":
                        bad = local_value > start
                    case _:  # "nearest"
                        bad = np.zeros(local_value.shape, dtype=bool)
                if np.any(bad):
                    raise KeyError("index not found")
                offset[mask] = 0
                continue
            end = self._get_value(self.tie_indices[seg] + length - 1)
            if is_datetime:
                start = start.astype(common_dtype)
                end = end.astype(common_dtype)
            offset[mask] = inverse_step(
                local_value, [0, length - 1], [start, end], num, den, method=method
            )
        return offset[0] if scalar else offset

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
        end = self._get_value(self.tie_indices[before] + self.tie_lengths[before] - 1)
        if np.any((reference > 0) & (value <= end)):
            raise KeyError("value is in an overlap region")

        # gap
        after = np.minimum(reference + 1, len(self.tie_values) - 1)
        end = self._get_value(
            self.tie_indices[reference] + self.tie_lengths[reference] - 1
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

        offset = self._resolve_offset(value, reference, method)
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

        # compute new tie values, tie lengths and sampling interval. The new
        # anchor is read straight off the parent via `_get_value`, so it is
        # bit-identical to what the parent reports at that same global index
        # (D3): only samples *after* the anchor, re-stepped from a fresh
        # local anchor, can land up to a tick away from a full re-derivation
        # against the parent -- accepted as sub-resolution, per D3.
        tie_values = self._get_value(self.tie_indices[mask] + lo)
        tie_lengths = (hi - lo) // step
        numerator, denominator = self._sampling_ratio

        # build new coordinate
        data = {
            "tie_values": tie_values,
            "tie_lengths": tie_lengths,
            "sampling_numerator": numerator * step,
            "sampling_denominator": denominator,
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
        n1, d1 = self._sampling_ratio
        n2, d2 = other._sampling_ratio
        # cross-multiply rather than divide: two rates that only agree once
        # rounded to a single stored value (F3) must compare unequal here.
        # Python ints are arbitrary precision, so this needs no kernel and
        # cannot overflow however large the (unbounded, per D2) denominators
        # get.
        if np.issubdtype(self.dtype, np.datetime64):
            n1, n2 = int(n1.astype("i8")), int(n2.astype("i8"))
        elif not np.issubdtype(self.dtype, np.floating):
            n1, n2 = int(n1), int(n2)
        if not n1 * int(d2) == n2 * int(d1):
            raise ValueError(
                "cannot concatenate coordinate with different sampling intervals"
            )
        tie_values = np.concatenate([self.tie_values, other.tie_values])
        tie_lengths = np.concatenate([self.tie_lengths, other.tie_lengths])
        numerator, denominator = self._sampling_ratio
        return self.__class__(
            {
                "tie_values": tie_values,
                "tie_lengths": tie_lengths,
                "sampling_numerator": numerator,
                "sampling_denominator": denominator,
            },
            self.dim,
        )

    @override
    def _to_dataset(self, dataset, attrs):
        # Variation on CF-1.13 coordinate subsampling: a group names its
        # tie point coordinate variable and ends with the sampling
        # variable describing it. The sampling variable is a container
        # whose mapping puts the segment length variable in the tie point
        # index variable's slot, and whose spacing travels as attributes,
        # like the regular metadata of an interpolated coordinate.
        mapping = f"{self.name}_values: {self.name}_sampling"
        if "coordinate_sampling" in attrs:
            attrs["coordinate_sampling"] += " " + mapping
        else:
            attrs["coordinate_sampling"] = mapping
        tie_values = (
            self.tie_values.astype("M8[ns]")
            if np.issubdtype(self.tie_values.dtype, np.datetime64)
            else self.tie_values
        )
        sampling_attrs = {
            # interpolated dimension: segment length variable, subsampled dimension
            "tie_point_mapping": f"{self.dim}: {self.name}_lengths {self.name}_points",
            **encode_delta("sampling_interval", self.sampling_interval),
        }
        dataset.update(
            {
                f"{self.name}_sampling": ((), np.nan, sampling_attrs),
                f"{self.name}_lengths": (f"{self.name}_points", self.tie_lengths),
                f"{self.name}_values": (f"{self.name}_points", tie_values),
            }
        )
        return dataset, attrs

    @classmethod
    @override
    def _collect_from_dataset(cls, dataset, name):
        coords = {}
        mapping = dataset[name].attrs.pop("coordinate_sampling", None)
        if mapping is not None:
            for first, sampling in re.findall(r"(\w+): (\w+)", mapping):
                sampling_attrs = dataset[sampling].attrs
                dim, second, third = re.match(
                    r"(\w+): (\w+) (\w+)", sampling_attrs["tie_point_mapping"]
                ).groups()
                if "sampling_interval" in sampling_attrs:
                    coord, values, lengths = (
                        sampling.removesuffix("_sampling"),
                        first,
                        second,
                    )
                    interval = decode_delta("sampling_interval", sampling_attrs)
                else:
                    # the spelling that predates the CF-shaped grammar: the
                    # group named the coordinate rather than its tie point
                    # variable, the mapping listed both tie point variables,
                    # and the interval was the sampling variable's own value
                    coord, values, lengths = first, second, third
                    interval = dataset[sampling].values[()]
                    if "units" in sampling_attrs and "dtype" in sampling_attrs:
                        interval = np.timedelta64(
                            interval, UNITS_TO_CODE[sampling_attrs["units"]]
                        ).astype(sampling_attrs["dtype"])
                data = {
                    "tie_values": dataset[values].values,
                    "tie_lengths": dataset[lengths].values,
                    "sampling_interval": interval,
                }
                coords[coord] = Coordinate(data, dim)
        return coords

    def _ratio_data(self):
        """Rate entries for a fresh ``data`` dict.

        Preserves the exact ``sampling_numerator``/``sampling_denominator``
        pair rather than rebuilding through the lossy, already-divided
        ``sampling_interval`` spelling.
        """
        numerator, denominator = self._sampling_ratio
        return {"sampling_numerator": numerator, "sampling_denominator": denominator}

    def __add__(self, other):
        return self.__class__(
            {
                "tie_values": self.tie_values + other,
                "tie_lengths": self.tie_lengths,
                **self._ratio_data(),
            },
            self.dim,
        )

    def __sub__(self, other):
        return self.__class__(
            {
                "tie_values": self.tie_values - other,
                "tie_lengths": self.tie_lengths,
                **self._ratio_data(),
            },
            self.dim,
        )

    def issampled(self):
        """Return ``True`` (this is a :class:`SampledCoordinate`).

        .. deprecated:: 0.2.9
            Use ``isinstance(coord, SampledCoordinate)`` instead.
        """
        warnings.warn(
            "Coordinate.issampled() is deprecated; use "
            "isinstance(coord, SampledCoordinate) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return True

    @override
    def get_sampling_interval(self, cast=True):
        if len(self) < 2:
            return None
        numerator, denominator = self._sampling_ratio
        if cast and np.issubdtype(self.dtype, np.datetime64):
            # Divide last: converting the exact tick numerator to seconds
            # before dividing by the denominator avoids rounding to a tick
            # first, so signal processing receives the exact rate rather
            # than a representation-truncated one.
            return (numerator / np.timedelta64(1, "s")) / denominator
        return divide_sampling_ratio(numerator, denominator, self.dtype)

    @override
    def to_regular(self, sampling_interval=None, tolerance=None):
        """Regular by construction: validate any explicit spacing and return a copy.

        See :meth:`AxisCoordinate.to_regular` for the parameter contract.
        """
        if sampling_interval is not None:
            sampling_interval = parse_scalar_delta(sampling_interval, self.dtype)
            tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
            if np.abs(sampling_interval - self.sampling_interval) > tolerance:
                raise ValueError(
                    "`sampling_interval` does not match the stored sampling interval"
                )
        return self.copy()

    @override
    def simplify(self, tolerance=None, *, reduce=True, regularize=False):
        """Fuse adjacent segments whose junction drift is within *tolerance*.

        The coordinate is regular by construction (it carries a single
        ``sampling_interval``), so *regularize* is a no-op; fusing happens only
        when *reduce* is set. See :meth:`Coordinate.simplify` for the parameter
        contract.

        Integer and datetime axes fuse via ``xinterp.simplify_step`` (D5): a
        run of segments merges while the spread of its junction offsets from
        the shared rate stays within ``2 * tolerance``, then the run's tie
        value is re-anchored to the Chebyshev centre of those offsets -- so a
        surviving value may move by up to *tolerance*, twice what the former
        anchor-pinned walk fused. Float axes (``simplify_step`` needs integer
        or datetime ties) keep that anchor-pinned walk, values never moved.
        """
        if tolerance is False or not reduce:
            return self.copy()
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
        numerator, denominator = self._sampling_ratio
        if np.issubdtype(self.dtype, np.floating):
            tie_values = [self.tie_values[0]]
            tie_lengths = [self.tie_lengths[0]]
            for value, length in zip(self.tie_values[1:], self.tie_lengths[1:]):
                delta = value - (
                    tie_values[-1] + self.sampling_interval * tie_lengths[-1]
                )
                if np.abs(delta) <= tolerance:
                    tie_lengths[-1] += length
                else:
                    tie_values.append(value)
                    tie_lengths.append(length)
            tie_values = np.array(tie_values)
            tie_lengths = np.array(tie_lengths)
        else:
            is_datetime = np.issubdtype(self.dtype, np.datetime64)
            unit = np.datetime_data(self.dtype)[0] if is_datetime else None
            # tolerance may carry a coarser or finer timedelta64 unit than
            # tie_values; align it before dropping to raw integer ticks. The
            # rate itself is passed as the exact (numerator, denominator)
            # pair -- not the possibly-floored `sampling_interval` -- so a
            # fractional-tick rate does not silently launder here (D3).
            td_dtype = f"timedelta64[{unit}]" if is_datetime else None
            tie_values_ticks = (
                self.tie_values.astype("i8") if is_datetime else self.tie_values
            )
            num, den = self._step_rate()
            tol_ticks = int(
                tolerance.astype(td_dtype).astype("i8") if is_datetime else tolerance
            )
            keep, fused = simplify_step(
                tie_values_ticks, self.tie_lengths, num, den, tol_ticks
            )
            tie_lengths = np.add.reduceat(self.tie_lengths, np.flatnonzero(keep))
            tie_values = fused.astype(f"M8[{unit}]") if is_datetime else fused
        return self.__class__(
            {
                "tie_values": tie_values,
                "tie_lengths": tie_lengths,
                "sampling_numerator": numerator,
                "sampling_denominator": denominator,
            },
            self.dim,
        )

    @override
    def _split_candidates(self):
        if np.issubdtype(self.dtype, np.floating):
            # no exact rate to protect (D2): a float axis' denominator is
            # always 1, so this multiplication carries no rounding to speak of
            deltas = self.tie_values[1:] - (
                self.tie_values[:-1] + self.sampling_interval * self.tie_lengths[:-1]
            )
        else:
            num, den = self._step_rate()
            # deviation_step(...)[i] is exactly `tie_values[i + 1] -
            # forward_step(tie_indices[i + 1], ...)`, i.e. the junction jump
            # this candidate reports, computed without materialising
            # `sampling_interval * tie_lengths` (F1/F7's overflow)
            raw = deviation_step(self.tie_indices, self.tie_values, num, den)
            if np.issubdtype(self.dtype, np.datetime64):
                unit = np.datetime_data(self.dtype)[0]
                deltas = raw.astype(f"timedelta64[{unit}]")
            else:
                deltas = raw.astype(self.dtype)
        return self.tie_indices[1:], deltas
