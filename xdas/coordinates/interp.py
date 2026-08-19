"""
:class:`InterpCoordinate`: piecewise-linear coordinate.

Defined by tie points, using ``xinterp`` for forward and inverse interpolation.
Optionally carries a nominal ``sampling_interval`` (and ``tolerance``) making the
coordinate *regular* and providing a clean sample rate for signal-processing routines.
"""

import re
import warnings

import numpy as np
from typing_extensions import override
from xinterp import forward_points, infer_step, inverse_points, simplify_points

from .core import (
    AxisCoordinate,
    Coordinate,
    decode_delta,
    divide_sampling_ratio,
    encode_delta,
    is_monotonic_increasing,
    parse_data_dim,
    parse_sampling_ratio,
    parse_scalar_delta,
)


class InterpCoordinate(AxisCoordinate, ctype="interpolated"):
    """
    Piecewise-linear coordinate described by tie points (CF subsampling, 8.3).

    Following the CF conventions for compression by coordinate subsampling.
    Values between tie points are recovered by linear interpolation (via
    ``xinterp``), which also enables label-based selection through
    :meth:`~Coordinate.to_index`.  The index axis is split into *continuous
    areas* separated by *discontinuities*; a discontinuity is encoded as two
    consecutive tie points at adjacent indices (a gap of one).

    When *data* contains a ``sampling_interval`` key the coordinate also
    enforces a nominal sample spacing, making it *regular*
    (:meth:`isregular` returns ``True``) and giving signal-processing routines a
    clean sample rate.  A ``tolerance`` key may accompany it to allow bounded
    jitter around that rate.

    Parameters
    ----------
    data : dict with keys ``tie_indices`` and ``tie_values``
        ``tie_indices`` : sequence of int
            Positions of the tie points.  Must start at 0 and be strictly
            increasing.
        ``tie_values`` : sequence of float or datetime64
            Values at the tie points.  Must be strictly increasing to enable
            label-based selection.  Length must match ``tie_indices``.
        ``sampling_interval`` : scalar, optional
            Nominal sample spacing.  When provided the coordinate is
            *regular* and :meth:`get_sampling_interval` returns it directly.
            Stored internally as ``sampling_numerator`` / ``sampling_denominator``
            (an exact, gcd-reduced ratio); either spelling may be passed instead,
            provided together.
        ``tolerance`` : scalar, optional
            Allowed jitter around ``sampling_interval``.  Checked for
            consistency with the tie points at construction.  Ignored when
            ``sampling_interval`` is absent.
    dim : str, optional
        Name of the dimension this coordinate is associated with.
    dtype : dtype-like, optional
        Desired dtype for ``tie_values``.

    Notes
    -----
    Regularity is judged on the continuous areas only: a tie-point gap of one
    index (``den == 1``) is a CF discontinuity and carries no sampling-rate
    information.  A ``sampling_interval`` (exactly, its underlying
    ``numerator / denominator`` pair) is valid when, for every continuous
    segment, the accumulated drift ``|denominator * num - numerator * den|``
    stays within ``2 * tolerance * denominator`` (each tie value may jitter
    by ±``tolerance``).
    A coordinate with no continuous area (e.g. ``tie_indices=[0, 1, 2]``) has no
    inferable spacing, so an explicitly provided one is stored as-is.  Use
    :meth:`simplify` to canonicalise a coordinate and acquire a spacing from
    the continuous areas within an accuracy budget.

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
            raise TypeError(
                "`data` must be dict-like with `tie_indices` and `tie_values` "
                "(and optionally `sampling_interval` / `tolerance`)"
            )

        tie_indices = np.asarray(data["tie_indices"])
        tie_values = np.asarray(data["tie_values"], dtype=dtype)
        sampling_interval = data.get("sampling_interval", None)
        sampling_numerator = data.get("sampling_numerator", None)
        sampling_denominator = data.get("sampling_denominator", None)
        tolerance = data.get("tolerance", None)

        # check shapes
        if not tie_indices.ndim == 1:
            raise ValueError("`tie_indices` must be 1D")
        if not tie_values.ndim == 1:
            raise ValueError("`tie_values` must be 1D")
        if not len(tie_indices) == len(tie_values):
            raise ValueError("`tie_indices` and `tie_values` must have the same length")

        # check dtypes
        if tie_indices.shape != (0,):
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

        # store base data
        tie_indices = tie_indices.astype(int)
        self.data = {"tie_indices": tie_indices, "tie_values": tie_values}
        self.dim = dim

        # optional regular sampling
        self._assign_sampling_interval(
            sampling_interval, sampling_numerator, sampling_denominator, tolerance
        )

    @property
    def tie_indices(self):
        """Integer array of tie-point positions (starts at 0, strictly increasing)."""
        return self.data["tie_indices"]

    @property
    def tie_values(self):
        """Array of tie-point values (numeric or datetime64, strictly increasing)."""
        return self.data["tie_values"]

    @property
    def _sampling_ratio(self):
        """``(numerator, denominator)`` pair backing :attr:`sampling_interval`.

        Both are ``None`` together when the coordinate is not regular. This is the
        one place (besides :func:`~.core.parse_scalar_delta`) that assumes the rate
        is a single scalar rather than a per-segment array; per-segment intervals
        are expected later.
        """
        return self.data["sampling_numerator"], self.data["sampling_denominator"]

    @property
    def sampling_interval(self):
        """Nominal sample spacing, or ``None`` when the coordinate is not regular."""
        numerator, denominator = self._sampling_ratio
        return divide_sampling_ratio(numerator, denominator, self.dtype)

    @property
    def tolerance(self):
        """Allowed jitter around :attr:`sampling_interval`, or ``None``."""
        return self.data["tolerance"]

    @property
    @override
    def dtype(self):
        return self.tie_values.dtype

    @classmethod
    @override
    def from_block(cls, start, size, step, dim=None, dtype=None):
        start = np.asarray(start, dtype=dtype)
        step = parse_scalar_delta(step, start.dtype)
        if size < 2:
            # A single (or zero) sample cannot span two tie points; keep the
            # declared spacing as metadata.
            data = {
                "tie_indices": [0][:size],
                "tie_values": [start][:size],
                "sampling_interval": step,
            }
        else:
            end = start + step * (size - 1)
            data = {
                "tie_indices": [0, size - 1],
                "tie_values": [start, end],
                "sampling_interval": step,
            }
        return cls(data, dim=dim, dtype=dtype)

    @override
    def __len__(self):
        if len(self.tie_indices) > 0:
            return int(self.tie_indices[-1]) + 1
        else:
            return 0

    @staticmethod
    @override
    def _isvalid(data):
        match data:
            case {"tie_indices": _, "tie_values": _, **rest} if set(rest) <= {
                "sampling_interval",
                "sampling_numerator",
                "sampling_denominator",
                "tolerance",
            }:
                return True
            case _:
                return False

    @override
    def _is_monotonic_increasing(self):
        # every step is a tie-value difference divided by a positive index
        # difference, so the whole axis increases exactly when the tie values do
        return bool(is_monotonic_increasing(self.tie_values))

    @override
    def _get_value(self, index):
        return forward_points(index, self.tie_indices, self.tie_values)

    @override
    def _get_indexer(self, value, method=None):
        if isinstance(value, str):
            value = np.datetime64(value)
        else:
            value = np.asarray(value)
        tie_values = self.tie_values
        # A label at finer datetime resolution than the tie values (e.g. a
        # millisecond query against a second-resolution axis) would otherwise
        # be rejected outright; compute in the common, finer resolution
        # instead of truncating the query.
        if np.issubdtype(self.dtype, np.datetime64) and value.dtype != self.dtype:
            common = np.promote_types(self.dtype, value.dtype)
            tie_values = tie_values.astype(common)
            value = value.astype(common)
        try:
            indexer = inverse_points(value, self.tie_indices, tie_values, method)
        except ValueError as e:
            if str(e) == "fp must be strictly increasing":
                raise ValueError(
                    "overlaps were found in the coordinate. If this is due to some "
                    "jitter in the tie values, consider smoothing the coordinate by "
                    "including some tolerance. This can be done by "
                    "`da[dim] = da[dim].simplify(tolerance)`, or by specifying a "
                    "tolerance when opening multiple files. If the overlaps are "
                    "genuine, resolve them with `xdas.trim_overlaps(da)`, which "
                    "drops the duplicated samples, or cut them apart with "
                    "`xdas.split(da, 'overlaps')`, which keeps every copy."
                )
            else:  # pragma: no cover
                raise
        return indexer

    @override
    def _slice(self, index_slice):
        start_index, stop_index, step_index = (
            index_slice.start,
            index_slice.stop,
            index_slice.step,
        )
        if stop_index - start_index <= 0:
            data = {"tie_indices": [], "tie_values": []}
        elif (stop_index - start_index) <= step_index:
            data = {"tie_indices": [0], "tie_values": [self._get_value(start_index)]}
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
                tie_values = self._get_value(start_index + tie_indices)
                tie_indices //= step_index

            data = {"tie_indices": tie_indices, "tie_values": tie_values}
        if self.sampling_interval is not None:
            # Scaling the exact numerator by the (integer) step is itself
            # exact -- no division involved -- unlike the divided-down
            # `sampling_interval` scalar this used to multiply.
            numerator, denominator = self._sampling_ratio
            data = {
                **data,
                "sampling_numerator": numerator * step_index,
                "sampling_denominator": denominator,
                "tolerance": self.tolerance,
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
        data = {
            "tie_indices": np.append(self.tie_indices, other.tie_indices + len(self)),
            "tie_values": np.append(self.tie_values, other.tie_values),
        }
        # Strict primitive: preserve the regular contract only when both sides
        # advertise the exact same spacing; otherwise the merged coord is
        # irregular by construction. The joining tie pair has ``den == 1`` (a
        # CF discontinuity) so each side's segments validate independently,
        # and ``max(tolerance)`` bounds the union. Reconciling slightly
        # different rates is the job of user-facing routines (see
        # :func:`concat_coords`, which delegates to :meth:`simplify`).
        numerator, denominator = self._sampling_ratio
        other_numerator, other_denominator = other._sampling_ratio
        if numerator is not None and other_numerator is not None:
            # Cross-multiply rather than divide: two rates that only agree
            # once rounded to a single stored value must compare unequal
            # here. Python ints are arbitrary precision, so this needs no
            # kernel and cannot overflow however large the (unbounded, per
            # D2) denominators get.
            if np.issubdtype(self.dtype, np.datetime64):
                lhs, rhs = (
                    int(numerator.astype("i8")),
                    int(other_numerator.astype("i8")),
                )
            elif not np.issubdtype(self.dtype, np.floating):
                lhs, rhs = int(numerator), int(other_numerator)
            else:
                lhs, rhs = numerator, other_numerator
            if lhs * int(other_denominator) == rhs * int(denominator):
                data = {
                    **data,
                    "sampling_numerator": numerator,
                    "sampling_denominator": denominator,
                    "tolerance": max(self.tolerance, other.tolerance),
                }
        return self.__class__(data, self.dim)

    @override
    def _to_dataset(self, dataset, attrs):
        # CF-1.13: a group names its tie point coordinate variables and
        # ends with the interpolation variable describing them
        mapping = f"{self.name}_values: {self.name}_interpolation"
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
            # interpolated dimension: index variable, subsampled dimension
            "tie_point_mapping": f"{self.dim}: {self.name}_indices {self.name}_points",
            # xdas reconstructs in float64 (int64 nanoseconds for datetimes)
            "computational_precision": "64",
        }
        if self.sampling_interval is not None:
            # The numerator, not the divided-down `sampling_interval`: a
            # denominator of 1 (today's files, and every whole-tick rate)
            # makes the two identical, so this is byte-for-byte what earlier
            # versions wrote; a denominator > 1 is what makes the round trip
            # exact instead of floored.
            numerator, denominator = self._sampling_ratio
            interp_attrs.update(encode_delta("sampling_interval", numerator))
            if denominator != 1:
                interp_attrs["sampling_interval_denominator"] = int(denominator)
        if self.tolerance is not None:
            interp_attrs.update(encode_delta("tolerance", self.tolerance))
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
            for coord, dim, indices, values in _parse_interpolation(mapping, dataset):
                data = {
                    "tie_indices": dataset[indices].values,
                    "tie_values": dataset[values].values,
                }
                # the oldest files spelled the mapping without writing an
                # interpolation variable at all
                interp_attrs = (
                    dataset[f"{coord}_interpolation"].attrs
                    if f"{coord}_interpolation" in dataset
                    else {}
                )
                if "sampling_interval" in interp_attrs:
                    # A missing denominator defaults to 1, so files written
                    # before this round trip existed load unchanged.
                    data["sampling_numerator"] = decode_delta(
                        "sampling_interval", interp_attrs
                    )
                    data["sampling_denominator"] = interp_attrs.get(
                        "sampling_interval_denominator", 1
                    )
                    data["tolerance"] = decode_delta("tolerance", interp_attrs)
                coords[coord] = Coordinate(data, dim)
        return coords

    def __add__(self, other):
        data = {"tie_indices": self.tie_indices, "tie_values": self.tie_values + other}
        if self.sampling_interval is not None:
            # A translation does not change the rate: pass the exact pair
            # through unchanged rather than the divided-down scalar.
            numerator, denominator = self._sampling_ratio
            data = {
                **data,
                "sampling_numerator": numerator,
                "sampling_denominator": denominator,
                "tolerance": self.tolerance,
            }
        return self.__class__(data, self.dim)

    def __sub__(self, other):
        data = {"tie_indices": self.tie_indices, "tie_values": self.tie_values - other}
        if self.sampling_interval is not None:
            numerator, denominator = self._sampling_ratio
            data = {
                **data,
                "sampling_numerator": numerator,
                "sampling_denominator": denominator,
                "tolerance": self.tolerance,
            }
        return self.__class__(data, self.dim)

    def isinterp(self):
        """Return ``True`` (this is an :class:`InterpCoordinate`).

        .. deprecated:: 0.2.9
            Use ``isinstance(coord, InterpCoordinate)`` instead.
        """
        warnings.warn(
            "Coordinate.isinterp() is deprecated; use "
            "isinstance(coord, InterpCoordinate) instead.",
            FutureWarning,
            stacklevel=2,
        )
        return True

    @override
    def get_sampling_interval(self, cast=True):
        numerator, denominator = self._sampling_ratio
        if numerator is None:
            return None
        if cast and np.issubdtype(self.dtype, np.datetime64):
            # Divide last: converting the exact tick numerator to seconds
            # before dividing by the denominator avoids rounding to a tick
            # first, so signal processing receives the exact rate rather
            # than a representation-truncated one.
            return (numerator / np.timedelta64(1, "s")) / denominator
        return divide_sampling_ratio(numerator, denominator, self.dtype)

    def _assign_sampling_interval(
        self,
        sampling_interval,
        sampling_numerator,
        sampling_denominator,
        tolerance=None,
    ):
        """Parse, validate and store the sampling ratio and its tolerance.

        ``None`` clears all three; a value is kept only if consistent with the
        tie points (see :meth:`_is_valid_sampling_interval`).
        """
        numerator, denominator = parse_sampling_ratio(
            sampling_interval, sampling_numerator, sampling_denominator, self.dtype
        )
        if numerator is None:
            if tolerance is not None:
                raise ValueError(
                    "`tolerance` cannot be set without a `sampling_interval`"
                )
            self.data["sampling_numerator"] = None
            self.data["sampling_denominator"] = None
            self.data["tolerance"] = None
            return

        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)

        # Normalise the tolerance to the tie-value resolution so an in-memory
        # coordinate matches the one read back after serialisation (which always
        # encodes timedeltas at the coordinate's datetime resolution).
        # `parse_sampling_ratio` already does the same for `numerator`.
        if np.issubdtype(self.dtype, np.datetime64):
            unit = np.datetime_data(self.dtype)[0]
            tolerance = tolerance.astype(f"timedelta64[{unit}]")

        if self._is_valid_sampling_interval(numerator, denominator, tolerance):
            self.data["sampling_numerator"] = numerator
            self.data["sampling_denominator"] = denominator
            self.data["tolerance"] = tolerance
        else:
            raise ValueError(
                "`sampling_interval` and `tolerance` are not consistent with "
                "the `tie_indices` and `tie_values`"
            )

    def _is_valid_sampling_interval(self, numerator, denominator, tolerance):
        """Whether the exact rate fits every continuous area within *tolerance*.

        Judged without dividing: cross-multiplies rather than computing
        ``numerator / denominator`` first, so validity is ``|denominator *
        num_seg - numerator * den_seg| <= 2 * tolerance * denominator`` for
        every continuous segment -- the old ``dmin <= sampling_interval <=
        dmax`` form divided first and so hid its own rounding (F2). With no
        continuous area `np.all([])` is vacuously True, accepting an explicit
        spacing as metadata (e.g. a two-tie-point block).
        """
        num, den = self._continuous_segments()
        if num.size == 0:
            return True
        denominator = int(denominator)
        if not np.issubdtype(self.dtype, np.floating):
            # Exact dtypes: work in Python ints (arbitrary precision) so the
            # cross-multiply cannot overflow however large the segment span or
            # the (unbounded, per D2) denominator get.
            if np.issubdtype(self.dtype, np.datetime64):
                num = num.view("i8")
                numerator = int(numerator.view("i8"))
                tolerance = int(tolerance.view("i8"))
            else:
                numerator = int(numerator)
                tolerance = int(tolerance)
            num = num.astype(object)
            den = den.astype(object)
        drift = denominator * num - numerator * den
        bound = 2 * tolerance * denominator
        return bool(np.all(np.abs(drift) <= bound))

    def _infer_regular(self):
        """
        Estimate the nominal spacing and tightest tolerance for this coordinate.

        Private helper behind :meth:`simplify` and :meth:`to_regular`: returns
        the spacing that minimises the worst per-segment drift and the smallest
        tolerance that would still validate it, without enforcing either on the
        coordinate.

        Returns
        -------
        numerator : scalar or None
            Exact-rate numerator minimising
            ``max_i |(numerator / denominator) * den_i - num_i|`` over the
            continuous segments. ``None`` when no continuous segment is
            available (every tie-point gap is a ``den == 1`` CF
            discontinuity).
        denominator : int or None
            Exact-rate denominator paired with `numerator`. ``None`` exactly
            when `numerator` is ``None``. Always 1 for float ties (D2: a float
            axis has no tick for a denominator to refer to).
        tolerance : scalar or None
            Half the worst residual drift at ``numerator / denominator``, plus
            a few ULPs so the value stays valid under re-validation. ``None``
            when `numerator` is ``None``.

        Notes
        -----
        Spacing is judged on continuous areas only, ``den == 1`` gaps being CF
        discontinuities (see :meth:`_continuous_segments`). For such a segment
        ``i``, ``num_i`` is the change in ``tie_values`` and ``den_i`` the
        change in ``tie_indices``. The quantity ``si * den_i - num_i`` is the
        drift accumulated between the regular grid and the tie values at the
        end of that segment, and :meth:`_is_valid_sampling_interval` accepts
        ``si`` exactly when every such drift stays within ``2 * tolerance``.

        The inferred spacing minimises the worst-case drift::

            si* = argmin_si  max_i |si * den_i - num_i|

        This convex, piecewise-linear objective is a length-weighted Chebyshev
        center of the per-segment rates ``r = num / den``. Its minimum is
        reached where the two most disagreeing segments balance, so over all
        pairs the binding one maximises
        ``den_i * den_j * |r_i - r_j| / (den_i + den_j)`` and the optimum is
        the rate of that merged pair::

            si* = (num_i + num_j) / (den_i + den_j)

        That pair is found in ``O(n log n)`` via :func:`_chebyshev_center_pair`
        (for float ties) or via ``xinterp.infer_step`` (for integer and
        datetime ties, in exact rational arithmetic). The matching tolerance
        is half the worst drift, since validity compares the drift against
        ``2 * tolerance``.
        """
        num, den = self._continuous_segments()
        if num.size == 0:
            return None, None, None
        if np.issubdtype(num.dtype, np.floating):
            # xinterp.infer_step needs integer or datetime ties; float axes
            # (distance, a few thousand channels at most) are too small to
            # need a compiled kernel for the same Chebyshev-centre search.
            # A float axis has no tick, so the denominator is always 1 (D2).
            pos_idx, neg_idx = _chebyshev_center_pair(num, den.astype(float))
            numerator = (num[pos_idx] + num[neg_idx]) / (den[pos_idx] + den[neg_idx])
            drift = np.abs(numerator * den - num).max()
            # A few ULPs of slack so re-validation cannot reject the value
            # just derived from the same quantities.
            tolerance = drift / 2 + 4 * np.spacing(np.abs(num).max())
            return numerator, np.int64(1), tolerance
        # Integer or datetime: reconstruct a synthetic continuous tie sequence
        # from the (num, den) segments -- discontinuities (den == 1) carry no
        # rate information and must stay excluded -- and let infer_step find
        # the exact length-weighted Chebyshev centre in integer arithmetic.
        is_datetime = np.issubdtype(num.dtype, np.timedelta64)
        x = np.concatenate(([0], np.cumsum(den))).astype("u8")
        unit = np.datetime_data(num.dtype)[0] if is_datetime else None
        f0 = np.timedelta64(0, unit) if is_datetime else np.zeros((), dtype=num.dtype)
        tie_values = np.concatenate(([f0], f0 + np.cumsum(num)))
        n, d, worst = infer_step(x, tie_values)
        half = -(-int(worst) // 2)  # ceil(worst / 2), exact integer arithmetic
        if is_datetime:
            numerator = np.timedelta64(int(n), unit)
            tolerance = np.timedelta64(half, unit).astype("timedelta64[ns]")
        else:
            numerator = num.dtype.type(n)
            tolerance = half
        return numerator, np.int64(d), tolerance

    @override
    def to_regular(self, sampling_interval=None, tolerance=None):
        """Enforce a nominal sampling interval, inferring it when omitted.

        The inferred spacing is the length-weighted Chebyshev center of the
        per-segment rates (see :meth:`_infer_regular`). Raises when no spacing
        can be inferred (no continuous area, i.e. every tie-point gap is a
        ``den == 1`` CF discontinuity) or when the spacing does not fit the tie
        points within *tolerance*. See :meth:`AxisCoordinate.to_regular` for
        the parameter contract.
        """
        if tolerance is None:
            tolerance = self.tolerance
        if sampling_interval is not None:
            # Explicit spacing: the caller's own scalar, stored as given.
            data = {
                "tie_indices": self.tie_indices,
                "tie_values": self.tie_values,
                "sampling_interval": sampling_interval,
                "tolerance": tolerance,
            }
            return self.__class__(data, self.dim)
        # No explicit spacing: keep the coordinate's own exact ratio if it is
        # already regular, else infer one -- both as an exact
        # (numerator, denominator) pair, never the divided-down scalar, so a
        # fractional-tick rate survives (see :meth:`_infer_regular`).
        numerator, denominator = self._sampling_ratio
        if numerator is None:
            numerator, denominator, _ = self._infer_regular()
        if numerator is None:
            raise ValueError(
                "cannot infer a sampling interval: the coordinate has no "
                "continuous area; pass `sampling_interval` explicitly"
            )
        data = {
            "tie_indices": self.tie_indices,
            "tie_values": self.tie_values,
            "sampling_numerator": numerator,
            "sampling_denominator": denominator,
            "tolerance": tolerance,
        }
        return self.__class__(data, self.dim)

    @override
    def simplify(self, tolerance=None, *, reduce=True, regularize=False):
        """Canonicalise within *tolerance*: drop tie points, then promote to regular.

        The *reduce* stage runs a one-pass greedy sleeve (see :func:`_sleeve`)
        to drop tie points whose removal shifts the curve by no more than
        *tolerance*. The CF 8.3 structure is preserved as an emergent property
        of that bound: real discontinuities are kept (any spanning line crosses
        them by far more than *tolerance*), soft ones are fused into a single
        ramp, and synchronisation tie points survive because removing them
        would, by definition, drift more than *tolerance*. Surviving values
        are never moved.

        The *regularize* stage promotes the result to *regular* when the
        surviving continuous segments admit a single ``sampling_interval`` within
        *tolerance* (the internal Chebyshev fit's worst residual stays inside the
        budget). The promotion is per-continuous-segment and sign-agnostic, so
        two same-rate segments joined by a CF overlap are still described by one
        spacing. An already-regular coordinate keeps its spacing only while the
        surviving tie points still honour it within the coordinate's own
        declared tolerance: tolerance means instrumental jitter and nothing
        else, is set once at construction, and is never widened afterwards
        (D3) -- a reduce pass whose fused jump exceeds the declared budget
        drops the coordinate's regularity rather than stretching the number
        that describes it.

        See :meth:`Coordinate.simplify` for the parameter contract.
        """
        if tolerance is False:
            return self.copy()
        if tolerance is None:
            # Default the budget to the coordinate's own declared jitter.
            tolerance = self.tolerance
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
        if reduce:
            tie_indices, tie_values = _sleeve(
                self.tie_indices, self.tie_values, tolerance
            )
        else:
            tie_indices, tie_values = self.tie_indices, self.tie_values
        data = {"tie_indices": tie_indices, "tie_values": tie_values}
        if self.sampling_interval is not None:
            numerator, denominator = self._sampling_ratio
            reduced = self.__class__(data, self.dim)
            if not reduce or reduced._is_valid_sampling_interval(
                numerator, denominator, self.tolerance
            ):
                data = {
                    **data,
                    "sampling_numerator": numerator,
                    "sampling_denominator": denominator,
                    "tolerance": self.tolerance,
                }
            return self.__class__(data, self.dim)
        # Otherwise try to promote: infer the best spacing on the surviving
        # continuous segments and keep it only if it validates within the budget.
        if regularize:
            reduced = self.__class__(data, self.dim)
            numerator, denominator, _ = reduced._infer_regular()
            if numerator is not None and reduced._is_valid_sampling_interval(
                numerator, denominator, tolerance
            ):
                data = {
                    **data,
                    "sampling_numerator": numerator,
                    "sampling_denominator": denominator,
                    "tolerance": tolerance,
                }
        return self.__class__(data, self.dim)

    @override
    def _split_candidates(self):
        """Discontinuity split points, each paired with its step's deviation from the neighbouring interval."""
        tie_intervals = np.diff(self.tie_values) / np.diff(self.tie_indices)
        (positions,) = np.nonzero(np.diff(self.tie_indices) == 1)
        references = np.where(
            positions > 0,
            positions - 1,
            np.minimum(positions + 1, len(tie_intervals) - 1),
        )
        deltas = tie_intervals[positions] - tie_intervals[references]
        return self.tie_indices[positions + 1], deltas

    def _continuous_segments(self):
        """Per-segment value/index spans ``(num, den)`` for the continuous areas.

        A ``den == 1`` gap is a CF discontinuity (section 8.3), not a segment, so
        it is excluded and carries no sampling-rate information.
        """
        num = np.diff(self.tie_values)
        den = np.diff(self.tie_indices)
        mask = den != 1
        return num[mask], den[mask]


def _parse_interpolation(mapping, dataset):
    """
    Yield ``(name, dim, indices, values)`` per group of *mapping*.

    Reads a ``coordinate_interpolation`` attribute in either spelling.
    CF-1.13 words each group ``tie_point_coordinate_variable: [...]
    interpolation_variable``, the coordinate name and the interpolated
    dimension then coming from the interpolation variable and its
    ``tie_point_mapping``; xdas wrote ``dimension: index_variable
    value_variable`` before the format break. Only the CF spelling ends
    a group with a variable carrying ``interpolation_name``, which is
    what tells the two apart. The tie point coordinate variable name is
    taken from the group as written, whatever it is.

    Parameters
    ----------
    mapping : str
        The attribute value to parse.
    dataset : xarray.Dataset
        The dataset the named variables live in.

    Yields
    ------
    tuple of str
        Coordinate name, interpolated dimension, tie point index
        variable and tie point coordinate variable.
    """
    groups, tie_points = [], []
    for word in mapping.split():
        if word.endswith(":"):
            tie_points.append(word[:-1])
        else:
            groups.append((tie_points, word))
            tie_points = []
    if all(
        len(tie_points) == 1
        and word in dataset
        and "interpolation_name" in dataset[word].attrs
        for tie_points, word in groups
    ):
        for (values,), interpolation in groups:
            name = interpolation.removesuffix("_interpolation")
            dim, indices, _ = re.match(
                r"(\w+): (\w+) (\w+)",
                dataset[interpolation].attrs["tie_point_mapping"],
            ).groups()
            yield name, dim, indices, values
    else:
        for dim, indices, values in re.findall(r"(\w+): (\w+) (\w+)", mapping):
            yield dim, dim, indices, values


def _epsilon_ratio(dtype, epsilon):
    """
    Express *epsilon* as an exact rational number of storage ticks.

    An :class:`InterpCoordinate` reconstructs values as ``round(exact chord)``
    at the storage resolution (:func:`xinterp.forward_points`), so a tie point can sit
    up to half a tick off the exact line and still be the only representable
    value there.  Collinearity must therefore be judged with half a tick of
    slack, which is the ``+ 1`` and ``* 2`` below: the returned pair is
    ``epsilon + 1/2`` in ticks for exact dtypes, and plain ``epsilon`` for
    floats, which carry no tick.

    Returns
    -------
    numerator, denominator
        ``epsilon`` in ticks as ``numerator / denominator``, denominator
        positive.
    """
    if np.issubdtype(dtype, np.datetime64):
        unit, count = np.datetime_data(dtype)
        one = np.timedelta64(count, unit)
        # bring both to the finer unit so a sub-tick epsilon is not truncated
        common = np.promote_types(one.dtype, epsilon.dtype)
        ticks = int(one.astype(common).view("i8"))
        eps = int(epsilon.astype(common).view("i8"))
        return 2 * eps + ticks, 2 * ticks
    if np.issubdtype(dtype, np.integer):
        return 2 * int(epsilon) + 1, 2
    return epsilon, 1.0


def _sleeve(x, y, epsilon):
    """
    Reduce the piecewise-linear curve *(x, y)* with a one-pass greedy sleeve.

    Points are dropped when the segment connecting the surviving neighbours
    passes within *epsilon* of them. The walk is left to right (the direction
    acquisition produces tie points): from the current anchor it maintains the
    intersection of every dropped point's ±*epsilon* slope cone — the sleeve —
    and emits a knot exactly when a candidate leaves it. Knots are original
    points, so surviving values are never moved, and any point whose removal
    would drift the curve by more than *epsilon* (a discontinuity edge, a
    synchronisation tie) empties the sleeve and survives. One pass, O(n)
    whatever the number of surviving points — where Douglas-Peucker
    degenerates quadratically once discontinuities or jitter make many
    points survive.

    Integer and datetime values are compared with exact cross-multiplied
    integer arithmetic, the budget widened by the half tick the storage
    resolution costs (see :func:`_epsilon_ratio`), so a zero *epsilon* drops
    exactly the points the coordinate cannot tell apart from collinear. Float
    values use float arithmetic.

    The walk keeps a point as soon as the chord to it leaves the running cone,
    which is conservative: a curve that some single chord would fit within
    *epsilon* may still keep interior points. It never moves or drops a point
    the budget does not allow, only occasionally fewer than it could. That is
    why the fast path below is not redundant with the walk: it is a global
    test the greedy walk cannot always reach on its own.

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
    if len(x) < 3:
        return x, y
    # Fast path: one chord spans the whole curve (the fully continuous case,
    # resolved vectorized). `forward_points` reconstructs exactly what the
    # reduced coordinate would return, so this measures the real shift and
    # needs no allowance of its own -- and being a global test it also
    # catches curves the incremental walk below is too conservative to
    # collapse.
    deviation = np.abs(y - forward_points(x, x[[0, -1]], y[[0, -1]]))
    if deviation.max() <= epsilon:
        return x[[0, -1]], y[[0, -1]]
    en, ed = _epsilon_ratio(y.dtype, epsilon)
    keep = simplify_points(x, y, en, ed)
    return x[keep], y[keep]


def _chebyshev_center_pair(num, den):
    """
    Segment indices binding the length-weighted Chebyshev center, in O(n log n).

    Returns the pair maximising ``den_i den_j |r_i - r_j| / (den_i + den_j)`` with
    ``r = num / den``, equivalently the lowest point of the upper envelope of the
    ``2 n`` lines ``±(den_i si - num_i)``. That vertex is the meeting of the
    binding negative- and positive-slope lines, found with the convex-hull trick
    instead of the O(n^2) pairwise scan.

    Parameters
    ----------
    num : numpy.ndarray
        Per-segment numerators as floats (seconds for datetime axes).
    den : numpy.ndarray
        Per-segment denominators as floats, all strictly positive.

    Returns
    -------
    pos_idx, neg_idx : int
        Segment indices of the binding positive- and negative-slope lines. A
        single segment trivially selects itself (``pos_idx == neg_idx``).
    """
    seg = np.arange(len(den))
    # Positive-slope lines (den si - num) and negative-slope lines (num - den si).
    slopes = np.concatenate([den, -den])
    intercepts = np.concatenate([-num, num])
    idx = np.concatenate([seg, seg])
    # Process lines by ascending slope, equal slopes ordered by descending
    # intercept so the dominant one comes first.
    order = np.lexsort((-intercepts, slopes))
    return _upper_envelope_min_pair(slopes, intercepts, idx, order)


def _upper_envelope_min_pair(slopes, intercepts, idx, order):
    """Binding (positive, negative) line indices at the upper-envelope minimum.

    Plain Python, not compiled: only reached for float ``InterpCoordinate``
    axes (``xinterp.infer_step`` handles integer and datetime ties in exact
    arithmetic instead), which are distance axes of a few thousand channels
    at most -- far too small to need a JIT kernel.
    """
    n = order.size
    hull_s = np.empty(n, dtype=slopes.dtype)
    hull_b = np.empty(n, dtype=intercepts.dtype)
    hull_i = np.empty(n, dtype=idx.dtype)
    m = 0  # current hull size
    for t in range(n):
        k = order[t]
        s, b, seg = slopes[k], intercepts[k], idx[k]
        # Equal slopes: the dominant (larger intercept) one came first; skip rest.
        if m > 0 and hull_s[m - 1] == s:
            continue
        # Drop any line the convex-hull trick proves can never be the maximum.
        while m >= 2:
            s1, b1 = hull_s[m - 2], hull_b[m - 2]
            s2, b2 = hull_s[m - 1], hull_b[m - 1]
            if (b - b1) / (s1 - s) <= (b2 - b1) / (s1 - s2):
                m -= 1
            else:
                break
        hull_s[m], hull_b[m], hull_i[m] = s, b, seg
        m += 1
    # The envelope is convex with slope increasing along x; its minimum sits at
    # the negative-to-positive slope transition, between the binding pair. The
    # scan is safely bounded because `_chebyshev_center_pair` always feeds in
    # both `+den_i` and `-den_i` lines, so at least one slope of each sign
    # reaches the hull.
    t = 0
    while hull_s[t] < 0.0:
        t += 1
    return hull_i[t], hull_i[t - 1]
