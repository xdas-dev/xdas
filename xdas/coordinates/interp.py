"""
:class:`InterpCoordinate`: piecewise-linear coordinate.

Defined by tie points, using ``xinterp`` for forward and inverse interpolation.
Optionally carries a nominal ``sampling_interval`` (and ``tolerance``) making the
coordinate *regular* and providing a clean sample rate for signal-processing routines.
"""

import re

import numpy as np
from numba import njit
from typing_extensions import override
from xinterp import forward, inverse

from .core import (
    AxisCoordinate,
    Coordinate,
    decode_delta,
    encode_delta,
    is_monotonic_increasing,
    parse_data_dim,
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
    information.  A ``sampling_interval`` is valid when, for every continuous
    segment, the accumulated drift ``|sampling_interval * den - num|`` stays
    within ``2 * tolerance`` (each tie value may jitter by ±``tolerance``).
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
        self._assign_sampling_interval(sampling_interval, tolerance)

    @property
    def tie_indices(self):
        """Integer array of tie-point positions (starts at 0, strictly increasing)."""
        return self.data["tie_indices"]

    @property
    def tie_values(self):
        """Array of tie-point values (numeric or datetime64, strictly increasing)."""
        return self.data["tie_values"]

    @property
    def sampling_interval(self):
        """Nominal sample spacing, or ``None`` when the coordinate is not regular."""
        return self.data["sampling_interval"]

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
                "tolerance",
            }:
                return True
            case _:
                return False

    @override
    def _is_monotonic_increasing(self):
        return not self.get_split_indices("overlaps", tolerance=False).size

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
            data = {
                **data,
                "sampling_interval": self.sampling_interval * step_index,
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
        if (
            self.sampling_interval is not None
            and self.sampling_interval == other.sampling_interval
        ):
            data = {
                **data,
                "sampling_interval": self.sampling_interval,
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
            interp_attrs.update(
                encode_delta("sampling_interval", self.sampling_interval)
            )
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
                    data["sampling_interval"] = decode_delta(
                        "sampling_interval", interp_attrs
                    )
                    data["tolerance"] = decode_delta("tolerance", interp_attrs)
                coords[coord] = Coordinate(data, dim)
        return coords

    def __add__(self, other):
        data = {"tie_indices": self.tie_indices, "tie_values": self.tie_values + other}
        if self.sampling_interval is not None:
            data = {
                **data,
                "sampling_interval": self.sampling_interval,
                "tolerance": self.tolerance,
            }
        return self.__class__(data, self.dim)

    def __sub__(self, other):
        data = {"tie_indices": self.tie_indices, "tie_values": self.tie_values - other}
        if self.sampling_interval is not None:
            data = {
                **data,
                "sampling_interval": self.sampling_interval,
                "tolerance": self.tolerance,
            }
        return self.__class__(data, self.dim)

    @override
    def get_sampling_interval(self, cast=True):
        delta = self.sampling_interval
        if delta is None:
            return None
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

    def _assign_sampling_interval(self, sampling_interval, tolerance=None):
        """Parse, validate and store the sampling interval and its tolerance.

        ``None`` clears both; a value is kept only if consistent with the tie
        points (see :meth:`_is_valid_sampling_interval`).
        """
        if sampling_interval is None:
            if tolerance is not None:
                raise ValueError(
                    "`tolerance` cannot be set without a `sampling_interval`"
                )
            self.data["sampling_interval"] = None
            self.data["tolerance"] = None
            return

        sampling_interval = parse_scalar_delta(sampling_interval, self.dtype)
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)

        # Normalise datetime deltas to the tie-value resolution so an in-memory
        # coordinate matches the one read back after serialisation (which always
        # encodes timedeltas at the coordinate's datetime resolution).
        if np.issubdtype(self.dtype, np.datetime64):
            unit = np.datetime_data(self.dtype)[0]
            sampling_interval = sampling_interval.astype(f"timedelta64[{unit}]")
            tolerance = tolerance.astype(f"timedelta64[{unit}]")

        if self._is_valid_sampling_interval(sampling_interval, tolerance):
            self.data["sampling_interval"] = sampling_interval
            self.data["tolerance"] = tolerance
        else:
            raise ValueError(
                "`sampling_interval` and `tolerance` are not consistent with "
                "the `tie_indices` and `tie_values`"
            )

    def _is_valid_sampling_interval(self, sampling_interval, tolerance):
        """Whether *sampling_interval* fits every continuous area within *tolerance*."""
        num, den = self._continuous_segments()
        # Bound the per-segment accumulated drift: each tie value may jitter by
        # ±tolerance, so a segment span may be off by up to 2 * tolerance. With no
        # continuous area `np.all([])` is vacuously True, accepting an explicit
        # spacing as metadata (e.g. a two-tie-point block). Datetime bounds use
        # integer division and are only accurate to the dtype resolution.
        dmin = (num - 2 * tolerance) / den
        dmax = (num + 2 * tolerance) / den
        valid = np.all((dmin <= sampling_interval) & (sampling_interval <= dmax))
        return bool(valid)

    def _minimal_tolerance(self, sampling_interval):
        """Smallest tolerance for which *sampling_interval* validates, or ``None``.

        Private helper behind :meth:`simplify`. Validity bounds the accumulated
        drift of every continuous segment by ``2 * tolerance``, so the tightest
        admissible value is half the worst drift, rounded up at the dtype
        resolution. Returns ``None`` when even that value fails to validate,
        which callers treat as "this spacing cannot be declared".
        """
        num, den = self._continuous_segments()
        if num.size == 0:
            # No continuous area: any tolerance is vacuously valid.
            return parse_scalar_delta(None, self.dtype, default_zero=True)
        drift = np.abs(num - sampling_interval * den).max()
        if np.issubdtype(self.dtype, np.datetime64):
            # Integer resolution: round up so the halved drift is not truncated.
            unit = np.datetime_data(self.dtype)[0]
            counts = int(drift / np.timedelta64(1, unit))
            tolerance = np.timedelta64(-(-counts // 2), unit)
        else:
            # A few ULPs of slack so re-validation cannot reject the value we
            # just derived from the same quantities (as in :meth:`_infer_regular`).
            tolerance = drift / 2 + 4 * np.spacing(np.abs(num).max())
        if not self._is_valid_sampling_interval(sampling_interval, tolerance):
            return None  # pragma: no cover
        return tolerance

    def _infer_regular(self):
        """
        Estimate the nominal spacing and tightest tolerance for this coordinate.

        Private helper behind :meth:`simplify` and :meth:`to_regular`: returns
        the spacing that minimises the worst per-segment drift and the smallest
        tolerance that would still validate it, without enforcing either on the
        coordinate.

        Returns
        -------
        sampling_interval : scalar or None
            Spacing minimising ``max_i |sampling_interval * den_i - num_i|``
            over the continuous segments. ``None`` when no continuous segment
            is available (every tie-point gap is a ``den == 1`` CF
            discontinuity).
        tolerance : scalar or None
            Half the worst residual drift at ``sampling_interval``, plus a few
            ULPs so the value stays valid under re-validation. ``None`` when
            ``sampling_interval`` is ``None``.

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
        rather than scanning all pairs. The matching tolerance is half the
        worst drift, since validity compares the drift against
        ``2 * tolerance``.
        """
        num, den = self._continuous_segments()
        if num.size == 0:
            return None, None
        # Float seconds for datetime axes pick the binding pair without
        # integer/timedelta overflow; the final values stay in the native dtype.
        is_datetime = np.issubdtype(num.dtype, np.timedelta64)
        num_seconds = num / np.timedelta64(1, "s") if is_datetime else num.astype(float)
        pos_idx, neg_idx = _chebyshev_center_pair(num_seconds, den.astype(float))
        sampling_interval = (num[pos_idx] + num[neg_idx]) / (
            den[pos_idx] + den[neg_idx]
        )
        si_seconds = (
            sampling_interval / np.timedelta64(1, "s")
            if is_datetime
            else float(sampling_interval)
        )
        drift = np.abs(si_seconds * den - num_seconds).max()
        # A few ULPs of slack so re-validation cannot reject the returned pair.
        tolerance = drift / 2 + 4 * np.spacing(np.abs(num_seconds).max())
        if is_datetime:
            tolerance = np.timedelta64(int(np.ceil(tolerance * 1e9)), "ns")
        return sampling_interval, tolerance

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
        # Default each unspecified argument to the stored regular config; an
        # explicit value still overrides it.
        if sampling_interval is None:
            sampling_interval = self.sampling_interval
        if tolerance is None:
            tolerance = self.tolerance
        if sampling_interval is None:
            sampling_interval, _ = self._infer_regular()
        if sampling_interval is None:
            raise ValueError(
                "cannot infer a sampling interval: the coordinate has no "
                "continuous area; pass `sampling_interval` explicitly"
            )
        data = {
            "tie_indices": self.tie_indices,
            "tie_values": self.tie_values,
            "sampling_interval": sampling_interval,
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
        spacing. An already-regular coordinate keeps its spacing regardless of
        *regularize* and just widens its stored tolerance to absorb any jump
        fused by the reduce stage.

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
            # Already regular: keep the spacing. A reduce pass may fuse a soft
            # discontinuity into a ramp whose absorbed jump exceeds the declared
            # jitter; keep the declared tolerance when it still validates and
            # only widen by the spent budget when it does not, so a lossless or
            # seam-only reduction round-trips the exact regular metadata.
            reduced = self.__class__(data, self.dim)
            if reduce and not reduced._is_valid_sampling_interval(
                self.sampling_interval, self.tolerance
            ):
                # Reduction fused a discontinuity, so the surviving tie points
                # now drift further from the nominal grid than the declared
                # jitter allows. Widen to the least value that describes them:
                # the budget bounds how far *values* may move, not how much
                # drift fusing a seam exposes, so it is no bound at all here.
                new_tolerance = reduced._minimal_tolerance(self.sampling_interval)
                if new_tolerance is None:  # pragma: no cover
                    # Numerical safety net: the spacing cannot be declared for
                    # the reduced tie points, so stay irregular rather than let
                    # the constructor raise.
                    return reduced
            else:
                new_tolerance = self.tolerance
            data = {
                **data,
                "sampling_interval": self.sampling_interval,
                "tolerance": new_tolerance,
            }
            return self.__class__(data, self.dim)
        # Otherwise try to promote: infer the best spacing on the surviving
        # continuous segments and keep it only if it validates within the budget.
        if regularize:
            reduced = self.__class__(data, self.dim)
            sampling_interval, _ = reduced._infer_regular()
            if sampling_interval is not None and reduced._is_valid_sampling_interval(
                sampling_interval, tolerance
            ):
                data = {
                    **data,
                    "sampling_interval": sampling_interval,
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

    Integer and datetime values are compared with exact (arbitrary-precision)
    cross-multiplied integer arithmetic, so a zero *epsilon* drops exactly the
    collinear points; float values use float arithmetic.

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
    if np.issubdtype(y.dtype, np.datetime64):
        # exact integer arithmetic, with epsilon brought to the finer of the
        # two time units so sub-unit tolerances are not truncated
        unit, count = np.datetime_data(y.dtype)
        one = np.timedelta64(count, unit)
        common = np.promote_types(one.dtype, epsilon.dtype)
        scale = int(one.astype(common).view("i8"))
        values = (y.view("i8") * scale).tolist()
        eps = int(epsilon.astype(common).view("i8"))
        margin = one
    else:
        values = y.tolist()
        eps = epsilon
        margin = 1 if np.issubdtype(y.dtype, np.integer) else 0.0
    # fast path: one chord spans the whole curve (the fully continuous case,
    # resolved vectorized). `forward` rounds to the value unit, so the chord
    # is only trusted beyond a one-unit margin; near the boundary the exact
    # loop below decides
    deviation = np.abs(y - forward(x, x[[0, -1]], y[[0, -1]]))
    if deviation.max() + margin <= epsilon:
        return x[[0, -1]], y[[0, -1]]
    positions = x.tolist()
    keep = np.zeros(len(x), dtype=bool)
    keep[0] = keep[-1] = True
    ax, ay = positions[0], values[0]
    # the sleeve: feasible slope interval from the anchor, as exact
    # (numerator, positive denominator) rationals; None while unbounded
    lo = hi = None
    for i in range(1, len(positions)):
        dx = positions[i] - ax
        dy = values[i] - ay
        if (lo is None or lo[0] * dx <= dy * lo[1]) and (
            hi is None or dy * hi[1] <= hi[0] * dx
        ):
            # the chord anchor -> i passes within epsilon of every dropped
            # point; tighten the sleeve with this point's own cone
            if lo is None or (dy - eps) * lo[1] > lo[0] * dx:
                lo = (dy - eps, dx)
            if hi is None or (dy + eps) * hi[1] < hi[0] * dx:
                hi = (dy + eps, dx)
        else:
            keep[i - 1] = True
            ax, ay = positions[i - 1], values[i - 1]
            dx = positions[i] - ax
            dy = values[i] - ay
            lo = (dy - eps, dx)
            hi = (dy + eps, dx)
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


@njit(cache=True)
def _upper_envelope_min_pair(slopes, intercepts, idx, order):  # pragma: no cover
    """Binding (positive, negative) line indices at the upper-envelope minimum."""
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
