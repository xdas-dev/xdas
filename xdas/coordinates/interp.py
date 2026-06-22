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
    :meth:`to_regular` to infer or enforce a spacing from the continuous areas.

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

        # store base data
        tie_indices = tie_indices.astype(int)
        self.data = dict(tie_indices=tie_indices, tie_values=tie_values)
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
        end = start + step * (size - 1)
        return cls(
            {
                "tie_indices": [0, size - 1],
                "tie_values": [start, end],
                "sampling_interval": step,
            },
            dim=dim,
            dtype=dtype,
        )

    @override
    def __len__(self):
        if len(self.tie_indices) > 0:
            return int(self.tie_indices[-1] - self.tie_indices[0] + 1)
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
                tie_values = [self._get_value(start_index + idx) for idx in tie_indices]
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
        if self.sampling_interval != other.sampling_interval:
            raise ValueError(
                "cannot append coordinate with different sampling interval"
            )
        if self.sampling_interval is not None:
            tolerance = (
                max(self.tolerance, other.tolerance)
                if self.tolerance is not None and other.tolerance is not None
                else None
            )
            data = {
                **data,
                "sampling_interval": self.sampling_interval,
                "tolerance": tolerance,
            }
        return self.__class__(data, self.dim)

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
            for dim, indices, values in re.findall(r"(\w+): (\w+) (\w+)", mapping):
                data = {
                    "tie_indices": dataset[indices].values,
                    "tie_values": dataset[values].values,
                }
                interp_attrs = dataset[f"{dim}_interpolation"].attrs
                if "sampling_interval" in interp_attrs:
                    data["sampling_interval"] = decode_delta(
                        "sampling_interval", interp_attrs
                    )
                    data["tolerance"] = decode_delta("tolerance", interp_attrs)
                coords[dim] = Coordinate(data, dim)
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

    def to_regular(self, sampling_interval=None, tolerance=None):
        """
        Return a copy of this coordinate with an enforced nominal sampling interval.

        Parameters
        ----------
        sampling_interval : scalar, optional
            Nominal sample spacing to enforce. When omitted it is inferred as the
            spacing that best satisfies :meth:`_is_valid_sampling_interval`, i.e.
            the one minimising the worst per-segment drift (see Notes).
        tolerance : scalar or ``"auto"``, optional
            Tolerated jitter around *sampling_interval*. Defaults to a
            dtype-dependent epsilon, so a genuinely irregular axis raises
            :exc:`ValueError`. Pass ``"auto"`` to set the smallest tolerance that
            keeps *sampling_interval* valid (see Notes).

        Returns
        -------
        InterpCoordinate
            A new coordinate with :attr:`sampling_interval` set, or with it left
            unset when no spacing can be inferred (no continuous area, i.e. every
            tie-point gap is a ``den == 1`` CF discontinuity).

        Notes
        -----
        Spacing is judged on continuous areas only, ``den == 1`` gaps being CF
        discontinuities (see :meth:`_continuous_segments`). For such a segment
        ``i``, ``num_i`` is the change in ``tie_values`` and ``den_i`` the change
        in ``tie_indices``. The
        quantity ``si * den_i - num_i`` is the drift accumulated between the
        regular grid and the tie values at the end of that segment, and
        :meth:`_is_valid_sampling_interval` accepts ``si`` exactly when every
        such drift stays within ``2 * tolerance``.

        The inferred spacing minimises the worst-case drift::

            si* = argmin_si  max_i |si * den_i - num_i|

        This convex, piecewise-linear objective is a length-weighted Chebyshev
        center of the per-segment rates ``r = num / den``. Its minimum is reached
        where the two most disagreeing segments balance, so over all pairs the
        binding one maximises ``den_i * den_j * |r_i - r_j| / (den_i + den_j)``
        and the optimum is the rate of that merged pair::

            si* = (num_i + num_j) / (den_i + den_j)

        That pair is found in ``O(n log n)`` via :func:`_chebyshev_center_pair`
        rather than scanning all pairs.

        The matching auto tolerance is half that worst drift, since validity
        compares the drift against ``2 * tolerance``::

            tolerance = max_i |si* * den_i - num_i| / 2
        """
        # An already-regular coordinate keeps its spacing unless one is forced.
        if self.sampling_interval is not None and sampling_interval is None:
            return self.copy()

        # Spacing is judged on the continuous areas only; `den == 1` gaps are CF
        # discontinuities (see `_continuous_segments`).
        num, den = self._continuous_segments()

        if sampling_interval is None and num.size > 0:
            # Per-segment numerators as plain floats (seconds for datetime axes),
            # used only to pick the binding pair without integer/timedelta overflow.
            num_seconds = (
                num / np.timedelta64(1, "s")
                if np.issubdtype(num.dtype, np.timedelta64)
                else num.astype(float)
            )
            i, j = _chebyshev_center_pair(num_seconds, den.astype(float))
            # Balance point of the binding pair, kept in the native dtype.
            sampling_interval = (num[i] + num[j]) / (den[i] + den[j])

        if isinstance(tolerance, str):
            if tolerance != "auto":
                raise ValueError(f"unknown tolerance {tolerance!r}, expected 'auto'")
            if sampling_interval is None or num.size == 0:
                tolerance = None
            else:
                # Half the worst drift (validity needs `2 * tolerance >= drift`),
                # in float, plus a few ULPs so the re-validation cannot reject it.
                is_datetime = np.issubdtype(num.dtype, np.timedelta64)
                num_seconds = (
                    num / np.timedelta64(1, "s") if is_datetime else num.astype(float)
                )
                si_seconds = (
                    sampling_interval / np.timedelta64(1, "s")
                    if is_datetime
                    else float(sampling_interval)
                )
                drift = np.abs(si_seconds * den - num_seconds).max()
                tolerance = drift / 2 + 4 * np.spacing(np.abs(num_seconds).max())
                if is_datetime:
                    tolerance = np.timedelta64(int(np.ceil(tolerance * 1e9)), "ns")

        data = {
            "tie_indices": self.tie_indices,
            "tie_values": self.tie_values,
            "sampling_interval": sampling_interval,
            "tolerance": tolerance,
        }
        return self.__class__(data, self.dim)

    @override
    def simplify(self, tolerance=None):
        """Drop redundant tie points within *tolerance* via Douglas-Peucker.

        The CF 8.3 structure is preserved as an emergent property of that bound:
        real discontinuities are kept (any spanning line crosses them by far more
        than *tolerance*), soft ones are fused into a single ramp, and
        synchronisation tie points survive because removing them would, by
        definition, drift more than *tolerance*. Surviving values are never
        moved.

        See :meth:`Coordinate.simplify` for the parameter contract.
        """
        if tolerance is False:
            return self.copy()
        tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
        tie_indices, tie_values = _douglas_peucker(
            self.tie_indices, self.tie_values, tolerance
        )
        data = {"tie_indices": tie_indices, "tie_values": tie_values}
        if self.sampling_interval is not None:
            data = {
                **data,
                "sampling_interval": self.sampling_interval,
                "tolerance": self.tolerance,
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
    i, j : int
        Segment indices of the binding positive- and negative-slope lines. A
        single segment trivially selects itself (``i == j``).
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
    # the negative-to-positive slope transition, between the binding pair.
    t = 0
    while hull_s[t] < 0.0:
        t += 1
    return hull_i[t], hull_i[t - 1]
