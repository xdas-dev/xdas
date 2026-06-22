"""
:class:`InterpCoordinate`: piecewise-linear coordinate.

Defined by tie points, using ``xinterp`` for forward and inverse interpolation.
Optionally carries a nominal ``sampling_interval`` (and ``tolerance``) making the
coordinate *regular* and providing a clean sample rate for signal-processing routines.
"""

import re

import numpy as np
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
    Piecewise-linear coordinate described by tie points (CF convention).

    Values between tie points are recovered by linear interpolation.
    Discontinuities are represented by two consecutive tie points at adjacent
    indices.  Supports label-based selection via :meth:`~Coordinate.to_index`.

    When *data* contains a ``sampling_interval`` key the coordinate also
    enforces a nominal sample spacing, making it *regular*
    (:meth:`isregular` returns ``True``).  A ``tolerance`` key may
    accompany it to allow bounded jitter around that rate.

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

    def _assign_sampling_interval(self, sampling_interval, tolerance=None):
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

    def _is_valid_sampling_interval(self, sampling_interval, tolerance=None):
        num = np.diff(self.tie_values)
        den = np.diff(self.tie_indices)
        mask = den != 1
        num = num[mask]
        den = den[mask]
        dmin = (num - 2 * tolerance) / den
        dmax = (num + 2 * tolerance) / den
        valid = np.all((dmin <= sampling_interval) & (sampling_interval <= dmax))
        return bool(valid)

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
            unset when no spacing can be inferred (no segment spans more than one
            sample).

        Notes
        -----
        For a non-unit segment ``i`` between two tie points, ``num_i`` is the
        change in ``tie_values`` and ``den_i`` the change in ``tie_indices``. The
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

        The matching auto tolerance is half that worst drift, since validity
        compares the drift against ``2 * tolerance``::

            tolerance = max_i |si* * den_i - num_i| / 2
        """
        # An already-regular coordinate keeps its spacing unless one is forced.
        if self.sampling_interval is not None and sampling_interval is None:
            return self.copy()

        num = np.diff(self.tie_values)
        den = np.diff(self.tie_indices)
        # Only multi-sample segments carry rate information; unit gaps are
        # ignored, consistently with `_is_valid_sampling_interval`.
        mask = den != 1
        num = num[mask]
        den = den[mask]

        if sampling_interval is None and num.size > 0:
            # Per-segment rates as plain floats (seconds for datetime axes), used
            # only to pick the binding pair without integer/timedelta overflow.
            num_seconds = (
                num / np.timedelta64(1, "s")
                if np.issubdtype(num.dtype, np.timedelta64)
                else num.astype(float)
            )
            den_float = den.astype(float)
            rate = num_seconds / den_float
            # height_ij = den_i den_j |r_i - r_j| / (den_i + den_j); the diagonal
            # is zero, so a single segment trivially selects itself.
            height = (
                den_float[:, None]
                * den_float[None, :]
                * np.abs(rate[:, None] - rate[None, :])
                / (den_float[:, None] + den_float[None, :])
            )
            i, j = np.unravel_index(np.argmax(height), height.shape)
            # Balance point of the binding pair, kept in the native dtype.
            sampling_interval = (num[i] + num[j]) / (den[i] + den[j])

        if isinstance(tolerance, str):
            if tolerance != "auto":
                raise ValueError(f"unknown tolerance {tolerance!r}, expected 'auto'")
            if sampling_interval is None or num.size == 0:
                tolerance = None
            else:
                # Validity requires `2 * tolerance >= max drift`, so halve the
                # worst drift. Work in float (seconds for datetime axes) and add a
                # few ULPs at value scale so the division-based re-validation
                # cannot reject the result on rounding alone.
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
        tie_intervals = np.diff(self.tie_values) / np.diff(self.tie_indices)
        (positions,) = np.nonzero(np.diff(self.tie_indices) == 1)
        references = np.where(
            positions > 0,
            positions - 1,
            np.minimum(positions + 1, len(tie_intervals) - 1),
        )
        deltas = tie_intervals[positions] - tie_intervals[references]
        return self.tie_indices[positions + 1], deltas


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
