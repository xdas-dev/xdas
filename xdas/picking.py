from itertools import pairwise

import numpy as np
from numba import njit, prange
from scipy.fft import next_fast_len
from scipy.integrate import trapezoid

from .core.coordinates import get_sampling_interval
from .core.dataarray import DataArray
from .core.datacollection import DataMapping, DataSequence


class WaveFront(DataSequence):
    """Sequence of 1D horizon picks forming a wavefront.

    A WaveFront is a collection of non-overlapping 1D ``DataArray`` horizons
    sharing the same single dimension and dtype. Horizons are sorted by their
    coordinate range and can be interpolated, differenced, and summarized
    statistically.

    Parameters
    ----------
    horizons : sequence of DataArray
        Sequence of 1D horizons. Each element must be a ``DataArray`` with a
        single dimension. All horizons must share the same dimension name and
        dtype, and must not overlap in coordinate range.

    Attributes
    ----------
    dim : str or None
        The single dimension name shared by all horizons. ``None`` if empty.
    dtype : numpy.dtype or None
        The dtype of the horizon values. ``None`` if empty.
    length : float
        Total coordinate span across all horizons, computed as the sum of
        ``end - start`` for each horizon.

    Notes
    -----
    - Horizons are validated to be 1D, have identical ``dims`` and ``dtype``,
      and be non-overlapping. Overlaps raise ``ValueError``.
    - Coordinate dtype ``datetime64`` is supported for interpolation via
      conversion to float internally when necessary.
    """

    def __init__(self, horizons):
        if len(horizons) == 0:
            super().__init__(horizons, "horizon")
            self.dim = None
            self.dtype = None
            return

        if not all(horizon.ndim == 1 for horizon in horizons):
            raise ValueError("All horizons must be 1D")

        (dim,) = horizons[0].dims
        if not all(horizon.dims == (dim,) for horizon in horizons):
            raise ValueError("All horizons must have the same dimension")

        dtype = horizons[0].dtype
        if not all(horizon.dtype == dtype for horizon in horizons):
            raise ValueError("All horizons must have the same dtype")

        horizons = sorted(horizons, key=lambda horizon: horizon[dim][0].values)
        for a, b in pairwise(horizons):
            if a[dim][-1].values > b[dim][0].values:
                raise ValueError("Horizons are overlapping")

        super().__init__(horizons, "horizon")
        self.dim = dim
        self.dtype = dtype

    @property
    def coords(self):
        """Concatenate coordinates from all horizons.

        Returns
        -------
        dict
            Mapping ``{dim: numpy.ndarray}`` with concatenated coordinate
            values from all horizons.
        """
        return {
            self.dim: np.concatenate([horizon[self.dim].values for horizon in self])
        }

    @property
    def length(self):
        """Total coordinate span across all horizons.

        Returns
        -------
        float
            Sum of ``end - start`` for each horizon.
        """
        return sum(
            horizon[self.dim][-1].values - horizon[self.dim][0].values
            for horizon in self
        )

    @classmethod
    def from_picks(cls, picks, gap_threshold, min_points=2):
        """Build a wavefront from pick points grouped by gaps.

        The input picks dataframe must have exactly two columns:
        ``[value_column, dim_column]``. Picks are sorted by the coordinate
        column and grouped into horizons where consecutive picks are separated
        by less than or equal to ``gap_threshold``; larger gaps start new
        horizons. Short horizons with fewer than ``min_points`` are discarded.

        Parameters
        ----------
        picks : pandas.DataFrame
            DataFrame with two columns: values and coordinates along a single
            dimension. Duplicates are dropped before grouping.
        gap_threshold : float
            Threshold on coordinate differences to split groups into separate
            horizons.
        min_points : int, default=2
            Minimum number of picks required to keep a horizon.

        Returns
        -------
        WaveFront
            WaveFront instance composed of the resulting horizons.
        """
        value_column, dim_column = picks.columns
        picks = picks.drop_duplicates()
        picks = picks.sort_values(dim_column)
        groups = picks[dim_column].diff().gt(gap_threshold).cumsum()
        horizons = [
            DataArray(
                group[value_column].values,
                coords={dim_column: group[dim_column].values},
                dims=(dim_column,),
            )
            for _, group in picks.groupby(groups)
            if len(group) >= min_points
        ]
        return cls(horizons)

    def simplify(self, tolerance):
        """Simplify horizons using an interp-based Douglas–Peucker algorithm.

        Removes unnecessary points from each horizon so that the simplified
        wavefront approximates the original within ``tolerance``. The
        algorithm is applied independently to each horizon, using linear
        interpolation between endpoints (via ``numpy.interp``) to estimate
        the piecewise-linear approximation and recursively splitting segments
        where the maximum deviation exceeds ``tolerance``. Endpoints are
        always preserved.

        Parameters
        ----------
        tolerance : float or numpy.timedelta64
            Maximum allowed deviation between the original horizon and its
            simplified version. If horizon values have ``datetime64`` dtype,
            provide a ``numpy.timedelta64`` tolerance (e.g.,
            ``np.timedelta64(50, 'ms')``). For numeric dtypes, provide a
            float.

        Returns
        -------
        WaveFront
            A new wavefront with simplified horizons, preserving endpoints.

        Notes
        -----
        - For ``datetime64`` values, interpolation uses internal conversion to
          float nanoseconds for deviation checks, and results are cast back to
          ``datetime64``. The provided ``tolerance`` should be a
          ``numpy.timedelta64`` for consistent comparison.
        """
        if len(self) == 0:
            return WaveFront([])
        horizons = []
        for horizon in self:
            x = horizon[self.dim].values
            y = horizon.values
            xs, ys = _douglas_peucker(x, y, tolerance)
            horizons.append(DataArray(ys, coords={self.dim: xs}, dims=(self.dim,)))
        return WaveFront(horizons)

    def interp(self, coords):
        """Interpolate the wavefront at given coordinates.

        For each request coordinate, interpolation is performed within the
        corresponding horizon if the coordinate falls inside its range;
        otherwise ``NaN`` (or ``NaT`` for ``datetime64``) is returned.

        Parameters
        ----------
        coords : array-like
            1D array of coordinates along ``self.dim`` to sample.

        Returns
        -------
        DataArray
            Values sampled at ``coords`` with ``coords`` set on ``self.dim``.
        """
        coords = np.asarray(coords)
        if coords.ndim != 1:
            raise ValueError("`coords` must be an 1D array-like object")
        if np.issubdtype(coords.dtype, np.datetime64):
            raise NotImplementedError("datetime64 coords are not supported yet")
        values = np.full(coords.shape, np.nan, dtype=self.dtype)
        for horizon in self:
            mask = (coords >= horizon[self.dim][0].values) & (
                coords <= horizon[self.dim][-1].values
            )
            if np.any(mask):
                values[mask] = self._interp(coords[mask], horizon)
        return DataArray(values, coords={self.dim: coords})

    def _interp(self, coords, horizon):
        if np.issubdtype(self.dtype, np.datetime64):
            fp = horizon.values.astype(float)
            f = np.interp(coords, horizon[self.dim].values, fp)
            return np.rint(f).astype(self.dtype)
        else:
            return np.interp(coords, horizon[self.dim].values, horizon.values)

    def diff(self, other):
        """Compute pointwise difference between two compatible wavefronts.

        Both wavefronts must share the same dimension and dtype. The union of
        their coordinates is formed, both are interpolated on this union, and
        differences are computed. Contiguous valid segments (where both values
        are finite/non-``NaT``) are returned as horizons in a new wavefront.

        Parameters
        ----------
        other : WaveFront
            Another wavefront to subtract from this one.

        Returns
        -------
        WaveFront
            Wavefront composed of horizons covering contiguous valid segments
            of the difference.

        Raises
        ------
        TypeError
            If ``other`` is not a ``WaveFront``.
        ValueError
            If dimensions or dtypes differ.
        """
        # check input compatibility
        if not isinstance(other, WaveFront):
            raise TypeError("`other` must be a WaveFront instance")
        if self.dim != other.dim:
            raise ValueError("WaveFronts must have the same dimension to compute diff")
        if self.dtype != other.dtype:
            raise ValueError("WaveFronts must have the same dtype to compute diff")

        # get union of coords and interpolate both wavefronts
        coords = {
            self.dim: np.unique(
                np.concatenate([self.coords[self.dim], other.coords[other.dim]])
            )
        }
        self_interp = self.interp(coords[self.dim])
        other_interp = other.interp(coords[other.dim])
        values = self_interp.values - other_interp.values

        # split on NaN / NaT entries in the result
        if np.issubdtype(self.dtype, np.datetime64):
            valid = ~np.isnat(self_interp.values) & ~np.isnat(other_interp.values)
        else:
            valid = np.isfinite(self_interp.values) & np.isfinite(other_interp.values)

        # create separate horizons for each contiguous valid segment
        groups = np.cumsum(~valid)
        horizons = [
            DataArray(
                values[groups == g][valid[groups == g]],
                coords={self.dim: coords[self.dim][groups == g][valid[groups == g]]},
                dims=(self.dim,),
            )
            for g in np.unique(groups[valid])
        ]
        return WaveFront(horizons)

    def mean(self):
        """Length-weighted mean value of the wavefront.

        The mean is computed by integrating each horizon via the trapezoidal
        rule and dividing the sum by the total ``length``.

        Returns
        -------
        float
            Mean value or ``nan`` if empty.
        """
        if len(self) == 0:
            return np.nan
        values = [
            trapezoid(horizon.values, horizon[self.dim].values) for horizon in self
        ]
        return np.sum(values) / self.length

    def var(self, *, mean=None):
        """Length-weighted variance of the wavefront.

        Parameters
        ----------
        mean : float, optional
            Precomputed mean value. If ``None``, it is computed via ``mean``.

        Returns
        -------
        float
            Variance value or ``nan`` if empty.
        """
        if len(self) == 0:
            return np.nan
        if mean is None:
            mean = self.mean()
        values = [
            _square_trapezoid(horizon.values - mean, horizon[self.dim].values)
            for horizon in self
        ]
        return np.sum(values) / self.length

    def std(self, *, mean=None):
        """Length-weighted standard deviation of the wavefront.

        Parameters
        ----------
        mean : float, optional
            Precomputed mean value. If ``None``, it is computed via ``mean``.

        Returns
        -------
        float
            Standard deviation.
        """
        return np.sqrt(self.var(mean=mean))


class WaveFrontCollection(DataMapping):
    """Mapping from labels to compatible wavefronts.

    A collection of labeled :class:`WaveFront` instances that all share the
    same dimension and dtype. Provides interpolation, difference, and
    aggregated statistics across the collection.

    Parameters
    ----------
    wavefronts : dict[str, WaveFront | sequence of DataArray]
        Mapping from labels to wavefronts or sequences of horizons that can be
        converted into a :class:`WaveFront`.

    Attributes
    ----------
    dim : str
        Shared single dimension name across all wavefronts.
    dtype : numpy.dtype
        Shared dtype across all wavefront values.
    """

    def __init__(self, wavefronts):
        wavefronts = {
            label: (
                wavefronts[label]
                if isinstance(wavefronts[label], WaveFront)
                else WaveFront(wavefronts[label])
            )
            for label in wavefronts
        }

        dims = set(wavefronts[label].dim for label in wavefronts)
        if len(dims) != 1:
            raise ValueError("All wavefronts must have the same dimension")
        (dim,) = dims

        dtype = set(wavefronts[label].dtype for label in wavefronts)
        if len(dtype) != 1:
            raise ValueError("All wavefronts must have the same dtype")
        (dtype,) = dtype

        super().__init__(wavefronts, "wavefront")
        self.dim = dim
        self.dtype = dtype

    @classmethod
    def from_picks(cls, picks, gap_threshold, min_points=2):
        """Build a collection of wavefronts from labeled picks.

        The input picks dataframe must have three columns
        ``[value_column, dim_column, label_column]``. Each label forms a
        separate wavefront via :meth:`WaveFront.from_picks`. Empty wavefronts
        are dropped.

        Parameters
        ----------
        picks : pandas.DataFrame
            DataFrame with value, coordinate, and label columns.
        gap_threshold : float
            Threshold on coordinate differences to split groups into horizons.
        min_points : int, default=2
            Minimum number of picks required to keep a horizon.

        Returns
        -------
        WaveFrontCollection
            Collection of labeled, non-empty wavefronts.
        """
        value_column, dim_column, label_column = picks.columns
        wavefronts = {}
        for label, group in picks.groupby(label_column):
            wf = WaveFront.from_picks(
                group[[value_column, dim_column]], gap_threshold, min_points=min_points
            )
            if len(wf) > 0:
                wavefronts[label] = wf
        return cls(wavefronts)

    def interp(self, coords):
        """Interpolate all wavefronts at given coordinates.

        Parameters
        ----------
        coords : array-like
            1D array of coordinates along ``self.dim`` to sample.

        Returns
        -------
        DataMapping
            Mapping from label to :class:`DataArray` of interpolated values.
        """
        return DataMapping(
            {label: wavefront.interp(coords) for label, wavefront in self.items()},
            "wavefront",
        )

    def diff(self, other):
        """Compute differences against another collection with same schema.

        Each label in ``self`` is differenced against the corresponding label
        in ``other`` via :meth:`WaveFront.diff`. Empty differences are dropped.

        Parameters
        ----------
        other : WaveFrontCollection
            Another collection to subtract from this one.

        Returns
        -------
        WaveFrontCollection
            Collection of non-empty difference wavefronts.

        Raises
        ------
        ValueError
            If type, dimension, or dtype compatibility fails.
        """
        # check input compatibility
        if not isinstance(other, WaveFrontCollection):
            raise ValueError("`other` must be a WaveFrontCollection")
        if self.dim != other.dim:
            raise ValueError("WaveFronts must have the same dimension to compute diff")
        if self.dtype != other.dtype:
            raise ValueError("WaveFronts must have the same dtype to compute diff")
        wavefronts = {}
        for label, wavefront in self.items():
            diff = wavefront.diff(other[label])
            if len(diff) > 0:
                wavefronts[label] = diff
        return WaveFrontCollection(wavefronts)

    def mean(self):
        """Length-weighted mean across all wavefronts.

        Returns
        -------
        float
            Aggregated mean weighted by each wavefront's ``length``.
        """
        values = np.array([wavefront.mean() for wavefront in self.values()])
        lengths = np.array([wavefront.length for wavefront in self.values()])
        return np.sum(values * lengths) / np.sum(lengths)

    def var(self):
        """Length-weighted variance across all wavefronts.

        Returns
        -------
        float
            Aggregated variance weighted by each wavefront's ``length``.
        """
        mean = self.mean()
        values = np.array([wavefront.var(mean=mean) for wavefront in self.values()])
        lengths = np.array([wavefront.length for wavefront in self.values()])
        return np.sum(values * lengths) / np.sum(lengths)

    def std(self):
        """Length-weighted standard deviation across all wavefronts.

        Returns
        -------
        float
            Aggregated standard deviation.
        """
        return np.sqrt(self.var())


def tapered_selection(da, start, end, window=None, size=None, dim="last"):
    """Select and taper segments defined by start/end coordinates.

    Coordinates with non-finite ``start`` or ``end`` values are ignored. If
    ``size`` is not provided, the output length along ``dim`` is the
    ``scipy.fft.next_fast_len`` of the largest selected window length. A
    symmetric taper ``window`` is split in half and applied to both ends of
    each selected segment. The ``window`` length must not exceed the smallest
    selected window length.

    Parameters
    ----------
    da : DataArray
        Input 2D array. Must contain ``dim`` as one of its dimensions.
    start : array-like
        1D array of start coordinates along ``dim`` for each row (other
        dimension). Non-finite values are ignored.
    end : array-like
        1D array of end coordinates along ``dim`` for each row (other
        dimension). Non-finite values are ignored.
    window : array-like, optional
        Taper window values. If odd-length, the central value is removed to
        make it even, resulting in symmetric halves applied to segment ends.
    size : int, optional
        Output length along ``dim``. If ``None``, determined via
        ``next_fast_len`` of the maximum ``end - start`` over valid rows.
    dim : str, default="last"
        Dimension along which selection and tapering happen.

    Returns
    -------
    DataArray
        Selected and tapered data with shapes ``{other_dim: N, dim: size}``,
        where ``N`` is the number of valid start/end pairs. The ``dim``
        coordinate is updated to span ``0`` to ``d * (size - 1)``, where
        ``d`` is the sampling interval along ``dim``.

    Raises
    ------
    ValueError
        If shapes mismatch, no valid pairs are found, or the taper ``window``
        is longer than a selected window.
    """
    # transpose so `dim` is last
    da = da.transpose(..., dim)

    # convert to numpy
    data = np.asarray(da)
    start = np.asarray(start)
    end = np.asarray(end)
    window = np.asarray(window if window is not None else [])

    # check shapes
    if not data.shape[:-1] == start.shape == end.shape:
        raise ValueError("shape mismatch between `da`, `start`, and `end`")

    # select valid start/end
    mask = np.isfinite(start) & np.isfinite(end)
    selection = np.nonzero(mask)[0]

    # check selection
    if selection.size == 0:
        raise ValueError("No valid start/end pairs found")

    # get selection indices
    startindex = da[dim].get_indexer(start[selection], method="bfill")
    endindex = da[dim].get_indexer(end[selection], method="ffill")
    stopindex = endindex + 1

    # determine output size
    if size is None:
        size = next_fast_len(max(stopindex - startindex))

    # check window size
    if min(stopindex - startindex) < window.size:
        raise ValueError("some selected windows are smaller than the window size")

    # make window even-sized (central value should be 1 so can be skipped)
    if window.size % 2 != 0:
        half_size = window.size // 2
        window = np.concatenate((window[:half_size], window[-half_size:]))

    # perform tapered selection
    data = _tapered_selection(
        data,
        selection,
        startindex,
        stopindex,
        size,
        window,
    )

    # update output coords
    coords = {}
    for name in da.coords:
        if da[name].dim == dim:
            if name == dim:
                coords[name] = {
                    "tie_indices": [0, size - 1],
                    "tie_values": [0.0, (size - 1) * get_sampling_interval(da, dim)],
                }
            else:
                pass  # skip non-dimensional coords for `dim`
        else:
            coords[name] = da[name][selection]

    # return output DataArray
    return DataArray(data, coords=coords, dims=da.dims)


@njit(parallel=True)
def _tapered_selection(data, sel, start, stop, size, window):
    out = np.zeros((sel.size, size), dtype=data.dtype)
    w = window.size // 2
    for i in prange(sel.size):
        j = 0
        n = stop[i] - start[i]
        p = sel[i]
        q = start[i]
        k = 0
        while j < w:
            out[i, j] = data[p, q] * window[k]
            j += 1
            q += 1
            k += 1
        while j < n - w:
            out[i, j] = data[p, q]
            j += 1
            q += 1
        while j < n:
            out[i, j] = data[p, q] * window[k]
            j += 1
            q += 1
            k += 1
    return out


def _douglas_peucker(x, y, epsilon):
    mask = np.ones(len(x), dtype=bool)
    stack = [(0, len(x))]
    while stack:
        start, stop = stack.pop()
        ysimple = _interp(
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


def _interp(x, xp, fp):
    if np.issubdtype(fp.dtype, np.datetime64):
        f = np.interp(x, xp, fp.astype(float))
        return np.rint(f).astype(fp.dtype)
    else:
        return np.interp(x, xp, fp)


def _square_trapezoid(y, x):
    x = np.asarray(x)
    y = np.asarray(y)
    dx = x[1:] - x[:-1]
    yi = y[:-1]
    yj = y[1:]
    return np.sum(dx * (yi * yi + yi * yj + yj * yj) / 3.0)
