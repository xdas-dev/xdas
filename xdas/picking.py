from itertools import pairwise

import numpy as np
from numba import njit, prange
from scipy.fft import next_fast_len
from scipy.integrate import trapezoid

from .core.coordinates import get_sampling_interval
from .core.dataarray import DataArray
from .core.datacollection import DataMapping, DataSequence


class WaveFront(DataSequence):
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
        return {
            self.dim: np.concatenate([horizon[self.dim].values for horizon in self])
        }

    @property
    def length(self):
        return sum(
            horizon[self.dim][-1].values - horizon[self.dim][0].values
            for horizon in self
        )

    @classmethod
    def from_picks(cls, picks, gap_threshold, min_points=2):
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

    def interp(self, coords):
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
        if len(self) == 0:
            return np.nan
        values = [
            trapezoid(horizon.values, horizon[self.dim].values) for horizon in self
        ]
        return np.sum(values) / self.length

    def var(self, *, mean=None):
        if len(self) == 0:
            return np.nan
        if mean is None:
            mean = self.mean()
        values = [
            square_trapezoid(horizon.values - mean, horizon[self.dim].values)
            for horizon in self
        ]
        return np.sum(values) / self.length

    def std(self, *, mean=None):
        return np.sqrt(self.var(mean=mean))


class WaveFrontCollection(DataMapping):
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
        return DataMapping(
            {label: wavefront.interp(coords) for label, wavefront in self.items()},
            "wavefront",
        )

    def diff(self, other):
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
        values = np.array([wavefront.mean() for wavefront in self.values()])
        lengths = np.array([wavefront.length for wavefront in self.values()])
        return np.sum(values * lengths) / np.sum(lengths)

    def var(self):
        mean = self.mean()
        values = np.array([wavefront.var(mean=mean) for wavefront in self.values()])
        lengths = np.array([wavefront.length for wavefront in self.values()])
        return np.sum(values * lengths) / np.sum(lengths)

    def std(self):
        return np.sqrt(self.var())


def tapered_selection(da, start, end, window=None, size=None, dim="last"):
    """
    Selects and tapers a DataArray based on `start` and `end` values.

    Coordinates with NaN or NaT `start` or `end` values are ignored. If no `size` is
    provided, the length of the resulting data is determined by the next fast length
    (for FFT) of the maximum distance between the start and end values. The tapering
    window is split in half and applied to the start and end of the selected data. The
    window size must be smaller than the smallest selected data window.

    Parameters
    ----------
    da : DataArray
        Input data array to select and taper. Must be 2D and have `dim` as one of its
        dimensions.
    start : array-like
        Start values along the other dimension than `dim` (must be 1D and have the
        same size) NaN or NaT values indicate coordinates to be ignored.
    end : array-like
        End values along the other dimension than `dim` (must be 1D and have the
        same size) NaN or NaT values indicate coordinates to be ignored.
    size : int, optional
        Size of the output data along `dim`. If None, it is determined by the next
        fast length of the maximum selected window.
    dim : str, optional
        Dimension along which to perform the selection and tapering. Default is 'last'.
    window : array-like, optional
        Tapering window to apply to the selected data.

    Returns
    -------
    DataArray
        A DataArray containing the selected and tapered data with sizes {other_dim: N,
        `dim`: `size`}, where N is the number of valid start/end pairs. The `dim`
        dimension becomes the last dimension and its coordinates run from 0 to
        d * (size - 1), where d is the sampling interval along `dim`.

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


def square_trapezoid(y, x):
    x = np.asarray(x)
    y = np.asarray(y)
    dx = x[1:] - x[:-1]
    yi = y[:-1]
    yj = y[1:]
    return np.sum(dx * (yi * yi + yi * yj + yj * yj) / 3.0)
