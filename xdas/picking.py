import numpy as np
from numba import njit, prange
from scipy.fft import next_fast_len

import xdas as xd


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
        same size) NaN or NaT values indicate coordinates to ignored.
    end : array-like
        End values along the other dimension than `dim` (must be 1D and have the
        same size) NaN or NaT values indicate coordinates to ignored.
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
                    "tie_values": [0.0, (size - 1) * xd.get_sampling_interval(da, dim)],
                }
            else:
                pass  # skip non-dimensional coords for `dim`
        else:
            coords[name] = da[name][selection]

    # return output DataArray
    return xd.DataArray(data, coords=coords, dims=da.dims)


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
