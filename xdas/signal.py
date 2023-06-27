import numpy as np
import scipy.signal as sp
import xarray as xr


def integrate(da, midpoints=False, dim="distance"):
    """
    Integrate along a given dimension.

    Parameters
    ----------
    da : DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    DataArray
        The integrated data.
    """
    d = np.median(np.diff(da[dim].values))
    out = da.cumsum(dim) * d
    if midpoints:
        s = da[dim].values
        s += d / 2
        out[dim] = s
    return out


def segment_mean(da, limits, window="hann", dim="distance"):
    """
    Piecewise mean removal.

    Parameters
    ----------
    da : DataArray
        The data that segment mean should be removed.
    limits : list of float
        The segments limits.
    window : str, optional
        The tapering windows to apply at each window, by default "hann".
    dim : str, optional
        The axis along which remove the segment means, by default "distance".

    Returns
    -------
    DataArray
        The data with segment means removed.
    """
    out = da.copy()
    for sstart, send in zip(limits[:-1], limits[1:]):
        key = dict(distance=slice(sstart, np.nextafter(send, -np.inf)))
        subset = out.loc[key]
        win = xr.DataArray(
            sp.get_window(window, subset.sizes[dim]),
            {dim: subset[dim]},
        )
        ref = (subset * win).sum(dim) / win.sum(dim)
        out.loc[key] -= ref
    return out


def sliding_mean(da, wlen, window="hann", pad_mode="reflect", dim="distance"):
    """
    Sliding mean removal.

    Parameters
    ----------
    da : DataArray
        The data that sliding mean should be removed.
    wlen : float
        Length of the sliding mean.
    window : str, optional
        Tapering window used, by default "hann"
    pad_mode : str, optional
        Padding mode used, by default "reflect"
    dim : str, optional
        The dimension along which remove the sliding mean, by default "distance"

    Returns
    -------
    DataArray
        The data with sliding mean removed.
    """
    d = np.median(np.diff(da[dim].values))
    n = round(wlen / d)
    if n % 2 == 0:
        n += 1
    win = sp.get_window(window, n)
    win /= np.sum(win)
    shape = tuple(-1 if d == dim else 1 for d in da.dims)
    win = np.reshape(win, shape)
    data = da.data
    pad_width = tuple((n // 2, n // 2) if d == dim else (0, 0) for d in da.dims)
    mean = sp.fftconvolve(np.pad(data, pad_width, mode=pad_mode), win, mode="valid")
    data = data - mean
    return da.copy(data=data)
