import numpy as np
import scipy.signal as sp
import xarray as xr


def get_sample_spacing(da, dim):
    """
    Returns the sample spacing along a given dimension.

    Parameters
    ----------
    da : DataArray or Database
        The data from which extract the sample spacing.
    dim : str
        The dimension along which get the sample spacing.

    Returns
    -------
    float
        The sample spacing.
    """
    d = (da[dim][-1] - da[dim][0]) / (len(da[dim]) - 1)
    d = np.asarray(d)
    if np.issubdtype(d.dtype, np.timedelta64):
        d = d / np.timedelta64(1, "s")
    d = d.item()
    return d


def detrend(da, type, dim):
    """
    Detrend data along given dimension

    Parameters
    ----------
    da : DataArray
        The data to detrend.
    type : str
        Either "linear" or "constant".
    dim : str
        The dimension along which to detrend the data.

    Returns
    -------
    DataArray
        The detrended data.
    """
    axis = da.get_axis_num(dim)
    data = sp.detrend(da.values, axis, type)
    return da.copy(data=data)


def iirfilter(da, freq, btype, corners=4, zerophase=False, dim="time"):
    """
    SOS IIR filtering along given dimension.

    data: DataArray
        Traces to filter.
    freq: float or list
        Cuttoff frequency or band corners [Hz].
    fs: float
        Sampling frequency [Hz].
    btype: {'bandpass', 'lowpass', 'highpass', 'bandstop'}
        The type of the filter.
    corners: int
        The order of the filter.
    zerophase: bool
        If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    dim: str, optional
        The dimension along which to filter.
    """
    axis = da.get_axis_num(dim)
    fs = 1.0 / get_sample_spacing(da, dim)
    sos = sp.iirfilter(corners, freq, btype=btype, ftype="butter", output="sos", fs=fs)
    if zerophase:
        data = sp.sosfiltfilt(sos, da.values, axis=axis)
    else:
        data = sp.sosfilt(sos, da.values, axis=axis)
    return da.copy(data=data)


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
    d = get_sample_spacing(da, dim)
    out = da.cumsum(dim) * d
    if midpoints:
        s = da[dim].values
        s += d / 2
        out[dim] = s
    return out


def segment_mean_removal(da, limits, window="hann", dim="distance"):
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


def sliding_mean_removal(da, wlen, window="hann", pad_mode="reflect", dim="distance"):
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
    d = get_sample_spacing(da, dim)
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
