import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.signal as sp

from . import config
from .database import Database


def parse_dim(db, dim):
    if dim == "first":
        return db.dims[0]
    elif dim == "last":
        return db.dims[-1]
    else:
        if dim in db.dims:
            return dim
        else:
            ValueError(f"{dim} not in db.dims")


def get_sampling_interval(db, dim):
    """
    Returns the sample spacing along a given dimension.

    Parameters
    ----------
    db : Database or DataArray or Database
        The data from which extract the sample spacing.
    dim : str
        The dimension along which get the sample spacing.

    Returns
    -------
    float
        The sample spacing.
    """
    d = (db[dim][-1].values - db[dim][0].values) / (len(db[dim]) - 1)
    d = np.asarray(d)
    if np.issubdtype(d.dtype, np.timedelta64):
        d = d / np.timedelta64(1, "s")
    d = d.item()
    return d


def parallelize(func, axis, parallel):
    n_workers = get_workers_count(parallel)
    if n_workers == 1:
        return func
    else:
        return multithread_along_axis(func, int(axis == 0), n_workers)


def get_workers_count(parallel):
    if parallel is None:
        return config.get("n_workers")
    elif isinstance(parallel, bool):
        if parallel:
            return os.cpu_count()
        else:
            return 1
    elif isinstance(parallel, int):
        return parallel
    else:
        raise TypeError("`parallel` must be either bool or int.")


def multithread_along_axis(func, axis, n_workers):
    def wrapper(x, *args, **kwargs):
        def fn(x):
            return func(x, *args, **kwargs)

        xs = np.array_split(x, n_workers, axis)
        with ThreadPoolExecutor(n_workers) as executor:
            ys = list(executor.map(fn, xs))
        return multithreaded_concatenate(ys, axis, n_workers=n_workers)

    return wrapper


def multithreaded_concatenate(arrays, axis=0, out=None, dtype=None, n_workers=None):
    arrays = [np.asarray(array, dtype) for array in arrays]

    ndim = set(array.ndim for array in arrays)
    if len(ndim) == 1:
        (ndim,) = ndim
    else:
        raise ValueError("arrays must have the same number of dimensions.")

    dtype = set(array.dtype for array in arrays)
    if len(dtype) == 1:
        (dtype,) = dtype
    else:
        raise ValueError("arrays must have the same dtype.")

    shapes = [list(array.shape) for array in arrays]
    section_sizes = [shape.pop(axis) for shape in shapes]
    subshape = set([tuple(shape) for shape in shapes])
    if len(subshape) == 1:
        (subshape,) = subshape
    else:
        raise ValueError("arrays must have the same shape on axes other than `axis`.")
    shape = list(subshape)
    shape.insert(axis, sum(section_sizes))
    shape = tuple(shape)

    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if not (out.ndim == ndim and out.dtype == dtype, out.shape == shape):
            raise ValueError("`out` does not match with provided arrays.")

    div_points = np.cumsum([0] + section_sizes, dtype=int)

    with ThreadPoolExecutor(n_workers) as executor:
        for idx, array in enumerate(arrays):
            start = div_points[idx]
            end = div_points[idx + 1]
            slices = tuple(
                slice(start, end) if n == axis else slice(None) for n in range(ndim)
            )
            executor.submit(out.__setitem__, slices, array)

    return out


def detrend(db, type="linear", dim="last"):
    """
    Detrend data along given dimension

    Parameters
    ----------
    db : Database or DataArray
        The data to detrend.
    type : str
        Either "linear" or "constant".
    dim : str
        The dimension along which to detrend the data.

    Returns
    -------
    Database or DataArray
        The detrended data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    data = sp.detrend(db.values, axis, type)
    return db.copy(data=data)


def taper(db, window="hann", fftbins=False, dim="last"):
    """
    Apply a tapering window along the given dimension

    Parameters
    ----------
    db : Database or DataArray
        The data to taper.
    window : str or tuple, optional
        The window to use, by default "hann"
    fftbins : bool, optional
        Weather to use a periodic windowing, by default False
    dim : str, optional
        Dimension along the which to taper, by default "time"

    Returns
    -------
    Database or DataArray
        The tapered data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    w = sp.get_window(window, db.shape[axis], fftbins=fftbins)
    shape = [-1 if ax == axis else 1 for ax in range(db.ndim)]
    w = w.reshape(shape)
    data = w * db.values
    return db.copy(data=data)


def filter(db, freq, btype, corners=4, zerophase=False, dim="last", parallel=None):
    """
    SOS IIR filtering along given dimension.

    da: DataArray
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
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    fs = 1.0 / get_sampling_interval(db, dim)
    sos = sp.iirfilter(corners, freq, btype=btype, ftype="butter", output="sos", fs=fs)
    if zerophase:
        func = lambda x, sos, axis: sp.sosfiltfilt(sos, x, axis)
        func = parallelize(func, axis, parallel)
    else:
        func = lambda x, sos, axis: sp.sosfilt(sos, x, axis)
        func = parallelize(func, axis, parallel)
    data = func(db.values, sos, axis=axis)
    return db.copy(data=data)


def lfilter(b, a, db, dim="last", parallel=None):
    """
    Scipy function apply a digital filter forward and backward to a signal.
    `Link text https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter/`_
    axis are now given by dim

    Original Scipy Parameters
    -------------------------
    b: array_like
        The numerator coefficient vector in a 1-D sequence.
    a: array_like
        The denominator coefficient vector in a 1-D sequence.
        If a[0] is not 1, then both a and b are normalized by a[0].

    New Xdas Parameters
    -------------------
    db: Database
        Traces to filter
    dim: str, optional
        The dimension along which to filter.
    parallel: bool or int, optional
        whether to parallelize the function, if true all cores are used,
        if false single core, if int n cores are used

    Returns
    -------
    y: Database
        The output of the digital filter

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xs
    >>> from scipy import signal

    Create dummy time-series
    >>> rng = np.random.default_rng()
    >>> t = np.linspace(-1, 1, 201)
    >>> x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
    ...      0.1*np.sin(2*np.pi*1.25*t + 1) +
    ...      0.18*np.cos(2*np.pi*3.85*t))
    >>> xn = x + rng.standard_normal(len(t)) * 0.08
    >>> shape=xn.shape()
    >>> db = xdas.Database(xn,{"time":t})

    Create an order 3 lowpass butterworth filter:
    >>> b, a = signal.butter(3, 0.05)
    >>> z = xp.lfilter(b, a, db, dim)

    Apply the filter again, to have a result filtered at an order the same as filtfilt:
    >>> z2 = xp.lfilter(b, a, z, dim])

    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    func = lambda x, b, a, axis: sp.lfilter(b, a, x, axis)
    func = parallelize(func, axis, parallel)
    data = func(db.values, b, a, axis)
    return db.copy(data=data)


def filtfilt(
    b,
    a,
    db,
    dim="last",
    padtype="odd",
    padlen=None,
    method="pad",
    irlen=None,
    parallel=None,
):
    """
    Scipy function filtfilt data along one-dimension with an IIR or FIR filter.
    `Link text https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt`_
    axis are now given by dim

    Original Scipy Parameters
    -------------------------
    b: array_like
        The numerator coefficient vector in a 1-D sequence.
    a: array_like
        The denominator coefficient vector in a 1-D sequence.
        If a[0] is not 1, then both a and b are normalized by a[0].
    padtype: str or None, optional
        Must be ‘odd’, ‘even’, ‘constant’, or None.
        This determines the type of extension to use for the padded signal.
          If padtype is None, no padding is used. The default is ‘odd’.
    padlen: int or None, optional
        The number of elements by which to extend x at both ends of axis before applying the filter.
        This value must be less than x.shape[axis] - 1. padlen=0 implies no padding.
        The default value is 3 * max(len(a), len(b)).
    method: str, optional
        Determines the method for handling the edges of the signal, either “pad” or “gust”.
        When method is “pad”, the signal is padded; the type of padding is determined
        by padtype and padlen, and irlen is ignored. When method is “gust”,
        Gustafsson’s method is used, and padtype and padlen are ignored.
    irlen: int or None, optional
        When method is “gust”, irlen specifies the length of the impulse response of the filter.
        If irlen is None, no part of the impulse response is ignored.
        For a long signal, specifying irlen can significantly improve the performance of the filter.

    New Xdas Parameters
    -------------------
    db: Database
        Traces to filter
    dim: str, optional
        The dimension along which to filter.
    parallel: bool or int, optional
        whether to parallelize the function, if true all cores are used, if false single core, if int n cores are used

    Returns
    -------
    y: Database
        The output of the digital filter

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xs
    >>> from scipy import signal

    Create dummy time-series
    >>> rng = np.random.default_rng()
    >>> t = np.linspace(-1, 1, 201)
    >>> x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
    ...      0.1*np.sin(2*np.pi*1.25*t + 1) +
    ...      0.18*np.cos(2*np.pi*3.85*t))
    >>> xn = x + rng.standard_normal(len(t)) * 0.08
    >>> shape=xn.shape()
    >>> db = xdas.Database(xn,{"time":t})

    Create an order 3 lowpass butterworth filter:
    >>> b, a = signal.butter(3, 0.05)
    >>> z = xp.filtfilt(b, a, db, dim, padlen=150)
    """

    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    func = lambda x, b, a, axis, padtype, padlen, method, irlen: sp.filtfilt(
        b, a, x, axis, padtype, padlen, method, irlen
    )
    func = parallelize(func, axis, parallel)
    data = func(db.values, b, a, axis, padtype, padlen, method, irlen)
    return db.copy(data=data)


def sosfilt(sos, db, dim="last", parallel=None):
    """
    Scipy function filter data along one dimension using cascaded second-order sections.
    `Link text https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html`_
    The initial conditions for the filter delays si is not implemented.
    axis are now given by dim

    Original Scipy Parameters
    -------------------------
    sos: array_like
        Array of second-order filter coefficients, must have shape (n_sections, 6).
        Each row corresponds to a second-order section, with the first three columns
        providing the numerator coefficients and the last three providing the denominator
        coefficients.

    New Xdas Parameters
    -------------------
    db: Database
        Traces to filter
    dim: str, optional
        The dimension along which to filter.
    parallel: bool or int, optional
        whether to parallelize the function, if true all cores are used, if false single core, if int n cores are used

    Returns
    -------
    y: Database
        The output of the digital filter

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xs
    >>> from scipy import signal

    Create dummy time-series
    >>> rng = np.random.default_rng()
    >>> t = np.linspace(-1, 1, 201)
    >>> x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
    ...      0.1*np.sin(2*np.pi*1.25*t + 1) +
    ...      0.18*np.cos(2*np.pi*3.85*t))
    >>> xn = x + rng.standard_normal(len(t)) * 0.08
    >>> shape=xn.shape()
    >>> db = xdas.Database(xn,{"time":t})

    Create an order 3 lowpass butterworth filter:
    fs = 1.0 / get_sampling_interval(db, dim)
    >>> sos = signal.butter(3, 0.05,output='sos')
    >>> z = xp.sosfilt(sos, db, dim,parallel=None)
    """

    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    func = lambda x, sos, axis: sp.sosfilt(sos, x, axis)
    func = parallelize(func, axis, parallel)
    data = func(db.values, sos, axis)
    return db.copy(data=data)


def sosfiltfilt(sos, db, dim="last", padtype="odd", padlen=None, parallel=None):
    """
    Scipy function a forward-backward digital filter using cascaded second-order sections..
    `Link text https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html#scipy.signal.sosfiltfilt`_
    axis are now given by dim

    Original Scipy Parameters
    -------------------------
    sos: array_like
        Array of second-order filter coefficients, must have shape (n_sections, 6).
        Each row corresponds to a second-order section, with the first three columns
        providing the numerator coefficients and the last three providing the denominator
        coefficients.
    padtype: str or None, optional
        Must be ‘odd’, ‘even’, ‘constant’, or None. This determines the type of extension
        to use for the padded signal to which the filter is applied. If padtype is None,
        no padding is used. The default is ‘odd’.

    padlen: int or None, optional
        The number of elements by which to extend x at both ends of axis before applying
        the filter. This value must be less than x.shape[axis] - 1. padlen=0 implies no padding.
        The default value is something complicated see scipy.sosfiltfilt

    New Xdas Parameters
    -------------------
    db: Database
        Traces to filter
    dim: str, optional
        The dimension along which to filter.
    parallel: bool or int, optional
        whether to parallelize the function, if true all cores are used, if false single core, if int n cores are used

    Returns
    -------
    y: Database
        The output of the digital filter

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xs
    >>> from scipy import signal

    Create dummy time-series
    >>> rng = np.random.default_rng()
    >>> t = np.linspace(-1, 1, 201)
    >>> x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
    ...      0.1*np.sin(2*np.pi*1.25*t + 1) +
    ...      0.18*np.cos(2*np.pi*3.85*t))
    >>> xn = x + rng.standard_normal(len(t)) * 0.08
    >>> shape=xn.shape()
    >>> db = xdas.Database(xn,{"time":t})

    Create an order 3 lowpass butterworth filter:
    >>> sos = signal.butter(3, 0.05,output='sos')
    >>> z = xp.sosfiltfilt(sos, db, dim, parallel=None)
    """

    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    func = lambda x, sos, axis, padtype, padlen: sp.sosfiltfilt(
        sos, x, axis, padtype, padlen
    )
    func = parallelize(func, axis, parallel)
    data = func(db.values, sos, axis, padtype, padlen)
    return db.copy(data=data)

#def medfilt2d(input, dictionary)
    # dictionary is {'time':2, 'space':4}


def decimate(db, q, n=None, ftype=None, zero_phase=None, dim="last", parallel=None):
    """
    Downsample the signal after applying an anti-aliasing filter.

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with Hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    db : Database or DataArray
        The signal to be downsampled, as an N-dimensional dataarray.
    q : int
        The downsampling factor. When using IIR downsampling, it is recommended
        to call `decimate` multiple times for downsampling factors higher than
        13.
    n : int, optional
        The order of the filter (1 less than the length for 'fir'). Defaults to
        8 for 'iir' and 20 times the downsampling factor for 'fir'.
    ftype : str {'iir', 'fir'} or ``dlti`` instance, optional
        If 'iir' or 'fir', specifies the type of lowpass filter. If an instance
        of an `dlti` object, uses that object to filter before downsampling.
    dim : str, optional
        The dimension along which to decimate.
    zero_phase : bool, optional
        Prevent phase shift by filtering with `filtfilt` instead of `lfilter`
        when using an IIR filter, and shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.

    Returns
    -------
    Database or DataArray
        The down-sampled signal.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    func = parallelize(sp.decimate, axis, parallel)
    data = func(db.values, q, n, ftype, axis, zero_phase)
    return db[{dim: slice(None, None, q)}].copy(data=data)


def integrate(db, midpoints=False, dim="last"):
    """
    Integrate along a given dimension.

    Parameters
    ----------
    db : Database or DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    Database or DataArray
        The integrated data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    d = get_sampling_interval(db, dim)
    data = np.cumsum(db.values, axis=axis) * d
    out = db.copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


def differentiate(db, midpoints=False, dim="last"):
    """
    Differentiate along a given dimension.

    Parameters
    ----------
    db : Database or DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    Database or DataArray
        The integrated data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    d = get_sampling_interval(db, dim)
    data = np.diff(db.values, axis=axis) / d
    out = db.isel({dim: slice(None, -1)}).copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


def segment_mean_removal(db, limits, window="hann", dim="last"):
    """
    Piecewise mean removal.

    Parameters
    ----------
    db : Database or DataArray
        The data that segment mean should be removed.
    limits : list of float
        The segments limits.
    window : str, optional
        The tapering windows to apply at each window, by default "hann".
    dim : str, optional
        The axis along which remove the segment means, by default "distance".

    Returns
    -------
    Database or DataArray
        The data with segment means removed.
    """
    dim = parse_dim(db, dim)
    out = db.copy()
    axis = db.get_axis_num(dim)
    for sstart, send in zip(limits[:-1], limits[1:]):
        key = {dim: slice(sstart, np.nextafter(send, -np.inf))}
        data = out.loc[key].values
        win = sp.get_window(window, data.shape[axis])
        shape = tuple(-1 if a == axis else 1 for a in range(data.ndim))
        win = np.reshape(win, shape)
        ref = np.sum(data * win, axis=axis) / np.sum(win)
        out.loc[key] = out.loc[key].values - ref  # TODO: Add Database Arithmetics.
    return out


def sliding_mean_removal(db, wlen, window="hann", pad_mode="reflect", dim="last"):
    """
    Sliding mean removal.

    Parameters
    ----------
    db : Database or DataArray
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
    Database or DataArray
        The data with sliding mean removed.
    """
    dim = parse_dim(db, dim)
    d = get_sampling_interval(db, dim)
    n = round(wlen / d)
    if n % 2 == 0:
        n += 1
    win = sp.get_window(window, n)
    win /= np.sum(win)
    shape = tuple(-1 if d == dim else 1 for d in db.dims)
    win = np.reshape(win, shape)
    data = db.data
    pad_width = tuple((n // 2, n // 2) if d == dim else (0, 0) for d in db.dims)
    mean = sp.fftconvolve(np.pad(data, pad_width, mode=pad_mode), win, mode="valid")
    data = data - mean
    return db.copy(data=data)


def fft(db, n=None, dim={"last": "frequency"}, norm=None):
    ((olddim, newdim),) = dim.items()
    olddim = parse_dim(db, olddim)
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    f = np.fft.fftshift(np.fft.fftfreq(n, d))
    data = np.fft.fftshift(np.fft.fft(db.values, n, axis, norm), axis)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)


def rfft(db, n=None, dim={"last": "frequency"}, norm=None):
    ((olddim, newdim),) = dim.items()
    olddim = parse_dim(db, olddim)
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    f = np.fft.rfftfreq(n, d)
    data = np.fft.rfft(db.values, n, axis, norm)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)
