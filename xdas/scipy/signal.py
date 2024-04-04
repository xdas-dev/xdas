from inspect import signature

import numpy as np
import scipy.signal as sp

from ..atoms import atomized
from ..coordinates import Coordinate, get_sampling_interval
from ..core import collects, splits
from ..database import Database
from ..parallel import parallelize


@atomized
@splits
@collects
def detrend(db, type="linear", dim="last", parallel=None):
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

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = db.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.detrend)
    data = func(db.values, axis, type)
    return db.copy(data=data)


@atomized
@splits
@collects
def taper(db, window="hann", fftbins=False, dim="last", parallel=None):
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
    axis = db.get_axis_num(dim)
    w = sp.get_window(window, db.shape[axis], fftbins=fftbins)
    shape = [-1 if ax == axis else 1 for ax in range(db.ndim)]
    w = w.reshape(shape)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(np.multiply)
    data = func(db.values, w)
    return db.copy(data=data)


@atomized
@splits
@collects
def filter(
    db, freq, btype, corners=4, zerophase=False, dim="last", parallel=None
):  # TODO
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
    axis = db.get_axis_num(dim)
    fs = 1.0 / get_sampling_interval(db, dim)
    sos = sp.iirfilter(corners, freq, btype=btype, ftype="butter", output="sos", fs=fs)
    if zerophase:
        func = lambda x, sos, axis: sp.sosfiltfilt(sos, x, axis)
        func = parallelize(axis, parallel)(func)
    else:
        func = lambda x, sos, axis: sp.sosfilt(sos, x, axis)
        func = parallelize(axis, parallel)(func)
    data = func(db.values, sos, axis=axis)
    return db.copy(data=data)


@atomized
@splits
@collects
def hilbert(db, N=None, dim="last", parallel=None):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    db : Database
        Signal data. Must be real.
    N: int, optional
        Number of Fourier components. Default: `db.sizes[dim]`.
    dim: str, optional
        The dimension along which to transform. Default: last.
    parallel: bool or int, optional
        Whether to parallelize the function, if True all cores are used,
        if False single core, if int: number of cores.

    Returns
    -------
    Database
        Analytic signal of `db` along dim.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    In this example we use the Hilbert transform to determine the analytic signal.

    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> xp.hilbert(db, dim="time")
    <xdas.Database (time: 300, distance: 401)>
    [[ 0.0497+0.1632j -0.0635+0.0125j ...  0.1352-0.3107j -0.2832-0.0126j]
     [-0.1096-0.0335j  0.124 +0.0257j ... -0.0444+0.2409j  0.1378-0.2702j]
     ...
     [-0.1977+0.0545j -0.0533-0.1947j ...  0.3722+0.125j  -0.0127+0.1723j]
     [ 0.1221-0.1808j  0.1888+0.0368j ... -0.4517+0.1581j  0.0411+0.1512j]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = db.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.hilbert)
    data = func(db.values, N, axis)
    return db.copy(data=data)


@atomized
@splits
@collects
def resample(db, num, dim="last", window=None, domain="time", parallel=None):
    """
    Resample db to num samples using Fourier method along the given dimension.

    The resampled signal starts at the same value as db but is sampled with a spacing
    of len(db) / num * (spacing of db). Because a Fourier method is used, the signal is
    assumed to be periodic.

    Parameters
    ----------
    db: Database
        The data to be resampled.
    num: int
        The number of samples in the resampled signal.
    dim: str, optional
        The dimension along which to resample. Default is last.
    window: array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier domain. See below for
        details.
    domain: string, optional
        A string indicating the domain of the input x: `time` Consider the input db as
        time-domain (Default), `freq` Consider the input db as frequency-domain.

    Returns
    -------
    Database
        The resampled database.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    A synthetic database is resample from 300 to 100 samples along the time dimension.
    The 'hamming' window is used.

    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> xp.resample(db, 100, dim='time', window='hamming', domain='time')
    <xdas.Database (time: 100, distance: 401)>
    [[ 0.039988  0.04855  -0.08251  ...  0.02539  -0.055219 -0.006693]
     [-0.032913 -0.016732  0.033743 ...  0.028534 -0.037685  0.032918]
     [ 0.01215   0.064107 -0.048831 ...  0.009131  0.053133  0.019843]
     ...
     [-0.036508  0.050059  0.015494 ... -0.012022 -0.064922  0.034198]
     [ 0.054003 -0.013902 -0.084095 ...  0.008979  0.080804 -0.063866]
     [-0.042741 -0.03524   0.122637 ... -0.013453 -0.075183  0.093055]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.940
      * distance (distance): 0.000 to 10000.000
    """
    axis = db.get_axis_num(dim)
    dim = db.dims[axis]
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.resample)
    data, t = func(db.values, num, db[dim].values, axis, window, domain)
    new_coord = {"tie_indices": [0, num - 1], "tie_values": [t[0], t[-1]]}
    coords = {
        name: new_coord if name == dim else coord
        for name, coord in db.coords.items()
        if not (coord.dim == dim and not name == dim)  # don't handle non-dimensional
    }
    return Database(data, coords, db.dims, db.name, db.attrs)


@atomized
@splits
@collects
def resample_poly(
    db,
    up,
    down,
    dim="last",
    window=("kaiser", 5.0),
    padtype="constant",
    cval=None,
    parallel=None,
):
    """
    Resample db along the given dimension using polyphase filtering.

    The signal in `db` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. By default, values beyond the boundary of the signal are assumed
    to be zero during the filtering step.

    Parameters
    ----------
    db : Database
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    dim : int, optional
        The dimension of `db` that is resampled. Default is last.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by `scipy.signal.upfirdn`.
        Changes assumptions on values beyond the boundary. If `constant`,
        assumed to be `cval` (default zero). If `line` assumed to continue a
        linear trend defined by the first and last points. `mean`, `median`,
        `maximum` and `minimum` work as in `np.pad` and assume that the values
        beyond the boundary are the mean, median, maximum or minimum
        respectively of the array along the dimension.
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Returns
    -------
    Database
        The resampled data.

    Examples
    --------
    This example is made to resample the input database in the time domain at 100 samples
    with an original shape of 300 in time. The choosed window is a 'hamming' window.
    The database is synthetic data.

    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> xp.resample_poly(db, 2, 5, dim='time')
    <xdas.Database (time: 120, distance: 401)>
    [[-0.006378  0.012767 -0.002068 ... -0.033461  0.002603 -0.027478]
     [ 0.008851 -0.037799  0.009595 ...  0.053291 -0.0396    0.026909]
     [-0.034468  0.085153 -0.038036 ... -0.015803  0.030245  0.047028]
     ...
     [ 0.02834   0.053455 -0.155873 ...  0.033726 -0.036478 -0.016146]
     [ 0.015454 -0.062852  0.049064 ... -0.018409  0.113782 -0.072631]
     [-0.026921 -0.01264   0.087272 ...  0.001695 -0.147191  0.177587]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.950
      * distance (distance): 0.000 to 10000.000

    """
    axis = db.get_axis_num(dim)
    dim = db.dims[axis]
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.resample_poly)
    data = func(db.values, up, down, axis, window, padtype, cval)
    start = db[dim][0].values
    d = db[dim][-1].values - db[dim][-2].values
    end = db[dim][-1].values + d
    new_coord = Coordinate(
        {
            "tie_indices": [0, data.shape[axis]],
            "tie_values": [start, end],
        },
        dim,
    )
    new_coord = new_coord[:-1]
    coords = {
        name: new_coord if name == dim else coord
        for name, coord in db.coords.items()
        if not (coord.dim == dim and not name == dim)  # don't handle non-dimensional
    }
    return Database(data, coords, db.dims, db.name, db.attrs)


@atomized
# @splits
@collects
def lfilter(b, a, db, dim="last", zi=None, parallel=None):
    """
    Filter data along one-dimension with an IIR or FIR filter.

    Filter a data sequence, `db`, using a digital filter. The filter is a direct
    form II transposed implementation of the standard difference equation.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    db : Database
        An N-dimensional input database.
    dim : str, optional
        The dimension of the input data array along which to apply the
        linear filter. Default is last.
    zi : array_like or str, optional
        Initial conditions for the filter delays. If `zi` is None or ... then
        initial rest is assumed.
    parallel: bool or int, optional
        Whether to parallelize the function, if true: all cores are used, if false:
        single core, if int: n cores are used.
    Returns
    -------
    db : Database
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned. If `zi` is given or ... then `zf`
        holds the final filter delay values.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> b, a = sp.iirfilter(4, 0.5, btype="low")
    >>> xp.lfilter(b, a, db, dim='time')
    <xdas.Database (time: 300, distance: 401)>
    [[ 0.004668 -0.005968  0.007386 ... -0.0138    0.01271  -0.026618]
     [ 0.008372 -0.01222   0.022552 ... -0.041387  0.046667 -0.093521]
     [-0.008928  0.002764  0.012621 ... -0.032496  0.039645 -0.076117]
     ...
     [ 0.012576 -0.1661    0.196026 ...  0.048191 -0.014532 -0.033122]
     [-0.06294  -0.092234  0.316862 ...  0.045337 -0.139729  0.094086]
     [-0.035233  0.036613  0.044002 ... -0.053585 -0.121344  0.241415]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = db.get_axis_num(dim)
    across = int(axis == 0)
    if zi is ...:
        n_sections = max(len(a), len(b)) - 1
        shape = tuple(
            n_sections if _axis == axis else _size
            for _axis, _size in enumerate(db.shape)
        )
        zi = np.zeros(shape)
    if zi is None:
        func = parallelize((None, None, across), across, parallel)(sp.lfilter)
        data = func(b, a, db.values, axis, zi)
        return db.copy(data=data)
    else:
        func = parallelize(
            (None, None, across, None, across), (across, across), parallel
        )(sp.lfilter)
        data, zf = func(b, a, db.values, axis, zi)
        return db.copy(data=data), zf


@atomized
# @splits
@collects
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
    Apply a digital filter forward and backward to a signal.

    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.

    The function provides options for handling the edges of the signal.

    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    db : Database
        The array of data to be filtered.
    dim : srt, optional
        The dimension of `db` to which the filter is applied.
        Default is last.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `db` at both ends of
        `dim` before applying the filter.  This value must be less than
        ``db.sizes[dim] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Returns
    -------
    Database
        The filtered output with the same coordinates as `db`.


    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> b, a = sp.iirfilter(4, 0.5, btype="low")
    >>> xp.lfilter(b, a, db, dim='time')
    <xdas.Database (time: 300, distance: 401)>
    [[ 0.004668 -0.005968  0.007386 ... -0.0138    0.01271  -0.026618]
     [ 0.008372 -0.01222   0.022552 ... -0.041387  0.046667 -0.093521]
     [-0.008928  0.002764  0.012621 ... -0.032496  0.039645 -0.076117]
     ...
     [ 0.012576 -0.1661    0.196026 ...  0.048191 -0.014532 -0.033122]
     [-0.06294  -0.092234  0.316862 ...  0.045337 -0.139729  0.094086]
     [-0.035233  0.036613  0.044002 ... -0.053585 -0.121344  0.241415]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = db.get_axis_num(dim)
    func = lambda x, b, a, axis, padtype, padlen, method, irlen: sp.filtfilt(
        b, a, x, axis, padtype, padlen, method, irlen
    )
    func = parallelize(axis, parallel)(func)
    data = func(db.values, b, a, axis, padtype, padlen, method, irlen)
    return db.copy(data=data)


@atomized
# @splits
@collects
def sosfilt(sos, db, dim="last", zi=None, parallel=None):
    """
    Filter data along one dimension using cascaded second-order sections.

    Filter a data sequence, `db`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    db : Database
        An N-dimensional input database.
    dim : str, optional
        The dimension of the input database  along which to apply the
        linear filter. Default is -1.
    zi : array_like or str, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `db`, but with ``db.sizes[dim]``
        replaced by 2.  If `zi` is None,... , or is not given then initial rest
        (i.e. all zeros) is assumed.

    Returns
    -------
    y : Database
        The output of the digital filter.
    zi : ndarray, optional
        If `zi` is None, this is not returned. If `zi` is given or is ... then `zf`
        holds the final filter delay values.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> sos = sp.iirfilter(4, 0.5, btype="low", output="sos")
    >>> xp.sosfilt(sos, db, dim='time')
    <xdas.Database (time: 300, distance: 401)>
    [[ 0.004668 -0.005968  0.007386 ... -0.0138    0.01271  -0.026618]
     [ 0.008372 -0.01222   0.022552 ... -0.041387  0.046667 -0.093521]
     [-0.008928  0.002764  0.012621 ... -0.032496  0.039645 -0.076117]
     ...
     [ 0.012576 -0.1661    0.196026 ...  0.048191 -0.014532 -0.033122]
     [-0.06294  -0.092234  0.316862 ...  0.045337 -0.139729  0.094086]
     [-0.035233  0.036613  0.044002 ... -0.053585 -0.121344  0.241415]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = db.get_axis_num(dim)
    across = int(axis == 0)
    if zi is ...:
        n_sections = sos.shape[0]
        shape = (n_sections,) + tuple(
            2 if index == axis else element for index, element in enumerate(db.shape)
        )
        zi = np.zeros(shape)
    if zi is None:
        func = parallelize((None, across), across, parallel)(sp.sosfilt)
        data = func(sos, db.values, axis, zi)
        return db.copy(data=data)
    else:
        func = parallelize(
            (None, across, None, across + 1), (across, across + 1), parallel
        )(sp.sosfilt)
        data, zf = func(sos, db.values, axis, zi)
        return db.copy(data=data), zf


@atomized
@splits
@collects
def sosfiltfilt(sos, db, dim="last", padtype="odd", padlen=None, parallel=None):
    """
    A forward-backward digital filter using cascaded second-order sections.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    db : Database
        The data to be filtered.
    dim : str, optional
        The dimension of `db` to which the filter is applied.
        Default is last.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `db` at both ends of
        `dim` before applying the filter.  This value must be less than
        ``db.sizes[do,] - 1``.  ``padlen=0`` implies no padding.
        The default value is::

            3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                        (sos[:, 5] == 0).sum()))

        The extra subtraction at the end attempts to compensate for poles
        and zeros at the origin (e.g. for odd-order filters) to yield
        equivalent estimates of `padlen` to those of `filtfilt` for
        second-order section filters built with `scipy.signal` functions.

    Returns
    -------
    Database
        The filtered output with the same coordinates as `db`.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> sos = sp.iirfilter(4, 0.5, btype="low", output="sos")
    >>> xp.sosfiltfilt(sos, db, dim='time')
    <xdas.Database (time: 300, distance: 401)>
    [[ 0.04968  -0.063651  0.078731 ... -0.146869  0.135149 -0.283111]
     [-0.01724   0.018588 -0.037267 ...  0.025092 -0.107095  0.127912]
     [-0.004291 -0.002956 -0.032369 ...  0.078337 -0.150316  0.155965]
     ...
     [-0.068308 -0.06057   0.165391 ... -0.115742 -0.005265  0.19878 ]
     [-0.014123  0.039773 -0.063194 ... -0.090854 -0.053728  0.206149]
     [ 0.122203  0.188674 -0.231442 ...  0.304675 -0.452161  0.041323]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = db.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize((None, across), across, parallel)(sp.sosfiltfilt)
    data = func(sos, db.values, axis, padtype, padlen)
    return db.copy(data=data)


@atomized
@splits
@collects
def decimate(db, q, n=None, ftype="iir", zero_phase=None, dim="last", parallel=None):
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

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = db.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.decimate)
    data = func(db.values, q, n, ftype, axis, zero_phase)
    return db[{dim: slice(None, None, q)}].copy(data=data)


@atomized
@splits
@collects
def integrate(db, midpoints=False, dim="last", parallel=None):
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

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = db.get_axis_num(dim)
    d = get_sampling_interval(db, dim)
    func = lambda x: np.cumsum(x, axis=axis) * d
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(db.values)
    out = db.copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


@atomized
@splits
@collects
def differentiate(db, midpoints=False, dim="last", parallel=None):
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

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = db.get_axis_num(dim)
    d = get_sampling_interval(db, dim)
    func = lambda x: np.diff(x, axis=axis) / d
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(db.values)
    out = db.isel({dim: slice(None, -1)}).copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


@atomized
@collects
def segment_mean_removal(db, limits, window="hann", dim="last"):  # TODO: parallelize
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


@atomized
@splits
@collects
def sliding_mean_removal(
    db, wlen, window="hann", pad_mode="reflect", dim="last", parallel=None
):
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

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = db.get_axis_num(dim)
    d = get_sampling_interval(db, dim)
    n = round(wlen / d)
    if n % 2 == 0:
        n += 1
    win = sp.get_window(window, n)
    win /= np.sum(win)
    shape = tuple(-1 if a == axis else 1 for a in range(db.ndim))
    win = np.reshape(win, shape)
    pad_width = tuple((n // 2, n // 2) if a == axis else (0, 0) for a in range(db.ndim))
    func = lambda x: x - sp.fftconvolve(
        np.pad(x, pad_width, mode=pad_mode), win, mode="valid"
    )
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(db.values)
    return db.copy(data=data)


@atomized
@collects
def medfilt(db, kernel_dim):  # TODO: parallelize
    """
    Perform a median filter along given dimensions

    Apply a median filter to the input using a local window-size given by kernel_size.
    The array will automatically be zero-padded.

    Parameters
    ----------
    db : Database
        A database to filter.
    kernel_dim : dict
        A dictionary which keys are the dimensions over which to apply a median
        filtering and which values are the related kernel size in that direction.
        All values must be odd. If not all dims are provided, missing dimensions
        are associated to 1, i.e. no median filtering along that direction.
        At least one dimension must be passed.

    Returns
    -------
    Database
        The median filtered data.

    Examples
    --------
    A median filter is applied to some synthetic database with a median window size
    of 7 along the time dimension and 5 along the space dimension.

    >>> import xdas.scipy.signal as xp
    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> xp.medfilt(db, {"time": 7, "distance": 5})
    <xdas.Database (time: 300, distance: 401)>
    [[ 0.        0.        0.       ...  0.        0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]
     ...
     [ 0.        0.        0.       ... -0.000402  0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    if not all(dim in db.dims for dim in kernel_dim.keys()):
        raise ValueError("dims provided not in database")
    kernel_size = tuple(kernel_dim[dim] if dim in kernel_dim else 1 for dim in db.dims)
    data = sp.medfilt(db.values, kernel_size)
    return db.copy(data=data)


@atomized
@collects
def fft(db, n=None, dim={"last": "frequency"}, norm=None, parallel=None):
    ((olddim, newdim),) = dim.items()
    olddim = db.dims[db.get_axis_num(olddim)]
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    f = np.fft.fftshift(np.fft.fftfreq(n, d))
    func = lambda x: np.fft.fftshift(np.fft.fft(x, n, axis, norm), axis)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(db.values)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)


@atomized
@collects
def rfft(db, n=None, dim={"last": "frequency"}, norm=None, parallel=None):
    ((olddim, newdim),) = dim.items()
    olddim = db.dims[db.get_axis_num(olddim)]
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(np.fft.rfft)
    f = np.fft.rfftfreq(n, d)
    data = func(db.values, n, axis, norm)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)
