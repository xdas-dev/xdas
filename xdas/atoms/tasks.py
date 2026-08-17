"""
Task atoms: the public processing vocabulary with physical parameters only.

Every public parameter keeps its meaning when the sampling rate changes:
frequencies are in Hz, window lengths in seconds (or meters along distance).
Machine parameters (coefficients, factors, taps) live in the kernel layer
(:mod:`xdas.atoms.kernel`) and are designed here from the data at the first
call.

Each task atom has a function form generated with
:func:`~xdas.atoms.core.as_function`: ``resample(da, 50.0)`` applies eagerly,
``resample(..., 50.0)`` returns the atom, and passing an atom extends a
pipeline. Stateless operations are plain ``@atomized`` functions, which behave
identically; the split between functions and classes is invisible to users.
"""

import datetime as dt
import math
import warnings
from fractions import Fraction

import numpy as np
import scipy.fft
import scipy.signal as sp
from scipy.signal import ShortTimeFFT, get_window

from ..coordinates import get_sampling_interval
from ..core import DataArray, concat
from ..parallel import parallelize
from .core import Atom, Sequential, State, _whole_record, atomized
from .kernel import DownSample, SOSFilter, UpSample
from .signal import FIRFilter, IIRFilter

__all__ = [
    "STFT",
    "Differentiate",
    "Filter",
    "Integrate",
    "Resample",
    "ResamplePoly",
    "detrend",
    "differentiate",
    "filter",
    "hilbert",
    "integrate",
    "medfilt",
    "resample",
    "sliding_mean_removal",
    "stft",
    "taper",
]


class Filter(Atom):
    """
    Bandpass, lowpass or highpass filter with corner frequencies in Hz.

    The band is given as a pair of corner frequencies ``(low, high)`` in Hz,
    with ``None`` opening one end: ``(1.0, 10.0)`` is a bandpass, ``(1.0,
    None)`` a highpass and ``(None, 10.0)`` a lowpass.

    Parameters
    ----------
    freq : tuple of float or None
        The pair of corner frequencies (low, high) in Hz. Use None to open one
        end of the band.
    ftype : {"iir", "fir"}
        The filter implementation. "iir" designs a Butterworth filter applied
        in second-order sections; it is causal and streams chunk by chunk.
        "fir" designs a windowed-sinc linear-phase filter whose group delay is
        compensated on the coordinate, making it effectively zero-phase while
        remaining streamable.
    order : int
        The order of the IIR filter. Ignored for FIR filters, whose length is
        set by `transition`. Default is 4.
    transition : float, optional
        The FIR transition bandwidth in Hz. Default is 10% of the lowest given
        corner frequency. Ignored for IIR filters.
    zerophase : bool
        If True with an IIR filter, the filter is applied forwards and
        backwards, doubling the effective order and cancelling the phase
        shift. Exact zero-phase IIR filtering has no causal streaming form, so
        such an atom refuses chunked execution along its dimension; use
        ``ftype="fir"`` for a streamable (effectively) zero-phase filter. FIR
        filters ignore this parameter as they are always compensated.
    dim : str
        The dimension along which to filter. Default is "time".

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()
    >>> filtered = xd.filter(da, (1.0, 10.0))
    >>> atom = xd.filter(..., (None, 10.0), ftype="fir")
    >>> atom
    Filter(freq=(None, 10.0), ftype='fir')

    """

    def __init__(
        self, freq, ftype="iir", order=4, transition=None, zerophase=False, dim="time"
    ):
        super().__init__()
        try:
            low, high = freq
        except (TypeError, ValueError):
            raise TypeError(
                "`freq` must be a pair of corner frequencies, using None to "
                "open one end, e.g. (1.0, None) for a highpass"
            ) from None
        if low is None and high is None:
            raise ValueError("at least one corner frequency must be given")
        if ftype not in ("iir", "fir"):
            raise ValueError("`ftype` must be either 'iir' or 'fir'")
        self.freq = (low, high)
        self.ftype = ftype
        self.order = order
        self.transition = transition
        self.zerophase = zerophase
        self.dim = dim
        if low is None:
            self.btype = "lowpass"
            self.cutoff = high
        elif high is None:
            self.btype = "highpass"
            self.cutoff = low
        else:
            self.btype = "bandpass"
            self.cutoff = (low, high)
        if ftype == "fir":
            self.filter = FIRFilter(..., self.cutoff, self.btype, dim=self.dim)
            self.fs = State(...)
        elif not zerophase:
            self.filter = IIRFilter(self.order, self.cutoff, self.btype, dim=self.dim)

    def _check_chunk_dim(self, x, chunk_dim):
        """Zero-phase IIR has no causal streaming form: whole-record only."""
        if self.ftype == "iir" and self.zerophase:
            self._refuse_chunked_along(self.dim, chunk_dim, x)

    def initialize(self, da, **flags):
        """Measure the sampling rate to size the FIR filter from `transition`."""
        if self.ftype == "fir":
            self.fs = State(1.0 / get_sampling_interval(da, self.dim))
            self.initialize_from_state()

    def initialize_from_state(self):
        """Derive the FIR length from the transition bandwidth."""
        if self.ftype == "fir":
            if self.transition is None:
                transition = 0.1 * min(f for f in self.freq if f is not None)
            else:
                transition = self.transition
            numtaps = int(np.ceil(3.3 * self.fs / transition))
            self.filter.numtaps = numtaps // 2 * 2 + 1

    def call(self, da, **flags):
        """Apply the filter, delegating to the designed child atom."""
        if self.ftype == "iir" and self.zerophase:
            from ..signal import filter

            return filter(
                da,
                self.cutoff,
                self.btype,
                corners=self.order,
                zerophase=True,
                dim=self.dim,
            )
        return self.filter(da, **flags)


def _round_half_up(x):
    """Round *x* to the nearest integer, ties rounding up rather than to even."""
    return math.floor(x + 0.5)


def _snap_factors(ratio):
    """
    Return the (up, down) nesting the output grid inside or around the input's.

    ``ratio`` is the requested ``up/down`` (target rate over current rate). A
    nested grid is a plain multiple of the input's own sampling interval, so
    the choice is a rounding, not a search: ``down = round(1/ratio)`` for a
    coarser target, or symmetrically ``up = round(ratio)`` when the request is
    finer than the input (the input grid nests inside the output's instead).
    """
    f = 1 / ratio
    if f >= 1:
        return 1, max(1, _round_half_up(f))
    return max(1, _round_half_up(1 / f)), 1


def _solve_ratio(ratio, tolerance, maxup, maxdown):
    """
    Return the simplest ``(up, down)`` approximating *ratio* within *tolerance*.

    One candidate per numerator ``up`` in ``1..maxup``, with
    ``down = max(1, round(up / ratio))`` and candidates whose ``down`` exceeds
    ``maxdown`` (when given) dropped; the simplest kept candidate — the
    smallest ``(up, down)`` pair — wins. Raises :class:`ValueError` when none
    is within tolerance, with the closest candidate attached as ``.closest =
    (up, down, deviation)`` for a caller to build a richer message from.

    Parameters
    ----------
    ratio : float
        The requested ``up/down`` ratio (target rate over current rate).
        Must be positive.
    tolerance : float
        Maximum relative deviation from *ratio* a candidate may have.
    maxup : int
        Largest numerator to try.
    maxdown : int or None
        Largest denominator to accept, or ``None`` for no cap.

    Returns
    -------
    (int, int)
        The simplest ``(up, down)`` within tolerance.
    """
    if not ratio > 0:
        raise ValueError(f"the requested ratio must be positive, got {ratio:g}")
    best = None
    closest = None
    for up in range(1, maxup + 1):
        down = max(1, round(up / ratio))
        if maxdown is not None and down > maxdown:
            continue
        deviation = abs(up / down - ratio)
        if closest is None or deviation < closest[2]:
            closest = (up, down, deviation)
        if deviation <= tolerance * ratio and (best is None or (up, down) < best):
            best = (up, down)
    if best is not None:
        return best
    if closest is None:
        raise ValueError(
            f"no ratio with down <= maxdown ({maxdown}) is reachable within "
            f"maxup ({maxup}); raise `maxup` or `maxdown`"
        )
    up, down, deviation = closest
    error = ValueError(
        f"cannot reach the requested ratio ({ratio:.6g}) within "
        f"tolerance={tolerance:g}: the closest simple ratio is up={up}, "
        f"down={down} ({up / down:.6g}, {100 * (up / down - ratio) / ratio:+.2g}% "
        f"deviation); raise `maxup` (currently {maxup}) to search a more "
        'complex ratio, loosen `tolerance`, or use `snap=True` / `method="fft"`'
    )
    error.closest = (up, down, deviation)
    raise error


def _as_seconds(delta):
    """Return *delta* as a plain float in seconds, or unchanged if already one."""
    if isinstance(delta, np.timedelta64):
        return delta / np.timedelta64(1, "s")
    return float(delta)


def _target_ratio(rate, interval, up, down, delta):
    """
    Resolve which of the three target spellings was given, into a ratio.

    Exactly one of `rate`, `interval` or `up`/`down` must be given.

    Parameters
    ----------
    rate : float or None
        Target sampling rate, in the reciprocal of *delta*'s units.
    interval : float, numpy.timedelta64, datetime.timedelta or None
        Target sampling interval, in *delta*'s own units, or a typed
        duration when *delta* is a `numpy.timedelta64` (datetime dimension).
    up, down : int or None
        The exact rational ratio, machine spelling. Either may be given
        alone, the other then defaulting to 1.
    delta : float or numpy.timedelta64
        The measured sampling interval of the data, as returned by
        :func:`~xdas.get_sampling_interval` with ``cast=False``.

    Returns
    -------
    ratio : float
        The requested ``up/down`` (target rate over current rate).
    spelling : {"rate", "interval", "factor"}
        Which spelling was used, for phrasing error messages.
    up, down : int or None
        The exact pair when ``spelling == "factor"``, else ``None``.
    """
    named = [
        name
        for name, given in (
            ("rate", rate is not None),
            ("interval", interval is not None),
            ("up/down", up is not None or down is not None),
        )
        if given
    ]
    if not named:
        raise TypeError(
            "one of `rate`, `interval` or `up`/`down` must be given to name "
            "the resampling target"
        )
    if len(named) > 1:
        raise TypeError(
            "only one of `rate`, `interval` or `up`/`down` may be given to "
            f"name the resampling target, got {' and '.join(named)}"
        )
    if up is not None or down is not None:
        up = 1 if up is None else up
        down = 1 if down is None else down
        if not (isinstance(up, (int, np.integer)) and not isinstance(up, bool)):
            raise ValueError(f"`up` must be an integer, got {up!r}")
        if not (isinstance(down, (int, np.integer)) and not isinstance(down, bool)):
            raise ValueError(f"`down` must be an integer, got {down!r}")
        if not up > 0:
            raise ValueError(f"`up` must be positive, got {up!r}")
        if not down > 0:
            raise ValueError(f"`down` must be positive, got {down!r}")
        return up / down, "factor", int(up), int(down)
    if rate is not None:
        if not rate > 0:
            raise ValueError(f"`rate` must be positive, got {rate!r}")
        return _as_seconds(delta) * rate, "rate", None, None
    if isinstance(interval, (np.timedelta64, dt.timedelta)):
        if not isinstance(delta, np.timedelta64):
            raise ValueError(
                "`interval` was given as a timedelta, which only makes sense "
                "on a datetime coordinate"
            )
        target = np.timedelta64(interval).astype(delta.dtype)
        target_ticks = int(target.astype(np.int64))
        if not target_ticks > 0:
            raise ValueError("`interval` must be positive")
        delta_ticks = int(delta.astype(np.int64))
        return float(Fraction(delta_ticks, target_ticks)), "interval", None, None
    if not interval > 0:
        raise ValueError(f"`interval` must be positive, got {interval!r}")
    return _as_seconds(delta) / interval, "interval", None, None


def _edge_resample(da, num, dim, window, edge):
    """
    Resample *da* to *num* samples along *dim*, treating the record's edges.

    ``"none"`` is plain :func:`xdas.signal.resample`, assuming periodicity.
    ``"linear"`` removes the straight line through the first and last sample
    before resampling and restores it, evaluated on the output grid, after.
    ``"mirror"`` resamples the even extension ``[x, x[::-1]]`` and keeps the
    first half, which alone removes the periodic-boundary discontinuity
    outright (the extension is continuous *and* periodic).
    """
    from ..signal import resample as _resample

    if edge not in ("mirror", "linear", "none"):
        raise ValueError(f"`edge` must be 'mirror', 'linear' or 'none', got {edge!r}")
    axis = da.get_axis_num(dim)
    n = da.sizes[dim]
    if edge == "none" or n < 2:
        return _resample(da, num, dim=dim, window=window)
    if edge == "linear":
        shape = [1] * da.ndim
        shape[axis] = n
        t = np.arange(n).reshape(shape)
        first = da.isel({dim: slice(0, 1)}).values
        last = da.isel({dim: slice(n - 1, n)}).values
        trend = first + (last - first) * (t / (n - 1))
        detrended = da.copy(data=da.values - trend)
        out = _resample(detrended, num, dim=dim, window=window)
        shape[axis] = num
        # Output sample k sits at the same normalised position along the
        # record as input sample k*(n-1)/(num-1): the trend is evaluated at
        # that fraction, not at the input index scale.
        t_out = np.arange(num).reshape(shape)
        fraction = t_out / (num - 1) if num > 1 else t_out
        out_trend = first + (last - first) * fraction
        return out.copy(data=out.values + out_trend)
    # mirror
    extended_values = np.concatenate(
        [da.values, np.flip(da.values, axis=axis)], axis=axis
    )
    coord = da.coords[dim]
    delta = get_sampling_interval(da, dim, cast=False)
    coords = da.coords.copy()
    coords[dim] = type(coord).from_block(coord.start, 2 * n, delta, dim=dim)
    extended = DataArray(extended_values, coords, da.dims, da.name, da.attrs)
    out = _resample(extended, 2 * num, dim=dim, window=window)
    return out.isel({dim: slice(0, num)})


class Resample(Atom):
    """
    Resample to any target sampling rate, by FIR, IIR or FFT filtering.

    The single entry point for what ``scipy.signal.decimate``,
    ``resample_poly`` and ``resample`` do, chosen by `method`. The target
    grid is named in exactly one of three ways: a rate, a sampling interval,
    or a plain rational factor.

    Parameters
    ----------
    rate : float, optional
        Target sampling rate, in Hz (or 1/m along distance).
    interval : float, numpy.timedelta64 or datetime.timedelta, optional
        Target sampling interval, in the coordinate's own units, or a typed
        duration on a datetime coordinate.
    up, down : int, optional
        The exact rational ratio, the machine spelling. Either may be given
        alone; ``down=16`` alone is integer decimation.
    method : {"fir", "iir", "fft"}
        The resampling implementation. "fir" (default) is a polyphase
        lowpass FIR filter, chunkable and exact. "iir" is a Chebyshev I
        filter applied at the upsampled rate, chunkable when
        ``zerophase=False``. "fft" resamples in the Fourier domain, landing
        within half an output sample of any target; it needs the whole
        record along `dim` (chunking along another dimension is fine) and
        refuses `snap`.
    snap : bool
        Stay on the original sample grid: every output sample sits at the
        position of an input sample, nested at the ratio closest to what was
        asked (``round(interval / delta)``). Refused by ``method="fft"``.
        Default is False.
    maxup : int
        Cap on the upsampling the solver may invent when searching for a
        simple ratio within `tolerance`. Default is 10.
    maxdown : int, optional
        Cap on the downsampling the solver may reach. Uncapped by default,
        since ``up=1`` (pure decimation) reaches any depth exactly.
    tolerance : float, optional
        Deviation accepted, relative to the requested ratio, when selecting
        among candidate ratios. Default is ``1e-5`` for "fir"/"iir"; for
        "fft" no default applies (its own quantisation floor is coarser), an
        explicitly passed value still checks the achieved rate.
    window : str or tuple, optional
        "fir": the FIR design window, compatible with
        ``scipy.signal.get_window``; default ``("kaiser", 5.0)``. "fft": the
        spectral window passed to ``scipy.signal.resample``; default is no
        window. Not used by "iir".
    numtaps : int, optional
        "fir" only: override the ``20 * max(up, down) + 1`` taps rule.
    order : int, optional
        "iir" only: the Chebyshev I filter order. Default is 8.
    zerophase : bool
        "iir" only: apply the filter forwards and backwards
        (``sosfiltfilt``), whole-record. Default is False (causal, streamable).
    edge : {"mirror", "linear", "none"}, optional
        "fft" only: how the record's edges are treated before the Fourier
        resampling, which otherwise assumes periodicity. Default "mirror".
    dim : str
        The dimension along which to resample. Default is "time".

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()  # 50 Hz
    >>> xd.resample(da, 20.0).sizes["time"]
    120
    >>> xd.resample(da, down=2, dim="time").sizes["time"]
    150

    """

    def __init__(
        self,
        rate=None,
        interval=None,
        up=None,
        down=None,
        method="fir",
        snap=False,
        maxup=10,
        maxdown=None,
        tolerance=None,
        window=None,
        numtaps=None,
        order=None,
        zerophase=False,
        edge=None,
        dim="time",
    ):
        super().__init__()
        if method not in ("fir", "iir", "fft"):
            raise ValueError(f"`method` must be 'fir', 'iir' or 'fft', got {method!r}")
        if numtaps is not None and method != "fir":
            raise ValueError("`numtaps` only applies to `method='fir'`")
        if order is not None and method != "iir":
            raise ValueError("`order` only applies to `method='iir'`")
        if zerophase and method != "iir":
            raise ValueError("`zerophase` only applies to `method='iir'`")
        if edge is not None and method != "fft":
            raise ValueError("`edge` only applies to `method='fft'`")
        if window is not None and method == "iir":
            raise ValueError("`window` does not apply to `method='iir'`")
        if snap and method == "fft":
            raise ValueError(
                '`snap` is not supported by `method="fft"`: it has no way to '
                "stay on the original sample grid"
            )
        if snap and (up is not None or down is not None):
            raise ValueError(
                "`snap` cannot be combined with an explicit `up`/`down` ratio"
            )
        self.rate = rate
        self.interval = interval
        self.up = up
        self.down = down
        self.method = method
        self.snap = snap
        self.maxup = maxup
        self.maxdown = maxdown
        self.tolerance = tolerance
        self.window = window
        self.numtaps = numtaps
        self.order = order
        self.zerophase = zerophase
        self.edge = edge
        self.dim = dim
        if method == "fir":
            fir_window = ("kaiser", 5.0) if window is None else window
            self.child = FIRFilter(..., ..., "lowpass", fir_window, dim=dim)
        elif method == "iir":
            self.child = Sequential(
                [
                    UpSample(1, dim=dim),
                    SOSFilter(..., dim=dim),
                    DownSample(1, dim=dim),
                ]
            )
        if method != "fft":
            self.delta = State(...)

    def _check_chunk_dim(self, x, chunk_dim):
        """Refuse chunking along `dim` for "fft" and zero-phase "iir"."""
        if self.method == "fft" or (self.method == "iir" and self.zerophase):
            self._refuse_chunked_along(self.dim, chunk_dim, x)

    def initialize(self, da, **flags):
        """Measure the sampling interval; "fft" designs nothing upfront."""
        if self.method == "fft":
            return
        self.delta = State(get_sampling_interval(da, self.dim, cast=False))
        self.initialize_from_state()

    def initialize_from_state(self):
        """Solve the ratio and design the FIR taps or IIR SOS from `delta`."""
        if self.method == "fft":
            return
        ratio, spelling, up, down = _target_ratio(
            self.rate, self.interval, self.up, self.down, self.delta
        )
        if spelling == "factor":
            pass
        elif self.snap:
            up, down = _snap_factors(ratio)
        else:
            tolerance = 1e-5 if self.tolerance is None else self.tolerance
            try:
                up, down = _solve_ratio(ratio, tolerance, self.maxup, self.maxdown)
            except ValueError as error:
                raise self._ratio_error(ratio, tolerance, error) from None
        self.up_ = up
        self.down_ = down
        if self.method == "fir":
            fs = 1 / _as_seconds(self.delta)
            numtaps = 20 * max(up, down) + 1 if self.numtaps is None else self.numtaps
            if self.numtaps is None and down > 100:
                warnings.warn(
                    f"resampling by down={down} designs a {numtaps}-tap FIR "
                    'filter; consider method="iir" (flat cost in `down`), '
                    "explicit `numtaps`, or two passes",
                    UserWarning,
                    stacklevel=3,
                )
            cutoff = min(fs * up / down, fs) / 2
            self.child.numtaps = numtaps
            self.child.cutoff = cutoff
            self.child.up = up
            self.child.down = down
        else:  # iir
            order = 8 if self.order is None else self.order
            sos = sp.cheby1(
                order, 0.05, 0.8 / max(up, down), btype="lowpass", output="sos"
            )
            self.sos_ = sos
            self.child[0].factor = up
            self.child[1].sos = sos
            self.child[2].factor = down

    def _ratio_error(self, ratio, tolerance, cause):
        """Wrap a solver failure with the ways out available at this call."""
        # Only reached from the non-`snap` branch of `initialize_from_state`
        # (`snap` resolves through `_snap_factors`, which never raises).
        closest = getattr(cause, "closest", None)
        if closest is None:
            return cause
        _, _, deviation = closest
        snap_up, snap_down = _snap_factors(ratio)
        ways_out = [
            (
                f"snap=True (up={snap_up}, down={snap_down}, stays on the "
                "original sample grid)"
            )
        ]
        ways_out.append(
            f"a looser `tolerance` (>= {100 * deviation / ratio:.2g}%) or a "
            f"larger `maxup` (> {self.maxup})"
        )
        ways_out.append('method="fft", which lands within half an output sample')
        return ValueError(f"{cause} Ways out: " + "; ".join(ways_out))

    def flush(self):
        """Drain the causal iir child's buffered tail; fir and fft never buffer."""
        if self.method == "iir":
            return self.child.flush()
        return []

    def call(self, da, **flags):
        """Apply the designed FIR/IIR pipeline, or resample in place for FFT."""
        if self.method == "fft":
            return self._call_fft(da)
        if self.up_ == 1 and self.down_ == 1:
            return da
        if self.method == "iir" and self.zerophase:
            from ..signal import sosfiltfilt

            up, down = self.up_, self.down_
            if up > 1:
                da = UpSample(up, dim=self.dim)(da)
            da = sosfiltfilt(self.sos_, da, dim=self.dim)
            if down > 1:
                da = DownSample(down, dim=self.dim)(da)
            return da
        return self.child(da, **flags)

    def _call_fft(self, da):
        """Design and apply the Fourier resampling for this record alone."""
        delta = get_sampling_interval(da, self.dim, cast=False)
        ratio, spelling, up, down = _target_ratio(
            self.rate, self.interval, self.up, self.down, delta
        )
        if spelling == "factor":
            ratio = up / down
        if ratio == 1:
            return da
        n = da.sizes[self.dim]
        num = max(1, round(n * ratio))
        if self.tolerance is not None:
            achieved = num / n
            deviation = abs(achieved - ratio) / ratio
            if deviation > self.tolerance:
                floor = 1 / (2 * num)
                raise ValueError(
                    f'method="fft" cannot reach the requested ratio '
                    f"({ratio:.6g}) within tolerance={self.tolerance:g} at "
                    f"n={n} samples: achieved {achieved:.6g} "
                    f"({100 * deviation:+.2g}% deviation); its own "
                    f"quantisation floor here is {floor:.2g}"
                )
        edge = "mirror" if self.edge is None else self.edge
        return _edge_resample(da, num, self.dim, self.window, edge)


class ResamplePoly(Resample):
    """
    Deprecated alias of ``Resample(method="fir")``, removed in 0.3.

    Parameters
    ----------
    target : float
        The target sampling rate of the new data.
    maxfactor : int
        Limit the initial upsampling by this factor, to avoid accidental
        memory overflow. Default: 100.
    window : str or tuple of string and parameter values
        The window function to apply before FIR filtering. Default:
        ``("kaiser", 5.0)``.
    dim : str or int
        The dimension along which the downsampling is applied. Default:
        ``last``.
    """

    def __init__(self, target, maxfactor=100, window=("kaiser", 5.0), dim="last"):
        warnings.warn(
            "ResamplePoly is deprecated and will be removed in 0.3, use "
            "Resample (or xdas.resample) instead",
            DeprecationWarning,
            stacklevel=3,
        )
        super().__init__(
            rate=target, method="fir", maxup=maxfactor, window=window, dim=dim
        )
        self.target = target
        self.maxfactor = maxfactor


class Integrate(Atom):
    """
    Integrate cumulatively along a dimension.

    Stateful: when processing chunk by chunk, the cumulative sum continues
    across chunks.

    Parameters
    ----------
    midpoints : bool
        Whether to move the coordinates by half a step. Default is False.
    dim : str
        The dimension along which to integrate. Default is "time".

    """

    def __init__(self, midpoints=False, dim="time"):
        super().__init__()
        self.midpoints = midpoints
        self.dim = dim
        self.carry = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        """Allocate the carried cumulative offset for chunked operation."""
        if chunk_dim == self.dim:
            axis = da.get_axis_num(self.dim)
            shape = tuple(
                1 if index == axis else size for index, size in enumerate(da.shape)
            )
            self.carry = State(np.zeros(shape))
        else:
            self.carry = State(None)

    def call(self, da, **flags):
        """Integrate the chunk and offset it by the carried cumulative sum."""
        from ..signal import integrate

        out = integrate(da, midpoints=self.midpoints, dim=self.dim)
        if self.carry is not None:
            axis = out.get_axis_num(self.dim)
            out = out.copy(data=out.values + self.carry)
            index = tuple(
                slice(-1, None) if a == axis else slice(None) for a in range(out.ndim)
            )
            self.carry = State(out.values[index])
        return out


class Differentiate(Atom):
    """
    Differentiate along a dimension.

    Stateful: when processing chunk by chunk, the last sample of each chunk is
    carried over so the difference across the seam is not lost. The output has
    one sample less than the input in total.

    Parameters
    ----------
    midpoints : bool
        Whether to move the coordinates by half a step. Default is False.
    dim : str
        The dimension along which to differentiate. Default is "time".

    """

    def __init__(self, midpoints=False, dim="time"):
        super().__init__()
        self.midpoints = midpoints
        self.dim = dim
        self.buffer = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        """Initialise the one-sample carry-over buffer for chunked operation."""
        if chunk_dim == self.dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        """Differentiate the chunk, prepending the buffered last sample."""
        from ..signal import differentiate

        if self.buffer is not None:
            x = concat([self.buffer, da], self.dim)
            self.buffer = State(da.isel({self.dim: slice(-1, None)}))
        else:
            x = da
        return differentiate(x, midpoints=self.midpoints, dim=self.dim)


@atomized
@_whole_record()
def detrend(da, type="linear", dim="time", parallel=None):
    """
    Remove a trend along the given dimension.

    Whole-record operation: the trend is fitted on the full record, so this
    atom refuses chunked execution along its dimension.

    Parameters
    ----------
    da : DataArray
        The data to detrend.
    type : str
        Either "linear" or "constant". Default is "linear".
    dim : str
        The dimension along which to detrend. Default is "time".
    parallel : bool or int, optional
        Number of threads to use.

    Returns
    -------
    DataArray
        The detrended data.

    """
    from ..signal import detrend

    return detrend(da, type, dim=dim, parallel=parallel)


@atomized
@_whole_record()
def taper(da, window="hann", fftbins=False, dim="time", parallel=None):
    """
    Apply a tapering window along the given dimension.

    Whole-record operation: the window spans the full record, so this atom
    refuses chunked execution along its dimension.

    Parameters
    ----------
    da : DataArray
        The data to taper.
    window : str or tuple, optional
        The window to use, by default "hann".
    fftbins : bool, optional
        Whether to use a periodic windowing, by default False.
    dim : str, optional
        The dimension along which to taper. Default is "time".
    parallel : bool or int, optional
        Number of threads to use.

    Returns
    -------
    DataArray
        The tapered data.

    """
    from ..signal import taper

    return taper(da, window=window, fftbins=fftbins, dim=dim, parallel=parallel)


@atomized
@_whole_record()
def hilbert(da, dim="time", parallel=None):
    """
    Compute the analytic signal, using the Hilbert transform.

    Whole-record operation: the transform is acausal, so this atom refuses
    chunked execution along its dimension.

    Parameters
    ----------
    da : DataArray
        Signal data. Must be real.
    dim : str, optional
        The dimension along which to transform. Default is "time".
    parallel : bool or int, optional
        Number of threads to use.

    Returns
    -------
    DataArray
        Analytic signal of `da` along `dim`.

    """
    from ..signal import hilbert

    return hilbert(da, dim=dim, parallel=parallel)


@atomized
@_whole_record()
def sliding_mean_removal(
    da, wlen, window="hann", pad_mode="reflect", dim="time", parallel=None
):
    """
    Remove a sliding mean.

    The window length is physical: seconds along time, meters along distance.
    Pending overlap-aware execution, this atom refuses chunked execution along
    its dimension.

    Parameters
    ----------
    da : DataArray
        The data that the sliding mean should be removed from.
    wlen : float
        Length of the sliding mean, in the units of the `dim` coordinate.
    window : str, optional
        Tapering window used, by default "hann".
    pad_mode : str, optional
        Padding mode used, by default "reflect".
    dim : str, optional
        The dimension along which to remove the sliding mean. Default is
        "time".
    parallel : bool or int, optional
        Number of threads to use.

    Returns
    -------
    DataArray
        The data with the sliding mean removed.

    """
    from ..signal import sliding_mean_removal

    return sliding_mean_removal(
        da, wlen, window=window, pad_mode=pad_mode, dim=dim, parallel=parallel
    )


@atomized
@_whole_record(dim_arg="kernel")
def medfilt(da, kernel):
    """
    Apply a median filter with kernel lengths in physical units.

    Parameters
    ----------
    da : DataArray
        The data to filter.
    kernel : dict
        Mapping of dimension name to kernel length in the units of that
        dimension's coordinate (seconds along time, meters along distance).
        Each length is converted to the nearest odd number of samples.
        Dimensions not listed are not filtered.

    Returns
    -------
    DataArray
        The median filtered data.

    """
    from ..signal import medfilt

    kernel_dim = {}
    for dim, length in kernel.items():
        size = max(1, round(length / get_sampling_interval(da, dim)))
        kernel_dim[dim] = size if size % 2 else size + 1
    return medfilt(da, kernel_dim)


class STFT(Atom):
    """
    Short-Time Fourier Transform with window length and hop in physical units.

    The window length is a target in the units of the `dim` coordinate
    (seconds along time): the actual length is the next fast FFT size of the
    corresponding number of samples, so transforms stay efficient whatever the
    sampling rate. Frames start at the first sample and advance by `hop`; only
    fully computable frames are ever emitted, so when processing chunk by
    chunk the unconsumed tail is buffered across chunks (and dropped at gaps
    and at the end of the stream), and chunked processing emits exactly the
    frames of the eager transform. The output gains a "frequency" dimension
    (one-sided for real data, centered two-sided for complex data) and the
    `dim` coordinate moves to the frame centers.

    Parameters
    ----------
    wlen : float
        Target window length, in the units of the `dim` coordinate (seconds
        along time). The actual length is ``scipy.fft.next_fast_len`` of the
        equivalent number of samples.
    hop : float, optional
        Step between frame starts, in the same units. Must be positive and at
        most `wlen`. Like the window length, it is snapped: to the nearest
        whole number of samples (at least one, at most the actual window
        length), so the frame grid can differ from the request. Default is
        half the actual window length.
    window : str or tuple
        The tapering window, compatible with ``scipy.signal.get_window``.
        Default is "hann".
    scaling : {"spectrum", "psd"}
        The scaling of the complex frames: "spectrum" preserves peak
        amplitudes ("magnitude" scaling of `scipy.signal.ShortTimeFFT`),
        "psd" makes the squared modulus a power spectral density. Default is
        "spectrum".
    nfft : int, optional
        Expert mode: the FFT length in samples, to zero pad the windowed
        frames. Must be at least the actual window length; a common choice is
        twice that length, and fast FFT sizes matter. Default is the actual
        window length (no padding).
    dim : str
        The dimension along which to transform. Default is "time".
    parallel : bool or int, optional
        Number of threads to use.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()  # 50 Hz
    >>> xd.stft(da, 2.0, hop=1.0).sizes
    {'time': 5, 'distance': 401, 'frequency': 51}

    """

    def __init__(
        self,
        wlen,
        hop=None,
        window="hann",
        scaling="spectrum",
        nfft=None,
        dim="time",
        parallel=None,
    ):
        super().__init__()
        if not wlen > 0:
            raise ValueError("`wlen` must be positive")
        if hop is not None and not 0 < hop <= wlen:
            raise ValueError("`hop` must be positive and at most `wlen`")
        if scaling not in ("spectrum", "psd"):
            raise ValueError("`scaling` must be 'spectrum' or 'psd'")
        self.wlen = wlen
        self.hop = hop
        self.window = window
        self.scaling = scaling
        self.nfft = nfft
        self.dim = dim
        self.parallel = parallel
        self.sft = State(...)
        self.buffer = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        """Design the transform from the measured sampling rate."""
        fs = 1.0 / get_sampling_interval(da, self.dim)
        nperseg = scipy.fft.next_fast_len(max(round(self.wlen * fs), 1))
        if chunk_dim != self.dim and da.sizes[self.dim] < nperseg:
            raise ValueError(
                f"the record is shorter along {self.dim!r} "
                f"({da.sizes[self.dim]} samples) than the window "
                f"({nperseg} samples)"
            )
        if self.hop is None:
            hop = max(nperseg // 2, 1)
        else:
            hop = min(max(round(self.hop * fs), 1), nperseg)
        nfft = nperseg if self.nfft is None else self.nfft
        if nfft < nperseg:
            raise ValueError(
                f"`nfft` ({nfft}) must be at least the window length in "
                f"samples ({nperseg})"
            )
        self.sft = State(
            ShortTimeFFT(
                get_window(self.window, nperseg),
                hop=hop,
                fs=fs,
                fft_mode="onesided" if np.isrealobj(da.values) else "centered",
                mfft=nfft,
                scale_to="magnitude" if self.scaling == "spectrum" else "psd",
                phase_shift=None,
            )
        )
        if chunk_dim == self.dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        """Emit every fully computable frame, buffering the unconsumed tail."""
        nperseg = self.sft.m_num
        hop = self.sft.hop
        if self.buffer is None:
            return self._transform(da)
        da = concat([self.buffer, da], self.dim)
        n = da.sizes[self.dim]
        if n < nperseg:
            self.buffer = State(da)
            return None
        nframes = (n - nperseg) // hop + 1
        consumed = nframes * hop
        out = da.isel({self.dim: slice(0, consumed - hop + nperseg)})
        self.buffer = State(da.isel({self.dim: slice(consumed, None)}))
        return self._transform(out)

    def flush(self):
        """Discard the buffered tail: only fully computable frames are emitted."""
        if isinstance(self.buffer, DataArray):
            self.buffer = State(self.buffer.isel({self.dim: slice(0, 0)}))
        return []

    def _transform(self, da):
        """Compute the windowed FFT of every full frame in *da*."""
        sft = self.sft
        axis = da.get_axis_num(self.dim)

        def func(x):
            frames = np.lib.stride_tricks.sliding_window_view(x, sft.m_num, axis=axis)
            slc = [slice(None)] * frames.ndim
            slc[axis] = slice(None, None, sft.hop)
            frames = sft.win * frames[tuple(slc)]
            if sft.onesided_fft:
                return scipy.fft.rfft(frames, n=sft.mfft, axis=-1)
            return scipy.fft.fftshift(
                scipy.fft.fft(frames, n=sft.mfft, axis=-1), axes=-1
            )

        across = int(axis == 0)
        func = parallelize(across, across, self.parallel)(func)
        data = func(da.values)

        coord_cls = type(da.coords[self.dim])
        dt = get_sampling_interval(da, self.dim, cast=False)
        t0 = da.coords[self.dim].values[0]
        time = coord_cls.from_block(
            t0 + (sft.m_num // 2) * dt, data.shape[axis], sft.hop * dt
        )
        freqs = coord_cls.from_block(sft.f[0], len(sft.f), sft.delta_f)
        coords = {}
        for name in da.coords:
            if name == self.dim:
                coords[self.dim] = time
            elif da[name].dim != self.dim:  # TODO: keep non-dimensional coordinates
                coords[name] = da.coords[name]
        coords["frequency"] = freqs
        return DataArray(data, coords, da.dims + ("frequency",), da.name, da.attrs)


filter = atomized(Filter)
resample = atomized(Resample)
integrate = atomized(Integrate)
differentiate = atomized(Differentiate)
stft = atomized(STFT)
