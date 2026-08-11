"""
Task atoms: the public processing vocabulary with physical parameters only.

Every public parameter keeps its meaning when the sampling rate changes:
frequencies are in Hz, window lengths in seconds (or meters along distance).
Machine parameters (coefficients, factors, taps) live in the kernel layer
(:mod:`xdas.atoms.kernel`) and are designed here from the data at the first
call.

Each task atom has a function form generated with
:func:`~xdas.atoms.core.as_function`: ``decimate(da, 50.0)`` applies eagerly,
``decimate(..., 50.0)`` returns the atom, and passing an atom extends a
pipeline. Stateless operations are plain ``@atomized`` functions, which behave
identically; the split between functions and classes is invisible to users.
"""

import numpy as np

from ..coordinates import get_sampling_interval
from ..core import concat
from .core import Atom, State, _whole_record, atomized
from .signal import FIRFilter, IIRFilter, ResamplePoly

__all__ = [
    "Decimate",
    "Differentiate",
    "Filter",
    "Integrate",
    "Resample",
    "decimate",
    "detrend",
    "differentiate",
    "filter",
    "hilbert",
    "integrate",
    "medfilt",
    "resample",
    "sliding_mean_removal",
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
    Filter(freq=(None, 10.0), ftype=fir, order=4, zerophase=False, dim=time, btype=lowpass, cutoff=10.0)
      FIRFilter(numtaps=Ellipsis, cutoff=10.0, btype=lowpass, window=hamming, scale=True, up=1, down=1, dim=time)
        Polyphase(taps=Ellipsis, up=1, down=1, dim=time)

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


class Decimate(Atom):
    """
    Decimate to a target sampling rate by an integer factor.

    Composite atom: a lowpass anti-alias FIR filter (group delay compensated
    on the coordinate) followed by integer downsampling. The current sampling
    rate must be an integer multiple of `target`; for rational ratios use
    :class:`Resample`.

    Parameters
    ----------
    target : float
        The target sampling rate in Hz (or in 1/m along distance).
    window : str or tuple
        The window used to design the anti-alias filter, compatible with
        ``scipy.signal.get_window``. Default is ``("kaiser", 5.0)``.
    dim : str
        The dimension along which to decimate. Default is "time".

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()  # 50 Hz
    >>> xd.decimate(da, 25.0).sizes["time"]
    150

    """

    def __init__(self, target, window=("kaiser", 5.0), dim="time"):
        super().__init__()
        self.target = target
        self.window = window
        self.dim = dim
        self.antialias = FIRFilter(..., ..., "lowpass", self.window, dim=self.dim)
        self.fs = State(...)

    def initialize(self, da, **flags):
        """Measure the sampling rate and design the anti-alias filter."""
        self.fs = State(1.0 / get_sampling_interval(da, self.dim))
        self.initialize_from_state()

    def initialize_from_state(self):
        """Derive the integer factor and anti-alias design from the rate."""
        factor = round(self.fs / self.target)
        if factor < 1 or abs(self.fs / self.target - factor) > 1e-6 * factor:
            raise ValueError(
                f"the sampling rate ({self.fs:g}) is not an integer multiple "
                f"of the target ({self.target:g}); use Resample for rational "
                "ratios"
            )
        self.antialias.numtaps = 20 * factor + 1
        self.antialias.cutoff = self.target / 2
        self.antialias.down = factor

    def call(self, da, **flags):
        """Anti-alias filter and downsample in one polyphase pass."""
        if self.antialias.down == 1:
            return da
        return self.antialias(da, **flags)


class Resample(ResamplePoly):
    """
    Resample to any target sampling rate by polyphase filtering.

    Task-layer name for :class:`~xdas.atoms.ResamplePoly`: the data is
    upsampled, lowpass FIR filtered and downsampled so that the ratio of the
    factors matches `target` over the current sampling rate.

    Parameters
    ----------
    target : float
        The target sampling rate in Hz (or in 1/m along distance).
    maxfactor : int
        Limit on the intermediate upsampling factor, to avoid accidental
        memory overflow. Default is 100.
    window : str or tuple
        The window used to design the FIR filter, compatible with
        ``scipy.signal.get_window``. Default is ``("kaiser", 5.0)``.
    dim : str
        The dimension along which to resample. Default is "time".

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()  # 50 Hz
    >>> xd.resample(da, 20.0).sizes["time"]
    120

    """

    def __init__(self, target, maxfactor=100, window=("kaiser", 5.0), dim="time"):
        super().__init__(target, maxfactor=maxfactor, window=window, dim=dim)


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


filter = atomized(Filter)
decimate = atomized(Decimate)
resample = atomized(Resample)
integrate = atomized(Integrate)
differentiate = atomized(Differentiate)
