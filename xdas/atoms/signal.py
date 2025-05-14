from fractions import Fraction

import numpy as np
import scipy.signal as sp

from ..core.coordinates import Coordinate, get_sampling_interval
from ..core.dataarray import DataArray
from ..core.routines import concatenate, split
from ..parallel import parallelize
from .core import Atom, State


class ResamplePoly(Atom):
    """
    Pipeline implementation of polyphase-filter resampling from the
    original sampling rate to the ``target`` sampling rate.

    This is achieved by an upsampling of the data,
    followed by the application of a low-pass FIR filter,
    and finally by downsampling of the data. The ratio of the
    up and downsampling factors equals the target sampling rate over
    the original sampling rate.

    Parameters
    ----------

    target : float
        The target sampling rate of the new data
    maxfactor : int
        Limit the initial upsampling by this factor, to avoid
        accidental memory overflow. Default: 100
    window : str or tuple of string and parameter values
        The window function to apply befor FIR filtering. If a
        tuple is given, it needs to be compatible with ``scipy.signal.get_window``.
        Default: ``("kaiser", 5.0)``
    dim : str or int
        The dimension along which the downsampling is applied.
        This is either an index, ``time`` or ``distance``, or ``last``.
        Default: ``last``

    Examples
    --------
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> from xdas.atoms import Sequential, ResamplePoly
    >>> da = wavelet_wavefronts()

    Using ``ResamplePoly`` directly:

    >>> # Downsample time to 1 Hz
    >>> da2 = ResamplePoly(target=1., dim="time")(da)
    >>> da2["time"].values
    array(['2022-12-31T23:59:50.000000000', '2022-12-31T23:59:51.000000000',
        '2022-12-31T23:59:52.000000000', '2022-12-31T23:59:53.000000000',
        '2022-12-31T23:59:54.000000000', '2022-12-31T23:59:55.000000000'],
        dtype='datetime64[ns]')

    Using ``ResamplePoly`` as an atom in ``Sequential``:

    >>> # Downsample distance to 100m spacing
    >>> sequence = Sequential([
    ...    ResamplePoly(target=1/100., window=("tukey", 0.1), dim="distance")
    ... ])
    >>> result = sequence(da)
    >>> result["distance"].values
    array([-1000.,  -900.,  -800.,  -700.,  -600.,  -500.,  -400.,  -300.,
            -200.,  -100.,     0.,   100.,   200.,   300.,   400.,   500.,
            600.,   700.,   800.,   900.,  1000.,  1100.,  1200.,  1300.,
            1400.,  1500.,  1600.,  1700.,  1800.,  1900.,  2000.,  2100.,
            2200.,  2300.,  2400.,  2500.,  2600.,  2700.,  2800.,  2900.,
            3000.,  3100.,  3200.,  3300.,  3400.,  3500.,  3600.,  3700.,
            3800.,  3900.,  4000.,  4100.,  4200.,  4300.,  4400.,  4500.,
            4600.,  4700.,  4800.,  4900.,  5000.,  5100.,  5200.,  5300.,
            5400.,  5500.,  5600.,  5700.,  5800.,  5900.,  6000.,  6100.,
            6200.,  6300.,  6400.,  6500.,  6600.,  6700.,  6800.,  6900.,
            7000.,  7100.,  7200.,  7300.,  7400.,  7500.,  7600.,  7700.,
            7800.,  7900.,  8000.,  8100.,  8200.,  8300.,  8400.,  8500.,
            8600.,  8700.,  8800.,  8900.,  9000.])

    .. warning::
        The default ``dim`` value ``last`` does not work...
    """

    def __init__(self, target, maxfactor=100, window=("kaiser", 5.0), dim="last"):
        super().__init__()
        self.target = target
        self.maxfactor = maxfactor
        self.window = window
        self.dim = dim
        self.upsampling = UpSample(..., dim=self.dim)
        self.firfilter = FIRFilter(..., ..., "lowpass", self.window, dim=self.dim)
        self.downsampling = DownSample(..., self.dim)
        self.fs = State(...)

    def initialize(self, da, **flags):
        self.fs = State(1.0 / get_sampling_interval(da, self.dim))
        self.initialize_from_state()

    def initialize_from_state(self):
        fraction = Fraction(self.target / self.fs)
        fraction = fraction.limit_denominator(self.maxfactor)
        fraction = 1 / (1 / fraction).limit_denominator(self.maxfactor)
        up = fraction.numerator
        down = fraction.denominator
        cutoff = min(self.target / 2, self.fs / 2)
        max_rate = max(up, down)
        numtaps = 20 * max_rate + 1
        self.upsampling.factor = up
        self.firfilter.numtaps = numtaps
        self.firfilter.cutoff = cutoff
        self.downsampling.factor = down

    def call(self, da, **flags):
        if self.upsampling.factor == 1 and self.downsampling.factor == 1:
            return da
        da = self.upsampling(da, **flags)
        da = self.firfilter(da, **flags)
        da = self.downsampling(da, **flags)
        return da


class IIRFilter(Atom):
    """
    Pipeline implementation of an IIR filter.

    Parameters
    ----------
    order : int
        The order (number of corners) of the IIR filter
    cutoff : float or tuple
        The frequency cut-off of the filter. In the case
        of a low/high-pass filter, ``cutoff`` is a single number.
        In the case of a bandpass filter, ``cutoff`` is a tuple of
        two number (the upper and lower cut-off frequency, resp.).
    btype : str
        The type of the filter band. Valid options are:
            - ``lowpass``: removing frequencies above ``cutoff``
            - ``highpass``: removing frequencies below ``cutoff``
            - ``bandpass`` (default): removing frequencies below ``cutoff[0]`` and above ``cutoff[1]``
    ftype : str
        The IIR filter type. Default: ``butter``
    stype : str
        Form of the output of the filter design. Default: ``sos``
    rp : ?
        ???. Default: ``None``
    rs : ?
        ???. Default: ``None``
    dim : str or int
        The dimension along which the downsampling is applied.
        This is either an index, ``time`` or ``distance``, or ``last``.
        Default: ``last``

    Examples
    --------
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> from xdas.atoms import Sequential, IIRFilter
    >>> da = wavelet_wavefronts()

    Using ``IIRFilter`` directly:

    >>> # Highpass > 1.5 Hz
    >>> da2 = IIRFilter(order=4, cutoff=1.5, btype="highpass", dim="time")(da)
    >>> da2
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.038812 -0.049615  0.061412 ... -0.114737  0.105669 -0.221302]
    [-0.104748  0.121279 -0.088378 ...  0.171324 -0.086691  0.216594]
    [ 0.082237 -0.120316  0.004964 ... -0.111284 -0.136088  0.185075]
    ...
    [ 0.178379  0.011591 -0.31838  ... -0.228471 -0.314301  0.436016]
    [-0.194726 -0.004863  0.116678 ... -0.156696  0.397589 -0.130106]
    [ 0.140117  0.197221 -0.268858 ...  0.322317 -0.414973 -0.055147]]
    Coordinates:
    * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
    * distance (distance): 0.000 to 10000.000

    Using ``IIRFilter`` as an atom in ``Sequential``:

    >>> # Bandpass between 1 and 10 Hz
    >>> sequence = Sequential([
    ...    IIRFilter(order=6, cutoff=(1.0, 10.0), btype="bandpass", dim="time")
    ... ])
    >>> result = sequence(da)
    >>> result
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.00031  -0.000396  0.00049  ... -0.000916  0.000844 -0.001767]
    [ 0.001484 -0.001998  0.002966 ... -0.005491  0.005625 -0.011501]
    [ 0.001948 -0.003366  0.006708 ... -0.012976  0.014296 -0.028643]
    ...
    [ 0.016432 -0.012658 -0.089414 ... -0.021061  0.168231 -0.118295]
    [ 0.004816 -0.044008  0.035511 ... -0.040328  0.144616 -0.064695]
    [-0.014048 -0.079786  0.180202 ...  0.013841 -0.048853  0.062074]]
    Coordinates:
    * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
    * distance (distance): 0.000 to 10000.000

    """

    def __init__(
        self,
        order,
        cutoff,
        btype="bandpass",
        ftype="butter",
        stype="sos",
        rp=None,
        rs=None,
        dim="last",
    ):
        super().__init__()
        self.order = order
        self.cutoff = cutoff
        self.btype = btype
        self.ftype = ftype
        self.stype = stype
        self.rp = rp
        self.rs = rs
        self.dim = dim
        if self.stype == "ba":
            self.iirfilter = LFilter(..., ..., self.dim)
        elif self.stype == "sos":
            self.iirfilter = SOSFilter(..., self.dim)
        else:
            raise ValueError()
        self.fs = State(...)

    def initialize(self, da, **flags):
        self.fs = State(1.0 / get_sampling_interval(da, self.dim))
        self.initialize_from_state()

    def initialize_from_state(self):
        coeffs = sp.iirfilter(
            self.order,
            self.cutoff,
            self.rp,
            self.rs,
            self.btype,
            False,
            self.ftype,
            self.stype,
            self.fs,
        )
        if self.stype == "ba":
            self.iirfilter.b, self.iirfilter.a = coeffs
        elif self.stype == "sos":
            self.iirfilter.sos = coeffs
        else:
            raise ValueError()

    def call(self, da, **flags):
        return self.iirfilter(da, **flags)


class FIRFilter(Atom):
    """
    Pipeline implementation of an FIR filter.

    Parameters
    ----------
    numtaps : int
        The order (number of taps) of the FIR filter
    cutoff : float or tuple
        The frequency cut-off of the filter. In the case
        of a low/high-pass filter, ``cutoff`` is a single number.
        In the case of a bandpass filter, ``cutoff`` is a tuple of
        two number (the upper and lower cut-off frequency, resp.).
    btype : str
        The type of the filter band. Valid options are:
            - ``lowpass``: removing frequencies above ``cutoff``
            - ``highpass``: removing frequencies below ``cutoff``
            - ``bandpass`` (default): removing frequencies below ``cutoff[0]`` and above ``cutoff[1]``
    window : str or tuple of string and parameter values
        The window function to apply befor FIR filtering. If a
        tuple is given, it needs to be compatible with ``scipy.signal.get_window``.
        Default: ``hamming``
    width : ?
        Default: ``None``
    scale : bool
        Default: ``True``
    dim : str or int
        The dimension along which the downsampling is applied.
        This is either an index, ``time`` or ``distance``, or ``last``.
        Default: ``last``

    Examples
    --------
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> from xdas.atoms import Sequential, FIRFilter
    >>> da = wavelet_wavefronts()

    Using ``FIRFilter`` directly:

    >>> # Highpass > 1.5 Hz
    >>> da2 = FIRFilter(numtaps=5, cutoff=1.5, btype="highpass", dim="time")(da)
    >>> da2
    <xdas.DataArray (time: 300, distance: 401)>
    [[-2.339751e-04  2.991040e-04 -3.702198e-04 ...  6.916895e-04
    -6.370217e-04  1.334117e-03]
    [-1.091503e-03  1.471451e-03 -2.193486e-03 ...  4.060728e-03
    -4.168370e-03  8.518611e-03]
    [ 5.014406e-02 -6.344995e-02  7.666315e-02 ... -1.428919e-01
    1.298806e-01 -2.729624e-01]
    ...
    [ 9.129921e-02 -1.841086e-01  2.547145e-03 ... -4.218528e-01
    3.117905e-01 -2.467233e-01]
    [-1.979881e-01 -8.168980e-03  5.458106e-01 ...  4.309588e-01
    -1.352775e-01 -3.427569e-02]
    [ 1.808382e-01 -2.270671e-02 -2.354151e-01 ... -1.836509e-01
    -3.396010e-01  4.366619e-01]]
    Coordinates:
    * time (time): 2022-12-31T23:59:59.960 to 2023-01-01T00:00:05.940
    * distance (distance): 0.000 to 10000.000

    Using ``FIRFilter`` as an atom in ``Sequential``:

    >>> # Bandpass between 1 and 10 Hz
    >>> sequence = Sequential([
    ...    FIRFilter(numtaps=6, cutoff=(1.0, 10.0), btype="bandpass", dim="time")
    ... ])
    >>> result = sequence(da)
    >>> result
    <xdas.DataArray (time: 300, distance: 401)>
    [[-0.000244  0.000312 -0.000386 ...  0.000722 -0.000665  0.001392]
    [ 0.00554  -0.007003  0.00828  ... -0.015509  0.013836 -0.029197]
    [ 0.012271 -0.017179  0.029934 ... -0.054504  0.060639 -0.12196 ]
    ...
    [ 0.056955 -0.078299 -0.089504 ... -0.020045  0.120977 -0.096129]
    [-0.027768 -0.105027  0.228342 ...  0.025277  0.035432 -0.081469]
    [-0.021963 -0.046354  0.186166 ...  0.051622 -0.163209  0.177261]]
    Coordinates:
    * time (time): 2022-12-31T23:59:59.960 to 2023-01-01T00:00:05.940
    * distance (distance): 0.000 to 10000.000

    """

    def __init__(
        self,
        numtaps,
        cutoff,
        btype="bandpass",
        window="hamming",
        width=None,
        scale=True,
        dim="last",
    ):
        super().__init__()
        self.numtaps = numtaps
        self.cutoff = cutoff
        self.btype = btype
        self.window = window
        self.width = width
        self.scale = scale
        self.dim = dim
        self.lfilter = LFilter(..., [1.0], self.dim)
        self.fs = State(...)

    def initialize(self, da, **flags):
        self.fs = State(1.0 / get_sampling_interval(da, self.dim))
        self.initialize_from_state()

    def initialize_from_state(self):
        taps = sp.firwin(
            self.numtaps,
            self.cutoff,
            width=self.width,
            window=self.window,
            pass_zero=self.btype,
            scale=self.scale,
            fs=self.fs,
        )
        self.lag = (len(taps) - 1) // 2
        self.lfilter.b = taps

    def call(self, da, **flags):
        da = self.lfilter(da, **flags)
        da[self.dim] -= get_sampling_interval(da, self.dim, cast=False) * self.lag
        return da


class LFilter(Atom):
    def __init__(self, b, a, dim="last", parallel=None):
        super().__init__()
        self.b = b
        self.a = a
        self.dim = dim
        self.parallel = parallel
        self.axis = State(...)
        self.zi = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        self.axis = State(da.get_axis_num(self.dim))
        if self.dim == chunk_dim:
            n_sections = max(len(self.a), len(self.b)) - 1
            shape = tuple(
                n_sections if name == self.dim else size
                for name, size in da.sizes.items()
            )
            self.zi = State(np.zeros(shape))
        else:
            self.zi = State(None)

    def call(self, da, **flags):
        across = int(self.axis == 0)
        if self.zi is None:
            func = parallelize((None, None, across), across, self.parallel)(sp.lfilter)
            data = func(self.b, self.a, da.values, self.axis)
        else:
            func = parallelize(
                (None, None, across, None, across), (across, across), self.parallel
            )(sp.lfilter)
            data, zf = func(self.b, self.a, da.values, self.axis, self.zi)
            self.zi = State(zf)
        return da.copy(data=data)


class SOSFilter(Atom):
    def __init__(self, sos, dim="last", parallel=None):
        super().__init__()
        self.sos = sos
        self.dim = dim
        self.parallel = parallel
        self.axis = State(...)
        self.zi = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        self.axis = State(da.get_axis_num(self.dim))
        if self.dim == chunk_dim:
            n_sections = self.sos.shape[0]
            shape = (n_sections,) + tuple(
                2 if index == self.axis else element
                for index, element in enumerate(da.shape)
            )
            self.zi = State(np.zeros(shape))
        else:
            self.zi = State(None)

    def call(self, da, **flags):
        across = int(self.axis == 0)
        if self.zi is None:
            func = parallelize((None, across), across, self.parallel)(sp.sosfilt)
            data = func(self.sos, da.values, self.axis)
        else:
            func = parallelize(
                (None, across, None, across + 1), (across, across + 1), self.parallel
            )(sp.sosfilt)
            data, zf = func(self.sos, da.values, self.axis, self.zi)
            self.zi = State(zf)
        return da.copy(data=data)


class DownSample(Atom):
    def __init__(self, factor, dim="last"):
        super().__init__()
        self.factor = factor
        self.dim = dim
        self.buffer = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        if chunk_dim == self.dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        if self.factor == 1:
            return da
        if self.buffer is not None:
            da = concatenate([self.buffer, da], self.dim)
            divpoint = da.sizes[self.dim] - da.sizes[self.dim] % self.factor
            da, buffer = split(da, [divpoint], self.dim)
            self.buffer = State(buffer)
        return da.isel({self.dim: slice(None, None, self.factor)})


class UpSample(Atom):
    def __init__(self, factor, scale=True, dim="last"):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.dim = dim

    def call(self, da, **flags):
        if self.factor == 1:
            return da
        shape = tuple(
            self.factor * size if dim == self.dim else size
            for dim, size in da.sizes.items()
        )
        slc = tuple(
            slice(None, None, self.factor) if dim == self.dim else slice(None)
            for dim in da.dims
        )
        data = np.zeros(shape, dtype=da.dtype)
        if self.scale:
            data[slc] = da.values * self.factor
        else:
            data[slc] = da.values
        coords = da.coords.copy()
        delta = get_sampling_interval(da, self.dim, cast=False)
        tie_indices = coords[self.dim].tie_indices * self.factor
        tie_values = coords[self.dim].tie_values
        tie_indices[-1] += self.factor - 1
        tie_values[-1] += (self.factor - 1) / self.factor * delta
        coords[self.dim] = Coordinate(
            {
                "tie_indices": tie_indices,
                "tie_values": tie_values,
            },
            self.dim,
        )
        return DataArray(data, coords, da.dims, da.name, da.attrs)
