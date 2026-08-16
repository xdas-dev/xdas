"""
Composite signal-processing atoms with physical (Hz) parameters.

Includes :class:`IIRFilter`, :class:`FIRFilter`. The stateful
machine-parameter primitives they orchestrate live in
:mod:`xdas.atoms.kernel`.
"""

import scipy.signal as sp

from ..coordinates import get_sampling_interval
from .core import Atom, State
from .kernel import LFilter, Polyphase, SOSFilter


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
        """Determine the sampling rate from *da* and recompute the IIR coefficients."""
        self.fs = State(1.0 / get_sampling_interval(da, self.dim))
        self.initialize_from_state()

    def initialize_from_state(self):
        """Recompute and store the IIR coefficients from the current design parameters."""
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
        """Delegate to the underlying :class:`LFilter` or :class:`SOSFilter` atom."""
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
    up, down : int
        Machine parameters of the polyphase form: the taps are designed at the
        upsampled rate ``up * fs`` and applied by :class:`Polyphase`, which
        keeps one output sample in ``down``. Both default to 1, i.e. plain
        filtering.
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
        up=1,
        down=1,
        dim="last",
    ):
        super().__init__()
        self.numtaps = numtaps
        self.cutoff = cutoff
        self.btype = btype
        self.window = window
        self.width = width
        self.scale = scale
        self.up = up
        self.down = down
        self.dim = dim
        self.polyphase = Polyphase(..., self.up, self.down, self.dim)
        self.fs = State(...)

    def initialize(self, da, **flags):
        """Determine the sampling rate from *da* and recompute the FIR taps."""
        self.fs = State(1.0 / get_sampling_interval(da, self.dim))
        self.initialize_from_state()

    def initialize_from_state(self):
        """Recompute the FIR taps from the current design parameters."""
        taps = sp.firwin(
            self.numtaps,
            self.cutoff,
            width=self.width,
            window=self.window,
            pass_zero=self.btype,
            scale=self.scale,
            fs=self.fs * self.up,
        )
        # Interpolation spreads the energy of one input sample over `up`
        # upsampled ones, which the taps must compensate.
        self.polyphase.taps = self.up * taps
        self.polyphase.up = self.up
        self.polyphase.down = self.down

    def call(self, da, **flags):
        """Apply the FIR taps to *da*, delay-corrected and resampled."""
        return self.polyphase(da, **flags)
