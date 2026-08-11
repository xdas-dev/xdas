"""
Kernel atoms: exact stateful chunked primitives with machine parameters.

This is the expert layer. Kernel atoms take machine parameters (filter
coefficients, integer factors) whose meaning depends on the sampling rate;
the public task atoms (:mod:`xdas.atoms.tasks`) design them from physical
parameters at the first call. They are the units used to prove that chunked
processing equals unchunked processing.

Includes :class:`LFilter`, :class:`SOSFilter`, :class:`DownSample`,
:class:`UpSample`, :class:`Polyphase`.
"""

import math

import numpy as np
import scipy.signal as sp

from ..coordinates import Coordinate, get_sampling_interval
from ..coordinates.core import parse_scalar_delta
from ..core import DataArray, concat, split
from ..parallel import parallelize
from .core import Atom, State


def _along(axis, ndim, slc):
    """Index tuple selecting *slc* along *axis* and everything else elsewhere."""
    return tuple(slc if index == axis else slice(None) for index in range(ndim))


class LFilter(Atom):
    """
    Stateful direct-form IIR/FIR filter using :func:`scipy.signal.lfilter`.

    Parameters
    ----------
    b : array-like
        Numerator polynomial coefficients.
    a : array-like
        Denominator polynomial coefficients.
    dim : str or int, optional
        Dimension to filter along.  Defaults to ``"last"``.
    parallel : int, bool, or None, optional
        Worker count for parallelisation.
    """

    def __init__(self, b, a, dim="last", parallel=None):
        super().__init__()
        self.b = b
        self.a = a
        self.dim = dim
        self.parallel = parallel
        self.axis = State(...)
        self.zi = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        """Set the filter axis and allocate the initial conditions buffer."""
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
        """Apply the filter to *da*, updating the state if chunked."""
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
    """
    Stateful second-order-sections IIR filter using :func:`scipy.signal.sosfilt`.

    Parameters
    ----------
    sos : array-like, shape (n_sections, 6)
        SOS filter coefficients as returned by e.g. :func:`scipy.signal.iirfilter`.
    dim : str or int, optional
        Dimension to filter along.  Defaults to ``"last"``.
    parallel : int, bool, or None, optional
        Worker count for parallelisation.
    """

    def __init__(self, sos, dim="last", parallel=None):
        super().__init__()
        self.sos = sos
        self.dim = dim
        self.parallel = parallel
        self.axis = State(...)
        self.zi = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        """Set the filter axis and allocate the SOS initial-conditions buffer."""
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
        """Apply the SOS filter to *da*, updating the state if chunked."""
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
    """
    Stateful integer downsampling by selecting every *factor*-th sample.

    Parameters
    ----------
    factor : int
        Downsampling factor.
    dim : str or int, optional
        Dimension to downsample along.  Defaults to ``"last"``.
    """

    def __init__(self, factor, dim="last"):
        super().__init__()
        self.factor = factor
        self.dim = dim
        self.buffer = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        """Initialise the carry-over buffer for chunked operation."""
        if chunk_dim == self.dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        """Downsample *da*, buffering the trailing partial stride when chunked."""
        if self.factor == 1:
            return da
        if self.buffer is not None:
            da = concat([self.buffer, da], self.dim)
            divpoint = da.sizes[self.dim] - da.sizes[self.dim] % self.factor
            da, buffer = split(da, [divpoint], self.dim)
            self.buffer = State(buffer)
        return da.isel({self.dim: slice(None, None, self.factor)})


class UpSample(Atom):
    """
    Integer upsampling by zero-insertion (and optional energy scaling).

    Parameters
    ----------
    factor : int
        Upsampling factor.
    scale : bool, optional
        If ``True``, scale inserted samples so energy is preserved.
    dim : str or int, optional
        Dimension to upsample along.  Defaults to ``"last"``.
    """

    def __init__(self, factor, scale=True, dim="last"):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.dim = dim

    def call(self, da, **flags):
        """Upsample *da* by inserting zeros between every original sample."""
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
        new_delta = delta / self.factor
        coord = coords[self.dim]
        tie_indices = coord.tie_indices * self.factor
        tie_values = coord.tie_values
        tie_indices[-1] += self.factor - 1
        tie_values[-1] += (self.factor - 1) * new_delta
        data_coord = {"tie_indices": tie_indices, "tie_values": tie_values}
        if coord.isregular():
            # The derived rate may not be exactly representable (integer datetime
            # resolutions truncate), so declare the representation error as jitter
            # on top of the inherited one; chunk seams then stay within tolerance.
            data_coord["sampling_interval"] = new_delta
            data_coord["tolerance"] = coord.tolerance + np.abs(
                delta - new_delta * self.factor
            )
        # An irregular input gives no rate to inherit and no jitter bound to
        # derive one from, so the result stays irregular rather than claiming a
        # precision the source never declared.
        coords[self.dim] = Coordinate(data_coord, self.dim)
        return DataArray(data, coords, da.dims, da.name, da.attrs)


class Polyphase(Atom):
    """
    Stateful polyphase resampler: upsample, FIR filter and downsample in one pass.

    Computes what :class:`UpSample` → FIR :class:`LFilter` → :class:`DownSample`
    computes, but only the output samples that survive the decimation, and
    without ever materialising the zero-stuffed signal
    (:func:`scipy.signal.upfirdn`). The linear-phase group delay of `taps` is
    removed from the coordinate, as :class:`~xdas.atoms.FIRFilter` does.

    The taps are cast down to the data dtype when the data is less precise, so
    float32 input stays float32 instead of being promoted by the filter.

    Chunked calls carry the filter memory and the output-grid phase across
    chunks, and every call emits every output sample the stream can support so
    far — nothing is held back, so no flush is needed.

    Parameters
    ----------
    taps : array-like
        FIR coefficients, designed at the *upsampled* rate ``up * fs``. At
        least `up` of them are needed, one per polyphase branch.
    up : int, optional
        Upsampling factor. Default is 1.
    down : int, optional
        Downsampling factor. Default is 1.
    dim : str or int, optional
        Dimension to resample along. Defaults to ``"last"``.
    parallel : int, bool, or None, optional
        Worker count for parallelisation.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.signal as sp
    >>> import xdas as xd
    >>> from xdas.atoms import Polyphase

    >>> da = xd.testing.dummy(shape=(100, 3))
    >>> taps = sp.firwin(21, 0.4)
    >>> Polyphase(taps, down=2, dim="time")(da).sizes["time"]
    50

    Splitting the input does not change the result:

    >>> eager = Polyphase(taps, up=2, down=5, dim="time")(da)
    >>> atom = Polyphase(taps, up=2, down=5, dim="time")
    >>> outs = [atom(chunk, chunk_dim="time") for chunk in xd.split(da, 7, "time")]
    >>> chunked = xd.concat(outs, "time")
    >>> bool(np.allclose(chunked.values, eager.values))
    True

    """

    def __init__(self, taps, up=1, down=1, dim="last", parallel=None):
        super().__init__()
        self.taps = taps
        self.up = up
        self.down = down
        self.dim = dim
        self.parallel = parallel
        self.axis = State(...)
        self.buffer = State(...)
        self.consumed = State(...)

    @property
    def lag(self):
        """Group delay of the taps, in upsampled samples."""
        return (np.asarray(self.taps).size - 1) // 2

    @property
    def phase(self):
        """Period, in input samples, of the output-grid phase."""
        return self.down // math.gcd(self.up, self.down)

    def _history_size(self):
        """Input samples to keep: the filter memory plus one phase period."""
        memory = -(-(np.asarray(self.taps).size - 1) // self.up)
        return memory + self.phase - 1

    def initialize(self, da, chunk_dim=None, **flags):
        """Set the axis and allocate the history buffer for chunked operation."""
        if np.asarray(self.taps).size < self.up:
            raise ValueError(
                f"at least {self.up} taps are needed to resample by {self.up}/"
                f"{self.down} (one per polyphase branch), got "
                f"{np.asarray(self.taps).size}"
            )
        self.axis = State(da.get_axis_num(self.dim))
        if self.dim == chunk_dim:
            shape = tuple(
                self._history_size() if name == self.dim else size
                for name, size in da.sizes.items()
            )
            self.buffer = State(np.zeros(shape, dtype=da.dtype))
            self.consumed = State(0)
        else:
            self.buffer = State(None)
            self.consumed = State(None)

    def call(self, da, **flags):
        """Resample *da*, carrying the filter memory and grid phase if chunked."""
        size = da.sizes[self.dim]
        if size == 0:
            return []
        axis = self.axis
        up, down = self.up, self.down
        chunked = self.buffer is not None
        start = self.consumed if chunked else 0
        # The surviving outputs are those whose upsampled index falls in this
        # chunk; both bounds are ceils since the grid starts on sample zero.
        first = -(-start * up // down)
        stop = -(-(start + size) * up // down)
        # Prepend enough past input to warm up the filter, choosing an amount
        # that puts the chunk start on the output grid so `upfirdn`, which
        # always emits from its own sample zero, lands on the global phase.
        memory = -(-(np.asarray(self.taps).size - 1) // self.up)
        nhist = memory + (start - memory) % self.phase
        values = da.values
        if chunked:
            history = self.buffer[
                _along(axis, values.ndim, slice(self.buffer.shape[axis] - nhist, None))
            ]
        else:
            shape = tuple(
                nhist if index == axis else length
                for index, length in enumerate(values.shape)
            )
            history = np.zeros(shape, dtype=values.dtype)
        data = self._resample(np.concatenate([history, values], axis), axis)
        if chunked:
            self.buffer = State(self._keep_history(values, axis))
            self.consumed = State(start + size)
        if stop <= first:
            return []
        offset = (start - nhist) * up // down
        data = data[_along(axis, data.ndim, slice(first - offset, stop - offset))]
        return DataArray(
            data, self._coords(da, first, stop, start), da.dims, da.name, da.attrs
        )

    def _resample(self, values, axis):
        """Run the polyphase filter over *values*, keeping the data precision."""
        taps = np.asarray(self.taps)
        if np.issubdtype(values.dtype, np.floating) and (
            values.dtype.itemsize < taps.dtype.itemsize
        ):
            taps = taps.astype(values.dtype)
        across = int(axis == 0)
        func = parallelize((None, across, None, None, None), across, self.parallel)(
            sp.upfirdn
        )
        return func(taps, values, self.up, self.down, axis)

    def _keep_history(self, values, axis):
        """Return the last `_history_size` input samples of the stream so far."""
        size = values.shape[axis]
        length = self.buffer.shape[axis]
        if size >= length:
            return values[_along(axis, values.ndim, slice(size - length, None))]
        kept = self.buffer[_along(axis, values.ndim, slice(size, None))]
        return np.concatenate([kept, values], axis)

    def _coords(self, da, first, stop, start):
        """Build the output coordinates on the resampled, delay-corrected grid."""
        coord = da.coords[self.dim]
        delta = get_sampling_interval(da, self.dim, cast=False)
        size = stop - first
        # Output `index` sits `index * down - lag` upsampled samples after the
        # start of the run, hence that many minus `start * up` after this chunk.
        shifts = np.array([first, stop - 1]) * self.down - self.lag - start * self.up
        grid = coord.start + self._upsampled(shifts, delta)
        origin, last = grid[0], grid[1]
        step = self._upsampled(self.down, delta)
        if size > 1:
            tie_indices, tie_values = [0, size - 1], [origin, last]
            drift = abs((last - origin) - (size - 1) * step)
        else:
            tie_indices, tie_values = [0], [origin]
            drift = 0 * step
        data = {"tie_indices": tie_indices, "tie_values": tie_values}
        if coord.isregular():
            # A rate that the coordinate resolution cannot represent exactly
            # makes the tie values drift from the nominal step; declare that
            # drift as jitter rather than refusing to call the output regular.
            tolerance = getattr(coord, "tolerance", None)
            base = parse_scalar_delta(tolerance, coord.dtype, default_zero=True)
            data["sampling_interval"] = step
            data["tolerance"] = base + drift
        coords = da.coords.copy()
        coords[self.dim] = Coordinate(data, self.dim)
        return coords

    def _upsampled(self, count, delta):
        """Return the span of *count* upsampled samples, at coordinate resolution."""
        if np.issubdtype(np.asarray(delta).dtype, np.timedelta64):
            return (count * delta) // self.up
        return count * delta / self.up
