"""
Kernel atoms: exact stateful chunked primitives with machine parameters.

This is the expert layer. Kernel atoms take machine parameters (filter
coefficients, integer factors) whose meaning depends on the sampling rate;
the public task atoms (:mod:`xdas.atoms.tasks`) design them from physical
parameters at the first call. They are the units used to prove that chunked
processing equals unchunked processing.

Includes :class:`LFilter`, :class:`SOSFilter`, :class:`DownSample`,
:class:`UpSample`, :class:`Polyphase`, :class:`Rechunk`.
"""

import math

import numpy as np
import scipy.signal as sp

from ..coordinates import Coordinate, get_sampling_interval
from ..coordinates.core import parse_scalar_delta
from ..core import DataArray, concat, split
from ..parallel import parallelize
from .core import Atom, State, atomized


def _along(axis, ndim, slc):
    """Index tuple selecting *slc* along *axis* and everything else elsewhere."""
    return tuple(slc if index == axis else slice(None) for index in range(ndim))


def _carry_labels(coords, name, positions):
    """
    Re-index the non-dimensional coordinates attached to *name*.

    An atom that changes the number of samples along a dimension has to say
    what became of the *other* coordinates attached to it — the station code
    of a channel, the latitude of a sensor — or they keep their input length
    and silently label the output with the wrong lanes. They are carried by
    taking, for each output sample, the input sample it is drawn from
    (*positions*): a label names a source sample rather than a position, so
    unlike the dimension coordinate it is not shifted by a filter's group
    delay. Depending on the sampling grid alone, that mapping is the same
    however the stream was chunked.

    Parameters
    ----------
    coords : Coordinates
        The output coordinates, modified in place.
    name : str
        The resampled dimension.
    positions : ndarray of int
        One input index per output sample, into this chunk.

    Returns
    -------
    Coordinates
        The same mapping, for chaining.
    """
    for key, coord in list(coords.items()):
        if key != name and coord.dim == name:
            coords[key] = coord[positions]
    return coords


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
        # `dim` may be the "first"/"last" alias, which never equals a real
        # dimension name: resolve it against the data before comparing, else
        # the seam state is silently never allocated.
        dim = self._resolve_dim(da) or self.dim
        if dim == chunk_dim:
            n_sections = max(len(self.a), len(self.b)) - 1
            shape = tuple(
                n_sections if name == dim else size for name, size in da.sizes.items()
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
        # Resolve the "first"/"last" alias before comparing (see `LFilter`).
        dim = self._resolve_dim(da) or self.dim
        if dim == chunk_dim:
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
        # Resolve the "first"/"last" alias before comparing (see `LFilter`).
        dim = self._resolve_dim(da) or self.dim
        if chunk_dim == dim:
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

    def flush(self):
        """
        Emit the buffered samples that fall on the output grid.

        The buffer always starts on an output sample (every emission consumes
        a whole number of strides), so its strided selection is the exact
        remainder of the downsampled stream.
        """
        if not isinstance(self.buffer, DataArray) or self.buffer.sizes[self.dim] == 0:
            return []
        out = self.buffer.isel({self.dim: slice(None, None, self.factor)})
        self.buffer = State(self.buffer.isel({self.dim: slice(0, 0)}))
        return [out]


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
        # The "first"/"last" alias never matches a real dimension name, so it
        # has to be resolved before it names an axis or a coordinate.
        name = self._resolve_dim(da) or self.dim
        shape = tuple(
            self.factor * size if dim == name else size
            for dim, size in da.sizes.items()
        )
        slc = tuple(
            slice(None, None, self.factor) if dim == name else slice(None)
            for dim in da.dims
        )
        data = np.zeros(shape, dtype=da.dtype)
        if self.scale:
            data[slc] = da.values * self.factor
        else:
            data[slc] = da.values
        coords = da.coords.copy()
        delta = get_sampling_interval(da, name, cast=False)
        new_delta = delta / self.factor
        coord = coords[name]
        # Copies: the tie arrays are the input coordinate's own storage.
        tie_indices = np.asarray(coord.tie_indices) * self.factor
        tie_values = np.asarray(coord.tie_values).copy()
        if tie_indices.size == 1:
            # A one-sample chunk has a single tie, and the upsampled block
            # still spans `factor` samples: it takes a second tie to say so.
            tie_indices = np.append(tie_indices, self.factor - 1)
            tie_values = np.append(
                tie_values, tie_values[-1] + (self.factor - 1) * new_delta
            )
        else:
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
        coords[name] = Coordinate(data_coord, name)
        # Each inserted sample is drawn from the input sample it follows.
        positions = np.arange(shape[da.get_axis_num(name)]) // self.factor
        _carry_labels(coords, name, positions)
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
    >>> chunked = xd.concat(list(atom.iter_chunks(xd.split(da, 7, "time"))), "time")
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
        # Resolve the "first"/"last" alias before comparing (see `LFilter`).
        dim = self._resolve_dim(da) or self.dim
        if dim == chunk_dim:
            shape = tuple(
                self._history_size() if name == dim else size
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
        # The "first"/"last" alias would name a new dimension if it reached the
        # `Coordinate` built below, so resolve it here.
        name = self._resolve_dim(da) or self.dim
        coord = da.coords[name]
        delta = get_sampling_interval(da, name, cast=False)
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
        coords[name] = Coordinate(data, name)
        # Output `index` is drawn from input sample `index * down / up`, which
        # `first`/`stop` keep inside this chunk by construction.
        positions = np.rint(np.arange(first, stop) * self.down / self.up)
        positions = np.clip(positions.astype(int) - start, 0, da.sizes[name] - 1)
        return _carry_labels(coords, name, positions)

    def _upsampled(self, count, delta):
        """Return the span of *count* upsampled samples, at coordinate resolution."""
        if np.issubdtype(np.asarray(delta).dtype, np.timedelta64):
            return (count * delta) // self.up
        return count * delta / self.up


class Rechunk(Atom):
    """
    Merge and split incoming chunks to a fixed size along a dimension.

    Chunk sizes are a performance knob, not science, so they are given in
    samples (as in ``process(chunks=...)``): the canonical use is restoring a
    workable cadence after a decimation shrank the chunks. Rechunking never
    merges across a discontinuity — the seam handling of the base class
    flushes the partial buffer at every gap — so chunks stay internally
    regular through this stage. Each call returns zero or more chunks of
    exactly the target size; :meth:`flush` drains the remainder.

    Eager calls (whole records) pass through unchanged: re-joining the emitted
    chunks would reproduce the input.

    Parameters
    ----------
    chunks : dict
        Mapping of a unique dimension name to the target chunk size in
        samples, e.g. ``{"time": 1000}``.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.atoms import Rechunk
    >>> da = xd.testing.dummy(shape=(100, 10))
    >>> atom = Rechunk({"time": 30})
    >>> [out.sizes["time"] for out in atom.iter_chunks(xd.split(da, 4, "time"))]
    [30, 30, 30, 10]

    """

    def __init__(self, chunks):
        super().__init__()
        if not (isinstance(chunks, dict) and len(chunks) == 1):
            raise TypeError(
                "`chunks` must be a dict that maps a unique "
                "dimension to a unique size: {'dim': int}"
            )
        ((dim, size),) = chunks.items()
        if not (isinstance(size, int) and size > 0):
            raise ValueError("the chunk size must be a strictly positive integer")
        self.dim = dim
        self.size = size
        self.buffer = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        """Initialise the carry-over buffer for chunked operation."""
        # Resolve the "first"/"last" alias before comparing (see `LFilter`).
        dim = self._resolve_dim(da) or self.dim
        if chunk_dim == dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        """Emit full-size chunks from the buffered stream, keep the remainder."""
        if self.buffer is None:
            return da
        da = concat([self.buffer, da], self.dim)
        divpoint = da.sizes[self.dim] - da.sizes[self.dim] % self.size
        out = [
            da.isel({self.dim: slice(index, index + self.size)})
            for index in range(0, divpoint, self.size)
        ]
        self.buffer = State(da.isel({self.dim: slice(divpoint, None)}))
        return out

    def flush(self):
        """Emit the remaining partial chunk."""
        if not isinstance(self.buffer, DataArray) or self.buffer.sizes[self.dim] == 0:
            return []
        out = self.buffer
        self.buffer = State(out.isel({self.dim: slice(0, 0)}))
        return [out]


rechunk = atomized(Rechunk)
