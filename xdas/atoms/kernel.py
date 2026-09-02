"""
Kernel atoms: exact stateful chunked primitives with machine parameters.

This is the expert layer. Kernel atoms take machine parameters (filter
coefficients, integer factors) whose meaning depends on the sampling rate;
the public operation atoms (:mod:`xdas.atoms.operations`) design them from
physical parameters at the first call. They are the units used to prove that
chunked processing equals unchunked processing.

Includes :class:`LFilter`, :class:`SOSFilter`, :class:`DownSample`,
:class:`UpSample`, :class:`Polyphase`, :class:`Rechunk`.
"""

import math

import numpy as np
import scipy.signal as sp

from ..coordinates import Coordinate, InterpCoordinate, get_sampling_interval
from ..coordinates.core import parse_scalar_delta, quantization_tolerance, step_value
from ..core import DataArray, concat, split
from ..parallel import parallelize
from .core import Atom, State, atomized


def _along(axis, ndim, slc):
    """Index tuple selecting *slc* along *axis* and everything else elsewhere."""
    return tuple(slc if index == axis else slice(None) for index in range(ndim))


def _trim_along(da, dim, n):
    """Keep at most the last *n* samples of *da* along *dim*."""
    if da.sizes[dim] <= n:
        return da
    return da.isel({dim: slice(da.sizes[dim] - n, None)})


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
        dim = self._dim_name(da)
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
    sos : array-like
        SOS filter coefficients, of shape ``(n_sections, 6)``, as returned by
        e.g. :func:`scipy.signal.iirfilter`.
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
        dim = self._dim_name(da)
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
        dim = self._dim_name(da)
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
        name = self._dim_name(da)
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
        coord = coords[name]
        # Copies: the tie arrays are the input coordinate's own storage.
        tie_indices = np.asarray(coord.tie_indices) * self.factor
        tie_values = np.asarray(coord.tie_values).copy()
        anchor_index = int(tie_indices[-1])
        anchor_value = tie_values[-1]
        # A one-sample chunk has a single tie, and the upsampled block still
        # spans `factor` samples: it takes a second tie to say so.
        extended_index = anchor_index + self.factor - 1
        if coord.isregular():
            # Dividing the denominator rather than rounding the delta is
            # exact -- (num, den * factor) is the new rate exactly, nothing
            # to declare as jitter. Floats have no tick (D2: denominator
            # always 1), so the division stays a plain float one.
            numerator, denominator = coord._sampling_ratio
            if np.issubdtype(coord.dtype, np.floating):
                new_numerator, new_denominator = numerator / self.factor, 1
            else:
                new_numerator, new_denominator = (
                    numerator,
                    int(denominator) * self.factor,
                )
            extended_value = step_value(
                anchor_value,
                extended_index - anchor_index,
                new_numerator,
                new_denominator,
                coord.dtype,
            )
        else:
            # No declared rate to inherit, but the padded samples between
            # this chunk's last real sample and its boundary still need a
            # plausible placement; infer one exactly as any other
            # signal-processing call would (may warn or raise).
            delta = get_sampling_interval(da, name, cast=False)
            extended_value = anchor_value + (self.factor - 1) * (delta / self.factor)
        if tie_indices.size == 1:
            tie_indices = np.append(tie_indices, extended_index)
            tie_values = np.append(tie_values, extended_value)
        else:
            tie_indices[-1] = extended_index
            tie_values[-1] = extended_value
        data_coord = {"tie_indices": tie_indices, "tie_values": tie_values}
        if coord.isregular():
            data_coord["sampling_numerator"] = new_numerator
            data_coord["sampling_denominator"] = new_denominator
            # The trailing tie was placed by rounding to the coordinate's own
            # tick resolution, which -- unlike the old truncated-delta
            # laundering -- costs at most a fraction of a tick, not up to a
            # whole sample period.
            quantization = quantization_tolerance(
                tie_indices, tie_values, new_numerator, new_denominator, coord.dtype
            )
            # A sampled coordinate is regular but carries no jitter of its own.
            base = parse_scalar_delta(
                getattr(coord, "tolerance", None), coord.dtype, default_zero=True
            )
            data_coord["tolerance"] = base + quantization
        # An irregular input gives no rate to inherit, so the result stays
        # irregular rather than claiming a precision the source never declared.
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
    (:func:`scipy.signal.upfirdn`). By default the linear-phase group delay of
    `taps` is removed from the coordinate, as :class:`~xdas.atoms.FIRFilter`
    does; the values are the leading samples of ``upfirdn`` untouched.

    With ``compensate=True`` the group delay is taken out of the *data*
    instead: the output then lands on the canonical grid (origin unchanged,
    ``delta * down / up`` spacing, ``ceil(n * up / down)`` samples) and matches
    :func:`scipy.signal.resample_poly` sample for sample. This is the mode
    :class:`~xdas.atoms.FIRFilter` selects when it is resampling. Chunked, it
    reads ahead of the canonical grid, so the last output samples of each run
    are held back and drained by :meth:`flush`.

    The taps are cast down to the data dtype when the data is less precise, so
    float32 input stays float32 instead of being promoted by the filter.

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
    compensate : bool, optional
        Take the group delay out of the data and land on the canonical grid
        (``scipy.signal.resample_poly`` semantics) rather than shifting the
        coordinate. Default is False.

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

    def __init__(self, taps, up=1, down=1, dim="last", parallel=None, compensate=False):
        super().__init__()
        self.taps = taps
        self.up = up
        self.down = down
        self.dim = dim
        self.parallel = parallel
        self.compensate = compensate
        self.axis = State(...)
        self.buffer = State(...)
        self.consumed = State(...)
        self.recent = State(...)
        self.emitted = State(...)

    @property
    def lag(self):
        """Group delay of the taps, in upsampled samples."""
        return (np.asarray(self.taps).size - 1) // 2

    @property
    def _pre_pad(self):
        """Taps zero-padding matching scipy.signal.resample_poly's ``n_pre_pad``."""
        half_len = (np.asarray(self.taps).size - 1) // 2
        return self.down - half_len % self.down

    @property
    def _pre_remove(self):
        """Padded-tap group delay in output samples (scipy's ``n_pre_remove``)."""
        half_len = (np.asarray(self.taps).size - 1) // 2
        return (half_len + self._pre_pad) // self.down

    def _filter_len(self):
        """Length of the filter actually convolved (padded, when resampling)."""
        length = np.asarray(self.taps).size
        return length + self._pre_pad if self.compensate else length

    @property
    def phase(self):
        """Period, in input samples, of the output-grid phase."""
        return self.down // math.gcd(self.up, self.down)

    def _memory(self):
        """Input samples the filter reaches back over."""
        return -(-(self._filter_len() - 1) // self.up)

    def _history_size(self):
        """Input samples to keep: the filter memory plus one phase period."""
        return self._memory() + self.phase - 1

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
        dim = self._dim_name(da)
        if dim == chunk_dim and self.compensate:
            # Delay-compensated resampling reads ahead of the canonical grid,
            # so a rolling window of recent input (values *and* labels) is
            # carried and the last output samples are held back for `flush`.
            self.recent = State(da.isel({dim: slice(0, 0)}))
            self.consumed = State(0)
            self.emitted = State(0)
            self.buffer = State(None)
        elif dim == chunk_dim:
            shape = tuple(
                self._history_size() if name == dim else size
                for name, size in da.sizes.items()
            )
            self.buffer = State(np.zeros(shape, dtype=da.dtype))
            self.consumed = State(0)
            self.recent = State(None)
            self.emitted = State(None)
        else:
            self.buffer = State(None)
            self.consumed = State(None)
            self.recent = State(None)
            self.emitted = State(None)

    def call(self, da, **flags):
        """Resample *da*, carrying the filter memory and grid phase if chunked."""
        if self.compensate:
            return self._call_resample(da, **flags)
        return self._call_filter(da, **flags)

    def _call_filter(self, da, **flags):
        """Filter, delay-compensating on the coordinate; emit every sample."""
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
        memory = self._memory()
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
        coords = self._output_coords(da, start, first, stop, self.lag)
        return DataArray(data, coords, da.dims, da.name, da.attrs)

    def _call_resample(self, da, **flags):
        """Rate change: emit only samples whose window fits the input so far."""
        size = da.sizes[self.dim]
        if size == 0:
            return []
        axis = self.axis
        up, down, npr = self.up, self.down, self._pre_remove
        if self.recent is None:
            # Eager (or chunked along another dimension): the whole canonical
            # grid at once, `resample_poly`'s natural zero tail closing it out.
            n_out = -(-size * up // down)
            nhist = self._nhist(0)
            history = np.zeros(
                tuple(nhist if i == axis else n for i, n in enumerate(da.values.shape)),
                dtype=da.dtype,
            )
            y = self._resample(np.concatenate([history, da.values], axis), axis, True)
            offset = (0 - nhist) * up // down
            return self._emit(y, offset, 0, n_out, da, 0)
        start = self.consumed
        recent, navail = self.recent, self.recent.sizes[self.dim]
        nhist = self._nhist(start)
        y = self._resample(
            np.concatenate([self._warmup(recent.values, nhist, axis), da.values], axis),
            axis,
            True,
        )
        offset = (start - nhist) * up // down
        # Highest `upfirdn` index whose window does not reach past the last real
        # sample received; canonical index `k` maps to `y[k - offset + npr]`.
        safe = ((nhist + size) * up - 1 + self._pre_pad) // down
        hi, lo = safe + offset - npr + 1, self.emitted
        frame = self._join_recent(recent, da)
        self.recent = State(_trim_along(frame, self.dim, self._history_size()))
        self.consumed = State(start + size)
        if hi <= lo:
            return []
        self.emitted = State(hi)
        return self._emit(y, offset, lo, hi, frame, start - navail)

    def flush(self):
        """Emit the held-back tail on trailing zeros, to ``ceil(n_in*up/down)``."""
        if not self.compensate or not isinstance(self.recent, DataArray):
            return []
        consumed = self.consumed
        n_out = -(-consumed * self.up // self.down)
        if self.emitted >= n_out:
            return []
        axis = self.axis
        recent, navail = self.recent, self.recent.sizes[self.dim]
        nhist = self._nhist(consumed)
        hist = self._warmup(recent.values, nhist, axis)
        tail = np.zeros(
            tuple(
                self._memory() + self.phase + 1 if i == axis else n
                for i, n in enumerate(hist.shape)
            ),
            dtype=hist.dtype,
        )
        y = self._resample(np.concatenate([hist, tail], axis), axis, True)
        offset = (consumed - nhist) * self.up // self.down
        lo = self.emitted
        self.emitted = State(n_out)
        return [self._emit(y, offset, lo, n_out, recent, consumed - navail)]

    def _nhist(self, start):
        """Warm-up depth that puts *start* on the output grid (see `_call_filter`)."""
        memory = self._memory()
        return memory + (start - memory) % self.phase

    def _warmup(self, values, nhist, axis):
        """Take the last *nhist* input samples, zero-padding the front if short."""
        if values.shape[axis] >= nhist:
            return values[
                _along(axis, values.ndim, slice(values.shape[axis] - nhist, None))
            ]
        pad = tuple(
            nhist - values.shape[axis] if i == axis else n
            for i, n in enumerate(values.shape)
        )
        return np.concatenate([np.zeros(pad, dtype=values.dtype), values], axis)

    def _emit(self, y, offset, lo, hi, frame, frame_start):
        """Slice canonical outputs ``[lo, hi)`` out of *y* and label them."""
        npr = self._pre_remove
        block = y[
            _along(self.axis, y.ndim, slice(lo - offset + npr, hi - offset + npr))
        ]
        coords = self._output_coords(frame, frame_start, lo, hi, 0)
        return DataArray(block, coords, frame.dims, frame.name, frame.attrs)

    def _join_recent(self, recent, da):
        """Concat the buffer and *da*, re-attaching non-dim labels concat drops."""
        name = self._dim_name(da)
        merged = concat([recent, da], name)
        extra = {
            key: (name, np.concatenate([recent.coords[key].values, coord.values]))
            for key, coord in da.coords.items()
            if key != name and coord.dim == name
        }
        return merged.assign_coords(extra) if extra else merged

    def _resample(self, values, axis, padded=False):
        """Run the polyphase filter over *values*, keeping the data precision.

        *padded* prepends ``_pre_pad`` zeros to the taps, the scipy.signal.
        resample_poly trick that makes the group delay a whole output sample.
        """
        taps = np.asarray(self.taps)
        if np.issubdtype(values.dtype, np.floating) and (
            values.dtype.itemsize < taps.dtype.itemsize
        ):
            taps = taps.astype(values.dtype)
        if padded:
            taps = np.concatenate([np.zeros(self._pre_pad, dtype=taps.dtype), taps])
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

    def _output_coords(self, frame, frame_start, lo, hi, lag):
        """Coordinates for canonical output indices ``[lo, hi)`` of a run.

        *frame* holds the coordinates the non-dim labels are drawn from, its
        first sample at input index *frame_start*; *lag* upsampled samples of
        group delay come off the axis (0 when the data carries it instead).
        """
        # Resolve the "first"/"last" alias before it names a `Coordinate` dim.
        name = self._dim_name(frame)
        coord = frame.coords[name]
        size = hi - lo
        # Output `k` sits `k*down - lag` upsampled samples after the run start,
        # hence that many minus `frame_start*up` after this frame.
        shift = lo * self.down - lag - frame_start * self.up
        if coord.isregular():
            # (num, den*up) is the exact upsampled rate, (num*down, den*up) the
            # exact output rate -- neither needs rounding.
            numerator, denominator = coord._sampling_ratio
            if np.issubdtype(coord.dtype, np.floating):
                up_ratio = (numerator / self.up, 1)
                out_ratio = (numerator * self.down / self.up, 1)
            else:
                up_ratio = (numerator, int(denominator) * self.up)
                out_ratio = (numerator * self.down, int(denominator) * self.up)
            origin = step_value(coord.start, shift, *up_ratio, coord.dtype)
        else:
            # No declared rate to inherit; infer one (may warn or raise).
            delta = get_sampling_interval(frame, name, cast=False)
            origin = coord.start + self._upsampled(shift, delta)
        coords = frame.coords.copy()
        if coord.isregular():
            # `from_block` lays the ties from `origin` at the exact output rate
            # and declares the tick quantization the far one costs. A sampled
            # coordinate has no jitter of its own, hence the `getattr`.
            coords[name] = InterpCoordinate.from_block(
                origin,
                size,
                out_ratio,
                dim=name,
                tolerance=getattr(coord, "tolerance", None),
            )
        elif size <= 1:
            coords[name] = Coordinate(
                {"tie_indices": [0][:size], "tie_values": [origin][:size]}, name
            )
        else:
            last_shift = (hi - 1) * self.down - lag - frame_start * self.up
            last = coord.start + self._upsampled(last_shift, delta)
            coords[name] = Coordinate(
                {"tie_indices": [0, size - 1], "tie_values": [origin, last]}, name
            )
        # Output `k` is drawn from input sample `k*down//up`. Flooring -- never
        # rounding -- keeps the position inside the frame: rounding a
        # half-integer up can walk past the last sample it holds.
        positions = (np.arange(lo, hi) * self.down) // self.up - frame_start
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
        dim = self._dim_name(da)
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
