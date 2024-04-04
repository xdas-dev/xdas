from fractions import Fraction

import numpy as np
import scipy.signal as sp

from ..core.coordinates import Coordinate, get_sampling_interval
from ..core.database import Database
from ..core.routines import concatenate, split
from ..parallel import parallelize
from .core import Atom, State


class ResamplePoly(Atom):
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

    def initialize(self, db, **kwargs):
        self.fs = State(1.0 / get_sampling_interval(db, self.dim))
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

    def call(self, db, **kwargs):
        db = self.upsampling(db, **kwargs)
        db = self.firfilter(db, **kwargs)
        db = self.downsampling(db, **kwargs)
        return db


class IIRFilter(Atom):
    def __init__(
        self,
        order,
        cutoff,
        btype="band",
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

    def initialize(self, db, **kwargs):
        self.fs = State(1.0 / get_sampling_interval(db, self.dim))
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

    def call(self, db, **kwargs):
        return self.iirfilter(db, **kwargs)


class FIRFilter(Atom):
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

    def initialize(self, db, **kwargs):
        self.fs = State(1.0 / get_sampling_interval(db, self.dim))
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

    def call(self, db, **kwargs):
        db = self.lfilter(db, **kwargs)
        db[self.dim] -= get_sampling_interval(db, self.dim, cast=False) * self.lag
        return db


class LFilter(Atom):
    def __init__(self, b, a, dim="last", parallel=None):
        super().__init__()
        self.b = b
        self.a = a
        self.dim = dim
        self.parallel = parallel
        self.axis = State(...)
        self.zi = State(...)

    def initialize(self, db, chunk=None, **kwargs):
        self.axis = State(db.get_axis_num(self.dim))
        if self.dim == chunk:
            n_sections = max(len(self.a), len(self.b)) - 1
            shape = tuple(
                n_sections if name == self.dim else size
                for name, size in db.sizes.items()
            )
            self.zi = State(np.zeros(shape))
        else:
            self.zi = State(None)

    def call(self, db, **kwargs):
        across = int(self.axis == 0)
        if self.zi is None:
            func = parallelize((None, None, across), across, self.parallel)(sp.lfilter)
            data = func(self.b, self.a, db.values, self.axis)
        else:
            func = parallelize(
                (None, None, across, None, across), (across, across), self.parallel
            )(sp.lfilter)
            data, zf = func(self.b, self.a, db.values, self.axis, self.zi)
            self.zi = State(zf)
        return db.copy(data=data)


class SOSFilter(Atom):
    def __init__(self, sos, dim="last", parallel=None):
        super().__init__()
        self.sos = sos
        self.dim = dim
        self.parallel = parallel
        self.axis = State(...)
        self.zi = State(...)

    def initialize(self, db, chunk=None, **kwargs):
        self.axis = State(db.get_axis_num(self.dim))
        if self.dim == chunk:
            n_sections = self.sos.shape[0]
            shape = (n_sections,) + tuple(
                2 if index == self.axis else element
                for index, element in enumerate(db.shape)
            )
            self.zi = State(np.zeros(shape))
        else:
            self.zi = State(None)

    def call(self, db, **kwargs):
        across = int(self.axis == 0)
        if self.zi is None:
            func = parallelize((None, across), across, self.parallel)(sp.sosfilt)
            data = func(self.sos, db.values, self.axis)
        else:
            func = parallelize(
                (None, across, None, across + 1), (across, across + 1), self.parallel
            )(sp.sosfilt)
            data, zf = func(self.sos, db.values, self.axis, self.zi)
            self.zi = State(zf)
        return db.copy(data=data)


class DownSample(Atom):
    def __init__(self, factor, dim="last"):
        super().__init__()
        self.factor = factor
        self.dim = dim
        self.buffer = State(...)

    def initialize(self, db, chunk=None, **kwargs):
        if chunk == self.dim:
            self.buffer = State(db.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, db, **kwargs):
        if self.buffer is not None:
            db = concatenate([self.buffer, db], self.dim)
            divpoint = db.sizes[self.dim] - db.sizes[self.dim] % self.factor
            db, buffer = split(db, [divpoint], self.dim)
            self.buffer = State(buffer)
        return db.isel({self.dim: slice(None, None, self.factor)})


class UpSample(Atom):
    def __init__(self, factor, scale=True, dim="last"):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.dim = dim

    def call(self, db, **kwargs):
        shape = tuple(
            self.factor * size if dim == self.dim else size
            for dim, size in db.sizes.items()
        )
        slc = tuple(
            slice(None, None, self.factor) if dim == self.dim else slice(None)
            for dim in db.dims
        )
        data = np.zeros(shape, dtype=db.dtype)
        if self.scale:
            data[slc] = db.values * self.factor
        else:
            data[slc] = db.values
        coords = db.coords.copy()
        delta = get_sampling_interval(db, self.dim, cast=False)
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
        return Database(data, coords, name=db.name, attrs=db.attrs)
