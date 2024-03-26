import numpy as np
import scipy.signal as sp

from xdas.coordinates import get_sampling_interval


class State:
    def __init__(self, state):
        self.state = state


class Filter:
    def __init__(self, *args):
        self._state = {}
        self._filters = {}

    def __setattr__(self, name, value):
        match value:
            case State(state=state):
                self._state[name] = state
                super().__setattr__(name, state)
            case Filter():
                self._filters[name] = value
                super().__setattr__(name, value)
            case other:
                super().__setattr__(name, other)

    @property
    def state(self):
        return self._state | {
            name: filter.state for name, filter in self._filters.items() if filter.state
        }

    def initialize(self, db, chunk=None):
        return NotImplemented

    def call(self, db, chunk=None):
        return NotImplemented

    def __call__(self, db, chunk=None):
        if not self.state:
            self.initialize(db, chunk)
        return self.call(db, chunk)


class IIRFilter(Filter):
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

    def initialize(self, db, chunk=None):
        fs = 1.0 / get_sampling_interval(db, self.dim)
        coeffs = sp.iirfilter(
            self.order,
            self.cutoff,
            self.rp,
            self.rs,
            self.btype,
            False,
            self.ftype,
            self.stype,
            fs,
        )
        match self.stype:
            case "ba":
                self.iirfilter = LFilter(*coeffs, self.dim)
            case "sos":
                self.iirfilter = SOSFilter(coeffs, self.dim)

    def call(self, db, chunk=None):
        return self.iirfilter(db, chunk)


class LFilter(Filter):
    def __init__(self, b, a, dim):
        super().__init__()
        self.b = b
        self.a = a
        self.dim = dim

    def initialize(self, db, chunk=None):
        self.axis = db.get_axis_num(self.dim)
        if self.dim == chunk:
            n_sections = max(len(self.a), len(self.b)) - 1
            shape = tuple(
                n_sections if name == self.dim else size
                for name, size in db.sizes.items()
            )
            self.zi = State(np.zeros(shape))

    def call(self, db, chunk=None):
        if hasattr(self, "zi"):
            data, state = sp.lfilter(self.b, self.a, db, self.axis, self.zi)
            self.zi = State(state)
        else:
            data, state = sp.lfilter(self.b, self.a, db, self.axis)
        return db.copy(data=data)


class SOSFilter(Filter):
    def __init__(self, sos, dim):
        super().__init__()
        self.sos = sos
        self.dim = dim

    def initialize(self, db, chunk=None):
        self.axis = db.get_axis_num(self.dim)
        if self.dim == chunk:
            n_sections = self.sos.shape[0]
            shape = (n_sections,) + tuple(
                2 if index == self.axis else element
                for index, element in enumerate(db.shape)
            )
            self.zi = State(np.zeros(shape))

    def call(self, db, chunk=None):
        if hasattr(self, "zi"):
            data, zi = sp.sosfilt(self.sos, db.values, self.axis, self.zi)
            self.zi = State(zi)
        else:
            data = sp.sosfilt(self.sos, db.values, self.axis)
        return db.copy(data=data)
