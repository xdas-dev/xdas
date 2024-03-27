import numpy as np
import scipy.signal as sp

from .coordinates import Coordinate, get_sampling_interval
from .core import open_datacollection
from .database import Database
from .datacollection import DataCollection


class State:
    def __init__(self, state):
        self.state = state


class Filter:
    def __init__(self):
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

    @property
    def filters(self):
        return {"data": self._filters} | {
            "children": {
                name: filter.filters
                for name, filter in self._filters.items()
                if filter.filters
            }
        }

    def initialize(self, x, **kwargs):
        return NotImplemented

    def initialize_from_state(self):
        return NotImplemented

    def call(self, x, **kwargs):
        return NotImplemented

    def __call__(self, x, **kwargs):
        if not self.state:
            self.initialize(x, **kwargs)
        return self.call(x, **kwargs)

    def save_state(self, path):
        DataCollection(self.state).to_netcdf(path)

    def set_state(self, state):
        for key, value in state.items():
            if isinstance(value, Database):
                setattr(
                    self, key, State(value.__array__())
                )  # TODO: shouldn't need __array__
                self.initialize_from_state()
            else:
                filter = getattr(self, key)
                filter.set_state(value)

    def load_state(self, path):
        state = open_datacollection(path).load()
        self.set_state(state)


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
        if self.stype == "ba":
            self.iirfilter = LFilter(..., ..., self.dim)
        elif self.stype == "sos":
            self.iirfilter = SOSFilter(..., self.dim)
        else:
            raise ValueError()

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


class LFilter(Filter):
    def __init__(self, b, a, dim):
        super().__init__()
        self.b = b
        self.a = a
        self.dim = dim

    def initialize(self, db, chunk=None, **kwargs):
        self.axis = State(db.get_axis_num(self.dim))
        if self.dim == chunk:
            n_sections = max(len(self.a), len(self.b)) - 1
            shape = tuple(
                n_sections if name == self.dim else size
                for name, size in db.sizes.items()
            )
            self.zi = State(np.zeros(shape))

    def call(self, db, **kwargs):
        if hasattr(self, "zi"):
            data, zi = sp.lfilter(self.sos, db.values, self.axis, self.zi)
            self.zi = State(self.zi.copy(data=zi))
        else:
            data = sp.lfilter(self.sos, db.values, self.axis)
        return db.copy(data=data)


class SOSFilter(Filter):
    def __init__(self, sos, dim):
        super().__init__()
        self.sos = sos
        self.dim = dim

    def initialize(self, db, chunk=None, **kwargs):
        self.axis = State(db.get_axis_num(self.dim))
        if self.dim == chunk:
            n_sections = self.sos.shape[0]
            shape = (n_sections,) + tuple(
                2 if index == self.axis else element
                for index, element in enumerate(db.shape)
            )
            self.zi = State(np.zeros(shape))

    def call(self, db, **kwargs):
        if hasattr(self, "zi"):
            data, zf = sp.sosfilt(self.sos, db.values, self.axis, self.zi)
            self.zi = State(zf)
        else:
            data = sp.sosfilt(self.sos, db.values, self.axis)
        return db.copy(data=data)
