from collections.abc import Callable
from fractions import Fraction
from functools import wraps
from typing import Any

import numpy as np
import scipy.signal as sp

from .coordinates import Coordinate, get_sampling_interval
from .core import concatenate, open_datacollection, split
from .database import Database
from .datacollection import DataCollection


class Sequential(list):
    """
    A class to handle a sequence of operations. Each operation is represented by an
    Atom class object, which contains the function and its arguments.

    Sequence inherits from list, and therefore behaves as it.

    Parameters
    ----------
    atoms: list
        The sequence of operations. Each element must either be an Atom, a Sequence, or
        an unitary callable.
    name: str
        A label given to this sequence.

    Examples
    --------
    >>> from xdas import PartialAtom, PartialStateAtom, Sequential
    >>> import xdas.signal as xp
    >>> import numpy as np

    >>> sequence = Sequential(
    ...     [
    ...         PartialAtom(xp.taper, dim="time"),
    ...         PartialStateAtom(xp.lfilter, [1.0], [0.5], ..., dim="time", state="zi"),
    ...         PartialAtom(np.square),
    ...     ],
    ...     name="Low frequency energy",
    ... )
    >>> sequence
    Low frequency energy:
      0: taper(..., dim=time)
      1: lfilter([1.0], [0.5], ..., dim=time)  [stateful]
      2: square(...)

    >>> sequence = Sequential(
    ...     [
    ...         PartialAtom(xp.decimate, 16, dim="distance"),
    ...         sequence,
    ...     ]
    ... )
    >>> sequence
    Sequence:
      0: decimate(..., 16, dim=distance)
      1:
        Low frequency energy:
          0: taper(..., dim=time)
          1: lfilter([1.0], [0.5], ..., dim=time)  [stateful]
          2: square(...)

    >>> from xdas.synthetics import generate
    >>> db = generate()
    >>> sequence(db)
    <xdas.Database (time: 300, distance: 26)>
    [[0.000000e+00 0.000000e+00 0.000000e+00 ... 0.000000e+00 0.000000e+00
      0.000000e+00]
     [4.828612e-30 9.894384e-17 3.227090e-14 ... 1.774912e-13 2.119855e-13
      1.793267e-13]
     [3.109880e-29 1.530490e-17 1.554244e-14 ... 9.062270e-14 1.612131e-11
      2.394051e-12]
     ...
     [2.056248e-28 2.659485e-15 6.235765e-13 ... 1.724542e-11 5.370072e-13
      6.058423e-12]
     [1.570712e-29 4.312953e-16 8.089942e-14 ... 5.433228e-13 6.281834e-13
      6.815881e-14]
     [0.000000e+00 0.000000e+00 0.000000e+00 ... 0.000000e+00 0.000000e+00
      0.000000e+00]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """

    def __init__(self, atoms: Any, name: str | None = None) -> None:
        super().__init__()
        for atom in atoms:
            if not isinstance(atom, (PartialAtom, Sequential)):
                atom = PartialAtom(atom)
            self.append(atom)
        self.name = name

    def __call__(self, x: Any) -> Any:
        for atom in self:
            x = atom(x)
        return x

    def __repr__(self) -> str:
        width = len(str(len(self)))
        name = self.name if self.name is not None else "sequence"
        s = f"{name.capitalize()}:\n"
        for idx, value in enumerate(self):
            label = f"  {idx:{width}}: "
            if isinstance(value, PartialAtom):
                s += label + repr(value) + "\n"
            else:
                s += label + "\n"
                s += "\n".join(f"    {e}" for e in repr(value).split("\n")[:-1]) + "\n"
        return s

    def reset(self) -> None:
        """Resets the state of all StateAtom of the sequence."""
        for atom in self:
            if isinstance(atom, PartialStateAtom):
                atom.reset()


class State:
    def __init__(self, state):
        self.state = state


class Atom:
    def __init__(self):
        super().__setattr__("_config", {})
        super().__setattr__("_state", {})
        super().__setattr__("_filters", {})

    def __repr__(self):
        name = self.__class__.__name__
        sig = ", ".join(
            f"{key}={value}" for key, value in self._config.items() if value is not None
        )
        s = f"{name}({sig})"
        for name, filter in self._filters.items():
            s += "\n" + "\n".join(f"  {e}" for e in repr(filter).split("\n"))
        return s

    def __setattr__(self, name, value):
        match value:
            case State(state=state):
                self._state[name] = state
                super().__setattr__(name, state)
            case Atom():
                self._filters[name] = value
                super().__setattr__(name, value)
            case other:
                self._config[name] = value
                super().__setattr__(name, other)

    @property
    def state(self):
        return self._state | {
            name: filter.state for name, filter in self._filters.items() if filter.state
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


class PartialAtom(Atom):
    """
    Base class for an xdas operation, to be used in conjunction with Sequence. Each
    Atom should be seen as an elementary operation to apply to the data, such as
    tapering, multiplication, integration, etc. More complex operations (fk-analysis,
    strain-to-displacement conversion, ...) can be written as a sequence of Atoms,
    executed in the right order. Each Atom can optionally be labelled with the `name`
    argument for easy identification in long sequences.

    Parameters
    ----------
    func : Callable
        The function that is called. It must take a unique data object as first
        argument and returns a unique output. Subsequent arguments are given by `*args`
        and `**kwargs`. If the data is not passed as first argument, an Ellipsis must
        be provided in the `*args` parameters.
    *args : Any
        Positional arguments to pass to `func`. If the data to process is passed as the
        nth argument, the nth element of `args` must contain an Ellipis (`...`).
    name : str
        Name to identify the function.
    **kwargs : Any
        Keyword arguments to pass to `func`.

    StateAtom uses one reserved keyword arguments that cannot by passed to `func`:
    'name'.


    Examples
    --------
    >>> from xdas import PartialAtom
    >>> import xdas.signal as xp
    >>> PartialAtom(xp.decimate, 2, dim="time", name="downsampling")
    decimate(..., 2, dim=time)

    >>> import numpy as np
    >>> PartialAtom(np.square)
    square(...)

    """

    def __init__(
        self, func: Callable, *args: Any, name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__()
        if not callable(func):
            raise TypeError("`func` should be callable")
        if not any(arg is ... for arg in args):
            args = (...,) + args
        if sum(arg is ... for arg in args) > 1:
            raise ValueError("`*args` must contain at most one Ellipsis")
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __call__(self, x: Any) -> Any:
        args = tuple(x if arg is ... else arg for arg in self.args)
        return self.func(*args, **self.kwargs)

    def __repr__(self) -> str:
        func = getattr(self.func, "__name__", "<function>")
        args = []
        for value in self.args:
            if value is ...:
                args.append("...")
            elif len(str(value)) > 10:
                args.append(f"<{type(value).__name__}>")
            else:
                args.append(str(value))
        kwargs = []
        for key, value in self.kwargs.items():
            if len(str(value)) > 10:
                value = f"<{type(value).__name__}>"
            kwargs.append(f"{key}={value}")
        params = ", ".join(args + kwargs)
        return f"{func}({params})"


class PartialStateAtom(PartialAtom):
    """
    A subclass of Atom that provides some logic for handling data states, which need to
    be updated throughout the execution chain. An example of a stateful operation is a
    recursive filter, passing on the state from t to t+1.

    The StateAtom class assumes that the stateful function takes two inputs: some data
    and a state. The stateful function must return the processed data and modified
    state.

    >>> def func(x, *args, state="zi", **kwargs):
    ...     return x, zf

    Here, `state` is a given keyword argument that contains the state, which can differ
    from one function to the next. The user must pass a dict `{"zi": value}` to provide
    the initial state. If no initial state is provided, the "init" string will be used.
    That special string indicates to the function to initialize the state and to return
    it along with the result. All xdas statefull function accepts this convention.

    StateAtom uses reserved keyword arguments that cannot by passed to `func`:
    'name' and 'state'.

    Parameters
    ----------
    func : Callable
        The function to call. It takes the data object to process as the first argument
        and the actual state as second argument. It returns the processed data along
        with the updated state. Subsequent arguments are given by `*args` and
        `**kwargs`. If the data is not passed as first argument, an Ellipsis must
        be provided in the `*args` parameters.
    *args : Any
        Positional arguments to pass to `func`. If the data to process is passed as the
        nth argument, the nth element of `args` must contain an Ellipis (`...`).
    state: Any
        The initial state that will be passed at the `func`. If `state` is a dict, the
        key indicates the keyword argument used by `func` for state passing, and the
        value contains the state. The "init" string is used to indicate than a new
        state must be initialized. If `state` is a string, it will use the default
        "init" code by default for that keyword argument.
    name : Hashable
        Name to identify the function.
    **kwargs : Any
        Keyword arguments to pass to `func`.

    Examples
    --------
    >>> from xdas import PartialStateAtom
    >>> import xdas.signal as xp
    >>> import scipy.signal as sp

    >>> sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")

    By default, `state` is the expected keyword argument and 'init' is the send value
    to ask for initialisation.

    >>> PartialStateAtom(xp.sosfilt, sos, ..., dim="time", state="zi")
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

    To manually specify the keyword argument and initial value a dict must be passed to
    the state keyword argument.

    >>> PartialStateAtom(xp.sosfilt, sos, ..., dim="time", state={"zi": "init"})
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

    """

    def __init__(self, func, *args, state, name=None, **kwargs):
        super().__init__(func, *args, name=name, **kwargs)
        if isinstance(state, str):
            state = {state: "init"}
        if isinstance(state, tuple):
            state = dict(zip(state, ["init"] * len(state)))
        for key, value in state.items():
            if key == "state":
                raise KeyError("`state` is a reserved word")
            setattr(self, key, State(value))

    def __repr__(self) -> str:
        return super().__repr__() + "  [stateful]"

    def __call__(self, x: Any) -> Any:
        args = tuple(x if arg is ... else arg for arg in self.args)
        kwargs = self.kwargs | self._state
        x, *state = self.func(*args, **kwargs)
        for key, value in zip(self._state, state):
            setattr(self, key, State(value))
        return x

    def reset(self):
        for key in self._state:
            setattr(self, key, State("init"))


def atomized(func):
    """
    Make the function return an Atom if `...` is passed.

    Functions that receive a `state` keyword argument which is not None will be
    considered as statefull atoms.

    Parameters
    ----------
    func: callable
        A function with a main data input that will trigger atomization if `...` is
        passed. If it has a state keword argument this later will trigger statefull
        atomization if anything but None is passed.

    Returns
    -------
    callable
        The atomized function. This latter has the same documentation and names than
        the original function. If `...` is passed as a positional argument, returns an
        Atom object. If a further `state` keyword argument is pass with a value other
        than `None`, retruns a StateAtom object.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if any(arg is ... for arg in args):
            state = {key: "init" for key, value in kwargs.items() if value is ...}
            if state:
                return PartialStateAtom(func, *args, state=state, **kwargs)
            else:
                return PartialAtom(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


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


class LFilter(Atom):
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
            data, zf = sp.lfilter(self.b, self.a, db.values, self.axis, self.zi)
            self.zi = State(zf)
        else:
            data = sp.lfilter(self.b, self.a, db.values, self.axis)
        return db.copy(data=data)


class SOSFilter(Atom):
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


class Resample(Atom):
    def __init__(
        self, target, maxup=1000, maxdown=1000, window=("kaiser", 5.0), dim="last"
    ):
        super().__init__()
        self.target = target
        self.maxup = maxup
        self.maxdown = maxdown
        self.dim = dim
        self.upsampling = None
        self.firfilter = None
        self.downsampling = None

    def initialize(self, db, chunk=None, **kwargs):
        ...

    def call(self, db, **kwargs):
        db = self.upsampling(db)
        db = self.firfilter(db)
        db = self.downsampling(db)
        return db


class DownSample(Atom):
    def __init__(self, factor, dim="last"):
        super().__init__()
        self.factor = factor
        self.dim = dim

    def initialize(self, db, chunk=None, **kwargs):
        if chunk == self.dim:
            self.buffer = State(db.isel({self.dim: slice(0, 0)}))

    def call(self, db, **kwargs):
        if hasattr(self, "buffer"):
            print(self.buffer)
            db = concatenate([self.buffer, db], self.dim)
            divpoint = db.sizes[self.dim] - db.sizes[self.dim] % self.factor
            db, buffer = split(db, [divpoint], self.dim)
            self.buffer = State(buffer)
        return db.isel({self.dim: slice(None, None, self.factor)})


class UpSample(Atom):
    def __init__(self, factor, dim="last"):
        super().__init__()
        self.factor = factor
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
        data[slc] = db.values
        coords = db.coords.copy()
        delta = get_sampling_interval(db, self.dim, cast=False)
        tie_indices = coords[self.dim].tie_indices * self.factor
        tie_values = coords[self.dim].tie_values
        tie_indices[-1] += self.factor - 1
        print(delta)
        print((self.factor - 1) / self.factor)
        tie_values[-1] += (self.factor - 1) / self.factor * delta
        coords[self.dim] = Coordinate(
            {
                "tie_indices": tie_indices,
                "tie_values": tie_values,
            },
            self.dim,
        )
        return Database(data, coords, name=db.name, attrs=db.attrs)


class FIRFilter(Atom):
    def __init__(
        self,
        order,
        cutoff,
        btype="bandpass",
        wtype="hamming",
        width=None,
        scale=True,
        dim="last",
    ):
        super().__init__()
        self.order = order
        self.cutoff = cutoff
        self.btype = btype
        self.wtype = wtype
        self.width = width
        self.scale = scale
        self.dim = dim
        self.lfilter = LFilter(..., [1.0], self.dim)

    def initialize(self, db, **kwargs):
        self.fs = State(1.0 / get_sampling_interval(db, self.dim))
        self.initialize_from_state()

    def initialize_from_state(self):
        taps = sp.firwin(
            self.order,
            self.cutoff,
            self.width,
            self.wtype,
            self.btype,
            self.scale,
            self.fs,
        )
        self.lfilter.b = taps

    def call(self, db, **kwargs):
        return self.lfilter(db, **kwargs)
