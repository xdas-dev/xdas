from collections.abc import Callable
from functools import wraps
from typing import Any

from sympy import sequence

from ..core.dataarray import DataArray
from ..core.datacollection import DataCollection
from ..core.routines import open_datacollection


class State:
    """
    A class to declare a new state or to update a preexising one into an Atom object.

    Parameters
    ----------
    state: Any
        The state to be passed to the Atom object.

    Examples
    --------

    In practice the State object is used when implementing new Atom objects. Bellow a
    dummy example without any class declaration.

    >>> from xdas.atoms import Atom, State

    Let's create an empty Atom object:

    >>> self = Atom()
    >>> self.state
    {}

    We can declare and initialize a new state:

    >>> self.count = State(0)
    >>> self.state
    {'count': 0}

    To update the state, we must use the State object again. The state is updated by:

    >>> self.count = State(self.count + 1)
    >>> self.state
    {'count': 1}

    """

    def __init__(self, state):
        self.state = state


class Atom:
    """
    The base class for atoms. Used to implement new Atom objects.

    Atoms are the building blocks to perform massive computations. They represent
    processing units that can be combined to create complex data processing pipelines.
    Each Atom object is a callable that takes a unique input data object and returns a
    unique processed data object. Atoms can be stateful meaning that they can store
    some memory between calls to ensure the continuity of the processing across chunks.
    The memory of the Atom is stored in the state attribute. The state is a dictionary
    that is updated during each execution of the Atom. The Atom object is initialized
    with an initial state at the first call. In subsequent calls, the state is updated
    **if and only if the `chunk_dim` flag is provided** along with the dimension along
    wich chunking was performed. If this flag is not provided, the state is reset
    between calls. The Atom can be reset manually to its initial state by calling the
    `reset` method.

    When implementing a new Atom object, the user must sublcass the Atom class, and
    at minima define the `initialize` and the `call` methods. The `initialize` method
    is called at the first call to the Atom and is used to initialize the Atom with the
    input data. The `call` method is called at each subsequent call to the Atom and is
    used to perform the main processing logic. The Atom class handles when an how those
    two methods are called.

    To reduce the size of the state that need to be stored, a good practive is to also
    define the `initialize_from_state` method. This method is called in the
    `initialize` as soon as the minimal set of states is initialized. The other states
    that are usefull for the processing but that can be recomputed from the minimal set
    are initialized in the `initialize_from_state` method.

    Attributes
    ----------
        state: dict
            Returns the current state of the atom recursively including
            the state of nested atoms.
        initialized: bool
            Wether the atom has been initialized or not.

    Methods
    -------
        initialize(x, **flags)
            Initializes the atom with the given input.
        initialize_from_state()
            Initializes the atom from its minimal state.
        call(x, **flags)
            Performs the main processing logic of the atom.
        reset() 
            Resets the atom to its initial state.

    """

    def __init__(self):
        object.__setattr__(self, "_config", {})
        object.__setattr__(self, "_state", {})
        object.__setattr__(self, "_atoms", {})

    def __repr__(self):
        name = self.__class__.__name__
        sig = ", ".join(
            f"{key}={value}" for key, value in self._config.items() if value is not None
        )
        s = f"{name}({sig})"
        for name, filter in self._atoms.items():
            s += "\n" + "\n".join(f"  {e}" for e in repr(filter).split("\n"))
        return s

    def __setattr__(self, name, value):
        match value:
            case State(state=state):
                self._state[name] = state
                object.__setattr__(self, name, state)
            case Atom():
                self._atoms[name] = value
                object.__setattr__(self, name, value)
            case other:
                self._config[name] = value
                object.__setattr__(self, name, other)

    @property
    def state(self):
        return self._state | {
            name: filter.state for name, filter in self._atoms.items() if filter.state
        }

    @property
    def initialized(self):
        return all(value is not ... for value in self._state.values())

    def initialize(self, x, **flags): ...

    def initialize_from_state(self): ...

    def call(self, x, **flags): ...

    def __call__(self, x, **flags):
        chunk_dim = flags.get("chunk_dim", None)
        if not self.initialized or chunk_dim is None:
            self.initialize(x, **flags)
        y = self.call(x, **flags)
        if not chunk_dim:
            self.reset()
        return y

    def reset(self):
        for key in self._state:
            setattr(self, key, State(...))
        for _, filter in self._atoms.items():
            filter.reset()

    def save_state(self, path):
        DataCollection(self.state).to_netcdf(path)

    def set_state(self, state):
        for key, value in state.items():
            if isinstance(value, DataArray):
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


class Sequential(Atom, list):
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
    >>> from xdas.atoms import Partial, Sequential
    >>> import xdas.signal as xp
    >>> import numpy as np

    Basic usage:

    >>> seq = Sequential(
    ...     [
    ...         Partial(xp.taper, dim="time"),
    ...         Partial(xp.lfilter, [1.0], [0.5], ..., dim="time", zi=...),
    ...         Partial(np.square),
    ...     ],
    ...     name="Low frequency energy",
    ... )
    >>> seq
    Low frequency energy:
      0: taper(..., dim=time)
      1: lfilter([1.0], [0.5], ..., dim=time)  [stateful]
      2: square(...)

    Nested sequences:

    >>> seq = Sequential(
    ...     [
    ...         Partial(xp.decimate, 16, dim="distance"),
    ...         seq,
    ...     ]
    ... )
    >>> seq
    Sequence:
      0: decimate(..., 16, dim=distance)
      1:
        Low frequency energy:
          0: taper(..., dim=time)
          1: lfilter([1.0], [0.5], ..., dim=time)  [stateful]
          2: square(...)

    Applying the sequence to data:

    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()
    >>> seq(da)
    <xdas.DataArray (time: 300, distance: 26)>
    [[0.000000e+00 0.000000e+00 0.000000e+00 ... 0.000000e+00 0.000000e+00
      0.000000e+00]
     [5.925923e-10 3.640952e-11 1.315744e-11 ... 4.024388e-14 3.245748e-12
      7.679807e-12]
     [3.497487e-09 1.191342e-10 4.021543e-11 ... 7.463132e-12 9.458801e-11
      2.833292e-10]
     ...
     [2.331440e-08 5.785383e-10 4.336722e-11 ... 1.827452e-11 5.099163e-10
      1.320907e-09]
     [1.826236e-09 5.470673e-11 1.045146e-11 ... 1.561169e-13 3.063598e-14
      7.832009e-16]
     [0.000000e+00 0.000000e+00 0.000000e+00 ... 0.000000e+00 0.000000e+00
      0.000000e+00]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """

    def __init__(self, atoms: Any, name: str | None = None) -> None:
        super().__init__()
        for key, atom in enumerate(atoms):
            if not isinstance(atom, Atom):
                atom = Partial(atom)
            self.append(atom)
            self._atoms[key] = atom
        self.name = name

    def call(self, x: Any, **flags) -> Any:
        for atom in self:
            x = atom(x, **flags)
        return x

    def __repr__(self) -> str:
        width = len(str(len(self)))
        name = self.name if self.name is not None else "sequence"
        s = f"{name.capitalize()}:\n"
        for idx, value in enumerate(self):
            label = f"  {idx:{width}}: "
            if isinstance(value, Partial):
                s += label + repr(value) + "\n"
            else:
                s += label + "\n"
                s += "\n".join(f"    {e}" for e in repr(value).split("\n")[:-1]) + "\n"
        return s

    def reset(self) -> None:
        """Resets the state of all StateAtom of the sequence."""
        for atom in self:
            if isinstance(atom, Partial):
                atom.reset()


class Partial(Atom):
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
    >>> from xdas.atoms import Partial
    >>> import xdas.signal as xp
    >>> Partial(xp.decimate, 2, dim="time", name="downsampling")
    decimate(..., 2, dim=time)

    >>> import numpy as np
    >>> Partial(np.square)
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
        self.kwargs = {}
        self.name = name
        for key, value in kwargs.items():
            if value is ...:
                setattr(self, key, State(...))
            elif isinstance(value, State):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

    @property
    def stateful(self):
        return bool(self._state)

    def call(self, x: Any, **flags) -> Any:
        args = tuple(x if arg is ... else arg for arg in self.args)
        kwargs = self.kwargs | self._state
        if self.stateful:
            x, *state = self.func(*args, **kwargs)
            for key, value in zip(self._state, state):
                setattr(self, key, State(value))
            return x
        else:
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
        return f"{func}({params})" + ("  [stateful]" if self.stateful else "")


class StatePartial(Partial):  # TODO: Merge documentation
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
    the initial state. If no initial state is provided, the ... flag will be used.
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
        value contains the state. The ... flag is used to indicate than a new
        state must be initialized. If `state` is a string, it will use the default ...
        flag by default for that keyword argument.
    name : Hashable
        Name to identify the function.
    **kwargs : Any
        Keyword arguments to pass to `func`.

    Examples
    --------
    >>> from xdas.atoms import Partial, State
    >>> import xdas.signal as xp
    >>> import scipy.signal as sp

    >>> sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")

    By default, `state` is the expected keyword argument and 'init' is the send value
    to ask for initialisation.

    >>> Partial(xp.sosfilt, sos, ..., dim="time", zi=...)
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

    To manually specify the keyword argument and initial value a dict must be passed to
    the state keyword argument.

    >>> Partial(xp.sosfilt, sos, ..., dim="time", zi=State(...))
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

    """

    ...


def atomized(func):
    """
    Make the function return an Atom if `...` or an atom is passed as argument.

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
            return Partial(func, *args, **kwargs)
        elif objs := tuple(arg for arg in args if isinstance(arg, Atom)):
            if len(objs) == 1:
                input = objs[0]
            else:
                raise ValueError("Only one Atom object can be passed as function input")
            args = tuple(... if isinstance(arg, Atom) else arg for arg in args)
            output = Partial(func, *args, **kwargs)
            if isinstance(input, Sequential):
                return input.append(output)
            else:
                return Sequential([input, output])
        else:
            return func(*args, **kwargs)

    return wrapper
