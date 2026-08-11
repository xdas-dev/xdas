"""
Base classes for stateful processing atoms.

Includes :class:`Atom`, :class:`State`, :class:`Sequential`, :class:`Partial`,
the :func:`atomized` decorator, the :func:`as_function` generator of the
function form of an atom class, and the :func:`compose` primitive behind the
``>>`` operator.
"""

import importlib
import inspect
import re
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from ..core import DataArray, DataCollection, open_datacollection


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


class Atom(np.lib.mixins.NDArrayOperatorsMixin):
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

    Atoms compose into pipelines with the ``>>`` operator (see :func:`compose`)
    and trace ordinary numpy expressions: applying a ufunc to an atom appends
    the operation to the pipeline instead of computing it.

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
        fresh()
            Returns a stateless clone sharing the configuration.

    """

    def __init__(self):
        object.__setattr__(self, "_config", {})
        object.__setattr__(self, "_state", {})
        object.__setattr__(self, "_atoms", {})

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    __hash__ = object.__hash__

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
        """Dict of the current state, including nested atom states."""
        return self._state | {
            name: filter.state for name, filter in self._atoms.items() if filter.state
        }

    @property
    def initialized(self):
        """``True`` if every state key, nested atoms included, is initialised."""
        return all(value is not ... for value in self._state.values()) and all(
            atom.initialized for atom in self._atoms.values()
        )

    def initialize(self, x, **flags):
        """Initialise the atom from a first chunks of data."""
        return NotImplemented

    def initialize_from_state(self):
        """Initialise the atom from its current state."""
        return NotImplemented

    def call(self, x, **flags):
        """Process a chunk of data."""
        return NotImplemented

    def __call__(self, x, **flags):
        """Process input data, initializing state if needed and resetting after final chunk."""
        chunk_dim = flags.get("chunk_dim", None)
        self._check_chunk_dim(x, chunk_dim)
        if not self.initialized or chunk_dim is None:
            self.initialize(x, **flags)
        y = self.call(x, **flags)
        if not chunk_dim:
            self.reset()
        return y

    def _check_chunk_dim(self, x, chunk_dim):
        """Raise if this atom cannot process *x* chunked along *chunk_dim*."""

    def _refuse_chunked_along(self, dim, chunk_dim, x=None):
        """
        Raise if a whole-record operation is being chunked along its own dim.

        The guard for atoms that need the whole record along the dimension
        they work on: call it from :meth:`initialize` (or a
        :meth:`_check_chunk_dim` override) with the dimension the atom works
        along and the dimension the stream is chunked along. ``"first"`` and
        ``"last"`` aliases are resolved against *x* when given, so the
        comparison is never made on an unresolved alias.
        """
        if chunk_dim is None:
            return
        if x is not None and hasattr(x, "dims"):
            if dim == "first":
                dim = x.dims[0]
            elif dim == "last":
                dim = x.dims[-1]
        if dim is None or dim in ("first", "last") or dim == chunk_dim:
            name = (
                getattr(self, "name", None)
                or getattr(getattr(self, "func", None), "__name__", None)
                or type(self).__name__
            )
            raise ValueError(
                f"{name} needs the whole record along {dim!r} and cannot "
                f"process data chunked along {chunk_dim!r}: process the "
                f"stream unchunked, or chunk along another dimension"
            )

    def __rshift__(self, other):
        """Compose with *other* into a new pipeline: ``atom >> atom``."""
        if isinstance(other, Atom) or callable(other):
            return compose(self, other)
        return NotImplemented

    def __rrshift__(self, other):
        """Prepend a bare callable, or apply the pipeline: ``da >> atom``."""
        if callable(other):
            return compose(other, self)
        return self(other)

    __irshift__ = __rshift__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Append the ufunc to the pipeline instead of computing it (tracing)."""
        if ufunc is np.right_shift and method == "__call__" and len(inputs) == 2:
            # ``da >> atom``: the DataArray operand dispatched through numpy.
            left, right = inputs
            if right is self and not isinstance(left, Atom):
                return self.__rrshift__(left)
        if method != "__call__":
            return NotImplemented
        if "out" in kwargs:
            # In-place operators rebind the target name with their return
            # value, so tracing them keeps value semantics: drop the `out`
            # and append the out-of-place operation.
            if kwargs["out"] != (self,):
                return NotImplemented
            kwargs = {key: value for key, value in kwargs.items() if key != "out"}
        if sum(input is self for input in inputs) != 1 or any(
            isinstance(input, Atom) and input is not self for input in inputs
        ):
            # Never bail silently into computation: fan-in is a design
            # boundary, and the error must say so at the line that traced it.
            raise TypeError(
                "traced fan-in is not supported: a pipeline is a single "
                "chain, so a ufunc may involve one atom exactly once "
                "(combine the branches eagerly, or wrap the whole "
                "expression in a function and Partial it)"
            )
        args = tuple(... if input is self else input for input in inputs)
        return compose(self, Partial(ufunc, *args, **kwargs))

    def reset(self):
        """Reset all state entries to ``...`` (uninitialised sentinel)."""
        for key in self._state:
            setattr(self, key, State(...))
        for filter in self._atoms.values():
            filter.reset()

    def fresh(self):
        """
        Return a stateless clone of this atom: same config, no state.

        Config is shared *by reference* (a model is never deep-copied),
        nested atoms are recursed, and every state entry comes back
        uninitialised — where :meth:`reset` wipes this instance, ``fresh``
        leaves it untouched, so one configured atom can serve several
        independent runs.
        """
        clone = type(self).__new__(type(self))
        Atom.__init__(clone)
        for name, value in vars(self).items():
            if name in ("_config", "_state", "_atoms"):
                continue
            if name in self._atoms:
                setattr(clone, name, self._atoms[name].fresh())
            elif name in self._state:
                setattr(clone, name, State(...))
            elif name in self._config:
                setattr(clone, name, value)
            else:
                # Attributes opted out of the registries with
                # `object.__setattr__` (pure helpers) travel as they are.
                object.__setattr__(clone, name, value)
        return clone

    def save_state(self, path):
        """Serialise the current state to a NetCDF4 file at *path*."""
        DataCollection(self.state).to_netcdf(path)

    def set_state(self, state):
        """
        Restore the atom state from a previously saved state dict.

        Parameters
        ----------
        state : dict
            Mapping of state key → value as returned by :attr:`state`.
        """
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
        """Load the atom state from the NetCDF4 file at *path*."""
        state = open_datacollection(path).load()
        self.set_state(state)


class Sequential(Atom, list):
    """
    A class to handle a sequence of operations.

    Each operation is represented by an Atom class object, which contains the
    function and its arguments. Sequence inherits from list, and therefore
    behaves as it.

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
    >>> import xdas.signal as xs
    >>> import numpy as np

    Basic usage:

    >>> seq = Sequential(
    ...     [
    ...         Partial(xs.taper, dim="time"),
    ...         Partial(xs.lfilter, [1.0], [0.5], ..., dim="time", zi=...),
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
    ...         Partial(xs.decimate, 16, dim="distance"),
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
        """Pass *x* through each atom in order and return the final result."""
        for atom in self:
            x = atom(x, **flags)
        return x

    def fresh(self):
        """Return a stateless clone: each stage cloned, config shared."""
        return type(self)([atom.fresh() for atom in self], name=self.name)

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


class Partial(Atom):
    """
    Wraps a function into an Atom.

    It works similarly to `functools.partial` but with additional features. If the
    input is not the first argument, an `Ellipsis` (`...`) can be used to indicate the
    position of the input. A `name` argument can be used to better identify the
    resulting atom.

    Some level of state passing can be achieved by passing `...` for one or several
    keyword arguments. In that case, the function is expected to accept `...` as
    keyword arguments, to properly initialize the corresponding states and to return as
    many additional outputs as there are stateful arguments.

    Partial uses several reserved keyword arguments that cannot by passed to `func`:
    'func', 'name' and 'state'.

    Parameters
    ----------
    func : Callable
        The function that is called. One of its argument is used as the input of the
        resulting Atom object while other parameters are fixed. The position of the
        input is by default the first argument. Otherwise, an Ellipsis (`...`) must be
        provided in the `*args` parameters to indicate the position of the input.
        The function must return a unique output except if the function is stateful. In
        that case, the function must return the processed data as first output and the
        updated state as additional outputs.
    *args : Any
        Positional arguments to pass to `func`. If the data to process is passed as the
        nth argument, the nth element of `args` must contain an Ellipsis (`...`).
    name : str
        Name to identify the function.
    **kwargs : Any
        Keyword arguments to pass to `func`. If one of the keyword arguments is `...`,
        it will be treated as a passing state and initialized or updated at each call.


    Examples
    --------
    >>> import numpy as np
    >>> import scipy.signal as sp
    >>> import xdas.signal as xs
    >>> from xdas.atoms import Partial

    Examples of a stateless atom:

    >>> Partial(xs.decimate, 2, dim="time")
    decimate(..., 2, dim=time)

    >>> Partial(np.square)
    square(...)

    Examples of a stateful atom with input data as second argument:

    >>> sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
    >>> Partial(xs.sosfilt, sos, ..., dim="time", zi=...)
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

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
        # A whole-record function marked with `_whole_record` refuses chunked
        # execution along its working dimension; that dimension is resolved
        # from the call arguments so the guard can compare it with the
        # chunked one.
        dim_arg = getattr(func, "_whole_record_dim_arg", None)
        if dim_arg is not None:
            try:
                bound = inspect.signature(func).bind_partial(*self.args, **self.kwargs)
                bound.apply_defaults()
                self.dim = bound.arguments.get(dim_arg)
            except (TypeError, ValueError):
                self.dim = None

    def _check_chunk_dim(self, x, chunk_dim):
        """Refuse chunking along the working dim of a whole-record function."""
        if getattr(self.func, "_whole_record_dim_arg", None) is not None:
            self._refuse_chunked_along(self.dim, chunk_dim, x)

    @property
    def stateful(self):
        """``True`` if any keyword argument is being passed as state."""
        return bool(self._state)

    def call(self, x: Any, **flags) -> Any:
        """Call the wrapped function with *x* substituted at the ``...`` position."""
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

    def __reduce__(self):
        return self.from_state, (self.get_state(),)

    @classmethod
    def from_state(cls, state):
        """Reconstruct a :class:`Partial` from a serialised *state* dict."""
        func = getattr(
            importlib.import_module(state["func"]["module"]), state["func"]["name"]
        )
        return cls(func, *state["args"], name=state["name"], **state["kwargs"])

    def get_state(self):
        """Return a JSON-serialisable dict describing the wrapped function and args."""
        return {
            "func": {"module": self.func.__module__, "name": self.func.__name__},
            "args": self.args,
            "kwargs": self.kwargs,
            "name": self.name,
        }


def compose(input, output):
    """
    Chain *input* then *output* into a new Sequential with value semantics.

    Composition never mutates its operands: each call returns a fresh
    Sequential, so intermediate pipelines stay usable on their own. Bare
    callables are wrapped into Partial atoms. Unnamed Sequentials are
    flattened; a named input Sequential keeps its name, a named output
    Sequential stays nested.

    This is the primitive behind the ``>>`` operator and operator tracing.

    Parameters
    ----------
    input : Atom or callable
        The upstream atom or pipeline.
    output : Atom or callable
        The atom or pipeline to append.

    Returns
    -------
    Sequential
        A new pipeline running *input* then *output*.
    """
    if not isinstance(input, Atom):
        input = Partial(input)
    if not isinstance(output, Atom):
        output = Partial(output)
    head = list(input) if isinstance(input, Sequential) else [input]
    tail = (
        list(output)
        if isinstance(output, Sequential) and output.name is None
        else [output]
    )
    name = input.name if isinstance(input, Sequential) else None
    return Sequential(head + tail, name=name)


def atomized(func):
    """
    Make the function return an Atom if `...` or an atom is passed as argument.

    In case `...` is passed as a positional argument, the function is wrapped into a
    Partial object. If an Atom object is passed as a positional argument, a new
    Sequential is returned that chains that atom with the atomized function (the
    input atom is never mutated). Otherwise, the function is called as is.

    Applied to an Atom subclass, `atomized` instead generates its function
    form (see :func:`as_function`): a function taking the data as first
    argument followed by the class parameters, with the same `...`/atom
    dispatch as above.

    Parameters
    ----------
    func: callable or type
        The function to wrap as a Partial atom if any `...` or input atom is passed.
        It must handle the `...` argument as a placeholder for the input data and for
        the passing states. It must return a unique output except if the function is
        stateful. In that case, the function must return the processed data as first
        output and the updated state as additional outputs. If an Atom subclass is
        given, its function form is returned instead.

    Returns
    -------
    output or atom: Any or (Partial or Sequential)
        if no `...` or Atom object is passed as a positional argument, returns the
        output of the function. If an Atom object is passed as a positional argument,
        returns a new Sequential chaining the Atom object and the atomized function.
        If `...` is passed as a positional argument, returns a Partial object containing
        the atomized function. This latter has the same documentation and names than the
        original function.

    Examples
    --------
    >>> import numpy as np
    >>> from xdas.atoms import atomized

    Basic usage:

    >>> @atomized
    ... def square(x):
    ...     return x ** 2
    >>> square(2)
    4
    >>> square(...)
    square(...)

    Passing an Atom object as input:

    >>> square(square(...))
    Sequence:
      0: square(...)
      1: square(...)

    Passing a stateful function:

    >>> @atomized
    ... def cumsum(x, cum=None):
    ...     return_state = cum is not None
    ...     if cum is None or cum is ...:
    ...         cum = 0.0
    ...     out = np.cumsum(x) + cum
    ...     cum += out[-1]
    ...     if return_state:
    ...         return out, cum
    ...     else:
    ...         return out
    >>> cumsum(..., cum=...)
    cumsum(...)  [stateful]

    """
    if isinstance(func, type) and issubclass(func, Atom):
        return as_function(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Dispatch to Partial/Sequential when ``...`` or an Atom is passed, else call directly."""
        if any(arg is ... for arg in args):
            return Partial(func, *args, **kwargs)
        elif objs := tuple(arg for arg in args if isinstance(arg, Atom)):
            if len(objs) == 1:
                input = objs[0]
            else:
                raise ValueError("Only one Atom object can be passed as function input")
            args = tuple(... if isinstance(arg, Atom) else arg for arg in args)
            return compose(input, Partial(func, *args, **kwargs))
        else:
            return func(*args, **kwargs)

    return wrapper


def _whole_record(dim_arg="dim"):
    """
    Mark a function as needing the whole record along its working dimension.

    The decorator to apply at the definition site of a whole-record function,
    under :func:`atomized`: the resulting atoms refuse chunked execution
    along the dimension named by the *dim_arg* argument (resolved from the
    call arguments, aliases included), via
    :meth:`Atom._refuse_chunked_along`.
    """

    def decorator(func):
        func._whole_record_dim_arg = dim_arg
        return func

    return decorator


def as_function(cls):
    """
    Generate the function form of an Atom subclass.

    A lowercase function taking the data as first argument, then the class
    parameters: called with data it builds the atom and applies it eagerly,
    called with ``...`` in the data slot it returns the configured atom, and
    called with an atom or a pipeline it returns a new pipeline extended
    with this atom.

    Parameters
    ----------
    cls : type
        The :class:`Atom` subclass to generate the function form of.

    Returns
    -------
    callable
        The function form, named after the class in snake case.
    """
    parameters = [
        parameter
        for key, parameter in inspect.signature(cls.__init__).parameters.items()
        if key != "self"
    ]

    def wrapper(da, *args, **kwargs):
        atom = cls(*args, **kwargs)
        if da is ...:
            return atom
        if isinstance(da, Atom):
            return compose(da, atom)
        return atom(da)

    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls.__name__).lower()
    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__module__ = cls.__module__
    wrapper.__signature__ = inspect.Signature(
        [inspect.Parameter("da", inspect.Parameter.POSITIONAL_OR_KEYWORD), *parameters]
    )
    wrapper.__doc__ = (
        f"Apply a :class:`{cls.__name__}` atom to `da`.\n\n"
        "Passing ``...`` as `da` returns the atom itself; passing an atom or a\n"
        "pipeline returns a new pipeline extended with this atom. The other\n"
        f"parameters are those of :class:`{cls.__name__}`, documented below.\n\n"
    ) + (inspect.getdoc(cls) or "")
    return wrapper
