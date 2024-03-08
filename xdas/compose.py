from collections.abc import Callable
from typing import Any


class Sequence(list):
    """
    A class to handle a sequence of operations. Each operation is represented by an
    Atom class object, which contains the function and its arguments. For stateful
    operations, use the StateAtom class.

    Sequence inherits from list, and therefore behaves as it.

    Parameters
    ----------
    atoms: list
        The sequence of operations. Each element must either be an Atom
        (or a StateAtom), a Sequence, or an unitary callable.
    name: str
        A label given to this sequence.

    Examples
    --------
    >>> from xdas import Atom, StateAtom, Sequence
    >>> import xdas.signal as xp
    >>> import numpy as np

    >>> sequence = Sequence(
    ...     [
    ...         Atom(xp.taper, dim="time"),
    ...         StateAtom(xp.lfilter, [1.0], [0.5], ..., dim="time"),
    ...         Atom(np.square),
    ...     ],
    ...     name="Low frequency energy",
    ... )
    >>> sequence
    Low frequency energy:
      0: taper(..., dim=time)
      1: lfilter([1.0], [0.5], ..., dim=time)  [stateful]
      2: square(...)

    >>> sequence = Sequence(
    ...     [
    ...         Atom(xp.decimate, 16, dim="distance"),
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
        distance: 0.000 to 10000.000

    """

    def __init__(self, atoms: Any, name: str | None = None) -> None:
        super().__init__()
        for atom in atoms:
            if not isinstance(atom, (Atom, Sequence)):
                atom = Atom(atom)
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
            if isinstance(value, Atom):
                s += label + repr(value) + "\n"
            else:
                s += label + "\n"
                s += "\n".join(f"    {e}" for e in repr(value).split("\n")[:-1]) + "\n"
        return s

    def reset(self) -> None:
        """Resets the state of all StateAtom of the sequence."""
        for atom in self:
            if isinstance(atom, StateAtom):
                atom.reset()


class Atom:
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
    >>> from xdas import Atom

    >>> import xdas.signal as xp
    >>> Atom(xp.decimate, 2, dim="time", name="downsampling")
    decimate(..., 2, dim=time)

    >>> import numpy as np
    >>> Atom(np.square)
    square(...)

    """

    def __init__(
        self, func: Callable, *args: Any, name: str | None = None, **kwargs: Any
    ) -> None:
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


class StateAtom(Atom):
    """
    A subclass of Atom that provides some logic for handling data states, which need to
    be updated throughout the execution chain. An example of a stateful operation is a
    recursive filter, passing on the state from t to t+1.

    The StateAtom class assumes that the stateful function takes two inputs: some data
    and a state. The stateful function must return the processed data and modified
    state.

    >>> def func(x, *args, state="init", **kwargs):
    ...     return x, state

    Here, `state` is a given keyword argument that contains the state, which can differ
    from one function to the next. For example, in scipy.signal.sosfilt, the state
    argument is `zi`. In that case, the user must pass a dict `{"zi": "init"}` to
    indicate the name of the keyword argument related to the state. Here "init" is a
    'special' string that most xdas statefull function accepts. It asks to the function
    to initialize the state and to return it along with the result.

    StateAtom uses to reserved keyword arguments that cannot by passed to `func`:
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
        The initial state that will be passed at the `state` keyword argument of `func`.
        If `state` is a dict, the key indicates the keyword argument used by `func` for
        state passing, and the value contains the state. The "init" string is used to
        indicate than a new state must be initialized.
    name : Hashable
        Name to identify the function.
    **kwargs : Any
        Keyword arguments to pass to `func`.

    Examples
    --------
    >>> import xdas.signal as xp
    >>> import scipy.signal as sp

    >>> sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")

    By default, `state` is the expected keyword argument and 'init' is the send value
    to ask for initialisation.

    >>> StateAtom(xp.sosfilt, sos, ..., dim="time")
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

    To manually specify the keyword argument and initial value a dict must be passed to
    the state keyword argument.

    >>> StateAtom(xp.sosfilt, sos, ..., dim="time", state={"state": "init"})
    sosfilt(<ndarray>, ..., dim=time)  [stateful]

    """

    def __init__(
        self,
        func: Callable,
        *args: Any,
        state: Any = "init",
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(func, *args, name=name, **kwargs)
        if isinstance(state, dict):
            self.state = state
        else:
            self.state = {"state": state}

    def __repr__(self) -> str:
        return super().__repr__() + "  [stateful]"

    def __call__(self, x: Any) -> Any:
        args = tuple(x if arg is ... else arg for arg in self.args)
        x, *state = self.func(*args, **self.kwargs, **self.state)
        for key, value in zip(self.state, state):
            self.state[key] = value
        return x

    def reset(self) -> None:
        for key in self.state:
            self.state[key] = "init"
