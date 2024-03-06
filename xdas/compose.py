from collections import Counter, UserDict
from collections.abc import Callable, Hashable
from copy import copy
from typing import Any, Self, Type

from .database import Database
from .processing import DatabaseLoader, DatabaseWriter, ProcessingChain

ChainType = Type[ProcessingChain]
LoaderType = Type[DatabaseLoader]
WriterType = Type[DatabaseWriter]
DatabaseType = Type[Database]


class Sequence(list):
    """
    A class to handle a sequence of operations. Each operation is represented by an
    Atom class object, which contains the function and its arguments. For stateful
    operations, use the StateAtom class.

    Sequence inherits from list, and therefore behaves as it.

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
    ...     name="Low pass filter",
    ... )
    ... sequence

    >>> from xdas.synthetics import generate

    >>> db = generate()
    >>> sequence(db)

    >>> Sequence([sequence] * 2)

    """

    def __init__(self, atoms: Any, name: str | None = None) -> None:
        """
        Initialise the Sequence with an arbitrary number of Atom or StateAtom objects.
        """
        super().__init__()
        for atom in atoms:
            if not isinstance(atom, (Atom, Sequence)):
                atom = Atom(atom)
            self.append(atom)
        self.name = name

    def __call__(self, x):
        for atom in self:
            x = atom(x)
        return x

    def __repr__(self):
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
        The function to execute in the Sequence. It takes a unique data object as the
        first argument and returns a unique output. Subsequent arguments are given by
        `*args` and `**kwargs`.
    *arg : Any
        Positional arguments to pass to `func`.
    name : Hashable
        Name to identify the function.
    **kwargs : Any
        Keyword arguments to pass to `func`.


    Examples
    --------
    >>> from xdas import Atom

    >>> import xdas.signal as xp
    >>> atom = Atom(xp.decimate, 2, dim="time", name="downsampling")

    >>> import numpy as np
    >>> atom = Atom(np.square)

    """

    def __init__(
        self, func: Callable, *args: Any, name: str | None = None, **kwargs: Any
    ) -> None:
        if not ... in args:
            args = (...,) + args
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __call__(self, x) -> Any:
        args = tuple(x if arg is ... else arg for arg in self.args)
        return self.func(*args, **self.kwargs)

    def __repr__(self) -> str:
        func = getattr(self.func, "__name__", "<function>")
        args = ["..." if value is ... else str(value) for value in self.args]
        kwargs = [f"{key}={value}" for key, value in self.kwargs.items()]
        params = ", ".join(args + kwargs)
        return f"{func}({params})"


class StateAtom(Atom):
    """
    A subclass of Atom that provides some logic for handling data states, which need to
    be updated throughout the execution chain. An example of a stateful operation is a
    recursive filter, passing on the state from t to t+1.

    The StateAtom class assumes that the stateful function takes two inputs: some data
    and a state. The stateful function must return the modified Database and modified state, i.e.:

    >>> db, state = func(db, <state>=state, **kwargs)

    Here, <state> is a given keyword argument that contains the state,
    which can differ from one function to the next. For example, in
    scipy.signal.sosfilt, the state argument is `zi`, and so `state_arg` is `zi`.


    Parameters
    ----------
    func : Callable
        The function to execute in the Sequence. It takes the data object to process as
        the first argument and the actual state as second argument. It returns the
        processed data along with the updated state. Subsequent arguments are given by
        `*args` and `**kwargs`.
    *arg : Any
        Positional arguments to pass to `func`.
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

    >>> atom = StateAtom(xp.filter, dim="time", state="init")

    """

    def __init__(
        self, func: Callable, *args, state: Any = "init", name=None, **kwargs
    ) -> None:

        super().__init__(func, *args, name=name, **kwargs)
        if isinstance(state, dict):
            self.state = state
        else:
            self.state = {"state": state}

    def __repr__(self) -> str:
        return super().__repr__() + "  [stateful]"

    def __call__(self, x) -> Any:
        args = tuple(x if arg is ... else arg for arg in self.args)
        x, self.state = self.func(*args, **self.kwargs, **self.state)
        return x

    def reset(self):
        self.state = "init"


def process(
    sequence,
    data_loader: DatabaseType | LoaderType,
    data_writer: None | WriterType = None,
) -> None | DatabaseType:
    """
    A convenience method that executes the current
    Sequence directly, rather than explicitly requesting
    the ProcessingChain and executing that.

    Parameters
    ----------
    data_loader : Database | DatabaseLoader
        If an xarray Database is provided, the
        ProcessingChain is executed directly on the
        Database. If a DatabaseLoader is provided,
        chunked processing is applied.
    data_writer : None | DatabaseWriter
        If provided, the result of the ProcessingChain
        is passed on to the DatabaseWriter for
        persistent storage. If None, the result is
        kept in memory.

    Returns
    -------
    result : Database
        The result of the ProcessingChain execution

    """

    chain = sequence.get_chain()
    if isinstance(data_loader, Database):
        result = chain(data_loader)
    elif isinstance(data_loader, DatabaseLoader):
        result = chain.process(data_loader, data_writer)
    else:
        raise TypeError(f"data_loader type '{type(data_loader)}' not accepted")

    return result
