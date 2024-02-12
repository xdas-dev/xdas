from collections import UserDict, Counter
from collections.abc import Hashable, Callable
from typing import Any, Type

from .processing import ProcessingChain, DatabaseLoader, DatabaseWriter
from .database import Database

ChainType = Type[ProcessingChain]
LoaderType = Type[DatabaseLoader]
WriterType = Type[DatabaseWriter]
DatabaseType = Type[Database]


class Sequence(UserDict):
    """
    A class to handle a sequence of operations.
    Each operation is represented by an Atom class object,
    which contains the function and its arguments.
    For stateful operations, use the StateAtom class.

    Sequence inherits from UserDict, and therefore behaves
    as a dictionary. It is extended with convenience
    methods to reorder the dictionary. (By default, all
    dict objects in Python >= 3.7 maintain their insertion order).

    Example usage:
    ```
    op1 = xdas.Atom(xdas.signal.taper, dim="time")
    op2 = xdas.Atom(xdas.signal.taper, dim="space", name="spatial taper")
    op3 = xdas.Atom(numpy.square)
    op4 = xdas.Atom(my_func, arg1=1.0, arg2=[0, 1, 2])
    sequence = xdas.Sequence(op1, op2, op3, op4)
    sequence["abs"] = xdas.Atom(np.abs)
    del sequence["spatial taper"]
    print(sequence)  # Print a summary of the Sequence
    sequence.execute(db)
    ```

    Attributes
    ----------
    name_counter : Counter
        Counter object to keep track of (duplicate)
        dictionary keys

    """

    def __init__(self, *args) -> None:
        """
        Initialise the Sequence with an arbitrary number of
        Atom or StateAtom objects. Any non-[State]Atom objects
        are discarded.
        """

        # Counter for keeping track of the (unique) names
        self.name_counter = Counter()

        # Loop over atoms
        for atom in args:
            if not isinstance(atom, Atom):
                continue
            # Set atom name
            self._check_name(atom)
            # Add parent reference
            atom.parent = self

        # Reset counter (parent __init__ will check names again)
        self.name_counter = Counter()

        # Create keyword args from unique names
        init = dict((atom.name, atom) for atom in args)
        # Initialise parent class
        return super().__init__(init)
    
    
    def __setitem__(self, key: Hashable, val: Any) -> None:
        """Override set() method to ensure unique atom atom naming"""

        # Only allow [State]Atom instances
        if not isinstance(val, Atom):
            return
        
        # If the key has no length:
        # get its string representation
        if not hasattr(key, "__len__"):
            key = str(key)
        
        # If a key is provided
        if (len(key) > 0):
            # If no name is provided with the Atom
            # set the atom name to key
            if len(val.name) == 0:
                val.name = key

        # Bookkeeping (checking for name duplicates)
        key = self._check_name(val)
        return super().__setitem__(key, val)
    

    def __getitem__(self, key: Hashable) -> Any:
        """
        Get an item from the Sequence.
        Overrides `Sequence.get(key)`

        Parameters
        ----------
        key : Hashable
            The key corresponding with the item to delete.
            If an integer is provided, it is interpreted
            as the position number in the Sequence.
        """
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)
    
    
    def __delitem__(self, key: Hashable) -> None:
        """
        Delete an item from the Sequence.
        Overrides `del Sequence[key]`

        Parameters
        ----------
        key : Hashable
            The key corresponding with the item to delete.
            If an integer is provided, it is interpreted
            as the position number in the Sequence.
        """
        # TODO: update counter and rename keys?
        # self.name_counter[key] -= 1
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__delitem__(key)
    

    def __str__(self) -> str:
        # Get the atom representations and join
        items = [f"{i:<3} {item}" for i, (_, item) in enumerate(self.data.items())]
        itemstr = "\n".join(items)
        # Make a nice header
        header = "xdas sequence"
        line = "-" * len(header) + "\n"
        header += "\n"
        return line + header + line + itemstr
    
    
    def _check_name(self, atom: Any) -> Hashable:
        """
        Assign a key to an Atom object. If no key is given
        by the user (atom.name), create one based on the
        function name (atom.func.__name__). To ensure that 
        all keys in the sequence are unique, this method 
        uses `name_counter` to keep track of potential 
        duplicates (likely the result of applying the same
        function multiple times with different arguments).

        Parameters
        ----------
        atom : Atom
            The instantiated Atom object

        Returns
        -------
        name : str
            The key that has been assigned to atom
        
        """

        # Get atom name
        name = atom.name

        # If the name has no length:
        # get its string representation
        if not hasattr(name, "__len__"):
            name = str(name)

        if len(name) == 0:
            name = atom.func.__name__
        
        # Check for duplicates
        self.name_counter[name] += 1
        if self.name_counter[name] > 1:
            name = f"{name}{self.name_counter[name]}"
        
        # Update name (and self.name_counter)
        atom.name = name

        return name
    
    
    def _insert(self, pos: Hashable, atom: Any, locator: Callable) -> None:
        """
        Insert an Atom at a particular position in the Sequence.
        This method is called from a parent Atom object using the
        `Atom.insert_before` or `Atom.insert_after` methods. 
        Example:
            `Sequence[key].insert_before(Atom)`

        Parameters
        ----------
        pos : Hashable
            Key of the position at which `atom` needs 
            to be inserted. Whether the insertion is
            before or after is determined by `locator`
        atom : Atom
            The instantiated Atom object to be inserted
        locator : Callable
            A function passed on from the calling Atom
            that indicates whether the insertion is made
            before or after `pos`
        
        """

        # List of current keys
        keys = list(self.keys())
        # Locate insertion position
        pos = locator(keys, pos)
        # Add key:item to dictionary (checks for duplicate names)
        self.__setitem__(atom.name, atom)
        name = atom.name
        # Insert key at insertion position
        keys.insert(pos, name)
        
        # Reorder dict
        self.data = {key: self.get(key) for key in keys}
        pass

    def _move(self, key: Hashable, locator: Callable) -> None:
        """
        Move an Atom within the Sequence in a direction indicated
        by `locator`. This method is called from a parent Atom object 
        using the `Atom.move_up` or `Atom.move_down` methods. 
        Example:
            `Sequence[key].move_down()`

        Parameters
        ----------
        pos : Hashable
            Key of the position at which `atom` needs 
            to be inserted. Whether the insertion is
            before or after is determined by `locator`
        atom : Atom
            The instantiated Atom object to be inserted
        locator : Callable
            A function passed on from the calling Atom
            that indicates whether the insertion is made
            before or after `pos`
        
        """

        # List of current keys
        keys = list(self.keys())
        # Current position of key
        current_pos = keys.index(key)
        # Target position of key
        target_pos = locator(keys, key)

        # Moving key up
        if current_pos == target_pos:
            target_pos -= 1
            current_pos += 1
        # Moving key down
        else:
            target_pos += 1

        # Key is already in first spot, 
        # so moving up does nothing
        if target_pos < 0:
            return
        
        # Key is already in last spot, 
        # so moving down does nothing
        if target_pos > len(keys):
            return
        
        # Insert key in target position
        keys.insert(target_pos, key)
        # Delete old key position
        keys.pop(current_pos)
        # Reorder dict
        self.data = {key: self.get(key) for key in keys}
        pass

    def get_chain(self) -> ChainType:
        """
        Link Atoms into a processing chain
        
        Returns
        -------
        chain : ProcessingChain
            The ProcessingChain object that contains
            additional execution methods.
        
        """
        atoms = [atom for _, atom in self.data.items()]
        chain = ProcessingChain(atoms)
        return chain
    
    def execute(self, 
                data_loader: DatabaseType | LoaderType, 
                data_writer: None | WriterType=None) -> None | DatabaseType:
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
            returned to the user.

        Returns
        -------
        result : None | Database
            If `data_writer` is None, the result of the
            ProcessingChain execution is returned, else
            None is returned.
        
        """
        
        # Get the list of Atoms in the Sequence
        chain = self.get_chain()
        # If an xarray Database is provided, execute
        # the ProcessingChain directly.
        if isinstance(data_loader, Database):
            result = chain(data_loader)

        # If no DatabaseWriter is provided, 
        # return the results directly
        if data_writer is None:
            return result
        
        # TODO: add IO logic
        
        pass

class Atom:
    """
    Base class for an xdas operation, to be used in conjunction 
    with Sequence. Each Atom should be seen as an elementary 
    operation to apply to the data,such as tapering, 
    multiplication, integration, etc. More complex operations 
    (fk-analysis, strain-to-displacement conversion, ...) can be 
    written as a sequence of Atoms, executed in the right order.
    Each Atom can optionally be labelled with the `name` argument
    for easy identification in long sequences.

    Example usage:
    ```
    sequence = xdas.Sequence(
        xdas.Atom(xdas.signal.taper, dim="space"),
        xdas.Atom(xdas.signal.taper, dim="time", name="taper space"),
    )
    
    sequence[0].move_down()
    sequence["taper space"].insert_before(
        xdas.Atom(numpy.square),
    )
    sequence[-1].delete()
    ```
    
    Attributes
    ----------
    func : Callable
        The function to execute in the Sequence. It
        takes an xarray Database as the first argument
        and returns a modified copy of the Database.
        Subsequent arguments are given by `func`.
    name : Hashable
        Name to identify the function
    kwargs : Any
        Arguments to pass to `func`

    Methods
    -------
    delete():
        Deletes the selected Atom
    set_args(**kwargs):
        Override the selected Atom's keyword arguments
    insert_after(Atom):
        Inserts a new Atom behind of the selected Atom
    insert_before(Atom):
        Inserts a new Atom ahead of the selected Atom
    mode_down():
        Moves the selected Atom down in the Sequence
    mode_up():
        Moves the selected Atom up in the Sequence
    """

    def __init__(self, func: Callable, name: Hashable="", **kwargs: Any) -> None:
        self.func = func
        if hasattr(kwargs, "name"):
            del kwargs["name"]
        self.kwargs = kwargs
        self.name = name
        pass

    def __call__(self, db) -> Any:
        return self.func(db, **self.kwargs)

    def __str__(self) -> str:
        args = [f"{key}={val}" for key, val in self.kwargs.items()]
        argstr = ", ".join(args)
        return f"{self.name:<25}{self.func.__name__}({argstr})"
    
    def _locate_before(self, x, a):
        return x.index(a)
    
    def _locate_after(self, x, a):
        return x.index(a)+1
    
    def delete(self) -> None:
        self.parent.pop(self.name)
        pass

    def set_args(self, **kwargs) -> None:
        self.kwargs = kwargs
        pass

    def insert_after(self, atom) -> None:
        self.parent._insert(self.name, atom, locator=self._locate_after)
        pass

    def insert_before(self, atom) -> None:
        self.parent._insert(self.name, atom, locator=self._locate_before)
        pass

    def move_down(self) -> None:
        self.parent._move(self.name, locator=self._locate_after)
        pass

    def move_up(self) -> None:
        self.parent._move(self.name, locator=self._locate_before)
        pass


class StateAtom(Atom):
    """
    A subclass of Atom that provides some logic for handling
    data states, which need to be updated throughout the
    execution chain. An example of a stateful operation is
    a recursive filter, passing on the state from t to t+1.
    
    The Atom base class assumes that a given function takes
    an xarray Database as the first argument, and returns
    a modified copy of this Database. The StateAtom class
    assumes that the stateful function takes a Database and
    an initialized state, and returns the modified Database
    and modified state, i.e.:

    `db, state = func(db, state_arg=state, **kwargs)`
    
    Here, `state_arg` is the name of the keyword argument
    that contains the state, which can differ from one function
    to the next. For example, in scipy.signal.sosfilt, the
    state argument is `zi`, and so `state_arg` is `zi`.

    Example usage:
    ```
    state = np.zeros((10, 100))
    state_op = StateAtom(
        scipy.signal.sosfilt, axis=0, state_arg="zi", state=state
    )
    ```

    Methods
    -------
    initialize_state(state):
        Set the initial state.
    
    """

    _state_initialized = False

    def __init__(self, 
                 func: Callable, 
                 state_arg: Hashable,
                 state: None | Any=None, 
                 **kwargs) -> None:
        
        self._state_arg = state_arg
        self._state = state
        if state is not None:
            self._state_initialized = True
        
        super().__init__(func, **kwargs)

    def __str__(self) -> str:
        return super().__str__() + "  [stateful]"
    
    def __call__(self, db) -> Any:
        kwargs = self.kwargs.copy()
        kwargs.update(self._state_arg, self._state)
        db, state = self.func(db, **self.kwargs)
        self._set_state(state)
        return db
    
    def _set_state(self, state: Any) -> None:
        self._state = state
        pass

    def initialize_state(self, state: Any) -> None:
        self._set_state(state)
        self._state_initialized = True
        pass
