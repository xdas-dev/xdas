from collections import UserDict, Counter
from collections.abc import Hashable, Callable, Sequence
from typing import Any, Dict, Type
from functools import partial

from .processing import ProcessingChain, DatabaseLoader, DatabaseWriter
from .database import Database

ChainType = Type[ProcessingChain]
LoaderType = Type[DatabaseLoader]
WriterType = Type[DatabaseWriter]
DatabaseType = Type[Database]


class Sequence(UserDict):

    # Counter for keeping track of the (unique) names
    name_counter = Counter()

    def __init__(self, *args) -> None:

        # Loop over atoms
        for atom in args:
            # Check/update atom name
            self._check_name(atom)
            # Add parent reference
            atom.parent = self

        # Create keyword args from unique names
        init = dict((atom.name, atom) for atom in args)
        # print(kwargs)
        # Initialise parent class
        return super().__init__(init)
        pass
    
    def __setattr__(self, key: Hashable, val: Any) -> None:
        if isinstance(val, Atom):
            val.name = key
            key = self._check_name(val)
        return super().__setattr__(key, val)
    
    def __str__(self) -> str:
        # Get the atom representations and join
        items = [f"{i:<3} {item}" for i, (_, item) in enumerate(self.data.items())]
        itemstr = "\n".join(items)
        # Make a nice header
        header = "xdas sequence"
        line = "-" * len(header) + "\n"
        header += "\n"
        return line + header + line + itemstr

    
    def __getitem__(self, key: Hashable) -> Any:
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)
    
    def __delitem__(self, key: Any) -> None:
        return super().__delitem__(key)
    
    def _check_name(self, atom: Any) -> Hashable:

        # Get atom name
        name = atom.name
        if len(name) == 0:
            name = atom.func.__name__
        
        # Check for duplicates
        self.name_counter[name] += 1
        if self.name_counter[name] > 1:
            name = f"{name}{self.name_counter[name]}"
        
        # Update name (and self.name_counter)
        atom.name = name

        return name        
    
    def _insert(self, pos: Hashable, atom, locator: Callable) -> None:

        self._check_name(atom)
        name = atom.name

        # List of current keys
        keys = list(self.keys())
        # Locate insertion position
        pos = locator(keys, pos)
        # Insert key at insertion position
        keys.insert(pos, name)
        # Add key:item to dictionary
        self.__setitem__(name, atom)
        # Reorder dict
        self.data = {key: self.get(key) for key in keys}
        pass

    def _move(self, key: Hashable, locator: Callable) -> None:
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
        # Create a list of partial functions which
        # take the database as the first argument.
        # The lambda function is a trick to attach
        # `name` as a keyword without passing it
        # onto the function (which would raise an
        # exception). `name` is only used for the
        # representation of the chain.
        atoms = [partial(lambda x, name, **kwargs: val.func(x, **kwargs), **val.args, name=key) for key, val in self.data.items()]
        chain = ProcessingChain(atoms)
        return chain
    
    def execute(self, 
                data_loader: DatabaseType | LoaderType, 
                data_writer: None | WriterType=None):
        
        chain = self.get_chain()
        if isinstance(data_loader, Database):
            result = chain(data_loader)

        if data_writer is None:
            return result
        
        # TODO: add IO logic
        
        pass

class Atom:

    def __init__(self, func: Callable, name: Hashable="", **kwargs: Any) -> None:
        self.func = func
        self.kwargs = kwargs
        self.name = name
        pass

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

    # def set_name(self, name: Hashable) -> None:
    #     self.name = name
    #     pass

    def set_args(self, **kwargs) -> None:
        self.kwargs = kwargs
        pass

    def insert_before(self, atom) -> None:
        self.parent._insert(self.name, atom, locator=self._locate_before)
        pass

    def insert_after(self, atom) -> None:
        self.parent._insert(self.name, atom, locator=self._locate_after)
        pass

    def move_up(self) -> None:
        self.parent._move(self.name, locator=self._locate_before)
        pass

    def move_down(self) -> None:
        self.parent._move(self.name, locator=self._locate_after)
        pass


class StateAtom(Atom):

    def __init__(self, func: Callable, state: None | Any=None, **kwargs) -> None:
        # TODO: add a state
        self.state = state
        super().__init__(func, **kwargs)

    def __str__(self) -> str:
        return super().__str__() + "  [stateful]"
