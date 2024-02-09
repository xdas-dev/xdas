from collections import UserDict
from collections.abc import Hashable, Callable
from typing import Any, Dict, Type
from functools import partial

from .processing import ProcessingChain

ChainType = Type[ProcessingChain]


class Sequence(UserDict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass
    
    def __setitem__(self, key: Hashable, item: Dict) -> None:
        func, args = next(iter(item.items()))
        atom = SequenceAtom(self, key, func, args)
        return super().__setitem__(key, atom)

    def __str__(self) -> str:
        items = [repr(item) for _, item in self.data.items()]
        itemstr = "\n".join(items)
        header = "xdas sequence"
        line = "-" * len(header) + "\n"
        header += "\n"
        return line + header + line + itemstr
    
    @staticmethod
    def _locate_before(x, a):
        return x.index(a)
    
    @staticmethod
    def _locate_after(x, a):
        return x.index(a)+1
    
    def _insert(self, pos: Hashable, obj: Dict, locator: Callable) -> None:
        # Get key to insert and corresponding items
        key, item = next(iter(obj.items()))
        # List of current keys
        keys = list(self.keys())
        # Locate insertion position
        pos = locator(keys, pos)
        # Insert key at insertion position
        keys.insert(pos, key)
        # Add key:item to dictionary
        self.__setitem__(key, item)
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
        atoms = [partial(val.func, **val.args, name=key) for key, val in self.data.items()]
        chain = ProcessingChain(atoms)
        return chain

class SequenceAtom:

    def __init__(self, parent, key: Hashable, func: Any, args: Dict) -> None:
        self.parent = parent
        self.key = key
        self.func = func
        # TODO:
        # If func is string: import from .atoms
        # If func is a user func: parallelise
        #   if func has dim or axis argument, include this in parallel
        # Store self.func
        self.args = args
        pass

    def __repr__(self) -> str:
        args = [f"{key}={val}" for key, val in self.args.items()]
        argstr = ", ".join(args)
        return f"[{self.key}]: {self.func}({argstr})"
    
    def delete(self) -> None:
        self.parent.pop(self.key)
        pass

    def set_args(self, args: Dict) -> None:
        self.args = args
        pass

    def insert_before(self, item: Dict) -> None:
        self.parent._insert(self.key, item, locator=self.parent._locate_before)
        pass

    def insert_after(self, item: Dict) -> None:
        self.parent._insert(self.key, item, locator=self.parent._locate_after)
        pass

    def move_up(self) -> None:
        self.parent._move(self.key, locator=self.parent._locate_before)
        pass

    def move_down(self) -> None:
        self.parent._move(self.key, locator=self.parent._locate_after)
        pass

