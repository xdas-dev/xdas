"""
msgpack-based serialization for dask task graphs.

Handles tuples, slices, callables, ``methodcaller``, and ``itemgetter``
objects.
"""

import importlib

import msgpack
from dask.utils import itemgetter, methodcaller

codes = {
    "tuple": 1,
    "slice": 2,
    "callable": 3,
    "methodcaller": 4,
    "itemgetter": 5,
}


def encode(obj):
    """
    Msgpack *default* hook — encode non-native types as :class:`msgpack.ExtType`.

    Handles ``tuple``, ``slice``, ``callable``, :class:`methodcaller`, and
    :class:`itemgetter`.

    Parameters
    ----------
    obj : object
        Object to encode.

    Returns
    -------
    msgpack.ExtType
    """
    if isinstance(obj, tuple):
        code = codes["tuple"]
        obj = list(obj)
    elif isinstance(obj, slice):
        code = codes["slice"]
        obj = {"start": obj.start, "stop": obj.stop, "step": obj.step}
    elif isinstance(obj, methodcaller):
        code = codes["methodcaller"]
        obj = obj.method
    elif isinstance(obj, itemgetter):
        code = codes["itemgetter"]
        obj = obj.index
    elif callable(obj):
        code = codes["callable"]
        obj = {"module": obj.__module__, "name": obj.__name__}
    else:
        raise TypeError(f"Cannot encode object of type {type(obj)}")
    data = dumps(obj)
    return msgpack.ExtType(code, data)


def decode(code, data):
    """
    Msgpack *ext_hook* — decode an :class:`msgpack.ExtType` back to the original object.

    Parameters
    ----------
    code : int
        Extension type code (one of the values in :data:`codes`).
    data : bytes
        Raw msgpack bytes for the payload.

    Returns
    -------
    object
        The decoded Python object.
    """
    obj = loads(data)
    if code == codes["tuple"]:
        return tuple(obj)
    elif code == codes["slice"]:
        return slice(obj["start"], obj["stop"], obj["step"])
    elif code == codes["callable"]:
        return getattr(importlib.import_module(obj["module"]), obj["name"])
    elif code == codes["methodcaller"]:
        return methodcaller(obj)
    elif code == codes["itemgetter"]:
        return itemgetter(obj)
    else:
        raise ValueError(f"Unknown code {code}")


def dumps(obj):
    """Serialize *obj* to msgpack bytes, encoding extension types via :func:`encode`."""
    return msgpack.dumps(obj, default=encode, strict_types=True)


def loads(obj):
    """Deserialize msgpack *obj* bytes, restoring extension types via :func:`decode`."""
    return msgpack.loads(obj, strict_map_key=False, ext_hook=decode)
