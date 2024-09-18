from dask.array import Array

from . import serial


def dumps(arr):
    """Serialize a dask array."""
    return serial.dumps(to_dict(arr))


def loads(data):
    """Deserialize a dask array."""
    return from_dict(serial.loads(data))


def to_dict(arr):
    """Convert a dask array to a dictionary."""
    graph = arr.__dask_graph__().cull(arr.__dask_keys__())
    graph = fuse(graph)
    return {
        "dask": graph,  # TODO: fuse then encode fails...
        "name": arr.name,
        "chunks": arr.chunks,
        "dtype": str(arr.dtype),
    }


def from_dict(dct):
    """Convert a dictionary to a dask array."""
    return Array(**dct)


def fuse(graph):
    """Simpligy a graph by grouping intermediate empty computations."""
    dsk = {}
    ignore = set()
    for key, computation in graph.items():
        if key in ignore:
            continue
        while iskey(computation) and computation in graph:
            ignore.add(computation)
            computation = graph[computation]
        dsk[key] = computation
    return dsk


def iskey(obj):
    if isinstance(obj, str) and len(obj) > 0:
        return True
    elif (
        isinstance(obj, tuple)
        and len(obj) > 1
        and isinstance(obj[0], str)
        and all(isinstance(index, int) for index in obj[1:])
    ):
        return True
    else:
        return False
