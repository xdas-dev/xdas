from dask.array import Array


def to_dict(arr):
    """Convert a dask array to a dictionary."""
    return {
        "dask": fuse(encode(arr.__dask_graph__())),  # TODO: fuse then encode fails...
        "name": arr.name,
        "chunks": arr.chunks,
        "dtype": str(arr.dtype),
    }


def from_dict(dct, read_fn):
    """Convert a dictionary to a dask array."""
    dct = dct.copy()
    dct["dask"] = decode(dct["dask"], read_fn)
    return Array(**dct)


def encode(graph):
    """Encode a graph by replacing tuple chunks and functions with strings."""
    dsk = {}
    for key, computation in graph.items():
        match key:
            case (str(name), *indices) if all(
                isinstance(index, int) for index in indices
            ):
                key = name + "@" + "@".join(str(index) for index in indices)
        match computation:
            # TODO: put correct engine and check func name
            case (func, *args) if callable(func) and func.__name__ == "read_data":
                computation = "@read", args[0], "silixa"
            case (str(name), *indices) if all(
                isinstance(index, int) for index in indices
            ):
                computation = name + "@" + "@".join(str(index) for index in indices)
        dsk[key] = computation
    return dsk


def decode(graph, read_fn):
    """Decode a graph by replacing string chunks and functions with tuples."""
    dsk = {}
    for key, computation in graph.items():
        parts = key.split("@")
        key = (parts[0], *[int(part) for part in parts[1:]])
        match computation:
            case "@read", path, engine:
                computation = read_fn, path  # TODO: use engine
        dsk[key] = computation
    return dsk


def fuse(graph):
    """Simpligy a graph by grouping intermediate empty computations."""
    dsk = {}
    ignore = set()
    for key, computation in graph.items():
        if key in ignore:
            continue
        while isinstance(computation, str) and computation in graph:
            computation = graph[computation]
            ignore.add(computation)
        dsk[key] = computation
    return dsk
