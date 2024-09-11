from dask.array import Array


def to_dict(arr):
    """Convert a dask array to a dictionary."""
    return {
        "dask": fuse(encode(arr.__dask_graph__())),  # TODO: fuse then encode fails...
        "name": arr.name,
        "chunks": arr.chunks,
        "dtype": str(arr.dtype),
    }


def from_dict(dct):
    """Convert a dictionary to a dask array."""
    dct = dct.copy()
    dct["dask"] = decode(dct["dask"])
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
            case (func, *args) if callable(func):
                if func.__name__ == "read_data":
                    computation = "@read", args[0], func.__module__.split(".")[-1]
                else:
                    raise NotImplementedError(
                        f"Function {func.__name__} not supported."
                    )
            case (str(name), *indices) if all(
                isinstance(index, int) for index in indices
            ):
                computation = name + "@" + "@".join(str(index) for index in indices)
        dsk[key] = computation
    return dsk


def decode(graph):
    """Decode a graph by replacing string chunks and functions with tuples."""
    dsk = {}
    for key, computation in graph.items():
        if not iskey(key):
            raise ValueError(f"Invalid key: {key}")
        if "@" in key:
            parts = key.split("@")
            key = (parts[0], *[int(part) for part in parts[1:]])
        match computation:
            case "@read", path, engine:
                from .. import io

                module = getattr(io, engine)
                computation = module.read_data, path
            case str(name):
                if "@" in name:
                    parts = name.split("@")
                    computation = (parts[0], *[int(part) for part in parts[1:]])
        dsk[key] = computation
    return dsk


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
    match obj:
        case str(name) if name and not name.startswith("@") and not name.endswith("@"):
            return True
        case (str(name), *indices) if all(
            isinstance(index, int) for index in indices
        ) and iskey(name):
            return True
        case _:
            return False
