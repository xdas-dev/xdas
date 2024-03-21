import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from string import Formatter

import numpy as np
import xarray as xr
from tqdm import tqdm

from .coordinates import InterpCoordinate, get_sampling_interval
from .database import Database
from .datacollection import DataCollection
from .virtual import DataLayout, DataSource


def open_mfdatacollection(paths):
    if isinstance(paths, str):
        paths = sorted(glob(paths))
    elif isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"could not find {path}")
    else:
        raise ValueError(
            f"`paths` must be either a string or a list, found {type(paths)}"
        )
    if len(paths) == 0:
        raise FileNotFoundError("no file to open")
    if len(paths) > 100_000:
        raise NotImplementedError(
            "The maximum number of file that can be opened at once is for now limited "
            "to 100 000."
        )
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(open_datacollection, path) for path in paths]
        dcs = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        ]
    return combine(dcs)


def open_treedatacollection(paths, engine="netcdf"):
    """
    Open directory tree structures as data collections.

    The resulting data collection will be a nesting of dicts down to the lower level
    which will be a list of databases.

    Parameters
    ----------
    paths : str
        A path descriptor provided as a string containings placeholders that describes
        the tree structure.
        Two flavours of placeholder can be provided:
        - `{field}`: this level of the tree will behave as a dict. It will use the
        directory/file names as keys.
        - `[field]`: this level of the tree will behave as a list. The directory/file
        names are not considered (as if the placeholder was replaced by a `*`) and
        files are gathered and combined as if using `open_mfdatabase`.

        Several dict placeholders with different names can be provided. They must be
        followed by one or more list placeholders that must share a unique name.

    engine: str
        The type of file to open.

    Returns
    -------
    DataCollection
        The collected data.

    Examples
    --------
    >>> import xdas
    >>> paths = "/data/{node}/{cable}/[acquisition]/proc/[acquisition].h5"
    >>> xdas.open_mfdatacollection(paths, engine="asn") # doctest: +SKIP
    Node:
      CCN:
        Cable:
          N:
            Acquisition:
              0: <xdas.Database (time: ..., distance: ...)>
              1: <xdas.Database (time: ..., distance: ...)>
      SER:
        Cable:
          N:
            Acquisition:
              0: <xdas.Database (time: ..., distance: ...)>
          S:
            Acquisition:
              0: <xdas.Database (time: ..., distance: ...)>
              1: <xdas.Database (time: ..., distance: ...)>
              2: <xdas.Database (time: ..., distance: ...)>


    """
    placeholders = re.findall(r"[\{\[].*?[\}\]]", paths)

    seen = set()
    fields = tuple(
        placeholder[1:-1]
        for placeholder in placeholders
        if not (placeholder in seen or seen.add(placeholder))
    )

    wildcard = paths
    for placeholder in placeholders:
        wildcard = wildcard.replace(placeholder, "*")
    fnames = sorted(glob(wildcard))

    regex = paths
    regex = regex.replace(".", r"\.")
    for placeholder in placeholders:
        if placeholder.startswith("{") and placeholder.endswith("}"):
            regex = regex.replace(placeholder, f"(?P<{placeholder[1:-1]}>.+)")
        else:
            regex = regex.replace(placeholder, r".*")
    regex = re.compile(regex)

    tree = defaulttree(len(fields))
    for fname in fnames:
        match = regex.match(fname)
        bag = tree
        for field in fields[:-1]:
            bag = bag[match.group(field)]
        bag.append(fname)

    return collect(tree, fields, engine)


def collect(tree, fields, engine="netcdf"):
    """
    Collects the data from a tree of paths using `fields` as level names.

    Parameters
    ----------
    tree : nested dict of lists
        The paths grouped in a tree hierarchy.
    fields : tuple of str
        The names of the levels of the tree hierarchy.
    engine : str
        The engine used to open files.

    Returns
    -------
    DataCollection
        The collected data.
    """
    fields = list(fields)
    name = fields.pop(0)
    collection = DataCollection({}, name=name)
    for key, value in tree.items():
        if isinstance(value, list):
            dc = open_mfdatabase(value, engine, squeeze=False)
            dc.name = fields[0]
            collection[key] = dc
        else:
            collection[key] = collect(value, fields, engine)
    return collection


def defaulttree(depth):
    """Generate a default tree of lists with given depth."""
    if depth == 1:
        return list()
    else:
        return defaultdict(lambda: defaulttree(depth - 1))


def open_mfdatabase(
    paths, engine="netcdf", tolerance=None, squeeze=True, verbose=False
):
    """
    Open a multiple file database.

    Parameters
    ----------
    paths : str or list
        The path names given as a shell-style wildcards string or a list of paths.
    engine : str
        The engine to use to read the file.
    tolerance : float or timedelta64, optional
        The tolerance to consider that the end of a file is continuous with the begging
        of the following

    Returns
    -------
    Database
        The database containing all files data.

    Raises
    ------
    FileNotFound
        If no file can be found.
    """
    if isinstance(paths, str):
        paths = sorted(glob(paths))
    elif isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"could not find {path}")
    else:
        raise ValueError(
            f"`paths` must be either a string or a list, found {type(paths)}"
        )
    if len(paths) == 0:
        raise FileNotFoundError("no file to open")
    if len(paths) > 100_000:
        raise NotImplementedError(
            "The maximum number of file that can be opened at once is for now limited "
            "to 100 000."
        )
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(open_database, path, engine=engine) for path in paths
        ]
        dbs = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        ]
    return aggregate(dbs, "time", tolerance, squeeze, None, verbose)


def combine(
    dcs, dim="first", tolerance=None, squeeze=False, virtual=None, verbose=False
):
    leaves = [dc for dc in dcs if isinstance(dc, list)]
    nodes = [dc for dc in dcs if isinstance(dc, dict)]
    if leaves and not nodes:
        dbs = [db for dc in leaves for db in dc]
        dc = aggregate(dbs, dim, tolerance, squeeze, virtual, verbose)
        dc.name = leaves[0].name
        return dc
    elif nodes and not leaves:
        (name,) = set(dc.name for dc in nodes)
        keys = sorted(set.union(*[set(dc.keys()) for dc in nodes]))
        return DataCollection(
            {key: combine([dc[key] for dc in dcs if key in dc]) for key in keys},
            name,
        )
    else:
        raise NotImplementedError("cannot combine mixed node/leave levels for now")


def aggregate(
    dbs, dim="first", tolerance=None, squeeze=False, virtual=None, verbose=False
):
    dbs = sorted(dbs, key=lambda db: db[dim][0].values)
    out = []
    bag = []
    for db in dbs:
        if not bag:
            bag = [db]
        elif db.coords.drop(dim).equals(bag[-1].coords.drop(dim)) and (
            get_sampling_interval(db, dim) == get_sampling_interval(bag[-1], dim)
        ):
            bag.append(db)
        else:
            out.append(bag)
            bag = [db]
    out.append(bag)
    collection = DataCollection(
        [concatenate(bag, dim, tolerance, virtual, verbose) for bag in out]
    )
    if squeeze and len(collection) == 1:
        return collection[0]
    else:
        return collection


def concatenate(dbs, dim="first", tolerance=None, virtual=None, verbose=None):
    """
    Concatenate several databases along a given dimension.

    Parameters
    ----------
    dbs : list
        List of databases to concatenate.
    dim : str
        The dimension along which concatenate.
    tolerance : float of timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, zero by default.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        databases are virtual. By default tries to create a virtual dataset if possible.

    Returns
    -------
    Database
        The concatenated database.
    """
    dbs = sorted(dbs, key=lambda db: db[dim][0].values)
    axis = dbs[0].get_axis_num(dim)
    dim = dbs[0].dims[axis]
    dims = dbs[0].dims
    dtype = dbs[0].dtype
    shape = tuple(
        sum([db.shape[a] for db in dbs]) if a == axis else dbs[0].shape[a]
        for a in range(len(dims))
    )
    if virtual is None:
        virtual = all(isinstance(db.data, DataSource) for db in dbs)
    if virtual:
        data = DataLayout(shape, dtype)
    else:
        data = np.zeros(shape, dtype)
    idx = 0
    tie_indices = []
    tie_values = []
    if verbose:
        iterator = tqdm(dbs, desc="Linking database")
    else:
        iterator = dbs
    for db in iterator:
        selection = tuple(
            slice(idx, idx + db.shape[axis]) if d == dim else slice(None) for d in dims
        )
        if virtual:
            data[selection] = db.data.vsource
        else:
            data[selection] = db.values
        tie_indices.extend(idx + db[dim].tie_indices)
        tie_values.extend(db[dim].tie_values)
        idx += db.shape[axis]
    coord = InterpCoordinate(
        {"tie_indices": tie_indices, "tie_values": tie_values}, dim
    )
    coord = coord.simplify(tolerance)
    coords = dbs[0].coords.copy()
    coords[dim] = coord
    return Database(data, coords)


def open_database(fname, group=None, engine="netcdf", **kwargs):
    """
    Open a database.

    Parameters
    ----------
    fname : str
        The path of the database.
    group : str, optional
        The file group where the database is located, by default None which corresponds
        to the root of the file.
    engine : str, optional
        The file format, by default "netcdf".

    Returns
    -------
    Database
        The opened database.

    Raises
    ------
    ValueError
        If the engine si not recognized.

    Raises
    ------
    FileNotFound
        If no file can be found.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError("no file to open")
    if engine == "netcdf":
        return Database.from_netcdf(fname, group=group, **kwargs)
    elif engine == "asn":
        from .io.asn import read

        return read(fname)
    elif engine == "optasense":
        from .io.optasense import read

        return read(fname)
    elif engine == "sintela":
        from .io.sintela import read

        return read(fname)
    elif callable(engine):
        return engine(fname)
    else:
        raise ValueError("engine not recognized")


def open_datacollection(fname, **kwargs):
    """
    Open a DataCollection from a file.

    Parameters
    ----------
    fname : str
        The path of the DataCollection.

    Returns
    -------
    DataCollection
        The opened DataCollection.

    Raises
    ------
    FileNotFound
        If no file can be found.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError("no file to open")
    return DataCollection.from_netcdf(fname, **kwargs)


def asdatabase(obj, tolerance=None):
    """
    Try to convert given object to a database.

    Only support Database or DataArray as input.

    Parameters
    ----------
    obj : object
        The objected to convert
    tolerance : float or datetime64, optional
        For dense coordinates, tolerance error for interpolation representation, by
        default zero.

    Returns
    -------
    Database
        The object converted to a Database. Data is not copied.

    Raises
    ------
    ValueError
        _description_
    """
    if isinstance(obj, Database):
        return obj
    elif isinstance(obj, xr.DataArray):
        return Database.from_xarray(obj)
    else:
        raise ValueError("Cannot convert to database.")


def split(db, dim="first"):
    if not isinstance(db[dim], InterpCoordinate):
        raise TypeError("the dimension to split must have as type `InterpCoordinate`.")
    (points,) = np.nonzero(np.diff(db[dim].tie_indices, prepend=[0]) == 1)
    indices = [db[dim].tie_indices[point] for point in points]
    div_indices = [0] + indices + [db.sizes[dim]]
    return DataCollection(
        [
            db.isel({dim: slice(div_indices[idx], div_indices[idx + 1])})
            for idx in range(len(div_indices) - 1)
        ]
    )


def chunk(db, nchunk, dim="first"):
    nsamples = db.sizes[dim]
    if not isinstance(nchunk, int):
        raise TypeError("`n` must be an integer")
    if nchunk <= 0:
        raise ValueError("`n` must be larger than 0")
    if nchunk >= nsamples:
        raise ValueError("`n` must be smaller than the number of samples")
    chunk_size, extras = divmod(nsamples, nchunk)
    chunks = [0] + extras * [chunk_size + 1] + (nchunk - extras) * [chunk_size]
    div_points = np.cumsum(chunks, dtype=np.int64)
    return DataCollection(
        [
            db.isel({dim: slice(div_points[idx], div_points[idx + 1])})
            for idx in range(nchunk)
        ]
    )
