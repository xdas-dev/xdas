import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import xarray as xr
from tqdm import tqdm

from .coordinates import InterpCoordinate
from .database import Database
from .datacollection import DataCollection
from .virtual import DataLayout, DataSource


def open_mfdatabase(paths, engine="netcdf", tolerance=np.timedelta64(0, "us")):
    """
    Open a multiple file database.

    Parameters
    ----------
    paths: str or list
        The path names given as a shell-style wildcards string or a list of paths.
    engine: str
        The engine to use to read the file.
    tolerance: timedelta64
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
    return concatenate(dbs, tolerance=tolerance, verbose=True)


def concatenate(dbs, dim="time", tolerance=None, virtual=None, verbose=None):
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
    coords = dbs[0].coords
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
