from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from tqdm import tqdm

from .coordinates import Coordinate
from .database import Database, DataCollection
from .virtual import DataLayout


def open_mfdatabase(paths, engine="netcdf", tolerance=np.timedelta64(0, "us")):
    """
    Open a multiple file database.

    Parameters
    ----------
    paths: str
        The path names given using shell=style wildcards.
    engine: str
        The engine to use to read the file.
    tolerance: timedelta64
        The tolerance to consider that the end of a file is continuous with the begging
        of the following

    ReturnsDA
    -------
    Database
        The database containing all files data.
    """
    fnames = sorted(glob(paths))
    if len(fnames) > 100_000:
        raise NotImplementedError(
            "The maximum number of file that can be opened at once is for now limited "
            "to 100 000."
        )
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(open_database, fname, engine=engine) for fname in fnames
        ]
        dbs = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        ]
    return concatenate(dbs, tolerance=tolerance, virtual=True)


def concatenate(dbs, tolerance=None, virtual=False):
    """
    Concatenate several databases along the time dimension.

    Parameters
    ----------
    dbs : list
        List of databases to concatenate.
    tolerance : float of timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, zero by default.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        databases are virtual.

    Returns
    -------
    Database
        The concatenated database.
    """
    if tolerance is None:
        if np.issubdtype(dbs[0]["time"].dtype, np.datetime64):
            tolerance = np.timedelta64(0, "us")
        else:
            tolerance = 0.0 
    dbs = sorted(dbs, key=lambda db: db["time"][0])
    shape = (sum([db.shape[0] for db in dbs]), dbs[0].shape[1])
    dtype = dbs[0].dtype
    if virtual:
        layout = DataLayout(shape, dtype)
    else:
        layout = np.zeros(shape, dtype)
    idx = 0
    tie_indices = []
    tie_values = []
    for db in tqdm(dbs, desc="Linking database"):
        if virtual:
            layout[idx : idx + db.shape[0]] = db.data.vsource
        else:
            layout[idx : idx + db.shape[0]] = db.values
        tie_indices.extend(idx + db["time"].tie_indices)
        tie_values.extend(db["time"].tie_values)
        idx += db.shape[0]
    time = Coordinate(tie_indices, tie_values).simplify(tolerance)
    return Database(layout, {"time": time, "distance": dbs[0]["distance"]})


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
    """
    if engine == "netcdf":
        return Database.from_netcdf(fname, group=group, **kwargs)
    elif engine == "asn":
        from .io.asn import read

        return read(fname)
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
    """
    return DataCollection.from_netcdf(fname, **kwargs)
