import os

import numpy as np
import scipy.signal as sp

import xdas


def generate(
    dirpath=None,
    starttime="2023-01-01T00:00:00",
    resolution=(np.timedelta64(20, "ms"), 25.0),
):
    """
    Generate some dummy files to bu used in code testing.

    It generates a monolithic `sample.nc` file and a chunked files (`001.nc`, `002.nc`,
    `003.nc`).

    Parameters
    ----------
    dirpath : str, optional
        Directory where files will be written. If None, not file will be written.
    starttime : str
        The starttime of the file, will be parsed by `np.datetime64(starttime)`.
    resolution : (timedelta64, float)
        The temporal and spatial sampling intervals.

    Examples
    --------

    >>> import os
    >>> import xdas
    >>> from xdas.synthetics import generate
    >>> from tempfile import TemporaryDirectory

    >>> with TemporaryDirectory() as dirpath:
    ...     generate(
    ...         dirpath, 
    ...         starttime="2024-01-01T00:00:00", 
    ...         resolution=(np.timedelta64(5, "ms"), 10.0),
    ...     )
    ...     db_monolithic = xdas.open_database(os.path.join(dirpath, "sample.nc"))
    ...     db_chunked = xdas.open_mfdatabase(os.path.join(dirpath, "00*.nc"))
    ...     db_monolithic.equals(db_chunked)
    True

    """
    np.random.seed(42)
    span = (np.timedelta64(6, "s"), 10000.0)
    shape = (span[0] // resolution[0], int(span[1] // resolution[1]) + 1)
    starttime = np.datetime64(starttime).astype("datetime64[ns]")
    snr = 10
    vp = 4000
    vs = vp / 1.75
    xc = 5_000
    fc = 10.0
    t0 = 1.0
    t = np.arange(shape[0]) * resolution[0] / np.timedelta64(1, "s")
    s = np.arange(shape[1]) * resolution[1]
    d = np.hypot(xc, (s - np.mean(s)))
    ttp = d / vp
    tts = d / vs
    data = np.zeros(shape)
    for k in range(shape[1]):
        data[:, k] += sp.gausspulse(t - ttp[k] - t0, fc) / 2
        data[:, k] += sp.gausspulse(t - tts[k] - t0, fc)
    data /= np.max(np.abs(data), axis=0, keepdims=True)
    data += np.random.randn(*shape) / snr
    data = np.diff(data, prepend=0, axis=-1)
    data = np.diff(data, prepend=0, axis=0)
    db = xdas.Database(
        data=data,
        coords={
            "time": dict(
                tie_indices=[0, shape[0] - 1],
                tie_values=[starttime, starttime + resolution[0] * (shape[0] - 1)],
            ),
            "distance": dict(
                tie_indices=[0, shape[1] - 1],
                tie_values=[0.0, resolution[1] * (shape[1] - 1)],
            ),
        },
    )
    if dirpath is not None:
        db.to_netcdf(os.path.join(dirpath, "sample.nc"))
        dbs = chunk(db, 3)
        for idx, db in enumerate(dbs, start=1):
            db.to_netcdf(os.path.join(dirpath, f"{idx:03d}.nc"))
    else:
        return db


def chunk(db, nchunk, dim="first"):
    axis = db.get_axis_num(dim)
    nsamples = db.shape[axis]
    if not isinstance(nchunk, int):
        raise TypeError("`n` must be an integer")
    if nchunk <= 0:
        raise ValueError("`n` must be larger than 0")
    if nchunk >= nsamples:
        raise ValueError("`n` must be smaller than the number of samples")
    chunk_size, extras = divmod(nsamples, nchunk)
    chunks = [0] + extras * [chunk_size + 1] + (nchunk - extras) * [chunk_size]
    div_points = np.cumsum(chunks, dtype=np.int64)
    return [
        db.isel({dim: slice(div_points[idx], div_points[idx + 1])})
        for idx in range(nchunk)
    ]
