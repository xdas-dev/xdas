import numpy as np
import scipy.signal as sp

from .core.database import Database
from .core.routines import chunk


def generate(
    *,
    starttime="2023-01-01T00:00:00",
    resolution=(np.timedelta64(20, "ms"), 25.0),
    nchunk=None,
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
    ...     os.chdir(dirpath)
    ...     generate().to_netcdf("sample.nc")
    ...     for idx, db in enumerate(generate(nchunk=3), start=1):
    ...         db.to_netcdf(f"{idx:03}.nc")
    ...     db_monolithic = xdas.open_database("sample.nc")
    ...     db_chunked = xdas.open_mfdatabase("00*.nc")
    ...     db_monolithic.equals(db_chunked)
    True

    """
    # sampling
    starttime = np.datetime64(starttime).astype("datetime64[ns]")
    span = (np.timedelta64(6, "s"), 10000.0)  # (6 s, 10 km)
    shape = (span[0] // resolution[0], int(span[1] // resolution[1]) + 1)
    t = np.arange(shape[0]) * resolution[0] / np.timedelta64(1, "s")  # time values [s]
    s = np.arange(shape[1]) * resolution[1]  # distance values [m]

    # physical parameters
    snr = 10  # signal to noise ration
    vp = 4000  # P-wave speed [m/s]
    vs = vp / 1.75  # S-wave speed [m/s]
    xc = 5_000  # source distance along [m]
    fc = 10.0  # source central frequency [Hz]
    t0 = 1.0  # source origin time relative to start of file [s]

    d = np.hypot(xc, (s - np.mean(s)))  # channel distance to source [m]
    ttp = d / vp  # P-wave travel time [s]
    tts = d / vs  # S-wave travel time [s]
    data = np.zeros(shape)
    for k in range(shape[1]):
        data[:, k] += sp.gausspulse(t - ttp[k] - t0, fc) / 2  # P is twice weaker
        data[:, k] += sp.gausspulse(t - tts[k] - t0, fc)
    data /= np.max(np.abs(data), axis=0, keepdims=True)  # normalize
    np.random.seed(42)
    data += np.random.randn(*shape) / snr  # add noise

    # strain rate like response
    data = np.diff(data, prepend=0, axis=-1)
    data = np.diff(data, prepend=0, axis=0)

    # pack data and coordinates as Database or DataCollection if chunking.
    db = Database(
        data=data,
        coords={
            "time": {
                "tie_indices": [0, shape[0] - 1],
                "tie_values": [starttime, starttime + resolution[0] * (shape[0] - 1)],
            },
            "distance": {
                "tie_indices": [0, shape[1] - 1],
                "tie_values": [0.0, resolution[1] * (shape[1] - 1)],
            },
        },
    )
    if nchunk is not None:
        return chunk(db, nchunk)
    else:
        return db
