import numpy as np
import scipy.signal as sp

import xdas
import os


def generate(dirpath):
    """
    Generate some dummy files to bu used in code testing.

    It generates a monolithic `sample.nc` file and a chunked files (`001.nc`, `002.nc`,
    `003.nc`).

    Parameters
    ----------
    dirpath : str
        Directory where files will be written.
    
    Examples
    --------
    >>> import os
    >>> import xdas
    >>> from xdas.synthetics import generate
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as dirpath:
    ...     generate(dirpath)
    ...     db_monolithic = xdas.open_database(os.path.join(dirpath, "sample.nc"))
    ...     db_chunked = xdas.open_mfdatabase(os.path.join(dirpath, "00*.nc"))
    ...     db_monolithic.equals(db_chunked)
    True
    
    """
    shape = (300, 401)
    resolution = (np.timedelta64(20, "ms"), 25.0)
    starttime = np.datetime64("2023-01-01T00:00:00")
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
    db.to_netcdf(os.path.join(dirpath, "sample.nc"))
    db[:100].to_netcdf(os.path.join(dirpath, "001.nc"))
    db[100:200].to_netcdf(os.path.join(dirpath, "002.nc"))
    db[200:].to_netcdf(os.path.join(dirpath, "003.nc"))
