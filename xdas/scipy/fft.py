import numpy as np

from ..atoms import atomized
from ..coordinates import get_sampling_interval
from ..core import collects
from ..database import Database
from ..parallel import parallelize


@atomized
@collects
def fft(db, n=None, dim={"last": "frequency"}, norm=None, parallel=None):
    ((olddim, newdim),) = dim.items()
    olddim = db.dims[db.get_axis_num(olddim)]
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    f = np.fft.fftshift(np.fft.fftfreq(n, d))
    func = lambda x: np.fft.fftshift(np.fft.fft(x, n, axis, norm), axis)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(db.values)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)


@atomized
@collects
def rfft(db, n=None, dim={"last": "frequency"}, norm=None, parallel=None):
    ((olddim, newdim),) = dim.items()
    olddim = db.dims[db.get_axis_num(olddim)]
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(np.fft.rfft)
    f = np.fft.rfftfreq(n, d)
    data = func(db.values, n, axis, norm)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)
