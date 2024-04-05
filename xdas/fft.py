import numpy as np

from .atoms.core import atomized
from .core.coordinates import get_sampling_interval
from .core.dataarray import DataArray
from .parallel import parallelize


@atomized
def fft(da, n=None, dim={"last": "frequency"}, norm=None, parallel=None):
    ((olddim, newdim),) = dim.items()
    olddim = da.dims[da.get_axis_num(olddim)]
    if n is None:
        n = da.sizes[olddim]
    axis = da.get_axis_num(olddim)
    d = get_sampling_interval(da, olddim)
    f = np.fft.fftshift(np.fft.fftfreq(n, d))
    func = lambda x: np.fft.fftshift(np.fft.fft(x, n, axis, norm), axis)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(da.values)
    coords = {
        newdim if name == olddim else name: f if name == olddim else da.coords[name]
        for name in da.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in da.dims)
    return DataArray(data, coords, dims, da.name, da.attrs)


@atomized
def rfft(da, n=None, dim={"last": "frequency"}, norm=None, parallel=None):
    ((olddim, newdim),) = dim.items()
    olddim = da.dims[da.get_axis_num(olddim)]
    if n is None:
        n = da.sizes[olddim]
    axis = da.get_axis_num(olddim)
    d = get_sampling_interval(da, olddim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(np.fft.rfft)
    f = np.fft.rfftfreq(n, d)
    data = func(da.values, n, axis, norm)
    coords = {
        newdim if name == olddim else name: f if name == olddim else da.coords[name]
        for name in da.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in da.dims)
    return DataArray(data, coords, dims, da.name, da.attrs)
