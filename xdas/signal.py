import numpy as np
import scipy.signal as sp


class Task:
    def __init__(self, xarr, chain, chunk_dim, chunk_size):
        self.xarr = xarr
        self.chain = chain
        self.chunk_dim = chunk_dim
        self.chunk_size = chunk_size


class SignalProcessingChain:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, xarr):
        out = xarr
        for filter in self.filters:
            out = filter(out)
        return out


class SignalProcessingUnit:
    def __init__(self, **kwargs):
        for key, value in kwargs:
            setattr(self, key, value)
        self.initialized = False

    def __call__(self, xarr, chunks=None):
        if not self.initialized:
            self.initialize(xarr)
        data = self.process_data(xarr)
        coords = self.process_coords(xarr)
        return xarr.__class__(data=data, coords=coords)

    def initialize(self, xarr):
        pass

    def reset(self):
        self.initialized = False

    def process_data(self, xarr):
        return xarr.data

    def process_coords(self, xarr):
        return xarr.coords


class DataLoader:
    def __init__(self, xarr, chunks):
        self.xarr = xarr
        self.chunks = chunks

class DataWriter:
    def __init__(self, path):
        pass

class LFilter(SignalProcessingUnit):
    def __init__(self, b, a, dim):
        self.b = b
        self.a = a
        self.dim = dim
        self.zi = None

    def process_data(self, xarr):
        out, self.zi = sp.lfilter(self.b, self.a, xarr, axis=self.axis, zi=self.zi)
        return out

    def initialize(self, xarr):
        self.axis = xarr.get_axis_num(self.dim)
        zi = sp.lfilter_zi(self.b, self.a)
        n_sections = zi.shape[0]
        s = [n_sections if k == self.axis else 1 for k in range(xarr.ndim)]
        zi = zi.reshape(s)
        s = [n_sections if k == self.axis else xarr.shape[k] for k in range(xarr.ndim)]
        zi = np.broadcast_to(zi, s)
        self.zi = zi
