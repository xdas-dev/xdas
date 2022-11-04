import numpy as np
import scipy.signal as sp
from dask.distributed import Client


class SignalProcessingChain:
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, xarr):
        out = xarr
        for filter in self.filters:
            out = filter(out)
        return out

    def process(self, xarr, dim, chunk_size, parallel=True):
        div, mod = divmod(xarr.sizes[dim], chunk_size)
        if mod == 0:
            nchunks = div
        else:
            nchunks = div + 1
        result = []
        if parallel:
            client = Client(n_workers=1)
            queue = []
            for filter in self.filters:
                queue.append(client.submit(lambda: None))
            for k in range(nchunks):
                query = {dim: slice(k * chunk_size, (k + 1) * chunk_size)}
                out = client.scatter(xarr[query])
                for n, filter in enumerate(self.filters):
                    queue[n].result()
                    out = client.submit(filter, out)
                    queue[n] = out
                result.append(out)
            result = client.gather(result)
            client.close()
        else:
            for k in range(nchunks):
                query = {dim: slice(k * chunk_size, (k + 1) * chunk_size)}
                result.append(self(xarr[query]))
        return result


class SignalProcessingUnit:
    pass


class LFilter(SignalProcessingUnit):
    def __init__(self, b, a, dim):
        self.b = b
        self.a = a
        self.dim = dim
        self.zi = None
        self.axis = None

    def __call__(self, xarr):
        if (self.zi is None) or (self.axis is None):
            self.initialize(xarr)
        data, self.zi = sp.lfilter(
            self.b, self.a, xarr.data, axis=self.axis, zi=self.zi
        )
        return xarr.copy(data=data)

    def initialize(self, xarr):
        self.axis = xarr.get_axis_num(self.dim)
        zi = sp.lfilter_zi(self.b, self.a)
        n_sections = zi.shape[0]
        s = [n_sections if k == self.axis else 1 for k in range(xarr.ndim)]
        zi = zi.reshape(s)
        s = [n_sections if k == self.axis else xarr.shape[k] for k in range(xarr.ndim)]
        zi = np.broadcast_to(zi, s)
        self.zi = zi

    def reset(self):
        self.zi = None
        self.axis = None


class Decimate(SignalProcessingUnit):
    def __init__(self, q, dim):
        self.q = q
        self.dim = dim

    def __call__(self, xarr):
        return xarr[{self.dim: slice(None, None, self.q)}]
