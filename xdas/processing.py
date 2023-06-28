from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import scipy.signal as sp

from xdas.core import concatenate


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
            chunks = []
            for k in range(nchunks):
                query = {dim: slice(k * chunk_size, (k + 1) * chunk_size)}
                chunks.append(xarr[query])
            scheduler = Scheduler(chunks, self.filters)
            result = scheduler.compute()
        else:
            for k in range(nchunks):
                query = {dim: slice(k * chunk_size, (k + 1) * chunk_size)}
                result.append(self(xarr[query]))
        return concatenate(result, dim)


class Scheduler:
    def __init__(self, chunks, filters):
        self.chunks = chunks
        self.filters = filters
        self.buffer = {idx: chunk for idx, chunk in enumerate(chunks)}
        self.executor = None
        self.futures = {}

    def compute(self):
        with ThreadPoolExecutor() as executor:
            self.executor = executor
            task = {"chunk": 0, "filter": 0}
            future = self.submit_task(task)
            while self.futures:
                done, _ = wait(self.futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task = self.futures.pop(future)
                    self.buffer[task["chunk"]] = future.result()
                    next_tasks = self.get_next_tasks(task)
                    for next_task in next_tasks:
                        self.submit_task(next_task)
                print("Number of futures:", len(self.futures))
        return list(self.buffer.values())

    def submit_task(self, task):
        filter = self.filters[task["filter"]]
        chunk = self.buffer[task["chunk"]]
        future = self.executor.submit(filter, chunk)
        self.futures[future] = task
        print("New task:", task)
        return future

    def get_next_tasks(self, task):
        next_tasks = []
        if not task["chunk"] >= len(self.chunks) - 1:
            next_task = task.copy()
            next_task["chunk"] += 1
            next_tasks.append(next_task)
        if not task["filter"] >= len(self.filters) - 1:
            next_task = task.copy()
            next_task["filter"] += 1
            next_tasks.append(next_task)
        return next_tasks


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


class SOSFilter:
    def __init__(self, sos, dim):
        self.sos = sos
        self.dim = dim
        self.zi = None
        self.axis = None

    def __call__(self, xarr):
        if (self.zi is None) or (self.axis is None):
            self.initialize(xarr)
        data, self.zi = sp.sosfilt(self.sos, xarr, axis=self.axis, zi=self.zi)
        return xarr.copy(data=data)

    def initialize(self, xarr):
        self.axis = xarr.get_axis_num(self.dim)
        zi = sp.sosfilt_zi(self.sos)
        ndim = len(xarr.shape)
        n_sections = zi.shape[0]
        s = [n_sections] + [2 if k == self.axis else 1 for k in range(ndim)]
        zi = zi.reshape(s)
        s = [n_sections] + [2 if k == self.axis else xarr.shape[k] for k in range(ndim)]
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


class Writter(SignalProcessingUnit):
    def __init__(self, fname, duration=np.timedelta64(1, "m")):
        self.fname = fname
        self.duration = duration
        self.buffer = None

    def __call__(self, xarr):
        if self.buffer is None:
            self.buffer = xarr
        else:
            self.buffer = concatenate([self.buffer, xarr])
        if self.buffer["time"][-1] - self.buffer["time"][-1] > self.duration:
            self.buffer.sel(
                time=slice(
                    None,
                )
            )
