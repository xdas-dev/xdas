import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import scipy.signal as sp
from tqdm import tqdm

from .core import concatenate, open_database
from .monitor import Monitor
from .signal import multithreaded_concatenate


class DatabaseLoader:
    def __init__(self, db, chunks):
        self.db = db
        ((self.chunk_dim, self.chunk_size),) = chunks.items()
        self.queue = Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(self.task)

    def __len__(self):
        div, mod = divmod(self.db.sizes[self.chunk_dim], self.chunk_size)
        return div if mod == 0 else div + 1

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = (idx + 1) * self.chunk_size
        query = {
            dim: slice(start, end) if dim == self.chunk_dim else slice(None)
            for dim in self.db.dims
        }
        return self.db[query].load()

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.queue.get()
        if chunk is None:
            raise StopIteration
        else:
            return chunk

    @property
    def nbytes(self):
        return self.db.nbytes

    def task(self):
        for idx in range(len(self)):
            data = self[idx]
            self.queue.put(data)
        self.queue.put(None)


class DatabaseWriter:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.queue = Queue(maxsize=1)
        self.results = []
        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(self.task)

    def to_netcdf(self, db):
        self.queue.put(db)

    def task(self):
        while True:
            db = self.queue.get()
            if db is None:
                break
            path = self.get_path(db)
            if os.path.exists(path):
                raise OSError(f"the file '{path}' already exists.")
            else:
                db.to_netcdf(path)
            self.results.append(open_database(path))

    def get_path(self, db):
        datetime = np.datetime_as_string(db["time"][0], unit="s").replace(":", "-")
        fname = f"{datetime}.nc"
        return os.path.join(self.dirpath, fname)

    def result(self):
        self.queue.put(None)
        self.future.result()
        return concatenate(self.results)


class SOSFilter:
    def __init__(self, sos, dim, parallel=None):
        self.sos = sos
        self.dim = dim
        self.parallel = parallel if parallel is not None else os.cpu_count()
        self.state = None

    def __call__(self, db):
        axis = db.get_axis_num(self.dim)
        if self.state is None:
            self.state = self.initialize(db.shape, db.dtype, axis)
        states = np.array_split(self.state, self.parallel, 1 + int(axis == 0))
        datas = np.array_split(db.values, self.parallel, int(axis == 0))
        fn = lambda data, state: sp.sosfilt(self.sos, data, axis=axis, zi=state)
        with ThreadPoolExecutor(self.parallel) as executor:
            datas, states = zip(*executor.map(fn, datas, states))
        data = multithreaded_concatenate(datas, int(axis == 0), n_workers=self.parallel)
        self.state = multithreaded_concatenate(
            states, 1 + int(axis == 0), n_workers=self.parallel
        )
        return db.copy(data=data)

    def initialize(self, shape, dtype, axis):
        n_sections = self.sos.shape[0]
        zi_shape = (n_sections,) + tuple(
            2 if index == axis else element for index, element in enumerate(shape)
        )
        return np.zeros(zi_shape, dtype=dtype)

    def reset(self):
        self.state = None


class ProcessingChain:
    def __init__(self, filters):
        self.filters = filters
        self.data_loader = None
        self.data_writer = None

    def __call__(self, db):
        for filter in self.filters:
            db = filter(db)
        return db

    def reset(self):
        for filter in self.filters:
            if hasattr(filter, "reset"):
                filter.reset()

    def process(self, db, chunks, dirpath):
        self.reset()
        self.data_loader = DatabaseLoader(db, chunks)
        self.data_writer = DatabaseWriter(dirpath)
        monitor = Monitor(total=self.data_loader.nbytes)
        monitor.tic("read")
        for chunk in self.data_loader:
            monitor.tic("proc")
            result = self(chunk)
            monitor.tic("write")
            self.data_writer.to_netcdf(result)
            monitor.toc(chunk.nbytes)
            monitor.tic("read")
        monitor.close()
        return self.data_writer.result()
