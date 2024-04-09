import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..core.routines import concatenate, open_dataarray
from .monitor import Monitor


def process(seq, data_loader, data_writer):
    """
    Execute a chunked processing pipeline ``seq``,
    ingesting the data from ``data_loader`` and
    flushing the processed data through ``data_writer``.

    Parameters
    ----------
    seq : ``Sequential``
        The sequence of atomic operations to execute
    data_loader : ``DataArrayLoader``
    data_writer : ``DataArrayWriter``
    """
    seq.reset()
    if hasattr(data_loader, "nbytes"):
        total = data_loader.nbytes
    else:
        total = None
    monitor = Monitor(total=total)
    monitor.tic("read")
    for chunk in data_loader:
        monitor.tic("proc")
        result = seq(chunk)
        monitor.tic("write")
        data_writer.to_netcdf(result)
        monitor.toc(chunk.nbytes)
        monitor.tic("read")
    monitor.close()
    return data_writer.result()


class DataArrayLoader:
    """
    
    """
    def __init__(self, da, chunks):
        self.da = da
        ((self.chunk_dim, self.chunk_size),) = chunks.items()
        self.queue = Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(self.task)

    def __len__(self):
        div, mod = divmod(self.da.sizes[self.chunk_dim], self.chunk_size)
        return div if mod == 0 else div + 1

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = (idx + 1) * self.chunk_size
        query = {
            dim: slice(start, end) if dim == self.chunk_dim else slice(None)
            for dim in self.da.dims
        }
        return self.da[query].load()

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
        return self.da.nbytes

    def task(self):
        for idx in range(len(self)):
            data = self[idx]
            self.queue.put(data)
        self.queue.put(None)


class RealTimeLoader(Observer):
    def __init__(self, path, engine="netcdf"):
        super().__init__()
        self.path = path
        self.queue = Queue()
        self.handler = Handler(self.queue, engine)
        self.schedule(self.handler, self.path, recursive=True)
        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.queue.get()
        if chunk is None:
            raise StopIteration
        else:
            return chunk


class Handler(FileSystemEventHandler):
    def __init__(self, queue, engine):
        self.engine = engine
        self.queue = queue

    def on_closed(self, event):
        da = open_dataarray(event.src_path, engine=self.engine)
        self.queue.put(da.load())


class DataArrayWriter:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.queue = Queue(maxsize=1)
        self.results = []
        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(self.task)

    def to_netcdf(self, da):
        self.queue.put(da)

    def task(self):
        while True:
            da = self.queue.get()
            if da is None:
                break
            path = self.get_path(da)
            if os.path.exists(path):
                raise OSError(f"the file '{path}' already exists.")
            else:
                da.to_netcdf(path)
            self.results.append(open_dataarray(path))

    def get_path(self, da):
        datetime = np.datetime_as_string(da["time"][0].values, unit="s").replace(
            ":", "-"
        )
        fname = f"{datetime}.nc"
        return os.path.join(self.dirpath, fname)

    def result(self):
        self.queue.put(None)
        self.future.result()
        return concatenate(self.results)
