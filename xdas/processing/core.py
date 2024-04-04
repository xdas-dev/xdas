import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..core import concatenate, open_database
from .monitor import Monitor


def process(seq, data_loader, data_writer):
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
        db = open_database(event.src_path, engine=self.engine)
        self.queue.put(db.load())


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
        datetime = np.datetime_as_string(db["time"][0].values, unit="s").replace(
            ":", "-"
        )
        fname = f"{datetime}.nc"
        return os.path.join(self.dirpath, fname)

    def result(self):
        self.queue.put(None)
        self.future.result()
        return concatenate(self.results)
