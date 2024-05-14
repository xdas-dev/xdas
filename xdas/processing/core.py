import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..core.routines import concatenate, open_dataarray
from .monitor import Monitor


def process(atom, data_loader, data_writer):
    """
    Execute a chunked processing pipeline.

    Parameters
    ----------
    atom : callable
        The atomic operation to execute on each chunk of data.
    data_loader : DataArrayLoader
        The data loader object that provides the chunks of data.
    data_writer : DataArrayWriter
        The data writer object that writes the processed data.

    Returns
    -------
    result : object
        The result of the processing pipeline.

    Notes
    -----
    This function executes a chunked processing pipeline by ingesting the data from
    the `data_loader` and flushing the processed data through the `data_writer`.
    It iterates over the chunks of data provided by the `data_loader`, applies the
    `atom` function to each chunk, and writes the processed data using the `data_writer`.
    The progress of the processing is monitored using a `Monitor` object.

    """
    atom.reset()
    if hasattr(data_loader, "nbytes"):
        total = data_loader.nbytes
    else:
        total = None
    monitor = Monitor(total=total)
    monitor.tic("read")
    for chunk in data_loader:
        monitor.tic("proc")
        result = atom(chunk, chunk_dim=data_loader.chunk_dim)
        monitor.tic("write")
        data_writer.write(result)
        monitor.toc(chunk.nbytes)
        monitor.tic("read")
    monitor.close()
    return data_writer.result()


class DataArrayLoader:
    """
    A class to handle data chunked data ingestion.

    Parameters
    ----------
    da : ``DataArray``, ``DataCollection``
        The (virtual) DataArray or DataCollection that contains the data to be chunked
    chunks : dict
        The sizes of the chunks along each dimension. Needs to be of the form:
        ``{"dim": int}``. Each key needs to correspond with a dimension (either "time"
        or "distance"), and each value is an integer indicating the size of the chunk
        (in samples) along that dimension.

    Examples
    --------
    >>> import xdas
    >>> from xdas.processing import DataArrayLoader
    >>> da = xdas.open_dataarray(...)  # doctest: +SKIP

    Create chunks along the time dimension

    >>> chunks = {"time": 1000}
    >>> dl = DataArrayLoader(da, chunks)  # doctest: +SKIP

    Create chunks along both dimensions

    >>> chunks2 = {"time": 1000, "distance": 10}
    >>> dl2 = DataArrayLoader(da, chunks2)  # doctest: +SKIP

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
    """
    A class to handle chunked data egress.

    Parameters
    ----------
    dirpath : str or path
        The directory to store the output of a processing pipeline. The directory needs
        to exist and be empty.

    Examples
    --------
    >>> import os, shutil
    >>> from xdas.processing import DataArrayWriter

    >>> dirpath = "output"
    >>> if not os.path.exists(dirpath):
    ...     os.makedirs(dirpath) # doctest: +SKIP

    >>> dw = DataArrayWriter(dirpath) # doctest: +SKIP

    """

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.queue = Queue(maxsize=1)
        self.results = []
        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(self.task)

    def write(self, da):
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


class DataFrameWriter:
    """
    A class for writing pandas DataFrames to a CSV file asynchronously.

    Parameters
    ----------
    path : str
        The path to the CSV file.

    Attributes
    ----------
    path : str
        The path to the CSV file.
    parse_dates : list or bool
        A list of columns to parse as dates.
    queue : Queue
        A queue to hold the DataFrames to be written.
    executor : ThreadPoolExecutor
        A thread pool executor for asynchronous writing.
    future : Future
        A future object representing the result of the asynchronous task.

    Methods
    -------
    write(df)
        Writes a DataFrame to the queue for asynchronous writing.
    task()
        The asynchronous task that writes the DataFrames to the CSV file.
    result()
        Waits for the asynchronous task to complete and returns the DataFrame read from the CSV file.
    """

    def __init__(self, path, parse_dates=False):
        self.path = path
        self.parse_dates = parse_dates
        self.queue = Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(self.task)

    def write(self, df):
        """
        Writes a DataFrame to the queue for asynchronous writing.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be written.
        """
        self.queue.put(df)

    def task(self):
        """
        The asynchronous task that writes the DataFrames to the CSV file.
        """
        while True:
            df = self.queue.get()
            if df is None:
                break
            if not os.path.exists(self.path):
                df.to_csv(self.path, mode="w", header=True, index=False)
            else:
                df.to_csv(self.path, mode="a", header=False, index=False)

    def result(self):
        """
        Waits for the asynchronous task to complete and returns the DataFrame read from the CSV file.

        Returns
        -------
        pandas.DataFrame
            The DataFrame read from the CSV file.
        """
        self.queue.put(None)
        self.future.result()
        try:
            out = pd.read_csv(self.path, parse_dates=self.parse_dates)
        except pd.errors.EmptyDataError:
            out = pd.DataFrame()
        return out
