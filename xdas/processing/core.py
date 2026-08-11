"""
Core processing infrastructure for chunked pipeline execution.

Includes :class:`DataArrayLoader`, :class:`DataArrayWriter`,
:class:`DataFrameWriter`, :class:`StreamWriter`, :class:`ZMQPublisher`,
:class:`ZMQSubscriber`, :class:`RealTimeLoader`, and :func:`process`.
"""

import os
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from glob import glob
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory

import numpy as np
import obspy
import pandas as pd
import zmq
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..core import DataArray, concat, open_dataarray
from .monitor import Monitor


class _RayFuture:
    """A future handed out by :class:`ProcessPool`, resolved via the object store."""

    def __init__(self, pool, task):
        self._pool = pool
        self._task = task
        self._ref = None
        self._cancelled = False

    def result(self):
        """Block until the task ran and return its output (or raise its error)."""
        return self._pool._result(self)


class ProcessPool:
    """
    A pool of worker processes whose results cross through shared memory.

    Each task runs as a Ray task in its own process. The pool quacks like a
    :class:`~concurrent.futures.Executor` as far as the loader and writer
    need (``submit``/``shutdown``/context manager), but the data crossing
    back is never pickled through a pipe: a task result lands in Ray's
    shared-memory object store, written once by the worker, and ``result()``
    maps it zero-copy into the parent. Large task *arguments* — the chunk a
    writer sends out — take the same path, one memcpy into the store instead
    of a serialize-transfer-deserialize round. The price of zero-copy is
    immutability: array data coming out of the store is read-only, which
    atoms honor by allocating their outputs.

    ``max_workers`` is enforced by parking submissions beyond it and
    launching them as running tasks finish, mirroring how a process pool
    queues its backlog. Ray is initialized lazily on first use (an already
    initialized Ray, e.g. configured by the user, is left untouched).

    Parameters
    ----------
    max_workers : int
        Maximum number of tasks running concurrently.
    """

    def __init__(self, max_workers):
        try:
            import ray
        except ImportError:
            raise ImportError(
                "pool='processes' requires the ray package: pip install xdas[ray]"
            ) from None
        if not ray.is_initialized():
            ray.init(include_dashboard=False)
        self._ray = ray
        self._max_workers = max_workers
        self._remotes = {}
        self._pending = deque()
        self._running = []

    def submit(self, fn, /, *args, **kwargs):
        """
        Schedule ``fn(*args, **kwargs)`` as a task and return its future.

        Parameters
        ----------
        fn : callable
            The unit of work; large array arguments go to the object store.

        Returns
        -------
        _RayFuture
            A ``result()``-able handle, resolved zero-copy from the store.
        """
        future = _RayFuture(self, (fn, args, kwargs))
        self._pending.append(future)
        self._launch()
        return future

    def shutdown(self, wait=True):
        """
        Cancel parked tasks and (by default) wait for the running ones.

        The Ray runtime itself is left up: it is a session-wide resource,
        shared with the other pools of the run and with whatever the user
        configured before xdas started.

        Parameters
        ----------
        wait : bool, optional
            Whether to block until in-flight tasks complete.
        """
        while self._pending:
            self._pending.popleft()._cancelled = True
        if wait and self._running:
            self._ray.wait(self._running, num_returns=len(self._running))
        self._running.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.shutdown()

    def _remote(self, fn):
        """Wrap *fn* as a Ray remote function, once per distinct callable."""
        if fn not in self._remotes:
            self._remotes[fn] = self._ray.remote(fn)
        return self._remotes[fn]

    def _launch(self):
        """Drop finished tasks and start parked ones up to ``max_workers``."""
        if self._running:
            _, self._running = self._ray.wait(
                self._running, num_returns=len(self._running), timeout=0
            )
        while self._pending and len(self._running) < self._max_workers:
            future = self._pending.popleft()
            fn, args, kwargs = future._task
            future._ref = self._remote(fn).remote(*args, **kwargs)
            self._running.append(future._ref)

    def _result(self, future):
        """Resolve *future*, first waiting out the backlog ahead of it."""
        while future._ref is None:
            if future._cancelled:
                raise CancelledError()
            self._ray.wait(self._running, num_returns=1)
            self._launch()
        return self._ray.get(future._ref)


POOLS = {"threads": ThreadPoolExecutor, "processes": ProcessPool}
"""Worker pools available for chunk ingress and egress, name → factory."""


def get_pool(pool, max_workers):
    """
    Build the worker pool used to load or write chunks.

    Threads are enough when the work releases the GIL, but compressed HDF5
    does not: reading a chunk goes through one virtual layout, hence one h5py
    call holding the global HDF5 lock, so decompression of several chunks
    cannot overlap in-process and extra threads only contend. Worker
    processes each hold their own lock. What crosses to a worker is the
    *manifest* of the chunk (a sliced virtual array, kilobytes), not data,
    and the loaded chunk crosses *back* through a shared-memory object
    store: the worker writes it once, the parent maps it zero-copy
    (read-only). ``"processes"`` requires the optional ``ray`` dependency
    (``pip install xdas[ray]``).

    Parameters
    ----------
    pool : str
        Pool kind, ``"threads"`` (default everywhere) or ``"processes"``.
    max_workers : int
        Number of workers.

    Returns
    -------
    executor
        A :class:`~concurrent.futures.Executor`-like pool.

    Examples
    --------
    >>> from xdas.processing.core import get_pool
    >>> with get_pool("threads", 2) as pool:
    ...     pool.submit(abs, -1).result()
    1
    """
    if pool not in POOLS:
        raise ValueError(f"no worker pool named {pool!r}; available: {sorted(POOLS)}")
    return POOLS[pool](max_workers)


def _load(da):
    """Load a (virtual) DataArray. The unit of work shipped to ingress workers."""
    return da.load()


def _dump(chunk, path, encoding):
    """Write *chunk* to *path* and return it virtually. Egress unit of work."""
    chunk.to_netcdf(path, encoding=encoding)
    return open_dataarray(path)


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
    if hasattr(atom, "reset"):
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

    To optimize I/O latencies, chunks are loaded before they are used asynchronously
    in a buffer as soon as the iterator is created.

    Parameters
    ----------
    da : ``DataArray``
        The (virtual) DataArray that contains the data to be chunked
    chunks : dict
        The sizes of the chunks along each dimension. Needs to be of the form:
        ``{"dim": int}``. The key correspond with the dimension (usually "time"),
        and the value is an integer indicating the size of the chunk (in samples)
        along that dimension.
    max_buffers : int, default=1
        The maximum number of chunks to load into memory at the same time.
    max_workers : int, default=1
        The maximum number of workers used to load the chunks.
    pool : {"threads", "processes"}, default="threads"
        The kind of workers to load with. Compressed HDF5 decodes under the
        global HDF5 lock, so several threads do not decode concurrently and
        ``max_workers`` above one only pays off with ``"processes"``: each
        worker receives the manifest of its chunk (kilobytes), reads its own
        files, and returns the loaded chunk through a shared-memory object
        store — zero-copy for the parent, with chunk data arriving
        read-only. Requires the optional ``ray`` dependency. See
        :func:`get_pool`.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.processing import DataArrayLoader
    >>> da = xd.open_dataarray(...)  # doctest: +SKIP

    Create chunks along the time dimension

    >>> chunks = {"time": 1000}
    >>> dl = DataArrayLoader(da, chunks)  # doctest: +SKIP

    Iterate over the chunks

    >>> for chunk in dl:
    ...     process(chunk)  # doctest: +SKIP

    Decode four chunks at a time, one worker process each

    >>> dl = DataArrayLoader(da, chunks, 4, 4, pool="processes")  # doctest: +SKIP

    """

    def __init__(self, da, chunks, max_buffers=1, max_workers=1, pool="threads"):
        if not isinstance(da, DataArray):
            raise TypeError(f"`da` must by a DataArray object, not a {type(da)}")
        if not (isinstance(chunks, dict) and len(chunks) == 1):
            raise TypeError(
                "`chunks` must be a dict that maps a unique "
                "dimension to a unique size: {'dim': int}"
            )
        ((chunk_dim, chunk_size),) = chunks.items()
        chunk_dim = str(chunk_dim)
        chunk_size = int(chunk_size)
        if chunk_dim not in da.dims:
            raise ValueError(
                f"chunking dimension {chunk_dim} not found in `da` dimensions {da.dims}"
            )
        if chunk_size > da.sizes[chunk_dim]:
            raise ValueError(
                f"chunking size {chunk_size} is greater than `da` "
                f"size {da.sizes[chunk_dim]} along dim {chunk_dim}"
            )
        self.da = da
        self.chunk_dim = chunk_dim
        self.chunk_size = chunk_size
        self.max_buffers = max_buffers
        self.max_workers = max_workers
        self.pool = pool

    def __len__(self):
        div, mod = divmod(self.da.sizes[self.chunk_dim], self.chunk_size)
        return div if mod == 0 else div + 1

    def _select(self, idx):
        """Return chunk *idx* as a lazy selection: the manifest, not the data."""
        start = idx * self.chunk_size
        end = (idx + 1) * self.chunk_size
        query = {
            dim: slice(start, end) if dim == self.chunk_dim else slice(None)
            for dim in self.da.dims
        }
        return self.da[query]

    def __iter__(self):
        with get_pool(self.pool, self.max_workers) as executor:
            it = iter(range(len(self)))

            def submit(idx):
                # The task is the sliced virtual array, so a process worker
                # receives kilobytes and reads its own files.
                return executor.submit(_load, self._select(idx))

            futures = []
            try:
                for _ in range(self.max_buffers):
                    futures.append(submit(next(it)))
            except StopIteration:
                pass

            while futures:
                future = futures.pop(0)
                result = future.result()

                try:
                    futures.append(submit(next(it)))
                except StopIteration:
                    pass

                yield result

    @property
    def nbytes(self):
        """Total bytes of the underlying :class:`DataArray`."""
        return self.da.nbytes


class RealTimeLoader(Observer):
    """
    Real-time :class:`DataArray` loader that watches a directory for new files.

    Parameters
    ----------
    path : str or Path
        Directory to watch.
    engine : str or Engine, optional
        Engine used to open arriving files, given by name or as a configured
        :class:`~xdas.io.Engine` instance.  Defaults to ``"xdas"``.
    """

    def __init__(self, path, engine="xdas"):
        super().__init__()
        self.path = str(path) if isinstance(path, Path) else path
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
    """Watchdog event handler that loads closed files into a queue."""

    def __init__(self, queue, engine):
        self.engine = engine
        self.queue = queue

    def on_closed(self, event):
        """Load the newly-closed file and place it in the queue."""
        da = open_dataarray(event.src_path, engine=self.engine)
        self.queue.put(da.load())


class DataArrayWriter:
    """
    A class to handle chunked data egress.

    Parameters
    ----------
    dirpath : str or Path
        The directory to store the output of a processing pipeline. The directory needs
        to exist and be empty.
    encoding : dict
        The encoding to use when dumping the DataArrays to bytes.
    max_buffers : int, default=1
        The maximum number of chunks to load into memory at the same time.
    max_workers : int, default=1
        The maximum number of thread used to load the chunks.
    create_dirs : bool, optional
        Whether to create parent directories if they do not exist. Default is False.
    dim : str, optional
        Dimension the chunks follow each other along, used to join them at
        :meth:`result`. Defaults to ``"first"``, which is only right when the
        chunked dimension leads the output: a pipeline emitting it elsewhere
        must name it.
    pool : {"threads", "processes"}, default="threads"
        The kind of workers to write with. Compression happens under the
        global HDF5 lock, so as on the read side several threads do not
        compress concurrently; ``"processes"`` sends each chunk to a worker
        through the shared-memory object store — one memcpy at memory
        bandwidth — in exchange for parallel compression. Requires the
        optional ``ray`` dependency. See :func:`get_pool`.

    Examples
    --------
    >>> import xdas as xd
    >>> import xdas.processing as xp

    >>> expected = xd.DataArray(np.random.rand(1000, 100), dims=("time", "distance"))

    >>> dw = DataArrayWriter("some_path")  # doctest: +SKIP
    >>> for chunk in chunks:
    ...     dw.submit(chunk)  # doctest: +SKIP
    >>> result = dw.result  # doctest: +SKIP

    >>> assert result.equals(expected)  # doctest: +SKIP

    """

    def __init__(
        self,
        dirpath,
        encoding=None,
        max_buffers=1,
        max_workers=1,
        create_dirs=False,
        dim="first",
        pool="threads",
    ):
        dirpath = str(dirpath) if isinstance(dirpath, Path) else dirpath
        if create_dirs:
            os.makedirs(dirpath, exist_ok=True)
        if not os.path.exists(dirpath):
            raise OSError(f"no directory {dirpath}")
        self.dirpath = dirpath
        self.dim = dim
        self.encoding = encoding
        self.max_buffers = max_buffers
        self.max_workers = max_workers
        self.pool = pool
        self._executor = get_pool(pool, self.max_workers)
        self._futures = []
        self._results = []
        self._count = 0

    def submit(self, chunk):
        """
        Asynchronously write *chunk* to disk and register the path for later concat.

        Parameters
        ----------
        chunk : DataArray
            Processed data chunk to persist.
        """
        if not isinstance(chunk, DataArray):
            raise TypeError(f"`chunk` must by a DataArray object, not a {type(chunk)}")
        if not len(self._futures) < self.max_buffers:
            future = self._futures.pop(0)
            result = future.result()
            self._results.append(result)
        path = os.path.join(self.dirpath, f"{self._count:09d}")
        self._futures.append(self._executor.submit(_dump, chunk, path, self.encoding))
        self._count += 1

    def write(self, chunk):
        """Alias for :meth:`submit`."""
        return self.submit(chunk)

    def shutdown(self):
        """Shut down the internal thread pool."""
        self._executor.shutdown()

    def result(self):
        """Flush all pending writes and return the concatenated :class:`DataArray`."""
        while self._futures:
            future = self._futures.pop(0)
            result = future.result()
            self._results.append(result)
        self.shutdown()
        return concat(self._results, self.dim)


class DataFrameWriter:
    """
    A class for writing pandas DataFrames to a CSV file asynchronously.

    Parameters
    ----------
    path : str
        The path to the csv file.
    parse_dates : bool, int, optional
        Whether to parse dates when reopening the csv file at the end of the process
    create_dirs : bool, optional
        Whether to create parent directories if they do not exist. Default is False.

    Examples
    --------
    >>> import pandas as pd
    >>> import xdas.processing as xp

    >>> dw = xp.DataFrameWriter("output.csv")
    >>> for df in dfs:
    ...     dw.submit(dfs). # doctest: +SKIP
    >>> result = dw.result()  # doctest: +SKIP

    >>> expected = pd.concat(dfs, ignore_index=True)  # doctest: +SKIP
    >>> assert result.equals(expected)  # doctest: +SKIP

    """

    def __init__(self, path, parse_dates=None, create_dirs=False):
        dirpath = os.path.dirname(path)
        if create_dirs and dirpath:
            os.makedirs(dirpath, exist_ok=True)
        if dirpath and not os.path.exists(dirpath):
            raise OSError(f"no directory {dirpath}")
        self.path = str(path) if isinstance(path, Path) else path
        self.parse_dates = parse_dates
        self._executor = ThreadPoolExecutor(1)
        self._future = None

    def submit(self, df):
        """
        Asynchronously append *df* to the CSV file.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame chunk to write.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"`df` must by a DataFrame object, not a {type(df)}")
        if self._future is not None:
            self._future.result()
        self._future = self._executor.submit(self._write, df)

    def write(self, df):
        """Alias for :meth:`submit`."""
        return self.submit(df)

    def _write(self, df):
        if df is not None:  # pragma: no branch
            if not os.path.exists(self.path):
                df.to_csv(self.path, mode="w", header=True, index=False)
            else:
                df.to_csv(self.path, mode="a", header=False, index=False)

    def shutdown(self):
        """Shut down the internal thread pool."""
        self._executor.shutdown()

    def result(self):
        """Flush pending writes and return the full CSV as a :class:`pandas.DataFrame`."""
        self._future.result()
        self.shutdown()
        try:
            return pd.read_csv(self.path, parse_dates=self.parse_dates)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


class StreamWriter:
    """
    A class for writing obspy Streams to miniseed files asynchronously.

    Parameters
    ----------
    path : str
        The path of the miniseed file or the folder name where the miniseed files will
        be written.
    dataquality : str
        Data quality of the waveforms.
    kw_merge : dict
        Keyword arguments for merging the Streams, following the arguments of the
        obspy.core.stream.Stream.merge function.
    kw_write : dict
        Keyword arguments for writing the Streams, following the arguments of the
        obspy.core.stream.Stream.write function.
    output_format : str
        The output format of the miniseed files. Can be "flat" or "SDS".
        If "flat", the miniseed files will be written in a single file.
        If "SDS", the miniseed files will be written in the SDS file structure.
        For more information about SDS see:
        https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html

    Examples
    --------
    >>> import obspy
    >>> import numpy as np
    >>> import xdas as xd
    >>> import xdas.processing as xp

    Generate some DataArray:

    >>> data = np.random.randint(
    ...     low=-1000, high=1000, size=(1000, 10), dtype=np.int32
    ... )
    >>> starttime = np.datetime64("2023-01-01T00:00:00")
    >>> endtime = starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1)
    >>> distance = 5.0 * np.arange(data.shape[1])
    >>> da = xd.DataArray(
    ...     data=data,
    ...     coords={
    ...         "time": {
    ...             "tie_indices": [0, data.shape[0] - 1],
    ...             "tie_values": [starttime, endtime],
    ...             "sampling_interval": np.timedelta64(10, "ms"),
    ...         },
    ...         "distance": distance,
    ...     },
    ... )

    StreamWriter works great with the `DataArray.to_stream` method that can be used as
    an atom like this:

    >>> atom = lambda da, **kwargs: da.to_stream(
    ...     network="NT",
    ...     station="ST{:03}",
    ...     channel="HN1",
    ...     location="00",
    ...     dim={"distance": "time"},
    ... )
    >>> data_loader = xp.DataArrayLoader(da, chunks={"time": 100})

    This is how a StreamWriter can be used to write the data to a miniseed file:

    >>> kw_merge = {"method": 1}
    >>> kw_write = {"reclen": 4096}
    >>> data_writer = xp.StreamWriter(
    ...     "some_directory", "M", kw_merge, kw_write, output_format="SDS"
    ... )
    >>> result = xp.process(atom, data_loader, data_writer)

    The data will be written to the SDS file structure in the specified directory.

    >>> st = obspy.read("some_directory/2023/NT/*/HN1.D/NT.*.00.HN1.D.2023.001")

    Clean up:

    >>> import shutil
    >>> shutil.rmtree("some_directory")

    """

    def __init__(
        self, path, dataquality, kw_merge=None, kw_write=None, output_format="SDS"
    ):
        path = str(path) if isinstance(path, Path) else path
        if output_format == "SDS":
            os.makedirs(path, exist_ok=True)
            self.dirpath = path
            self.fname = None
        elif output_format == "flat":
            head, tail = os.path.split(path)
            if not os.path.exists(head):
                raise OSError(f"The directory {head} does not exist.")
            self.dirpath = head
            self.fname = tail
        else:
            raise ValueError(
                "output_format must be either 'SDS' or 'flat'. "
                f"Got {output_format} instead."
            )
        self.dataquality = dataquality
        self.kw_merge = kw_merge if kw_merge is not None else {}
        self.kw_write = kw_write if kw_write is not None else {}
        self.output_format = output_format
        self._executor = ThreadPoolExecutor(1)
        self._future = None

    def _to_SDS(self, st):
        for tr in st:
            new_st = obspy.Stream()
            new_st += tr
            new_st = new_st[0].split()
            for new_tr in new_st:
                if isinstance(new_tr.data, np.ma.masked_array):  # pragma: no cover
                    new_tr.data = new_tr.data.filled()
                new_tr.stats.mseed["dataquality"] = self.dataquality
            year = new_st[0].stats.starttime.year
            network = new_st[0].stats.network
            station = new_st[0].stats.station
            channel = new_st[0].stats.channel
            location = new_st[0].stats.location
            julday = new_st[0].stats.starttime.julday
            dirpath = os.path.join(
                self.dirpath, str(year), network, station, channel + ".D"
            )
            os.makedirs(dirpath, exist_ok=True)
            fname = f"{network}.{station}.{location}.{channel}.D.{year}.{julday:03d}"
            sds_path = os.path.join(dirpath, fname)
            new_st.write(sds_path, format="MSEED", **self.kw_write)

    def _to_flat(self, st):
        new_st = obspy.Stream()
        for tr in st:
            tmp_st = obspy.Stream()
            tmp_st += tr
            tmp_st = tmp_st[0].split()
            for new_tr in tmp_st:
                if isinstance(new_tr.data, np.ma.masked_array):  # pragma: no cover
                    new_tr.data = new_tr.data.filled()
                new_st += new_tr
        new_st.write(os.path.join(self.dirpath, self.fname), **self.kw_write)

    def submit(self, st):
        """
        Asynchronously write *st* to a temporary MiniSEED file.

        Parameters
        ----------
        st : obspy.Stream
            Stream chunk to persist.
        """
        if not isinstance(st, obspy.Stream):
            raise TypeError(f"`st` must by a DataFrame object, not a {type(st)}")
        if self._future is not None:
            self._future.result()
        self._future = self._executor.submit(self._write, st)

    def write(self, st):
        """Alias for :meth:`submit`."""
        return self.submit(st)

    def _write(self, st):
        st.write(f"{self.dirpath}/{st[0].stats.starttime}_tmp.mseed", **self.kw_write)

    def shutdown(self):
        """Shut down the internal thread pool."""
        self._executor.shutdown()

    def result(self):
        """Merge all temporary MiniSEED files and write the final output."""
        self._future.result()
        self.shutdown()
        pattern = f"{self.dirpath}/*_tmp.mseed"
        out = obspy.read(pattern)
        out = out.merge(**self.kw_merge)
        if self.output_format == "flat":
            self._to_flat(out)
        elif self.output_format == "SDS":  # pragma: no branch
            self._to_SDS(out)
        files_to_remove = glob(pattern)
        for file in files_to_remove:
            os.remove(file)
        return out


class ZMQPublisher:
    """
    A class for publishing DataArray chunks over ZeroMQ.

    Parameters
    ----------
    address : str
        The address to bind the publisher to.
    encoding : dict
        The encoding to use when dumping the DataArrays to bytes.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.processing import ZMQPublisher, ZMQSubscriber

    First we generate some data and split it into packets

    >>> packets = xd.split(xd.testing.dummy(), 10)

    We initialize the publisher at a given address

    >>> address = f"tcp://localhost:{xd.io.get_free_port()}"
    >>> publisher = ZMQPublisher(address)

    We can then publish the packets

    >>> for da in packets:
    ...     publisher.submit(da)

    To reduce the size of the packets, we can also specify an encoding

    >>> import hdf5plugin

    >>> address = f"tcp://localhost:{xd.io.get_free_port()}"
    >>> encoding = {"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=1e-6)}
    >>> publisher = ZMQPublisher(address, encoding)
    >>> for da in packets:
    ...     publisher.submit(da)

    """

    def __init__(self, address, encoding=None):
        self.address = address
        self.encoding = encoding
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(self.address)

    def submit(self, da):
        """
        Send a DataArray over ZeroMQ.

        Parameters
        ----------
        da : DataArray
            The DataArray to be sent.

        """
        self._socket.send(tobytes(da, self.encoding))

    def write(self, da):
        """Alias for :meth:`submit`."""
        self.submit(da)

    def result(self):
        """Return ``None`` — ZMQPublisher has no aggregated result."""
        return


class ZMQSubscriber:
    """
    A class for subscribing to DataArray chunks over ZeroMQ.

    Parameters
    ----------
    address : str
        The address to connect the subscriber to.

    Methods
    -------
    submit(da)
        Send a DataArray over ZeroMQ.

    Examples
    --------
    >>> import threading

    >>> import xdas as xd
    >>> from xdas.processing import ZMQSubscriber

    First we generate some data and split it into packets

    >>> da = xd.testing.dummy()
    >>> packets = xd.split(da, 10)

    We then publish the packets asynchronously

    >>> address = f"tcp://localhost:{xd.io.get_free_port()}"
    >>> publisher = ZMQPublisher(address)

    >>> def publish():
    ...     for packet in packets:
    ...         publisher.submit(packet)

    >>> threading.Thread(target=publish).start()

    Now let's receive the packets

    >>> subscriber = ZMQSubscriber(address)
    >>> packets = []
    >>> for n, da in enumerate(subscriber, start=1):
    ...     packets.append(da)
    ...     if n == 10:
    ...         break
    >>> da = xd.concat(packets)
    >>> assert da.equals(da)
    """

    def __init__(self, address):
        self.address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def __iter__(self):
        return self

    def __next__(self):
        message = self._socket.recv()
        return frombuffer(message)


def tobytes(da, encoding=None):
    """
    Serialise *da* to raw NetCDF4 bytes via a temporary file.

    Parameters
    ----------
    da : DataArray
        DataArray to serialise.
    encoding : dict, optional
        HDF5/NetCDF4 encoding options forwarded to :meth:`DataArray.to_netcdf`.

    Returns
    -------
    bytes
    """
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tmp.nc")
        da.to_netcdf(path, virtual=False, encoding=encoding)
        with open(path, "rb") as file:
            return file.read()


def frombuffer(da):
    """
    Deserialise raw NetCDF4 *da* bytes into a loaded :class:`DataArray`.

    Parameters
    ----------
    da : bytes
        Raw bytes as produced by :func:`tobytes`.

    Returns
    -------
    DataArray
    """
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tmp.nc")
        with open(path, "wb") as file:
            file.write(da)
        return open_dataarray(path).load()
