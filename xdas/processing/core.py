import os
from concurrent.futures import ThreadPoolExecutor
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

from ..core.dataarray import DataArray
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
        The maximum number of thread used to load the chunks.

    Examples
    --------
    >>> import xdas
    >>> from xdas.processing import DataArrayLoader
    >>> da = xdas.open_dataarray(...)  # doctest: +SKIP

    Create chunks along the time dimension

    >>> chunks = {"time": 1000}
    >>> dl = DataArrayLoader(da, chunks)  # doctest: +SKIP

    Iterate over the chunks

    >>> for chunk in dl:
    ...     process(chunk)  # doctest: +SKIP

    """

    def __init__(self, da, chunks, max_buffers=1, max_workers=1):
        if not isinstance(da, DataArray):
            raise TypeError(f"`da` must by a DataArray object, not a {type(da)}")
        if not isinstance(chunks, dict) and len(chunks) == 1:
            raise TypeError(
                "`chunks` must be a dict that maps a unique "
                "dimension to a unique size: {'dim': int}"
            )
        ((chunk_dim, chunk_size),) = chunks.items()
        chunk_dim = str(chunk_dim)
        chunk_size = int(chunk_size)
        if chunk_dim not in da.dims:
            raise ValueError(
                f"chunking dimension {chunk_dim} not "
                f"found in `da` dimensions {da.dims}"
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

    def __len__(self):
        div, mod = divmod(self.da.sizes[self.chunk_dim], self.chunk_size)
        return div if mod == 0 else div + 1

    def _get_chunk(self, idx):
        start = idx * self.chunk_size
        end = (idx + 1) * self.chunk_size
        query = {
            dim: slice(start, end) if dim == self.chunk_dim else slice(None)
            for dim in self.da.dims
        }
        return self.da[query].load()

    def __iter__(self):
        with ThreadPoolExecutor(self.max_workers) as executor:
            it = iter(range(len(self)))

            futures = []
            try:
                for _ in range(self.max_buffers):
                    futures.append(executor.submit(self._get_chunk, next(it)))
            except StopIteration:
                pass

            while futures:
                future = futures.pop(0)
                result = future.result()

                try:
                    futures.append(executor.submit(self._get_chunk, next(it)))
                except StopIteration:
                    pass

                yield result

    @property
    def nbytes(self):
        return self.da.nbytes


class RealTimeLoader(Observer):
    def __init__(self, path, engine="netcdf"):
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

    Examples
    --------
    >>> import xdas.processing as xp

    >>> expected = xd.DataArray(np.random.rand(1000, 100), dims=("time", "distance"))

    >>> dw = DataArrayWriter("some_path")
    >>> for chunk in chunks:
    ...     dw.submit(chunk)
    >>> result = dw.result

    >>> assert result.equals(expected)

    """

    def __init__(
        self, dirpath, encoding=None, max_buffers=1, max_workers=1, create_dirs=False
    ):
        dirpath = str(dirpath) if isinstance(dirpath, Path) else dirpath
        if create_dirs:
            os.makedirs(dirpath, exist_ok=True)
        if not os.path.exists(dirpath):
            raise OSError(f"no directory {dirpath}")
        self.dirpath = dirpath
        self.encoding = encoding
        self.max_buffers = max_buffers
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(self.max_workers)
        self._futures = []
        self._results = []
        self._count = 0

    def submit(self, chunk):
        if not isinstance(chunk, DataArray):
            raise TypeError(f"`chunk` must by a DataArray object, not a {type(chunk)}")
        if not len(self._futures) < self.max_buffers:
            future = self._futures.pop(0)
            result = future.result()
            self._results.append(result)
        self._futures.append(self._executor.submit(self._write, chunk, self._count))
        self._count += 1

    def write(self, chunk):
        return self.submit(chunk)

    def _write(self, chunk, count):
        path = os.path.join(self.dirpath, f"{count:09d}")
        chunk.to_netcdf(path, encoding=self.encoding)
        return open_dataarray(path)

    def shutdown(self):
        self._executor.shutdown()

    def result(self):
        while self._futures:
            future = self._futures.pop(0)
            result = future.result()
            self._results.append(result)
        self.shutdown()
        return concatenate(self._results)


class DataFrameWriter:
    """
    A class for writing pandas DataFrames to a CSV file asynchronously.

    Parameters
    ----------
    path : str
        The path to the csv file.
    parse_dates : bool, int, optional
        Weather to parse dates when reopening the csv file a the end of the process
    create_dirs : bool, optional
        Whether to create parent directories if they do not exist. Default is False.
    """

    def __init__(self, path, parse_dates=None, create_dirs=False):
        dirpath = os.path.dirname(path)
        if create_dirs:
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
        if dirpath and not os.path.exists(dirpath):
            raise OSError(f"no directory {dirpath}")
        self.path = str(path) if isinstance(path, Path) else path
        self.parse_dates = parse_dates
        self._executor = ThreadPoolExecutor(1)
        self._future = None

    def submit(self, chunk):
        if not isinstance(chunk, pd.DataFrame):
            raise TypeError(f"`chunk` must by a DataFrame object, not a {type(chunk)}")
        if self._future is not None:
            self._future.result()
        self._future = self._executor.submit(self._write, chunk)

    def write(self, chunk):
        return self.submit(chunk)

    def _write(self, chunk):
        if chunk is not None:
            if not os.path.exists(self.path):
                chunk.to_csv(self.path, mode="w", header=True, index=False)
            else:
                chunk.to_csv(self.path, mode="a", header=False, index=False)

    def shutdown(self):
        self._executor.shutdown()

    def result(self):
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
    >>> import xdas
    >>> import xdas.processing as xp

    Generate some DataArray:

    >>> data = np.random.randint(
    ...     low=-1000, high=1000, size=(1000, 10), dtype=np.int32
    ... )
    >>> starttime = np.datetime64("2023-01-01T00:00:00")
    >>> endtime = starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1)
    >>> distance = 5.0 * np.arange(data.shape[1])
    >>> da = xdas.DataArray(
    ...     data=data,
    ...     coords={
    ...         "time": {
    ...             "tie_indices": [0, data.shape[0] - 1],
    ...             "tie_values": [starttime, endtime],
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
        self.queue = Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(1)
        self.future = self.executor.submit(self.task)

    def to_SDS(self, st):
        """
        Convert and write the Stream to the SDS file structure.
        """
        for tr in st:
            new_st = obspy.Stream()
            new_st += tr
            new_st = new_st[0].split()
            for new_tr in new_st:
                if isinstance(new_tr.data, np.ma.masked_array):
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

    def to_flat(self, st):
        """
        Convert and write the Stream to a single miniseed file.
        """
        new_st = obspy.Stream()
        for tr in st:
            tmp_st = obspy.Stream()
            tmp_st += tr
            tmp_st = tmp_st[0].split()
            for new_tr in tmp_st:
                if isinstance(new_tr.data, np.ma.masked_array):
                    new_tr.data = new_tr.data.filled()
                new_st += new_tr
        new_st.write(os.path.join(self.dirpath, self.fname), **self.kw_write)

    def write(self, st):
        """
        Writes a Stream to the queue for asynchronous writing.

        Parameters
        ----------
        st : obspy.Stream
            The Stream to be written.
        """
        self.queue.put(st)

    def task(self):
        """
        The asynchronous task that writes the Stream to a temporary miniseed file.
        """
        while True:
            st = self.queue.get()
            if st is None:
                break
            st.write(
                f"{self.dirpath}/{st[0].stats.starttime}_tmp.mseed", **self.kw_write
            )

    def result(self):
        """
        Waits for the asynchronous task to complete, format the data into the chosen format, delete the temporary files
        and returns the one single merged Stream read from the temporary files.

        Returns
        -------
        obspy.Stream
            Returns one single merged Stream read from the temporary files.
        """
        self.queue.put(None)
        self.future.result()
        pattern = f"{self.dirpath}/*_tmp.mseed"
        out = obspy.read(pattern)
        out = out.merge(**self.kw_merge)
        if self.output_format == "flat":
            self.to_flat(out)
        elif self.output_format == "SDS":
            self.to_SDS(out)
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

    >>> packets = xd.split(xd.synthetics.dummy(), 10)

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
        self.submit(da)

    def result():
        return None


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

    >>> da = xd.synthetics.dummy()
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
    >>> da = xd.concatenate(packets)
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
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tmp.nc")
        da.to_netcdf(path, virtual=False, encoding=encoding)
        with open(path, "rb") as file:
            return file.read()


def frombuffer(da):
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tmp.nc")
        with open(path, "wb") as file:
            file.write(da)
        return open_dataarray(path).load()
