from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import scipy.signal as sp

from .core import concatenate, open_database


class SignalProcessingChain:
    """
    Chain of processing unit to apply to a Database.

    Parameters
    ----------
    filters : list of callable
        The chain of filters to apply. The filters can have some state memory to ensure
        continuity when repeatedly applied to consecutive chunks.
    """

    def __init__(self, filters):
        self.filters = filters

    def __call__(self, db):
        out = db
        for filter in self.filters:
            out = filter(out)
        return out

    def process(self, db, dim, chunk_size, parallel=True):
        """
        Process a Database by chunk along a given dimension.

        Parameters
        ----------
        db : Database
            The Database to process.
        dim : str
            Name of the the dimension along which to process the Database.
        chunk_size : int
            The number of samples per chunk.
        parallel : bool, optional
            Whether to use multithreading to improve performance, by default True.

        Returns
        -------
        Database
            The processed Database.
        """
        div, mod = divmod(db.sizes[dim], chunk_size)
        if mod == 0:
            nchunks = div
        else:
            nchunks = div + 1
        result = []
        if parallel:
            chunks = []
            for k in range(nchunks):
                query = {dim: slice(k * chunk_size, (k + 1) * chunk_size)}
                chunks.append(db[query])
            scheduler = Scheduler(chunks, self.filters)
            result = scheduler.compute()
        else:
            for k in range(nchunks):
                query = {dim: slice(k * chunk_size, (k + 1) * chunk_size)}
                result.append(self(db[query]))
        return concatenate(result, dim)


class Scheduler:
    """
    Sequential multithreading scheduler.

    As soon as one filter finishes to process a chunk, the scheduler submit the two
    newly available task: applying the same filter on the next chunk and applying the
    next filter to the same chunk.

    Parameters
    ----------
    chunks : list of Database
        The chunks to process.
    filters : list of callable.
        The chain of filters to apply. The filters can have some state memory to ensure
        continuity when repeatedly applied to consecutive chunks.
    """

    def __init__(self, chunks, filters):
        self.chunks = chunks
        self.filters = filters
        self.buffer = {idx: chunk for idx, chunk in enumerate(chunks)}
        self.executor = None
        self.futures = {}

    def compute(self):
        """
        Launch the processing.

        Returns
        -------
        list of Database
            The processed chunks.
        """
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
                # print("Number of futures:", len(self.futures))
        return list(self.buffer.values())

    def submit_task(self, task):
        """
        Submit a task.

        Parameters
        ----------
        task : dict
            The task to submit described by the number of the filter and the number of
            the chunk.

        Returns
        -------
        future
            The future related to the submitted task.
        """
        filter = self.filters[task["filter"]]
        chunk = self.buffer[task["chunk"]]
        future = self.executor.submit(filter, chunk)
        self.futures[future] = task
        # print("New task:", task)
        return future

    def get_next_tasks(self, task):
        """
        Given a finished task, gives the two new available possible tasks: applying the
        same filter on the next chunk and applying the next filter to the same chunk.

        Parameters
        ----------
        task : dict
            The finished task described as the number of the filter and the number of
            the chunk.

        Returns
        -------
        list of dict
            The new available tasks.
        """
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
    """
    Parent class to all signal processing units.
    """

    pass


class LFilter(SignalProcessingUnit):
    """
    Filter data along one-dimension with an IIR or FIR.

    Once the filter is initialized it can be used repeatedly on subsequent chunks of
    data as a normal python function. The state of the filter will be saved between
    each call to ensure continuity of the filtering. The state of the filter can be
    reset for a fresh start.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.
    dim : str
        Name of the dimension of the input database along which to apply the linear
        filter.
    """

    def __init__(self, b, a, dim):
        self.b = b
        self.a = a
        self.dim = dim
        self.zi = None
        self.axis = None

    def __call__(self, db):
        if (self.zi is None) or (self.axis is None):
            self.initialize(db)
        data, self.zi = sp.lfilter(self.b, self.a, db.data, axis=self.axis, zi=self.zi)
        return db.copy(data=data)

    def initialize(self, db):
        """
        Initialize the filter state.

        Parameters
        ----------
        db : Database
            Some sample of the kind of database to process to get the correct axis
            number.
        """
        self.axis = db.get_axis_num(self.dim)
        zi = sp.lfilter_zi(self.b, self.a)
        n_sections = zi.shape[0]
        s = [n_sections if k == self.axis else 1 for k in range(db.ndim)]
        zi = zi.reshape(s)
        s = [n_sections if k == self.axis else db.shape[k] for k in range(db.ndim)]
        zi = np.broadcast_to(zi, s)
        self.zi = zi

    def reset(self):
        """
        Reset state of the filter for a fresh start.
        """
        self.zi = None
        self.axis = None


class SOSFilter:
    """
    Filter data along one dimension using cascaded second-order sections.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients.
    dim : str
        Name of the dimension of the input database along which to apply the linear
        filter.
    """

    def __init__(self, sos, dim):
        self.sos = sos
        self.dim = dim
        self.zi = None
        self.axis = None

    def __call__(self, db):
        if (self.zi is None) or (self.axis is None):
            self.initialize(db)
        data, self.zi = sp.sosfilt(self.sos, db, axis=self.axis, zi=self.zi)
        return db.copy(data=data)

    def initialize(self, db):
        """
        Initialize the filter state.

        Parameters
        ----------
        db : Database
            Some sample of the kind of database to process to get the correct axis
            number.
        """
        self.axis = db.get_axis_num(self.dim)
        zi = sp.sosfilt_zi(self.sos)
        ndim = len(db.shape)
        n_sections = zi.shape[0]
        s = [n_sections] + [2 if k == self.axis else 1 for k in range(ndim)]
        zi = zi.reshape(s)
        s = [n_sections] + [2 if k == self.axis else db.shape[k] for k in range(ndim)]
        zi = np.broadcast_to(zi, s)
        self.zi = zi

    def reset(self):
        """
        Reset state of the filter for a fresh start.
        """
        self.zi = None
        self.axis = None


class Decimate(SignalProcessingUnit):
    def __init__(self, q, dim):
        self.q = q
        self.dim = dim

    def __call__(self, db):
        return db[{self.dim: slice(None, None, self.q)}]


class ChunkWriter(SignalProcessingUnit):
    """
    Write to disk chunk by chunk.

    Parameters
    ----------
    path : str
        Path were to store the chunks. The number of the chunk will be prepended.
    """

    def __init__(self, path):
        self.path = path
        self.chunk = None

    def __call__(self, db):
        if self.chunk is None:
            self.initialize(db)
        self.chunk += 1
        postfix = f"_{self.chunk:06d}.nc"
        fname = self.path + postfix
        db.to_netcdf(fname)
        return open_database(fname)

    def initialize(self, db):
        """
        Initialize the chunk numbering to zero.
        """
        self.chunk = 0

    def reset(self):
        """
        Reset the chunk numbering.
        """
        self.chunk = None
