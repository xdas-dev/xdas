"""
I/O engine for ASN HDF5 files and live ZMQ streams.

Includes :class:`ASNEngine` and a ZMQ-based real-time subscriber
(:class:`ZMQSubscriber`) for live ASN streams.
"""

import json
from bisect import bisect_left, bisect_right
from typing import ClassVar

import h5py
import numpy as np
import zmq

from ..coordinates import Coordinate, get_sampling_interval
from ..core import DataArray, concat_coords
from ..processing.core import SubscriptionTracker, ZMQEndpoint
from ..virtual import TileArray, VirtualSource
from .core import Engine


class ASNEngine(Engine, name="asn"):
    """Engine for reading ASN HDF5 files."""

    _supported_vtypes: ClassVar[list] = ["hdf5", "tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated", "sampled", "dense"],
        "distance": ["interpolated"],
    }

    def open_dataarray(self, fname):
        """Read an ASN HDF5 file *fname* and return a virtual :class:`DataArray`."""
        with h5py.File(fname, "r") as file:
            header = file["header"]
            demod = file["demodSpec"]

            t0 = np.datetime64(round(header["time"][()] * 1e9), "ns")
            dt = np.timedelta64(round(1e9 * header["dt"][()]), "ns")
            dx = float(header["dx"][()])  # Note: dx before (internal) downsampling!
            if self.vtype == "tiles":
                data = TileArray.from_tiles(
                    str(fname), file["data"].shape, file["data"].dtype, "asn"
                )
            else:
                data = VirtualSource(file["data"])

            # Get the optical distance for all the recorded channels (after downsampling)
            # Note that this vector is not continuous for more than one ROI
            all_dists = file["cableSpec"]["sensorDistances"][...]

            # One regular block per ROI, concatenated below
            roi_blocks = []

            # Channel spacing is dx times the ROI decimation. Some files omit
            # roiDec; those fall back to the spacing the ROI bounds imply.
            roi_decs = demod["roiDec"][...] if "roiDec" in demod else None

            # Loop over ROIs, get the start/stop index before downsampling
            for n_roi, (n_start, n_end) in enumerate(
                zip(demod["roiStart"], demod["roiEnd"])
            ):
                # ASN stores ROI end as an upper boundary. Use the last sampled distance
                # that does not exceed that boundary instead of indexing the insertion point.
                i_start, i_end = self._get_roi_bound_indices(
                    all_dists, n_start, n_end, dx
                )

                # Get the index where the ROI starts based on the position in the
                # distance vector. This solves the issue of rounding during decimation
                start = float(all_dists[i_start])
                size = i_end - i_start + 1
                if roi_decs is not None:
                    # Taking the spacing from the metadata keeps it bit-identical
                    # across ROIs that share a decimation, which is what lets
                    # concatenation preserve a regular axis.
                    step = dx * int(roi_decs[n_roi])
                elif size > 1:
                    step = (float(all_dists[i_end]) - start) / (i_end - i_start)
                else:
                    step = dx
                roi_blocks.append(
                    Coordinate[self.ctype["distance"]].from_block(
                        start, size, step, dim="distance"
                    )
                )

        nt = data.shape[0]
        time = Coordinate[self.ctype["time"]].from_block(t0, nt, dt, dim="time")
        # Concatenation keeps the declared spacing when every ROI agrees on it
        # and drops to irregular otherwise, so unevenly decimated files stay
        # honest without the engine having to test for it. Regularizing recovers
        # the spacing when ROI steps differ only by float rounding (which the
        # fallback above can produce), while genuinely different decimations
        # still fail the fit. Reducing is off so the ROI structure is preserved.
        distance = concat_coords(roi_blocks, reduce=False, regularize=True)
        return DataArray(data, {"time": time, "distance": distance})

    @staticmethod
    def load_tile(path, selection):
        """Read a source selection of the ``/data`` dataset of an ASN file."""
        with h5py.File(path, "r") as file:
            return file["/data"][selection]

    def _get_roi_bound_indices(self, all_dists, n_start, n_end, dx):
        start_index = bisect_left(all_dists, n_start * dx)
        if start_index >= len(all_dists):
            raise IndexError("ROI start lies beyond available sensor distances")

        end_index = bisect_right(all_dists, n_end * dx) - 1
        if end_index < 0:
            raise IndexError("ROI end lies before available sensor distances")

        return start_index, end_index


type_map = {
    "short": np.int16,
    "int": np.int32,
    "long": np.int64,
    "float": np.float32,
    "double": np.float64,
}


class ZMQSubscriber(ZMQEndpoint):
    """
    Iterator that pulls :class:`DataArray` chunks from a live ASN ZMQ publisher.

    Parameters
    ----------
    address : str
        ZMQ address of the publisher (e.g. ``"tcp://localhost:5555"``).
    timeout : float or None, optional
        How many seconds to wait at most for each message. None, the default,
        waits forever.

    Methods
    -------
    wait_until_subscribed()
        Block until the publisher has registered this subscription. Building
        the subscriber already does it.
    close()
        Release the socket and its context.
    """

    def __init__(self, address, timeout=None):
        """
        Initialize a ZMQStream object.

        Parameters
        ----------
        address : str
            The address to connect to.
        timeout : float or None, optional
            How many seconds to wait at most for each message.

        Examples
        --------
        >>> import threading

        >>> import xdas as xd
        >>> from xdas.io.asn import ZMQSubscriber

        >>> port = xd.io.get_free_port()
        >>> address = f"tcp://localhost:{port}"
        >>> publisher = ZMQPublisher(address)

        >>> da = xd.testing.dummy()
        >>> chunks = xd.split(da, 10)

        >>> def publish():
        ...     publisher.wait_for_subscribers()  # a replay, so no chunk is lost
        ...     for chunk in chunks:
        ...         publisher.submit(chunk)
        >>> thread = threading.Thread(target=publish)
        >>> thread.start()

        >>> subscriber = ZMQSubscriber(address)
        >>> for nchunk in range(10):
        ...     chunk = next(subscriber)
        ...     # do something with the chunk

        Both ends hold a socket until they are closed, by hand as here or by
        using them as context managers.

        >>> thread.join()
        >>> subscriber.close()
        >>> publisher.close()

        """
        self.address = address
        self.timeout = timeout
        self._subscribed = False
        self._connect(self.address)
        self.wait_until_subscribed()

    def __iter__(self):
        return self

    def __next__(self):
        message = self._get_message()
        if not self._is_packet(message):
            self._update_header(message)
            return self.__next__()
        else:
            return self._unpack(message)

    def _connect(self, address):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(address)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._context = context
        self._socket = socket

    def wait_until_subscribed(self):
        """
        Block until the publisher has registered this subscription.

        A publisher drops what it sends to a peer it does not know about yet,
        and a subscriber cannot tell from its own side whether its
        subscription has arrived — being connected is not being subscribed.
        Here the proof comes for free: the header describing the stream is the
        greeting an ASN publisher answers a new subscription with, in passing
        as it streams. It never waits for anyone, and receiving its header is
        proof that nothing it publishes from then on will be missed.

        This is done when the subscriber is built — a packet cannot be decoded
        before it — so calling it again returns immediately. The one stream
        that keeps a subscriber waiting is one that is not streaming: a
        publisher that has gone quiet, or has yet to send its first packet,
        acknowledges nobody.
        """
        # A subscriber that beat the first publication to the socket gets no
        # welcome message, and can be handed data before the header is sent.
        while not self._subscribed:
            message = self._get_message()
            if self._is_header(message):
                self._update_header(message)
                self._subscribed = True

    def _get_message(self):
        if self.timeout is not None and not self._socket.poll(
            round(1000 * self.timeout)
        ):
            raise TimeoutError(
                f"no message received from {self.address} after {self.timeout} seconds"
            )
        return self._socket.recv()

    @staticmethod
    def _is_header(message):
        """Whether *message* is a header rather than a data packet."""
        try:
            header = json.loads(message.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(header, dict) and "bytesPerPackage" in header

    def _is_packet(self, message):
        return len(message) == self.packet_size

    def _update_header(self, message):
        header = json.loads(message.decode("utf-8"))
        self.packet_size = 8 + header["bytesPerPackage"] * header["nPackagesPerMessage"]
        self.shape = (header["nPackagesPerMessage"], header["nChannels"])
        self.dtype = type_map[header["dataType"]]
        roiTable = header["roiTable"][0]
        di = (roiTable["roiStart"] // roiTable["roiDec"]) * header["dx"]
        de = (roiTable["roiEnd"] // roiTable["roiDec"]) * header["dx"]
        self.distance = {
            "tie_indices": [0, header["nChannels"] - 1],
            "tie_values": [di, de],
            "sampling_numerator": de - di,
            "sampling_denominator": header["nChannels"] - 1,
        }
        self.delta = float_to_timedelta(header["dt"], header["dtUnit"])

    def _unpack(self, message):
        t0 = np.frombuffer(message[:8], "datetime64[ns]").reshape(())
        data = np.frombuffer(message[8:], self.dtype).reshape(self.shape)
        time = {
            "tie_indices": [0, self.shape[0] - 1],
            "tie_values": [t0, t0 + (self.shape[0] - 1) * self.delta],
            "sampling_interval": self.delta,
        }
        return DataArray(data, {"time": time, "distance": self.distance})


class ZMQPublisher(SubscriptionTracker):
    """
    A class to stream data using ZeroMQ.

    Parameters
    ----------
    address : str
        The address to bind the ZeroMQ socket.

    Attributes
    ----------
    address : str
        The address where the ZeroMQ is bound to.
    nsubscribers : int
        The number of currently subscribed peers.

    Methods
    -------
    submit(da)
        Submits the data array for publishing.
    wait_for_subscribers(count, timeout)
        Blocks until *count* peers are subscribed, so that nothing published
        afterwards is dropped.
    close()
        Release the socket and its context.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.io.asn import ZMQPublisher

    >>> da = xd.testing.dummy()

    >>> port = xd.io.get_free_port()
    >>> address = f"tcp://localhost:{port}"
    >>> chunks = xd.split(da, 10)
    >>> with ZMQPublisher(address) as publisher:
    ...     for chunk in chunks:
    ...         publisher.submit(chunk)

    """

    def __init__(self, address):
        self.address = address
        self._nsubscribers = 0
        self._connect(address)
        self._header = None

    @property
    def header(self):
        """The last welcome-message header dict sent to new subscribers."""
        return self._header

    @header.setter
    def header(self, header):
        """Set the welcome-message header and push it to the ZMQ socket option."""
        self._header = header
        self._socket.setsockopt(
            zmq.XPUB_WELCOME_MSG, json.dumps(header).encode("utf-8")
        )

    def submit(self, da):
        """Publish *da* over ZMQ."""
        self._send(da)

    def write(self, da):
        """Alias for :meth:`submit`."""
        self._send(da)

    def _connect(self, address):
        context = zmq.Context()
        socket = context.socket(zmq.XPUB)
        socket.setsockopt(zmq.XPUB_VERBOSE, True)
        socket.bind(address)
        self._context = context
        self._socket = socket

    @staticmethod
    def _get_header(da):
        da = da.transpose("time", "distance")
        header = {
            "bytesPerPackage": da.dtype.itemsize * da.shape[1],
            "nPackagesPerMessage": da.shape[0],
            "nChannels": da.shape[1],
            "dataType": next((k for k, v in type_map.items() if v == da.dtype), None),
            "dx": get_sampling_interval(da, "distance"),
            "dt": get_sampling_interval(da, "time"),
            "dtUnit": "s",
            "dxUnit": "m",
            "roiTable": [{"roiStart": 0, "roiEnd": da.shape[1] - 1, "roiDec": 1}],
        }
        return header

    def _send(self, da):
        # Taking the subscriptions the socket has queued is what greets the
        # peers behind them with the header — ZeroMQ holds a welcome message
        # back until the application reads the subscription it answers — and
        # what keeps the subscriber count current. Neither costs any waiting.
        self._read_subscriptions(0.0)
        da = da.transpose("time", "distance")
        header = self._get_header(da)
        if header != self.header:
            # Peers that subscribed before there was a header to welcome them
            # with — including the very first one — only learn the layout if it
            # is sent down the stream, so the first submit publishes it too.
            self.header = header
            self._send_header()
        self._send_data(da)

    def _send_header(self):
        message = json.dumps(self.header).encode("utf-8")
        self._send_message(message)

    def _send_data(self, da):
        da = da.transpose("time", "distance")
        t0 = da["time"][0].values.astype("datetime64[ns]")
        data = da.values
        message = t0.tobytes() + data.tobytes()
        self._send_message(message)

    def _send_message(self, message):
        self._socket.send(message)


def float_to_timedelta(value, unit):
    """
    Convert a floating-point value to a timedelta object.

    Parameters
    ----------
    value : float
        The value to be converted.
    unit : str
        The unit of the value. Valid units are 'ns' (nanoseconds), 'us' (microseconds),
        'ms' (milliseconds), and 's' (seconds).

    Returns
    -------
    timedelta
        The converted timedelta object.

    Example
    -------
    float_to_timedelta(1.5, 'ms')  # doctest: +SKIP
    np.timedelta64(1500000,'ns')
    """
    conversion_factors = {
        "ns": 1e0,
        "us": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }
    conversion_factor = conversion_factors[unit]
    return np.timedelta64(round(value * conversion_factor), "ns")
