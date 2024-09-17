import json
import struct

import h5py
import numpy as np
import zmq

from xdas.core.coordinates import get_sampling_interval

from ..core.dataarray import DataArray
from ..virtual import VirtualSource


def read(fname):
    with h5py.File(fname, "r") as file:
        header = file["header"]
        t0 = np.datetime64(round(header["time"][()] * 1e9), "ns")
        dt = np.timedelta64(round(1e9 * header["dt"][()]), "ns")
        dx = header["dx"][()] * np.median(np.diff(header["channels"]))
        data = VirtualSource(file["data"])
    nt, nx = data.shape
    time = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
    distance = {"tie_indices": [0, nx - 1], "tie_values": [0.0, (nx - 1) * dx]}
    return DataArray(data, {"time": time, "distance": distance})


class ZMQSubscriber:
    """
    A class used to subscribe to a ZeroMQ stream.

    Parameters
    ----------
    address : str
        The address to connect to.

    Attributes
    ----------
    socket : zmq.Socket
        The ZeroMQ socket used for communication.
    packet_size : int
        The size of each packet in bytes.
    shape : tuple
        The shape of the data array.
    format : str
        The format string used for unpacking the data.
    distance : dict
        The distance information.
    dt : numpy.timedelta64
        The sampling time interval.
    nt : int
        The number of time samples per message.

    Methods
    -------
    connect(address)
        Connects to the specified address.
    get_message()
        Receives a message from the socket.
    is_packet(message)
        Checks if the message is a valid packet.
    update_header(message)
        Updates the header information based on the received message.
    stream_packet(message)
        Processes a packet and returns a DataArray object.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd
    >>> from xdas.io.asn import ZMQStream
    >>> import holoviews as hv
    >>> from holoviews.streams import Pipe
    >>> hv.extension("bokeh")

    >>> stream = ZMQStream("tcp://pisco.unice.fr:3333")

    >>> nbuffer = 100
    >>> buffer = np.zeros((nbuffer, stream.shape[1]))
    >>> pipe = Pipe(data=buffer)

    >>> bounds = (
    ...     stream.distance["tie_values"][0],
    ...     0,
    ...     stream.distance["tie_values"][1],
    ...     (nbuffer * stream.dt) / np.timedelta64(1, "s"),
    ... )

    >>> def image(data):
    ...     return hv.Image(data, bounds=bounds)

    >>> dmap = hv.DynamicMap(image, streams=[pipe])
    >>> dmap.opts(
    ...     xlabel="distance",
    ...     ylabel="time",
    ...     invert_yaxis=True,
    ...     clim=(-1, 1),
    ...     cmap="viridis",
    ...     width=800,
    ...     height=400,
    ... )
    >>> dmap

    >>> atom = xd.atoms.Sequential(
    ...     [
    ...         xd.signal.integrate(..., dim="distance"),
    ...         xd.signal.sliding_mean_removal(..., wlen=1000.0, dim="distance"),
    ...     ]
    ... )
    >>> for da in stream:
    ...     da = atom(da) / 100.0
    ...     buffer = np.concatenate([buffer, da.values], axis=0)
    ...     buffer = buffer[-nbuffer:None]
    ...     pipe.send(buffer)

    """

    def __init__(self, address):
        """
        Initializes a ZMQStream object.

        Parameters
        ----------
        address : str
            The address to connect to.
        """
        self.connect(address)
        message = self.get_message()
        self.update_header(message)

    def __iter__(self):
        return self

    def __next__(self):
        message = self.get_message()
        if not self.is_packet(message):
            self.update_header(message)
            return self.__next__()
        else:
            return self.unpack(message)

    def connect(self, address):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(address)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket = socket

    def get_message(self):
        return self.socket.recv()

    def is_packet(self, message):
        return len(message) == self.packet_size

    def update_header(self, message):
        header = json.loads(message.decode("utf-8"))

        self.packet_size = 8 + header["bytesPerPackage"] * header["nPackagesPerMessage"]
        self.shape = (header["nPackagesPerMessage"], header["nChannels"])
        self.dtype = np.float32 if header["dataType"] == "float" else np.int16

        roiTable = header["roiTable"][0]
        di = roiTable["roiStart"] * header["dx"]
        de = roiTable["roiEnd"] * header["dx"]
        self.distance = {
            "tie_indices": [0, header["nChannels"] - 1],
            "tie_values": [di, de],
        }

        self.dt = float_to_timedelta(header["dt"], header["dtUnit"])
        self.nt = header["nPackagesPerMessage"]

    def unpack(self, message):
        t0 = np.frombuffer(message[:8], "datetime64[ns]")
        data = np.frombuffer(message[8:], self.dtype).reshape(self.shape)
        time = {
            "tie_indices": [0, self.shape[0] - 1],
            "tie_values": [t0, t0 + (self.shape[0] - 1) * self.dt],
        }
        return DataArray(data, {"time": time, "distance": self.distance})


class ZMQPublisher:
    def __init__(self, address):
        self.connect(address)
        self._header = None

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, header):
        self._header = header
        self.socket.setsockopt(zmq.XPUB_WELCOME_MSG, json.dumps(header).encode("utf-8"))

    def submit(self, da):
        self.send(da)

    def write(self, da):
        self.send(da)

    def connect(self, address):
        context = zmq.Context()
        socket = context.socket(zmq.XPUB)
        socket.bind(address)
        self.socket = socket

    @staticmethod
    def get_header(da):
        header = {
            "bytesPerPackage": da.dtype.itemsize * da.shape[1],
            "nPackagesPerMessage": da.shape[0],
            "nChannels": da.shape[1],
            "dataType": "float" if da.dtype == np.float32 else "short",
            "dx": get_sampling_interval(da, "distance"),
            "dt": get_sampling_interval(da, "time"),
            "dtUnit": "s",
            "dxUnit": "m",
            "roiTable": [{"roiStart": 0, "roiEnd": da.shape[1] - 1, "roiDec": 1}],
        }
        return header

    def send(self, da):
        header = self.get_header(da)
        if self.header is None:
            self.header = header
        if not header == self.header:
            self.header = header
            self.send_header()
        self.send_data(da)

    def send_header(self):
        message = json.dumps(self.header).encode("utf-8")
        self.send_message(message)

    def send_data(self, da):
        t0 = da["time"][0].values.astype("datetime64[ns]")
        data = da.values
        message = t0.tobytes() + data.tobytes()
        self.send_message(message)

    def send_message(self, message):
        self.socket.send(message)


def float_to_timedelta(value, unit):
    """
    Converts a floating-point value to a timedelta object.

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
    >>> float_to_timedelta(1.5, 'ms')
    numpy.timedelta64(1500000,'ns')
    """
    conversion_factors = {
        "ns": 1e0,
        "us": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }
    conversion_factor = conversion_factors[unit]
    return np.timedelta64(round(value * conversion_factor), "ns")
