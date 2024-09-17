import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zmq

import xdas as xd
from xdas.io.asn import ZMQPublisher, ZMQSubscriber

executor = ThreadPoolExecutor()


def get_free_address():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    address = f"tcp://localhost:{port}"
    return address


coords = {
    "time": {
        "tie_indices": [0, 99],
        "tie_values": [
            np.datetime64("2020-01-01T00:00:00.000"),
            np.datetime64("2020-01-01T00:00:09.900"),
        ],
    },
    "distance": {"tie_indices": [0, 9], "tie_values": [0.0, 90.0]},
}

da_float32 = xd.DataArray(
    np.random.randn(100, 10).astype("float32"),
    coords,
)

da_int16 = xd.DataArray(
    np.random.randn(100, 10).astype("int16"),
    coords,
)


class TestZMQPublisher:
    def test_get_header(self):
        header = ZMQPublisher.get_header(da_float32)
        assert header["bytesPerPackage"] == 40
        assert header["nPackagesPerMessage"] == 100
        assert header["nChannels"] == 10
        assert header["dataType"] == "float"
        assert header["dx"] == 10.0
        assert header["dt"] == 0.1
        assert header["dtUnit"] == "s"
        assert header["dxUnit"] == "m"
        assert header["roiTable"] == [{"roiStart": 0, "roiEnd": 9, "roiDec": 1}]
        header = ZMQPublisher.get_header(da_int16)
        assert header["dataType"] == "short"

    def test_init_conect_set_header(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        pub.submit(da_float32)
        assert pub.header == ZMQPublisher.get_header(da_float32)

    def test_send_header(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        pub.submit(da_float32)
        socket = self.get_socket(address)
        pub.submit(da_float32)  # a packet must be sent once subscriber is connected
        assert socket.recv() == json.dumps(pub.header).encode("utf-8")

    def test_send_data(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        pub.submit(da_float32)
        socket = self.get_socket(address)
        pub.submit(da_float32)  # a packet must be sent once subscriber is connected
        socket.recv()  # header
        message = socket.recv()
        assert message[:8] == da_float32["time"][0].values.astype("M8[ns]").tobytes()
        assert message[8:] == da_float32.data.tobytes()
        pub.submit(da_int16)
        socket.recv()  # header
        message = socket.recv()
        assert message[:8] == da_int16["time"][0].values.astype("M8[ns]").tobytes()
        assert message[8:] == da_int16.data.tobytes()

    def test_send_chunks(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 10)
        pub.submit(chunks[0])
        time.sleep(0.001)
        socket = self.get_socket(address)
        for chunk in chunks[1:]:
            pub.submit(chunk)
            time.sleep(0.001)
        assert socket.recv() == json.dumps(pub.header).encode("utf-8")
        for chunk in chunks[1:]:  # first was sent before subscriber connected
            message = socket.recv()
            assert message[:8] == chunk["time"][0].values.astype("M8[ns]").tobytes()
            assert message[8:] == chunk.data.tobytes()

    def test_several_subscribers(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 10)
        pub.submit(chunks[0])
        time.sleep(0.001)
        socket1 = self.get_socket(address)
        for chunk in chunks[1:5]:
            pub.submit(chunk)
            time.sleep(0.001)
        socket2 = self.get_socket(address)
        for chunk in chunks[5:]:
            pub.submit(chunk)
            time.sleep(0.001)
        assert socket1.recv() == json.dumps(pub.header).encode("utf-8")
        for chunk in chunks[1:]:  # first was sent before subscriber connected
            message = socket1.recv()
            assert message[:8] == chunk["time"][0].values.astype("M8[ns]").tobytes()
            assert message[8:] == chunk.data.tobytes()
        assert socket2.recv() == json.dumps(pub.header).encode("utf-8")
        for chunk in chunks[5:]:  # first was sent before subscriber connected
            message = socket2.recv()
            assert message[:8] == chunk["time"][0].values.astype("M8[ns]").tobytes()
            assert message[8:] == chunk.data.tobytes()

    def test_change_header(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 10)
        pub.submit(chunks[0])
        time.sleep(0.001)
        socket = self.get_socket(address)
        for chunk in chunks[1:5]:
            pub.submit(chunk)
            header1 = pub.header
            time.sleep(0.001)
        for chunk in chunks[5:]:
            pub.submit(chunk.isel(distance=slice(0, 5)))
            header2 = pub.header
            time.sleep(0.001)
        assert socket.recv() == json.dumps(header1).encode("utf-8")
        for chunk in chunks[1:5]:  # first was sent before subscriber connected
            message = socket.recv()
            assert message[:8] == chunk["time"][0].values.astype("M8[ns]").tobytes()
            assert message[8:] == chunk.data.tobytes()
        assert socket.recv() == json.dumps(header2).encode("utf-8")
        for chunk in chunks[5:]:  # first was sent before subscriber connected
            message = socket.recv()
            assert message[:8] == chunk["time"][0].values.astype("M8[ns]").tobytes()
            assert message[8:] == chunk.isel(distance=slice(0, 5)).data.tobytes()

    def get_socket(self, address):
        socket = zmq.Context().socket(zmq.SUB)
        socket.connect(address)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        time.sleep(0.001)
        return socket


class TestZMQSubscriber:
    def test_one_chunk(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        chunks = [da_float32]
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        assert sub.packet_size == 4008
        assert sub.shape == (100, 10)
        assert sub.dtype == np.float32
        assert sub.distance == {"tie_indices": [0, 9], "tie_values": [0.0, 90.0]}
        assert sub.delta == np.timedelta64(100, "ms")
        result = next(sub)
        assert result.equals(da_float32)
        chunks = [da_int16]
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        result = next(sub)
        assert sub.packet_size == 2008
        assert sub.dtype == np.int16
        assert result.equals(da_int16)

    def test_several_chunks(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 5)
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        assert sub.packet_size == 808
        assert sub.shape == (20, 10)
        assert sub.dtype == np.float32
        assert sub.distance == {"tie_indices": [0, 9], "tie_values": [0.0, 90.0]}
        assert sub.delta == np.timedelta64(100, "ms")
        for chunk in chunks:
            result = next(sub)
            assert result.equals(chunk)

    def test_several_subscribers(self):
        address = get_free_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 5)
        thread = threading.Thread(target=self.publish, args=(pub, chunks[:2]))
        thread.start()
        sub1 = ZMQSubscriber(address)
        thread.join()
        thread = threading.Thread(target=self.publish, args=(pub, chunks[2:]))
        thread.start()
        sub2 = ZMQSubscriber(address)

        for chunk in chunks:
            result = next(sub1)
            assert result.equals(chunk)
        for chunk in chunks[2:]:
            result = next(sub2)
            assert result.equals(chunk)

    def publish(self, pub, chunks):
        for chunk in chunks:
            time.sleep(0.001)
            pub.submit(chunk)
