import json
import threading
import time

import numpy as np
import zmq

import xdas as xd
from xdas.io.asn import ZMQPublisher, ZMQSubscriber


def get_free_local_address():
    port = xd.io.get_free_port()
    return f"tcp://localhost:{port}"


coords = {
    "time": {
        "tie_indices": [0, 99],
        "tie_values": [
            np.datetime64("2020-01-01T00:00:00.000000000"),
            np.datetime64("2020-01-01T00:00:09.900000000"),
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
        header = ZMQPublisher._get_header(da_float32)
        assert header["bytesPerPackage"] == 40
        assert header["nPackagesPerMessage"] == 100
        assert header["nChannels"] == 10
        assert header["dataType"] == "float"
        assert header["dx"] == 10.0
        assert header["dt"] == 0.1
        assert header["dtUnit"] == "s"
        assert header["dxUnit"] == "m"
        assert header["roiTable"] == [{"roiStart": 0, "roiEnd": 9, "roiDec": 1}]
        header = ZMQPublisher._get_header(da_int16)
        assert header["dataType"] == "short"

    def test_init_conect_set_header(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        pub.submit(da_float32)
        assert pub.header == ZMQPublisher._get_header(da_float32)

    def test_send_header(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        pub.submit(da_float32)
        socket = self.get_socket(address)
        pub.submit(da_float32)  # a packet must be sent once subscriber is connected
        assert socket.recv() == json.dumps(pub.header).encode("utf-8")

    def test_send_data(self):
        address = get_free_local_address()
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
        address = get_free_local_address()
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
        address = get_free_local_address()
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
        address = get_free_local_address()
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
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = [da_float32]
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        assert sub.address == address
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
        address = get_free_local_address()
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
        address = get_free_local_address()
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

    def test_change_header(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 5)
        chunks = [chunk.isel(distance=slice(0, 5)) for chunk in chunks[:2]] + chunks[2:]
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        for chunk in chunks:
            result = next(sub)
            assert result.equals(chunk)
            
    def test_update_header(self):
        print("test_up")
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = [da_float32]
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        message = b'{\n    "bytesPerPackage": 64008,\n    "dataScale": 1,\n    "dataType": "float",\n    "dt": 0.01,\n    "dtUnit": "s",\n    "dx": 10.213001907746815,\n    "dxUnit": "m",\n    "experiment": "monaco-das-lig2024",\n    "gaugeLength": 20.42600381549363,\n    "gaugeLengthUnit": "m",\n    "instrument": "fsic036.fsi.lan",\n    "measurement": "monaco-longterm2025",\n    "measurementStartTime": "2025-07-08T12:08:31.709Z",\n    "muxPositions": [\n        {\n            "rx": 0,\n            "tx": 0\n        }\n    ],\n    "nChannels": 16002,\n    "nPackagesPerMessage": 10,\n    "roiTable": [\n        {\n            "roiDec": 10,\n            "roiEnd": 160010,\n            "roiStart": 0\n        }\n    ],\n    "sensitivities": [\n        {\n            "factor": 9112677.961649183,\n            "unit": "rad/(strain*m)"\n        }\n    ],\n    "sensorType": "D",\n    "spatialUnwrapRange": 615.21435546875,\n    "sweepLength": 0.0001,\n    "sweepLengthUnit": "s",\n    "switchChannel": 0,\n    "triggeredMeasurement": false,\n    "trustedTimeSource": false,\n    "unit": "rad/(s*m)",\n    "version": 2\n}\n'
        sub._update_header(message)
        assert sub.shape == (10,16002)

    def test_roiDec(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = [da_float32]
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        message = b'{\n    "bytesPerPackage": 64008,\n    "dataScale": 1,\n    "dataType": "float",\n    "dt": 0.01,\n    "dtUnit": "s",\n    "dx": 10,\n    "dxUnit": "m",\n    "experiment": "monaco-das-lig2024",\n    "gaugeLength": 20,\n    "gaugeLengthUnit": "m",\n    "instrument": "fsic036.fsi.lan",\n    "measurement": "monaco-longterm2025",\n    "measurementStartTime": "2025-07-08T12:08:31.709Z",\n    "muxPositions": [\n        {\n            "rx": 0,\n            "tx": 0\n        }\n    ],\n    "nChannels": 91,\n    "nPackagesPerMessage": 10,\n    "roiTable": [\n        {\n            "roiDec": 10,\n            "roiEnd": 90,\n            "roiStart": 0\n        }\n    ],\n    "sensitivities": [\n        {\n            "factor": 9112677.961649183,\n            "unit": "rad/(strain*m)"\n        }\n    ],\n    "sensorType": "D",\n    "spatialUnwrapRange": 615.21435546875,\n    "sweepLength": 0.0001,\n    "sweepLengthUnit": "s",\n    "switchChannel": 0,\n    "triggeredMeasurement": false,\n    "trustedTimeSource": false,\n    "unit": "rad/(s*m)",\n    "version": 2\n}\n'
        print(message)
        sub._update_header(message)
        print(sub.distance)
        assert sub.distance == {"tie_indices": [0, 9], "tie_values": [0, 90]}
    
    def test_iter(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 5)
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        sub = (chunk for _, chunk in zip(range(5), sub))
        result = xd.concatenate([chunk for chunk in sub])
        assert result.equals(da_float32)

    def publish(self, pub, chunks):
        for chunk in chunks:
            time.sleep(0.001)
            pub.submit(chunk)
