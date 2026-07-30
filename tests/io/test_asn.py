import json
import threading
import time

import h5py
import numpy as np
import pytest
import zmq

import xdas as xd
from xdas.io.asn import ZMQPublisher, ZMQSubscriber


def get_free_local_address():
    port = xd.io.get_free_port()
    return f"tcp://localhost:{port}"


da_float32 = xd.testing.dummy(shape=(100, 10), step=(0.1, 10.0), dtype="float32")
da_int16 = xd.testing.dummy(shape=(100, 10), step=(0.1, 10.0), dtype="int16")


class TestASNEngineROIBounds:
    def test_roi_start_beyond_sensor_distances(self):
        from xdas.io.asn import ASNEngine

        engine = ASNEngine()
        with pytest.raises(IndexError, match="ROI start lies beyond"):
            engine._get_roi_bound_indices(
                [0.0, 10.0, 20.0], n_start=5, n_end=3, dx=10.0
            )

    def test_roi_end_before_sensor_distances(self):
        from xdas.io.asn import ASNEngine

        engine = ASNEngine()
        with pytest.raises(IndexError, match="ROI end lies before"):
            engine._get_roi_bound_indices(
                [10.0, 20.0, 30.0], n_start=1, n_end=0, dx=10.0
            )


class TestASNEnginePublisher:
    def test_write_method(self):
        from xdas.io.asn import ZMQPublisher as ASNZMQPublisher

        address = get_free_local_address()
        pub = ASNZMQPublisher(address)
        pub.write(da_float32)


class TestASNEngine:
    def test_read_handles_exclusive_roi_end(self, tmp_path):
        path = tmp_path / "sample_asn.hdf5"
        with h5py.File(path, "w") as file:
            header = file.create_group("header")
            header["time"] = 0.0
            header["dt"] = 0.1
            header["dx"] = 10.0

            file.create_dataset("data", data=np.zeros((4, 4), dtype=np.float32))

            cable_spec = file.create_group("cableSpec")
            cable_spec["sensorDistances"] = np.array([0.0, 10.0, 20.0, 30.0])

            demod_spec = file.create_group("demodSpec")
            demod_spec["roiStart"] = np.array([0])
            demod_spec["roiEnd"] = np.array([4])

        da = xd.open_dataarray(path, engine="asn")

        assert da.shape == (4, 4)
        assert da["distance"][0].values == 0.0
        assert da["distance"][-1].values == 30.0
        # A uniform sensor grid is declared regular, like every other engine.
        assert da["distance"].isregular()
        assert da["distance"].get_sampling_interval() == 10.0

    @staticmethod
    def write_rois(path, rois, dx, with_dec=True):
        """Write an ASN file from ``(n_start, n_channels, dec)`` ROIs.

        Positions are given on the pre-decimation grid, as ASN stores them.
        """
        dists, roi_start, roi_end, decs = [], [], [], []
        for n_start, n_channels, dec in rois:
            channels = n_start + np.arange(n_channels) * dec
            dists += list(channels * dx)
            roi_start.append(n_start)
            roi_end.append(int(channels[-1]))
            decs.append(dec)
        with h5py.File(path, "w") as file:
            header = file.create_group("header")
            header["time"] = 0.0
            header["dt"] = 0.1
            header["dx"] = dx
            file.create_dataset(
                "data", data=np.zeros((4, len(dists)), dtype=np.float32)
            )
            cable_spec = file.create_group("cableSpec")
            cable_spec["sensorDistances"] = np.array(dists, dtype="float64")
            demod_spec = file.create_group("demodSpec")
            demod_spec["roiStart"] = np.array(roi_start, dtype="uint32")
            demod_spec["roiEnd"] = np.array(roi_end, dtype="uint32")
            if with_dec:
                demod_spec["roiDec"] = np.array(decs, dtype="uint32")
        return len(dists)

    def test_read_declares_metadata_spacing_across_rois(self, tmp_path):
        # ROIs sharing a decimation must keep one spacing whatever their
        # lengths: taking it from `dx * roiDec` keeps it bit-identical, while
        # re-deriving it from each ROI's bounds differs in the last ulp and
        # would drop the axis to irregular.
        path = tmp_path / "same_dec.hdf5"
        dx = 1.0213001907746815
        size = self.write_rois(path, [(0, 997, 15), (30000, 2003, 15)], dx)

        da = xd.open_dataarray(path, engine="asn")

        assert da.sizes["distance"] == size
        assert da["distance"].isregular()
        assert da["distance"].get_sampling_interval() == dx * 15

    def test_read_keeps_differently_decimated_rois_irregular(self, tmp_path):
        path = tmp_path / "mixed_dec.hdf5"
        dx = 1.0213001907746815
        self.write_rois(path, [(0, 500, 15), (30000, 500, 30)], dx)

        da = xd.open_dataarray(path, engine="asn")

        assert not da["distance"].isregular()
        assert da["distance"].get_sampling_interval() is None

    def test_read_regularizes_rois_differing_only_by_rounding(self, tmp_path):
        # Without roiDec the spacing is derived from each ROI's bounds, which
        # rounds differently per ROI. Those steps describe the same grid, so the
        # axis must stay regular rather than trip concatenation's exact match.
        path = tmp_path / "rounding.hdf5"
        dx = 1.0213001907746815
        self.write_rois(
            path,
            [(0, 997, 15), (30000, 2003, 15), (90000, 631, 15)],
            dx,
            with_dec=False,
        )

        da = xd.open_dataarray(path, engine="asn")

        assert da["distance"].isregular()
        assert da["distance"].get_sampling_interval() == pytest.approx(dx * 15)

    def test_read_single_channel_roi_without_dec(self, tmp_path):
        # No roiDec and a single channel: no spacing can be derived from the
        # bounds, so fall back to the raw channel spacing.
        path = tmp_path / "single_channel.hdf5"
        self.write_rois(path, [(0, 1, 15)], 10.0, with_dec=False)

        da = xd.open_dataarray(path, engine="asn")

        assert da.sizes["distance"] == 1
        assert da["distance"].get_sampling_interval() == 10.0

    def test_read_keeps_unevenly_decimated_rois_irregular(self, tmp_path):
        # Two ROIs decimated differently admit no single channel spacing.
        path = tmp_path / "two_roi_asn.hdf5"
        with h5py.File(path, "w") as file:
            header = file.create_group("header")
            header["time"] = 0.0
            header["dt"] = 0.1
            header["dx"] = 1.0

            file.create_dataset("data", data=np.zeros((4, 6), dtype=np.float32))

            cable_spec = file.create_group("cableSpec")
            cable_spec["sensorDistances"] = np.array(
                [0.0, 10.0, 20.0, 100.0, 130.0, 160.0]
            )

            demod_spec = file.create_group("demodSpec")
            demod_spec["roiStart"] = np.array([0, 100])
            demod_spec["roiEnd"] = np.array([20, 160])

        da = xd.open_dataarray(path, engine="asn")

        assert not da["distance"].isregular()
        assert da["distance"].get_sampling_interval() is None


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
        time.sleep(0.01)
        assert pub.header == ZMQPublisher._get_header(da_float32)

    def test_send_header(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        pub.submit(da_float32)
        time.sleep(0.01)
        socket = self.get_socket(address)
        pub.submit(da_float32)  # a packet must be sent once subscriber is connected
        time.sleep(0.01)
        assert socket.recv() == json.dumps(pub.header).encode("utf-8")

    def test_send_data(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        pub.submit(da_float32)
        time.sleep(0.01)
        socket = self.get_socket(address)
        pub.submit(da_float32)  # a packet must be sent once subscriber is connected
        time.sleep(0.01)
        socket.recv()  # header
        message = socket.recv()
        assert message[:8] == da_float32["time"][0].values.astype("M8[ns]").tobytes()
        assert message[8:] == da_float32.data.tobytes()
        pub.submit(da_int16)
        time.sleep(0.01)
        socket.recv()  # header
        message = socket.recv()
        assert message[:8] == da_int16["time"][0].values.astype("M8[ns]").tobytes()
        assert message[8:] == da_int16.data.tobytes()

    def test_send_chunks(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 10)
        pub.submit(chunks[0])
        time.sleep(0.01)
        socket = self.get_socket(address)
        for chunk in chunks[1:]:
            pub.submit(chunk)
            time.sleep(0.01)
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
        time.sleep(0.01)
        socket1 = self.get_socket(address)
        for chunk in chunks[1:5]:
            pub.submit(chunk)
            time.sleep(0.01)
        socket2 = self.get_socket(address)
        for chunk in chunks[5:]:
            pub.submit(chunk)
            time.sleep(0.01)
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
        time.sleep(0.01)
        socket = self.get_socket(address)
        for chunk in chunks[1:5]:
            pub.submit(chunk)
            header1 = pub.header
            time.sleep(0.01)
        for chunk in chunks[5:]:
            pub.submit(chunk.isel(distance=slice(0, 5)))
            time.sleep(0.01)
            header2 = pub.header
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
        time.sleep(0.01)
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
        assert sub.distance == {
            "tie_indices": [0, 9],
            "tie_values": [0.0, 90.0],
            "sampling_interval": 10.0,
        }
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
        assert sub.distance == {
            "tie_indices": [0, 9],
            "tie_values": [0.0, 90.0],
            "sampling_interval": 10.0,
        }
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

    def test_roiDec(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = [da_float32]
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        message = (
            b"{\n"
            b'    "bytesPerPackage": 64008,\n'
            b'    "dataScale": 1,\n'
            b'    "dataType": "float",\n'
            b'    "dt": 0.01,\n'
            b'    "dtUnit": "s",\n'
            b'    "dx": 10.213001907746815,\n'
            b'    "dxUnit": "m",\n'
            b'    "experiment": "monaco-das-lig2024",\n'
            b'    "gaugeLength": 20.42600381549363,\n'
            b'    "gaugeLengthUnit": "m",\n'
            b'    "instrument": "fsic036.fsi.lan",\n'
            b'    "measurement": "monaco-longterm2025",\n'
            b'    "measurementStartTime": "2025-07-08T12:08:31.709Z",\n'
            b'    "muxPositions": [\n'
            b"        {\n"
            b'            "rx": 0,\n'
            b'            "tx": 0\n'
            b"        }\n"
            b"    ],\n"
            b'    "nChannels": 16002,\n'
            b'    "nPackagesPerMessage": 10,\n'
            b'    "roiTable": [\n'
            b"        {\n"
            b'            "roiDec": 10,\n'
            b'            "roiEnd": 160010,\n'
            b'            "roiStart": 0\n'
            b"        }\n"
            b"    ],\n"
            b'    "sensitivities": [\n'
            b"        {\n"
            b'            "factor": 9112677.961649183,\n'
            b'            "unit": "rad/(strain*m)"\n'
            b"        }\n"
            b"    ],\n"
            b'    "sensorType": "D",\n'
            b'    "spatialUnwrapRange": 615.21435546875,\n'
            b'    "sweepLength": 0.0001,\n'
            b'    "sweepLengthUnit": "s",\n'
            b'    "switchChannel": 0,\n'
            b'    "triggeredMeasurement": false,\n'
            b'    "trustedTimeSource": false,\n'
            b'    "unit": "rad/(s*m)",\n'
            b'    "version": 2\n'
            b"}\n"
        )
        sub._update_header(message)
        assert sub.shape == (10, 16002)
        assert sub.distance == {
            "tie_indices": [0, 16001],
            "tie_values": [0.0, 163418.2435258568],
            "sampling_interval": 163418.2435258568 / 16001,
        }

    def test_iter(self):
        address = get_free_local_address()
        pub = ZMQPublisher(address)
        chunks = xd.split(da_float32, 5)
        threading.Thread(target=self.publish, args=(pub, chunks)).start()
        sub = ZMQSubscriber(address)
        sub = (chunk for _, chunk in zip(range(5), sub))
        result = xd.concat(list(sub))
        assert result.equals(da_float32)

    def publish(self, pub, chunks):
        time.sleep(0.01)
        for chunk in chunks:
            pub.submit(chunk)
            time.sleep(0.01)
