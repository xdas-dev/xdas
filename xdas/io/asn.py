from glob import glob

import h5py
import numpy as np

from ..core import Coordinate


def build_database(fname, paths, dgap=None):
    if isinstance(paths, str):
        paths = glob(paths)
    nsamples, nchannels, dt, dx = get_shared_metadata(paths[0])
    metadata = get_unique_metadata(paths)
    time = get_time(metadata, nsamples, dt, dgap)
    distance = get_distance(nchannels, dx)
    data = get_data(nchannels, nsamples, metadata)
    to_hdf(fname, time, distance, data)


def to_hdf(fname, time, distance, data):
    with h5py.File(fname, "w") as file:
        file.create_virtual_dataset("data", data, fillvalue=np.nan)
        file.create_dataset("time_tie_indices", data=time.tie_indices)
        file.create_dataset("time_tie_values", data=time.tie_values.astype("int"))
        file.create_dataset("distance_tie_indices", data=distance.tie_indices)
        file.create_dataset("distance_tie_values", data=distance.tie_values)


def get_data(nchannels, nsamples, metadata):
    layout = h5py.VirtualLayout(shape=(nsamples * len(metadata), nchannels), dtype="f4")
    for k, meta in enumerate(metadata):
        vsource = h5py.VirtualSource(
            meta["path"], "data", shape=(nsamples, nchannels), dtype="f4"
        )
        distance = k * nsamples
        layout[distance : distance + nsamples] = vsource
    return layout


def get_distance(nchannels, dx):
    tie_indices = [0, nchannels - 1]
    tie_values = [0, dx * nchannels]
    return Coordinate(tie_indices, tie_values)


def get_time(metadata, nsamples, dt, dgap):
    tie_indices = [0]
    for meta in metadata[:-1]:
        tie_indices.append(tie_indices[-1] + nsamples)
    tie_values = [meta["time"] for meta in metadata]
    time = Coordinate(tie_indices, tie_values)
    return correct_time(time, nsamples, dt, dgap)


def correct_time(time, nsamples, dt, dgap):
    tie_indices = []
    tie_values = []
    nfile = len(time.tie_indices)
    for k in range(nfile):
        tie_indices.append(time.tie_indices[k])
        tie_values.append(time.tie_values[k])
        nsample = nsamples
        flen = dt * nsample
        if k == nfile - 1 or time.tie_values[k + 1] - time.tie_values[k] - flen > dgap:
            tie_indices.append(time.tie_indices[k] + nsample - 1)
            tie_values.append(time.tie_values[k] + (nsample - 1) * dt)
    time = Coordinate(tie_indices, tie_values)
    time.simplify(dgap)
    return time


def get_unique_metadata(paths):
    metadata = []
    for path in paths:
        with h5py.File(path, "r") as file:
            metadata.append(
                {
                    "path": path,
                    "time": np.datetime64(round(file["/header/time"][()] * 1e6), "us"),
                }
            )
    metadata = sorted(metadata, key=lambda x: x["time"])
    return metadata


def get_shared_metadata(path):
    with h5py.File(path, "r") as file:
        header = file["header"]
        nsamples = header["nSamples"][()]
        nchannels = header["nChannels"][()]
        dt = np.timedelta64(round(1e6 * header["dt"][()]), "us")
        dx = header["dx"][()] * np.median(np.diff(header["channels"]))
    return nsamples, nchannels, dt, dx
