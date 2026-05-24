"""
HDF5 utility functions for compressing datasets while preserving file
structure and metadata.
"""

import h5py
import hdf5plugin


def compress(src_path: str, dst_path: str, dataset_location: str, encoding: dict):
    """
    Compress a specific dataset in an HDF5 file while preserving the rest of the file structure and metadata.

    Parameters
    ----------
    src_path : str
        Path to the original .hdf5 file.
    dst_path : str
        Path to save the compressed .hdf5 file.
    dataset_location : str
        Path to the dataset to compress inside the HDF5 file.
    encoding : dict
        Dictionary of encoding options for the dataset.
        Should contain the following keys:
        - 'compression': the compression algorithm to use and its parameters, part of the hdf5plugin library
        - 'chunks': the chunk size for the dataset, should be a tuple of integers, default to False for no chunking
    """

    if "chunks" in encoding.keys() and not encoding["chunks"]:
        encoding.pop("chunks")

    with h5py.File(src_path, "r") as src_file, h5py.File(dst_path, "w") as dst_file:

        dataset_name = "/" + dataset_location.lstrip("/")

        def _copy(src_group, dst_group, current_path):
            # Copy group attributes
            dst_group.attrs.update(src_group.attrs)
            for name, obj in src_group.items():
                obj_path = current_path.rstrip("/") + "/" + name
                # Compress the chosen dataset
                if obj_path == dataset_name:
                    data = src_file[dataset_name][()]
                    ds = dst_group.create_dataset(name, data=data, **encoding)
                    for key, val in src_file[dataset_name].attrs.items():
                        ds.attrs[key] = val
                # Copy the group
                elif isinstance(obj, h5py.Group):
                    grp = dst_group.create_group(name)
                    _copy(obj, grp, obj_path)
                # Copy the rest
                else:
                    src_group.file.copy(obj, dst_group, name=name)

        _copy(src_file, dst_file, "/")
