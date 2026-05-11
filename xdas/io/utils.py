import h5py
import hdf5plugin

def copy_except(src_file, dst_file, exclude_path):
    """
    Copy the contents of src_file to dst_file, excluding the object at exclude_path.

    Parameters
    ----------
    src_file : h5py.File
        The source HDF5 file to copy from.
    dst_file : h5py.File
        The destination HDF5 file to copy to.
    exclude_path : str
        The path of the object to exclude from copying, should be relative to the root of the HDF5 file and start with a "/".
    """
    exclude_path = "/" + exclude_path.lstrip("/")

    def _copy(src_group, dst_group, current_path):
        dst_group.attrs.update(src_group.attrs)
        for name, obj in src_group.items():
            obj_path = current_path.rstrip("/") + "/" + name
            if obj_path == exclude_path:
                continue
            if isinstance(obj, h5py.Group):
                grp = dst_group.create_group(name)
                _copy(obj, grp, obj_path)
            else:
                src_group.file.copy(obj, dst_group, name=name)

    _copy(src_file, dst_file, "/")

def compress(file_path:str, 
             save_path:str,
             dataset_location:str,
             encoding:dict):
    """
    Compress a specific dataset and save it in a new HDF5 file while preserving the rest of the original file structure and metadata.

    Parameters
    ----------
    file_path : str
        Path to the original .hdf5 file.
    save_path : str
        Path to save the compressed .hdf5 file.
    dataset_location : str
        The location of the dataset in the HDF5 file to compress.
    encoding : dict
        Dictionary of encoding options for the dataset.
        Should contain the following keys:
        - 'compression': the compression algorithm to use and its parameters, part of the hdf5plugin library
        - 'chunks': the chunk size for the dataset, should be a tuple of integers, default to False for no chunking                 
    """
    with h5py.File(file_path, 'r') as f_orig, h5py.File(save_path, 'w') as f_compressed:

        if not encoding["chunks"]:
            encoding.pop("chunks", "data")

        copy_except(f_orig, f_compressed, dataset_location)

        data = f_orig[dataset_location][()]
        f_compressed.create_dataset(dataset_location, data=data, **encoding)