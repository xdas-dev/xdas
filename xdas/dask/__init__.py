"""
Dask integration helpers for serializing and deserializing dask arrays
inside xdas HDF5 files.
"""

from .core import create_variable, dumps, loads
