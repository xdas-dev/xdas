"""
Dask integration helpers for serializing and deserializing dask arrays
inside xdas HDF5 files.
"""

__all__ = ["create_variable", "dumps", "loads"]

from .core import create_variable, dumps, loads
