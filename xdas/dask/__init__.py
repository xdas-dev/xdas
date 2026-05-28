"""
Dask integration helpers for xdas HDF5 files.

Serializes and deserializes dask arrays inside xdas HDF5 files.
"""

__all__ = ["create_variable", "dumps", "loads"]

from .core import create_variable, dumps, loads
