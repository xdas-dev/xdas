"""
Chunked processing pipeline: loaders, writers, real-time streaming, and
the :func:`process` orchestrator for larger-than-memory datasets.
"""

from .core import (
    DataArrayLoader,
    DataArrayWriter,
    DataFrameWriter,
    RealTimeLoader,
    StreamWriter,
    ZMQPublisher,
    ZMQSubscriber,
    process,
)
