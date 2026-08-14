"""
Chunked processing pipeline for larger-than-memory datasets.

Provides loaders, writers, real-time streaming, the :func:`process`
orchestrator and its :func:`get_source` / :func:`get_writer` dispatch.
"""

__all__ = [
    "DataArrayLoader",
    "DataArrayWriter",
    "DataFrameWriter",
    "RealTimeLoader",
    "ResultWriter",
    "StreamWriter",
    "ZMQPublisher",
    "ZMQSubscriber",
    "get_source",
    "get_writer",
    "process",
    "watch",
]

from .core import (
    DataArrayLoader,
    DataArrayWriter,
    DataFrameWriter,
    RealTimeLoader,
    ResultWriter,
    StreamWriter,
    ZMQPublisher,
    ZMQSubscriber,
    get_source,
    get_writer,
    process,
    watch,
)
