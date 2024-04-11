# xdas

[![Documentation Status](https://readthedocs.org/projects/xdas/badge/?version=latest)](https://xdas.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/xdas)](https://pypi.org/project/xdas/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/560867006.svg)](https://zenodo.org/badge/latestdoi/560867006)

*Xdas* is an open source Python library for managing, processing and visualizing **Distributed Acoustic Sensing (DAS)** data. It transforms any DAS format into a self-contained data that encapsulates both the data and coordinate metadata.

**Main features**:

- Composability of operations: build your own data processing pipeline with a mix of xdas, NumPy/SciPy, and custom routines.
- Numerous multi-threaded signal processing routines and tools to manage complex datasets, including I/O latency hiding.
- Convenient data selection across multiple files, with automatic interpolation of small data gaps
- Parsers for common file formats (NetCDF4,HDF5) and DAS manufacturers (ASN, Febus, OptaSense, Sintela).
- Extendability to other dense and heavy N-dimenional arrays such as large-N seismic arrays.

It uses the
[xarray](https://xarray.dev) API to work with labelled multi-dimensional arrays. 

## Installation

    pip install xdas

To install from a local repository:

    pip install -e .

To install from a local repository, including optional dependencies:

    pip install -e .[dev,docs,test]

## Documentation

The documentation for the stable and latest branches are available at: [https://xdas.readthedocs.io](https://xdas.readthedocs.io)

To compile the documentation locally:

    cd docs/ && make html

The compiled HTML documentation is then found in `docs/_build/html`

## Testing xdas

    cd tests/ && pytest
