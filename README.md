<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/_static/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/_static/logo-light.png">
    <img alt="Xdas Logo" height="250px">
</picture>
</div>

-----------------

[![Documentation Status](https://readthedocs.org/projects/xdas/badge/?version=latest)](https://xdas.readthedocs.io/en/latest/?badge=latest)
[![Tests Status](https://github.com/xdas-dev/xdas/actions/workflows/tests.yaml/badge.svg)](https://github.com/xdas-dev/xdas/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/xdas-dev/xdas/graph/badge.svg?token=00MD52JRA3)](https://codecov.io/gh/xdas-dev/xdas)
[![PyPI](https://img.shields.io/pypi/v/xdas)](https://pypi.org/project/xdas/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/560867006.svg)](https://zenodo.org/badge/latestdoi/560867006)

# Xdas: a Python Framework for Distributed Acoustic Sensing

*Xdas* is an python library for managing, processing and visualizing **Distributed Acoustic Sensing (DAS)** data. It reads any DAS format into self-described python abstractions that encapsulates both the data and the metadata (coordinates and attributes). Xdas reuses concepts of labeled N-dimensional arrays developped by the [Xarray](https://xarray.dev) library. It takes inspiration from [Dask](https://www.dask.org/) in term of lazy computing.

## Key Features

- Seamless manipulation of large multi-file datasets in their native DAS-specific format.
- Signal Processing: Multi-threaded implementations of common routines.
- Extensibility: build your pipeline with a mix of Xdas, NumPy/SciPy, and/or your own custom routines. 
- Larger-than-memory processing: apply your piplines with optimized I/O latencies.

Xdas can also be used in other context than DAS, for example in the context of other dense and heavy N-dimenional arrays such as large-N seismic arrays.

## Installation

Xdas can be fetched from [PyPI](https://pypi.org/project/xdas):

    pip install xdas

## Documentation

The documentation is available at: [https://xdas.readthedocs.io](https://xdas.readthedocs.io).

## Tutorials

A comprehensive series of tutorials can be found at: [https://github.com/xdas-dev/tutorials](https://github.com/xdas-dev/tutorials).

## Contributing

You can find information about contributing to Xdas in our [Contributing Guide](https://xdas.readthedocs.io/en/latest/contribute.html).

## Get in touch

- Ask usage questions and discuss any ideas on [GitHub Discussions](https://github.com/xdas-dev/xdas/discussions).
- Report bugs, suggest features or view the source code on [GitHub](https://github.com/xdas-dev/xdas).
- To follow the main announcements such as online trainning sessions please register to our [Newsletter](https://groups.google.com/g/xdas).

## Citing

If you use Xdas for your DAS data processing, please consider [citing the project](https://xdas.readthedocs.io/en/latest/cite.html).
