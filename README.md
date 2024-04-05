# xdas

[![Documentation Status](https://readthedocs.org/projects/xdas/badge/?version=latest)](https://xdas.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/xdas)](https://pypi.org/project/xdas/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/560867006.svg)](https://zenodo.org/badge/latestdoi/560867006)

xdas is a python library using the xarray API that allows to work with DAS data 
(Distributed Acoustic Sensing).

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
