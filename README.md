# xdas

[![Documentation Status](https://readthedocs.org/projects/xdas/badge/?version=latest)](https://xdas.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/xdas)](https://pypi.org/project/xdas/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/560867006.svg)](https://zenodo.org/badge/latestdoi/560867006)

xdas is a python library built around xarray that allows to work with DAS data 
(Distributed Acoustic Sensing).

## Installation

You can use ```pip``` to install xdas:

    pip install xdas


## Usage

Here how to read a febus file with some decimation:

```python
from xdas.io.febus import read
fname = "path_to_febus_file.h5"
database = read(fname, decimation=10)
```
The file will load coordinates as a Database wich has similar API than a xarray DataArray but with a very limited number of implemented methods. To get a classic xarray DataArray object:

```python
xarr = database.to_xarray()
```

To write and read the file using CF conventions:

```python
import xdas
database.to_netcdf("path.nc")
database = xdas.open_database("path.nc")
```
