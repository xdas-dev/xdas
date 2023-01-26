# xdas

xdas is a python library built around xarray that allows to work with DAS data 
(Distributed Acoustic Sensing).

## Installation

First install the dependencies:

    conda install dask h5py netcdf4 numpy scipy xarray

Then install ```xdas```:

    git clone https://github.com/xdas-dev/xdas.git
    (cd xdas && pip install -e .)

## Update

Pulling the change from github suffices:

    (cd xdas && git pull)
    (cd icoords && git pull)

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
from xdas.database import Database
database.to_netcdf("path.nc")
database = Database.from_netcdf("path.nc")
```
