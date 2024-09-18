---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
os.chdir("_data")
```

# Getting Started   

## Welcome to Xdas!

 *Xdas* is an open-source Python library for working with huge labeled N-dimensional arrays as used in Distributed Acoustic Sensing (DAS). The *Xdas* API is heavily inspired by the [*Xarray*](https://xarray.dev) project. It implements a subset of the Xarray functionality and extends it with features that allow to deal with large multi-file netCDF4/HDF5 datasets, usually with a very long dimension (usually time). It provides the classic signal processing tools to handle time-series that do not fit in memory. It also provides I/O capabilities with most DAS formats.

## Installing xdas

Xdas is a pure python package. It can easily be installed with `pip` from [PyPI](https://pypi.org/project/xdas):

`````{tab-set}
````{tab-item} Stable
```bash
pip install xdas
```
````
````{tab-item} Latest
```bash
pip install "git+https://github.com/xdas-dev/xdas.git@dev" --force-reinstall
```

````
`````

Xdas must first be imported along with other useful libraries:

```{code-cell}
import numpy as np
import xdas 
```

## Dataset virtual consolidation

Most instruments usually produces datasets made out of a multitude of files, each one containing a temporal chunk of the full acquisition. In Xdas you can virtually concatenate all those files to create a virtual dataset that allows to seamlessly access the entire dataset as if it was a unique file.

### Linking multiple files 

If you are considering a unique acquisition you can use {py:func}`~xdas.open_mfdataarray`. You can either pass a list of paths or a path pattern containing wildcards to specify which files must be linked together. The `engine` keyword indicates the format of the data. Xdas support a variety of DAS formats and it is easy to add support to any custom or missing format. See the [](user-guide/data-formats) section for more information. 

In the example here, we have three files of interest in the current working directory:

```{code-cell}
ls 00*.h5
```

We can link them like this:

```{code-cell}
da = xdas.open_mfdataarray("00*.h5", engine=None)
da
```

Xdas only loads the metadata from each file and returns a {py:class}`~xdas.DataArray` object. This object has mainly two attributes. First a `data` attribute that contain the data. Here a {py:class}`~xdas.VirtualStack` object that is a pointer to the different files we opened. Second, a `coords` attribute that contains the metadata related to how the space and the time are sampled. Here both dimensions are labeled using {py:class}`~xdas.InterpCoordinate` objects. Those allow to concisely store the time and space information, including potential gaps and overlaps. See the [](user-guide/interpolated-coordinates) section for more information. 

Note that if you want to create a single data collection object for multiple acquisitions (i.e. different instruments or several acquisition with different parameters), you can use the [DataCollection](user-guide/data-structure/datacollection) structure.  

```{note}
For Febus users, converting native files into Xdas NetCDF format generally improves I/O operations and reduce the amount of data by a factor two. This can be done by looping over Febus files and running: `xdas.open_dataarray("path_to_febus_file.h5", engine="febus").to_netcdf("path_to_xdas_file.nc", virtual=False)`. The converted files can then be linked as described above.
```

### Fixing small gaps and overlaps

If you do not have GPS synchronization during your DAS acquisition, you may have gaps or overlaps between files. With Xdas, you can define a tolerance to what extent you accept to shift the time of some data blocks to fix overlaps along the time dimension. In the case you have overlaps in time you may have errors when slicing the DataArray. 

```{code-cell} 
tolerance = np.timedelta64(30, "ms")  # usually enough for NTP synchronized experiments
da["time"] = da["time"].simplify(tolerance)
```
More important overlaps will need a manual intervention. Big gaps are not problematic as they do to break the bijection between time indices and values.

### Saving virtual dataset to disk

Once you are happy with your consolidated dataset, you can write it to disk using the Xdas NetCDF format:

```{code-cell} 
da.to_netcdf("da.nc", virtual=True)  # Xdas tries to write data virtually by default
```
Once this is done you and your collaborators will simply need to open that master file to access the whole dataset.

```{warning}
The created file only contains pointers to your data. If you move your data somewhere else your consolidated file will be broken. If this happens it will return only `numpy.nan` values.
```

## Exploration

Now that your dataset is ready to use, let's explore it!

### Read the virtual DataArray

The consolidated virtual dataset can be fetched as if it was a regular file:

```{code-cell} 
da = xdas.open_dataarray("da.nc")
da
```

Usually the amount of data linked in such a file is too big to be loaded into memory. When exploring a dataset, a common practice is to first make a selection of a small part of interest and then to load it into memory.

### Select the region of interest

Data arrays can be sliced using a label-based selection meaning that instead of providing indices we can slice the data by coordinates values:

```{code-cell}
da = da.sel(
    time=slice("2023-01-01T00:00:01", "2023-01-01T00:00:05"),
    distance=slice(1000, 9000),
)
da
```

### Load the data in memory

At this point we consider that the selection is small enough to be loaded into memory:

```{code-cell}
da = da.load()  # optional
da
```

### Write the data to disk with compression

In some case, it can be interesting to write down to disk the data with some compression. In this example, we use the Zfp compression from the *hdf5plugin* library which is a lossy compression that is particularly suited for floating point numbers. The recommended compression scheme is the *fixed accuracy mode* which ensure that your data is not altered by the compression above that threshold in absolute value. Be careful to choose a value which is much lower than your instrumental noise. Compression ratio of around 3-4 can usually be achieved in such a way. For big files, compressing by chunks can be useful to enhance slicing through the data (otherwise the entire data must be decompressed each time some part must be accessed).

```{code-cell}
import hdf5plugin
encoding = {"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=1e-6)}

da.to_netcdf("chunked_and_compressed.nc", encoding=encoding)
```

Reading compressed data is completely transparent, you do not need to specify anything.

```{code-cell}
xdas.open_dataarray("chunked_and_compressed.nc")
```

Note that you do not necessarily need to load it manually. Any step that requires to modify the data will automatically trigger the data importation.

### Visualization

Because DataArray objects are self-described (they encapsulate both the data an its related metadata), plotting your DataArray is a one line job:

```{code-cell}
da.plot(yincrease=False, vmin=-0.5, vmax=0.5)
```


## Signal processing

DataArray can be processed without having to extract the underlying N-dimensional array. Most numpy functions can be applied while preserving metadata. Xdas also wraps a large subset of [numpy](https://numpy.org/) and [scipy](https://scipy.org/) function by adding coordinates handling. You mainly need to replace `axis` arguments by `dim` ones and to provides dimensions by name and not by position.


### Numpy functions

You can apply most numpy functions to a data array. Xdas also have its own implementations that work by labels:

```{code-cell}
squared = np.square(da)
mean = xdas.mean(da, "time")
std = da.std("distance")
```

### Arithmetics 

You can manipulate data arrays objects as regular arrays, Xdas will check that dimensions and coordinates are consistent. 

```{code-cell}
squared = da * da
common_mode_removal = da - da.mean("distance")
```

### Scipy functions

Most scipy function from the `signal` and `fft` submodule have been implemented. The Xdas function are multithreaded. A `parallel` keyword argument can be passed to most of them to indicate the number of cores to use.

Bellow an example of spatial and temporal decimation:

```{code-cell}
import xdas.signal as xs 

decimated = xs.decimate(da, 2, ftype="fir", dim="distance", parallel=None)  # all cores by default
decimated = xs.decimate(decimated, 2, ftype="iir", dim="time", parallel=8)  # height cores

decimated.plot(yincrease=False, vmin=-0.25, vmax=0.25)
```

Here how to compute a FK diagram. Note that the DataArray object can be used to represent any number and kind of dimensions:

```{code-cell}
import xdas.fft as xfft

fk = xs.taper(da, dim="distance")
fk = xs.taper(fk, dim="time")
fk = xfft.rfft(fk, dim={"time": "frequency"})  # rename "time" -> "frequency"
fk = xfft.fft(fk, dim={"distance": "wavenumber"}) # rename "distance" -> "wavenumber"
fk = 20 * np.log10(np.abs(fk))
fk.plot(xlim=(-0.004, 0.004), vmin=-30, vmax=30, interpolation="antialiased")
```

### Saving results

Processed data can be saved to NetCDF. This time, because the data was changed, the data must be entirely written to disk. 

```{code-cell}
fk.to_netcdf("fk.nc")
```


## Massive processing using Atoms

The usual [numpy](https://numpy.org/)/[scipy](https://scipy.org/) way of processing data works great when the data of interest fit in memory. To deal with huge datasets, xdas introduce {py:class}`~xdas.atoms.Atom` objects. 

An {py:class}`~xdas.atoms.Atom` is a generic processing unit that takes one input and return one output. Atoms can store state information to ensure continuity from subsequent calls on contiguous chunks.
There are three ways to make atoms with xdas:

- Function can be *atomized* using the {py:class}`~xdas.atoms.Partial` class. All parameters except the input are fixed.
- The {py:mod}`xdas.atoms` module contains a set of predefined atoms. In particular most stateful atoms are implemented in that module.
- The user can subclass the {py:class}`~xdas.atoms.Atom` class and define its own atoms.

### Transforming a classic workflow into an atomic pipeline

Imagine you tested the following workflow on a small subset of your data:

```{code-cell}
from scipy.signal import iirfilter

b, a = iirfilter(4, 0.1, btype="high")

def process(da):
  da = xs.decimate(da, 2, ftype="fir", dim="distance")  # not impacted by chunking 
  da = xs.lfilter(b, a, da, dim="time")  # require state passing along time
  da = np.square(da)  # already a unary operator
  return da

monolithic = process(da)
```

To convert your workflow into an atomic pipeline you need:

1. to convert each processing step into an atom
2. to bundle all steps into a {py:class}`~xdas.atoms.Sequential` atom.

Converting each processing step into an atom depend on the nature of the step. In particular it depends wether the operation is **stateful** (it does rely on the history along the chunked dimension) or **stateless** (the operation can by applied separately on each chunk along the given dimension without any particular consideration). An example of a stateful operation is a recursive filter, passing on the state from t to t+1.Not that this stateful/less caracteristic depends on the chunking dimension.

- unary operators that are not stateful (that do not rely on the history along the chunked axis) can be used as is.
- functions that are not stateful must be wrapped with the {py:class}`~xdas.atoms.Partial` class.
- functions that **are stateful** must be replaced by an equivalent stateful object.

In practice, the atomized workflow can be implemented like below. The resulting atom is a callable that can be applied to any data array.

```{code-cell}
from xdas.atoms import Sequential, Partial, LFilter

atom = Sequential(
  [
    Partial(xs.decimate, 2, ftype="fir", dim="distance"),  # use Partial when stateless
    LFilter(b, a, dim="time"),  # use equivalent atom object if stateful
    np.square,  # do nothing if unary and stateless
  ]
)

atomic = atom(da)

assert atomic.equals(monolithic) # works as `process` but can by applied chunk by chunk
```

### Applying an atom chunk by chunk

While atoms can be used as an equivalent of functions to organize pipelines, their major selling points is their abilities to enable chunk processing. While chunk by chunk processing can be done manually, xdas provides the {py:mod}`xdas.processing` module to facilitate this operation. The user must define one data loader and one data writer. Then the {py:func}`~xdas.processing.process` function is used to run the computation. 

```{code-cell}
:tags: [remove-cell]

!mkdir output
```

In the example below the data array is loaded by chunks of 100 samples along the `"time"` dimension. Each chunk is processed by the atom that was defined above and each resulting processed chunk is saved in the `output` folder. Once the computation is completed, the data loader return a unified view on the output chunks. 

```{code-cell}
:tags: [remove-output]

from xdas.processing import process, DataArrayLoader, DataArrayWriter

dl = DataArrayLoader(da, chunks={"time": 100})
dw = DataArrayWriter("output")
chunked = process(atom, dl, dw)

assert chunked.equals(monolithic)  # again equal but could be applied to much bigger datasets
```

```{code-cell}
:tags: [remove-cell]

!rm -r output
```

This part was a short summary about atoms and chunk processing. To go deeper on the atom part you can head to the [](user-guide/atoms) section. To further study chunk processing you can head to the [](user-guide/processing) section.