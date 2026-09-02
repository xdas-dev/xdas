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
import xdas as xd
```

## Dataset virtual consolidation

Most instruments usually produces datasets made out of a multitude of files, each one containing a temporal chunk of the full acquisition. In Xdas you can virtually concatenate all those files to create a virtual dataset that allows to seamlessly access the entire dataset as if it was a unique file.

### Linking multiple files 

If you are considering a unique acquisition you can use {py:func}`~xdas.open`. You can either pass a list of paths or a path pattern containing wildcards to specify which files must be linked together. Xdas should automatically detect the file format. Xdas support a variety of DAS formats and it is easy to add support to any custom or missing format. See the [](user-guide/io/data-formats) section for more information. 

In the example here, we have three files of interest in the current working directory:

```{code-cell}
ls 00*.h5
```

We can link them like this:

```{code-cell}
da = xd.open("00*.h5")
da
```

Xdas only loads the metadata from each file and returns a {py:class}`~xdas.DataArray` object. This object has mainly two attributes. First a `data` attribute that contain the data. Here a {py:class}`~xdas.VirtualStack` object that is a pointer to the different files we opened. Second, a `coords` attribute that contains the metadata related to how the space and the time are sampled. Here both dimensions are labeled using {py:class}`~xdas.InterpCoordinate` objects. Those allow to concisely store the time and space information, including potential gaps and overlaps. See the [](user-guide/coordinates/interpolated-coordinates) section for more information. 

Note that if you want to create a single data collection object for multiple acquisitions (i.e. different instruments or several acquisition with different parameters), you can use the [DataCollection](user-guide/data-structures/datacollection) structure.  

```{note}
For Febus users, converting native files into Xdas NetCDF format generally improves I/O operations and reduce the amount of data by a factor two. This can be done by looping over Febus files and running: `xd.open("path_to_febus_file.h5").to_netcdf("path_to_xdas_file.nc", virtual=False)`. The converted files can then be linked as described above.
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
da = xd.open("da.nc")
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
xd.open("chunked_and_compressed.nc")
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
mean = xd.mean(da, "time")
std = da.std("distance")
```

### Arithmetics 

You can manipulate data arrays objects as regular arrays, Xdas will check that dimensions and coordinates are consistent. 

```{code-cell}
squared = da * da
common_mode_removal = da - da.mean("distance")
```

### Scipy functions

Most scipy function from the `signal` and `fft` submodule have been implemented. The Xdas functions are multithreaded, with the number of threads sized automatically from the amount of data to process; a `parallel` keyword argument lets you override this on most of them.

Below an example of spatial and temporal resampling using {py:func}`~xdas.resample`, the single entry point that covers what `scipy.signal.decimate`, `resample_poly` and `resample` do (`method="fir"` by default, `"iir"` or `"fft"` also available). A target can be specified as a rate, a sampling interval, or a plain `up`/`down` ratio:

```{code-cell}
import xdas.signal as xs 

resampled = xd.resample(da, down=2, dim="distance")  # halve the spatial sampling, FIR by default
resampled = xd.resample(resampled, rate=25.0, dim="time", method="iir")  # 25 Hz along time, IIR

resampled.plot(yincrease=False, vmin=-0.25, vmax=0.25)
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

An {py:class}`~xdas.atoms.Atom` is a generic processing unit that takes one input and return one output. Atoms can store state information to ensure continuity from subsequent calls on contiguous chunks. Every processing function xdas ships (`xd.resample`, `xs.lfilter`, `xd.trigger`, ...) has a matching atom: calling it with data applies it right away, while passing `...` in place of the data returns the atom itself, ready to be composed into a pipeline with the `>>` operator.

There are three ways to make atoms with xdas:

- Most functions in {py:mod}`xdas.signal`, {py:mod}`xdas.fft` and the top-level `xd.*` task functions already have an atom form, obtained by passing `...` as the data argument.
- Other functions can be *atomized* using the {py:class}`~xdas.atoms.Partial` class. All parameters except the input are fixed.
- The user can subclass the {py:class}`~xdas.atoms.Atom` class and define its own atoms.

### Transforming a classic workflow into an atomic pipeline

Imagine you tested the following workflow on a small subset of your data:

```{code-cell}
from scipy.signal import iirfilter

b, a = iirfilter(4, 0.1, btype="high")

def process(da):
  da = xd.resample(da, down=2, dim="distance")  # not impacted by chunking 
  da = xs.lfilter(b, a, da, dim="time")  # require state passing along time
  da = np.square(da)  # already a unary operator
  return da

monolithic = process(da)
```

Converting each processing step into an atom depends on the nature of the step, in particular whether the operation is **stateful** (it relies on the history along the chunked dimension) or **stateless** (each chunk along the chunked dimension can be processed independently). An example of a stateful operation is a recursive filter, passing its state from one chunk to the next. Note that this stateful/stateless characteristic depends on the chunking dimension.

- unary operators that are not stateful can be used as is.
- functions that have an atom form become atoms by passing `...` as their data argument.
- functions without an atom form that are stateless must be wrapped with the {py:class}`~xdas.atoms.Partial` class.
- operations that **are stateful** must be replaced by an equivalent stateful atom, such as {py:class}`~xdas.atoms.LFilter`.

In practice, the atomized workflow can be built by chaining atoms with `>>`. The resulting {py:class}`~xdas.atoms.Sequential` pipeline is a callable that can be applied to any data array:

```{code-cell}
from xdas.atoms import LFilter

atom = xd.resample(..., down=2, dim="distance") >> LFilter(b, a, dim="time") >> np.square
atom
```

```{code-cell}
atomic = atom(da)

assert atomic.equals(monolithic)  # works as `process` but can by applied chunk by chunk
```

### Applying an atom chunk by chunk

While atoms can be used as an equivalent of functions to organize pipelines, their major selling point is their ability to enable chunk processing. Every pipeline exposes a {py:meth}`~xdas.atoms.Atom.process` method that resolves whatever source and sink you give it — an in-memory or virtual {py:class}`~xdas.DataArray`, a file glob, a directory being written to in real time (see {py:func}`~xdas.watch`) — and streams chunks through, with no loader/writer boilerplate needed.

In the example below the data array is streamed by chunks of 100 samples along the `"time"` dimension, and each resulting processed chunk is saved in the `output` folder. The call returns a unified view on the output chunks once the computation is completed.

```{code-cell}
:tags: [remove-output]

chunked = atom.process(da, chunks={"time": 100}, out="output/")

assert chunked.equals(monolithic)  # again equal but could be applied to much bigger datasets
```

```{code-cell}
:tags: [remove-cell]

!rm -r output
```

This part was a short summary about atoms and chunk processing. To go deeper on the atom part you can head to the [](user-guide/pipeline/atoms) section. To further study chunk processing you can head to the [](user-guide/pipeline/processing) section. Xdas also ships built-in atoms for phase picking with pretrained [SeisBench](https://github.com/seisbench/seisbench) models (`xd.pick`, {py:class}`~xdas.atoms.Trigger`) — see the [](user-guide/pipeline/picking) guide.