---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
import xdas as xd
os.chdir("../../_data")
```

# Virtual Datasets

To deal with large multi-file dataset, *Xdas* uses the concept of virtual datasets. A virtual dataset is a file that contains pointers towards an arbitrary number of files that can then be accessed seamlessly as a single, contiguous dataset. 

*Xdas* uses several types of virtualization, selected with the `vtype` argument:

- `hdf5`: for HDF5 based formats, it leverages the [virtual datasets](https://docs.h5py.org/en/stable/vds.html) native capabilities of netCDF4/HDF5, which resolve the mapping in compiled C with no per-file Python overhead. The cost of that mapping does however grow with the number of linked files.
- `tiles`: a manifest of file-backed tiles stored as a plain array, decoded by the engine itself. It works with any format and keeps the file mapping inspectable.

Which types an engine offers is declared by its `_supported_vtypes` attribute; the first one listed is the default. See [](#choosing-a-virtualization-backend) for how to pick, and [](data-formats.md) for what each engine supports and defaults to.

A third backing, [Dask arrays](https://docs.Dask.org/en/stable/array.html), is deprecated and no longer used by any engine — see [](#dask-virtualization).

## HDF5 Virtualization

```{note}
Because netCDF4 are valid HDF5 files, the virtual dataset feature of HDF5 can be used with netCDF4 files.
```

In *xdas*, a {py:class}`VirtualSource` is a pointer towards a file, while a {py:class}`VirtualLayout` is table linking multiple {py:class}`VirtualSource`s. Below is an example of a virtual dataset linking three files:

![](/_static/virtual-datasets.svg)

In most cases, users do not need to deal with this object directly. 

```{note}
When opening a virtual dataset, this later will appear as a {py:class}`VirtualSource`. This is because HDF5 treats virtual dataset as regular files.
```

## Use cases

The generic {py:func}`xdas.open` funtion should cover all your needs. But you can specify how to handle individual files, multiple files, and virtual datasets, by picking one of the following routines:

| Function                             | Output                           | Description                                                                 |
|--------------------------------------|----------------------------------|-----------------------------------------------------------------------------|
| {py:func}`xdas.open_dataarray`       | {py:class}`~xdas.DataArray`      | Open a (virtual) file.                                               |
| {py:func}`xdas.open_mfdataarray`     | {py:class}`~xdas.DataArray`      | Open multiple (virtual) files and concatenate them.                         |
| {py:func}`xdas.open_mfdatacollection`| {py:class}`~xdas.DataCollection` | Open multiple (virtual) files, grouping and concatenating compatible files. |
| {py:func}`xdas.open_mfdatatree`      | {py:class}`~xdas.DataCollection` | Open a directory tree of files, organizing data in a data collection.       |
| {py:func}`xdas.open_datacollection`  | {py:class}`~xdas.DataCollection` | Open a (virtual) collection.                                         |

Please refer to the [](../data-structures/datacollection.md) section for the functions that return a data collection.

## Linking multi-file datasets

Multiple physical data files can be opened simultaneously with {py:func}`xdas.open`:

```{code-cell}
:tags: [remove-stdout,remove-stderr]

da = xd.open("00*.nc")
da
```

Here, `*` is a wildcard operator. `xdas.open` only creates file handles and loads basic metadata, but does not directly load the underlying DAS data in memory. Hence this method can open an arbitrary number
of files with no concern over memory allocation. Next, the DataArray can be written to disk as a single dataset. The `virtual` argument ensures that only the pointers to the original data files are written to disk (otherwise the whole data set will be written to disk):

```{code-cell}
da.to_netcdf("vds.nc", virtual=True)
```

It can then be read again as a single file using {py:func}`xdas.open`:

```{code-cell}
xd.open("vds.nc")
```

```{hint}
A virtual dataset can point to another virtual dataset. This can be beneficial for huge real time dataset where new data can be linked regularly by batches. Those batches can then be linked in a master virtual dataset. This avoids relinking all the files. 
```

```{warning}
When loading large part of a virtual dataset, you might end up with nan values. This normally happens when linked files are missing. But due to a [known limitation](https://forum.hdfgroup.org/t/virtual-datasets-and-open-file-limit/6757) of the HDF5 C library it can be due to the opening of too many files. Try increasing the number of possible file to open with the `ulimit` command. Or load smaller chunk of data. 
```

(tile-virtualization)=
## Tile Virtualization

With the `tiles` vtype, the mapping is not delegated to HDF5. *Xdas* stores it as a
{py:class}`xdas.virtual.TileArray`: a plain array manifest that records, for each tile, which
file it comes from and which part of that file it contributes. Reading a region resolves
which tiles it touches and asks the engine to decode each of them through its `load_tile`
method. The manifest is ordinary data, so it can be inspected, sliced and concatenated
like any other array, and it is stored as such inside the *Xdas* netCDF format.

(choosing-a-virtualization-backend)=
## Choosing a virtualization backend

For formats that HDF5 virtual datasets cannot serve, the choice is made for you. When an
engine supports both, the trade-off is essentially *who resolves the mapping*: the HDF5 C
library, or *Xdas* itself.

That choice decides how each cost scales with the size of the archive. Writing and
reopening an HDF5 virtual dataset both cost one operation per linked file, so both grow
with the file count; a tile manifest is written and read back as an array, so neither
does. Reading inverts the expectation one might have: resolving a region inside the C
library involves no Python at all, but its cost grows with how many mappings the dataset
*contains*, while a tile manifest is searched, so its cost grows only with how many tiles
the read *touches*. Modest file counts therefore favour HDF5, and the advantage moves to
tiles as the archive grows.

### HDF5 virtualization

**Advantages**

- Resolution happens inside the HDF5 C library, so reading involves no per-file Python
  call. On modest file counts this makes it the faster of the two to read.
- Any HDF5-aware tool can read the result, not only *Xdas*.
- Virtual datasets can point at other virtual datasets, so a growing archive can be
  linked in batches without relinking everything.
- A subset saved from a virtual dataset is very compact, because it refers to the dataset
  it was cut from rather than restating the underlying file list.

**Limitations**

- Building the mapping costs one HDF5 call per source file, so both the time and the
  memory needed to write a manifest grow in proportion to the number of files. Beyond
  some point, writing a single flat manifest stops being practical.
- Reopening a virtual dataset reads its whole mapping table, so opening cost also grows
  with the number of linked files. Deep archives therefore tend to require a pyramid of
  virtual datasets, which shifts that cost to read time and multiplies the number of
  manifest files to keep track of. The top of such a pyramid opens quickly precisely
  because it defers the work: the first read of a region then has to open the level below
  it, a toll that a short-lived process pays on every run.
- Read latency grows with the number of mappings the dataset holds, not only with the
  amount of data asked for, so the same request gets slower as the archive it lives in
  gets bigger. Past a large enough file count this outweighs the advantage of resolving
  in C, and reads become slower than the tile equivalent.
- Once written, the mapping is opaque: HDF5 presents a virtual dataset as a regular
  dataset, so the list of linked files can no longer be inspected or edited.
- Because a saved subset refers to its parent, extracts are not self-contained. Moving or
  deleting the parent breaks them, and each extract adds one more level of indirection.
- Strided (decimating) selection along the concatenation axis is not supported.
- Missing files are read as NaN rather than raising, and exceeding the C library's
  open-file limit produces the same symptom, which makes such problems easy to miss.

### Tile virtualization

**Advantages**

- Writing a manifest is an array write rather than a per-file operation, so it stays fast
  and light as the number of files grows. A single flat manifest remains workable at
  scales where an HDF5 one does not.
- Opening loads only the tile geometry, so open time stays low even for very large
  manifests, and no cost is deferred to the first read.
- Read latency depends on how much of the manifest a request touches, not on how large
  the manifest is, so reads do not get slower as the archive grows.
- The file list is data: it can be inspected, modified and saved again.
- Concatenation fuses manifests without reading any values.
- A saved subset names the data files it needs directly, so extracts are self-contained
  and never gain an extra level of indirection.
- Decimating along the stacking axis is supported: the stride is folded into the tile
  geometry and stays lazy.
- The per-file footprint of the manifest is smaller.

**Limitations**

- Decoding goes through Python, one call per tile touched. A request spread over a great
  many tiles therefore carries a per-tile overhead that the C library avoids, which is
  what makes HDF5 the quicker reader while file counts stay modest.
- For very small manifests the result can be larger than the HDF5 equivalent, since paths
  and geometry are written out explicitly instead of referring to a parent dataset.
- The manifest is only meaningful to *Xdas*.
- The engine must implement `load_tile`.

### Considerations that apply to both

- Building either manifest starts with reading the metadata of every file. That scan is
  dominated by disk access and is usually the bulk of the total build time, so it is not a
  criterion for choosing between the two.
- A scan holds at most 100 000 results at once. Tiles combines them every 100 000 files
  into a manifest whose per-file cost is negligible, so the batch is freed and a single
  call can scan any number of files. Fusing changes nothing for `hdf5`, which keeps one
  HDF5 virtual mapping per file whatever it does, so 100 000 is a ceiling for it instead
  of a batch size.
- The combined result does not depend on the scan order: files are sorted by their
  coordinate values, not by their names.
- Neither backend helps when a coordinate is not monotonic — for instance when files
  overlap in time. Label-based selection then falls back to a slow path in both cases, and
  is better addressed in the data itself.

```{hint}
As a rule of thumb, prefer `hdf5` when the dataset is modest in file count or when other
HDF5-based tooling has to read it, and `tiles` when the file count is large, when the
mapping needs to remain inspectable, when extracts must stand on their own, or when
decimated reads matter. The larger the archive, the stronger the case for `tiles`: it is
the only one of the two whose write, open and read costs do not all grow with the number
of files.
```

(dask-virtualization)=
## Dask Virtualization (deprecated)

```{deprecated} 0.2.9
Dask virtualization is no longer used by any engine and writing it is deprecated. The
formats that once relied on it — those HDF5 virtual datasets cannot serve — now use
[tile virtualization](#tile-virtualization) instead. Existing files that store a Dask
graph can still be read, so nothing on disk is lost, but new datasets should not be
written this way.
```

Formats that HDF5 could not virtualize used to be loaded as Dask arrays: an
N-dimensional stack of chunks, each with a task attached that produces its values,
serialized into the native *Xdas* netCDF format as a computation graph. Tiles replace it
with a manifest that describes the same mapping as plain array data, which is both more
compact and far quicker to build, and which does not go sluggish once the graph reaches
millions of tasks.
