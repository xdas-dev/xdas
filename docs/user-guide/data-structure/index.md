# Data Structure

Xdas leverages two main data structures. 

The workhorse data structures is the {py:class}`~xdas.DataArray` object, which is a N-dimensional labeled array object. In the context of DAS, it is typically used to encapsulated the data and metadata related to a unique acquisition (with potential small gaps and overlaps).

The second data structures is the {py:class}`~xdas.DataCollection` objects which is a nesting of `DataArray` objects. In the context of DAS it is typically used to combine multiple acquisition with potentially different sampling configurations together or to facilitate operations across different instruments. Another use case is the extraction and gathering of data windows related to detected earthquakes.

Note that the *Xdas* `DataArray` object follows its equivalent *Xarray* counterpart. It extends it in term of huge multi-file datasets handling at the cost of a limited subset of the Xarray API. The Xarray begin much more mature it might be interesting to make a detour into their [documentation](https://docs.xarray.dev/en/stable/user-guide/data-structures.html) to understands the concepts and goals behind the DataArray structure.

```{toctree}
:maxdepth: 1

dataarray
datacollection
```