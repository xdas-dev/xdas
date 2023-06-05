# Data Structures

*xdas* mainly uses two kind of data structures. The base kind of structure are the 
{py:class}`xarray.DataArray` and {py:class}`xdas.Database` object. Those are 
N-dimensional labeled array object. The second kind of structure are the 
{py:class}`xarray.Dataset` and {py:class}`xdas.DataCollection` objects that are 
mappings of DataArray and Database objects respectively. The *xdas* objects are their 
*xarray* counterpart that uses a subset of the *xarray* API but extend their capabilities 
in term handling of huge multi-file collection of data. When manipulating data that 
are small enough to be stored in-memory, the full capabilities of *xarray* objects are 
generally preferred. In a near future, we hope that the extra functionalities provided 
by the *xdas* object will be part of the *xarray* library to enjoy the best of both 
words.

The *xarray* objects structure description can be found in the 
[xarray documentation](https://docs.xarray.dev/en/stable//user-guide/data-structures). 
Here a focus on the *xdas* objects are presented.

## Database

{py:class}`xdas.Database` is the base class to load and manipulate big datasets to in 
*xdas*. It is mainly composed of two attributes: 

- `data`: any N-dimensional array-like object. Compared to *xarray* `xdas.Database` are
more flexible on the kind of array-like object that can be used. In particular, 
[](virtual-datasets.md) can be used.
- `coords`: a dict-like container of coordinates. Instead of *xarray* that uses dense
arrays to label each point, *xdas* uses [](interpolated-coordinates.md) that provides
an efficient representation of mainly evenly spaced data (with eventual gaps and small
sampling variations). 

![](/_static/database.svg)

## Dataset/DataCollection
