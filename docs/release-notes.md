# Release notes

## 0.2.9 (unreleased)

### New Features
- **Tile-backed virtual arrays.** The new `xdas.virtual.tiles` module exposes file archives as one lazy `TileArray`. Slicing (any step, including negative), integer indexing, `np.newaxis`, concatenation, and the numpy manipulation routines (the `transpose`, `flip`, `split`, `stack` and `atleast` families, `expand_dims`, `squeeze`, `roll`, `tile`, `delete`, `append`/`insert`) all stay lazy; whole-array reductions (`sum`, `mean`, `min`, `max`, …) stream one tile row at a time; reads touch only the tiles the selection overlaps (@atrabattoni).
- **`vtype="tiles"` on every HDF5 engine.** The open functions with `vtype="tiles"` return tile-backed arrays for the asn, febus, terra15, apsensing, prodml and native xdas engines. Silixa and MiniSEED always emit them now (replacing the serialized-dask-graph fallback, with time-axis push-down for Silixa), and Febus defaults to them — one tile per file, where the HDF5 backing needed one virtual mapping per data block. Custom engines add support by implementing `Engine.load_tile(path, selection, **params)` (@atrabattoni).
- Tile-backed arrays round-trip through the native xdas netCDF format: the manifest is stored as a compact `__tiles__` sibling group, relocatable by editing the single root path of its header (@atrabattoni).
- **Explicit engine configuration.** The open functions declare `engine`, `vtype` and `ctype`, and `engine` also accepts a configured `xdas.io.Engine` instance. Format-specific parameters (`overlaps`/`offset` for febus, `swapped_dims` for prodml, `tz` for terra15, `group` for the native format) are engine constructor parameters, validated up front (@atrabattoni).
- `open_mfdataarray` no longer caps the number of files when the resolved vtype consolidates its scan results, which `tiles` does; the 100 000 ceiling remains for `hdf5`, which builds one virtual mapping per file and so cannot be fused into anything smaller (@atrabattoni).
- **Streamed multi-file combining.** `open_mfdataarray` now fuses scan results every 100 000 files instead of holding one data array per file until the end, so memory no longer grows with the archive: results are accumulated without coordinate simplification (lossless in any arrival order) and sorted once at the end, giving the same result as before whatever the file naming. Acquisitions interleaved in time now group by compatibility, one array per acquisition, instead of splitting at each alternation. Since the batch size is also the ceiling, anything that opened in one call before still takes the single-batch path unchanged (@atrabattoni).
- **`xdas.sortby`.** Sort a tile- or stack-backed data array along a dimension by coordinate value, lazily: the blocks are permuted through the manifest without reading any data. This is how the streamed combine orders shuffled archives, exposed for standalone use (@atrabattoni).
- Constant tile geometry no longer costs one element per tile. A `sizes_k`, `starts_k` or `steps_k` column that holds a single value — what a scanned acquisition of equal-length files gives, and always the case for the absent origin and stride columns — is kept as a broadcast view, and its tile boundaries as a closed form instead of a full `cumsum`. Opening a 23-million-tile archive drops from 1.67 GB to 1.11 GB resident, and locating a tile along such an axis becomes a division instead of a binary search (@atrabattoni).
- **The `obspy` engine.** The miniseed engine is replaced by one named after the library rather than after a format: decoding is `obspy.read`, so miniSEED, SAC, GSE2, SEG-2 and everything else ObsPy supports now goes through it. It mirrors `obspy.read` exactly — each contiguous `Trace` becomes one lazy `DataArray`, and the collection mirrors the `Stream`, nested as `network / station / location / channel`. Files the old engine rejected (two sampling rates, duplicated ids, interleaved acquisitions) are now readable, and each tile points at an individual trace instead of the whole file. The `"miniseed"` engine is kept unchanged next to it, so views it wrote keep decoding and code written against it keeps running; auto-detection reaches `"obspy"` first (@atrabattoni).
- **`xdas.trim_overlaps`.** Resolve the overlaps of a data array or collection by dropping the duplicated samples, keeping the later copy by default (ObsPy's `merge(method=1, interpolation_samples=0)`) or the earlier one with `keep="first"`. Trimming lands on a sample boundary — nothing is resampled, interpolated or filled — and stays at the manifest level, so a lazy array stays lazy. This replaces the miniseed `ignore_last_sample` flag with its better form: the earlier copy goes only where an overlap genuinely exists, and clean seams are left alone. `xdas.split(da, "overlaps")` remains for keeping every copy (@atrabattoni).
- `DataCollection.select` is added as an alias of `query`, and `fields` now reports every level of the subtree rather than only the current one and its immediate children. Together with the nested collection the `obspy` engine returns, this gives `obspy.Stream.select` semantics: `dc.select(station="SX00*", channel="HH?")` (@atrabattoni).
- `xdas.concat` opening a *new* dimension now checks that the inputs agree on their other coordinates, and promotes the scalar ones that vary to a coordinate along that dimension. Stacking the components of a station is `xd.concat(traces, "channel")`, lazily, with `channel` becoming a real coordinate (@atrabattoni).
- `simplify` runs in linear time whatever the number of gaps: the reduce stage is now a one-pass sleeve instead of Douglas-Peucker, which degenerated quadratically on gap-rich coordinates (a 100 000-file gappy archive simplified in minutes; now milliseconds). The deviation guarantee is unchanged — dropped tie points stay within `tolerance` of the curve, surviving values never move — though the surviving tie-point selection may differ slightly on jittery axes (@atrabattoni).

### Breaking Changes
- **The stored tile format changed once, deliberately.** Everything about a tiling that is not a per-tile column now travels in a single JSON `header` attribute on the `__tiles__` group — the tile counts, the engine specification, the element type, the common source directory and the axis arrangement — replacing the `__tile_array__` placeholder attribute, the `root` / `axes` / `source_ndim` manifest variables and the per-column `ntiles` attributes. The manifest variables are now exactly the per-tile columns. The placeholder variable points at its describing group through a CF-`grid_mapping`-style `__tiling__` attribute rather than being tied to the group name. The reader does not accept the earlier spelling: rewrite existing tile-backed files with 0.2.8 or earlier still installed to read them, and this release to write them back (@atrabattoni).
- **Dask virtualization is removed**, reader and writer alike, along with the `xdas.dask` module: no engine has emitted it since tiles landed, and a `__dask_array__` graph can no longer be read. A Dask array remains valid `DataArray` data — it is now computed on write like any other eager array, and `virtual=True` rejects it (@atrabattoni).
- `TileArray` carries no user attributes: `TileArray.attrs` and the `attrs=` argument of `TileArray.from_tiles` are gone. It is a duck array, like the numpy array it stands in for — metadata belongs to the enclosing `DataArray`, where it always was in practice (@atrabattoni).
- Interpolated coordinates are stored the way CF-1.13 actually defines them, and files declare `Conventions = "CF-1.13"`: each `coordinate_interpolation` group names its tie point coordinate variable and ends with the interpolation variable, whose mapping attribute is the singular `tie_point_mapping` (interpolated dimension, tie point index variable, subsampled dimension) and which now carries the mandatory `computational_precision`. The earlier spelling — CF-shaped, but not valid against the grammar — is still read (@atrabattoni).
- Sampled coordinates are stored as a deliberate variation on that same CF grammar: a `coordinate_sampling` attribute whose groups name the tie point coordinate variable and end with a sampling variable, a container like the interpolation variable, whose `tie_point_mapping` puts the segment length variable in the tie point index variable's slot and whose `sampling_interval` travels as attributes, encoded like the regular metadata of an interpolated coordinate (@atrabattoni).
- Python 3.10 support is dropped and the numpy requirement is raised to 2.3: the tile manifests use `np.strings` routines introduced in numpy 2.3, which itself requires Python 3.11+. Python 3.10 reaches end of life in October 2026 (@atrabattoni).
- Passing a bare read function as `engine` now raises a `TypeError`: subclass `xdas.io.Engine` instead (see the data-formats documentation) (@atrabattoni).
- Misspelled or unsupported keyword arguments passed next to an engine name now raise a `TypeError` instead of being silently ignored, and combining `vtype`, `ctype` or engine keywords with an already configured engine instance raises a `ValueError` (@atrabattoni).
- **Opening a seismological file without naming an engine returns a nested collection, not a stacked array.** `xd.open(file)` on a three-component file used to return a `(3, 100)` array by guessing that the traces were synchronized; it now returns the `network / station / location / channel / acquisition` tree the `"obspy"` engine describes. `xd.concat(traces, "channel")` is the one-liner back, and the `dim="station"` multi-file idiom is replaced by the nesting plus `select`. Naming `engine="miniseed"` still gives the old shape, `ignore_last_sample` included, and `xd.open_dataarray` still falls through to it when the new engine cannot describe a file as a single array (@atrabattoni).
- `xd.open` now combines whether it opened one file or many, so the shape it returns no longer depends on the file count. Its leaf sequences are named `acquisition`, since after combining each element is one acquisition epoch — contiguous traces have fused and gaps have moved into the coordinate (@atrabattoni).
- `DataCollection.query` raises a `KeyError` on an indexer naming no level of the collection, instead of silently returning everything unchanged, and applies an indexer wherever its level sits in the tree rather than only at the root. `dc.query(time=slice(0, 5))`, which used to be a no-op, now raises: use `sel` to trim inside the leaves (@atrabattoni).
- A blank SEED location code is stored as `"--"`, the FDSN convention, since `""` cannot be a netCDF group name (@atrabattoni).

### Bug Fixes
- The miniseed `ctype` argument is now honored: it previously routed to an unused attribute and the reader always built interpolated time coordinates. The default is unchanged (@atrabattoni).
- A data collection keyed by zero-padded codes — a SEED location such as `"00"`, say — no longer reads back from netCDF as a sequence with its keys lost. A sequence is written under the canonical decimal spelling of its positions, so that is now what the reader compares against, instead of parsing the keys as integers (@atrabattoni).
- The miniSEED element type is read from the file's encoding rather than from the empty array `headonly=True` returns, which is always `float64`. A STEIM-compressed `int32` file used to be scanned as `float64` (@atrabattoni).
- Scanning miniSEED files is no longer forced to a single process (@atrabattoni).

## 0.2.8

### New Features
- **Regular coordinates.** A coordinate can now declare a nominal `sampling_interval` (with a `tolerance` bounding the allowed jitter). Query it with `isregular()` / `get_sampling_interval()`; promote an irregular coordinate with `to_regular()`. File engines, `from_block`, and the `fft`/`stft` outputs produce regular coordinates out of the box (@atrabattoni).
- Chunked and unchunked processing now yield identical coordinates: operations that derive a new rate record their rounding error in `tolerance`, and `simplify`/`concat` spend the declared tolerance by default, fusing chunk seams away (@atrabattoni).
- `simplify` gained `reduce` and `regularize` keywords, and the gaps/overlaps API now works on every axis coordinate, including dense ones (@atrabattoni).

### Deprecations
- The sampling interval is now declared metadata rather than a computed end-to-end average (which was silently wrong on jittery or gappy axes). Data saved by earlier versions carries no declared rate: querying it — e.g. through any signal-processing routine — still works for now, but the rate is inferred and a `FutureWarning` explains how to make the coordinate regular (`da[dim] = da[dim].to_regular(tolerance=...)`). A future release will raise instead (@atrabattoni).

### Bug Fixes
- Fix `Sequential.reset()` silently doing nothing: it only reset `Partial` atoms, so stateful atoms such as `IIRFilter` or `ResamplePoly` kept their state and a reused sequence returned wrong data (@atrabattoni).
- Fix `stft` ignoring the `"first"`/`"last"` dimension aliases — including its own default `dim` — which raised a size-conflict error instead of transforming the named axis (@atrabattoni).

### Refactoring
- Reworked the coordinate class hierarchy around two ABCs, `Coordinate` and `AxisCoordinate`. Use `isinstance(coord, AxisCoordinate)` instead of the removed `is*` predicates (@atrabattoni).
- Trimmed the coordinate API: several internal-leaning methods were removed or made private (@atrabattoni).
- `concat_coords` now simplifies its result by default, like `concat`; values are unchanged, only redundant tie points are dropped (@atrabattoni).
- Added `xdas.testing.dummy`, a configurable fixture generator replacing `xdas.synthetics.dummy` (@atrabattoni).
- Comply with ruff 0.16 and its much broader default rule set (@atrabattoni).
- The package version is now declared in a single place, `xdas/__init__.py` (@atrabattoni).

## 0.2.7

### Bug Fixes
- Fix a regression introduced in 0.2.6 where `is_monotonic` was significantly degrading `.sel` performance.
- Fix `xdas.concat` to gracefully handle empty inputs, preventing errors when selecting out-of-range data from a `DataCollection` (@atrabattoni).

### Documentation
- Achieved **100% docstring coverage** (excluding `__magic__` and private `_methods`) (@atrabattoni).
- Improved *User Guide* index (@atrabattoni).
- Added new *Sampled Coordinates* page (@atrabattoni).
- Enhanced *Processing* documentation (@atrabattoni).
- Improved *FAQ* page (@atrabattoni).
- Added missing API documentation for several methods (@atrabattoni).

### Refactoring
- Achieved **100% test coverage** across the codebase (@atrabattoni).
- Migrated development workflow from conda to [uv](https://docs.astral.sh/uv/) (@atrabattoni).
- *Reduced* test suite execution time by **~50%** (@atrabattoni).
- Migrated formatting tooling from `isort` + `black` to `ruff` **including docstring checks** (@atrabattoni).
- Ensure all ruff checks pass (@atrabattoni). 

## 0.2.6

### New features
- Add `xdas.open` that automatically infers which `xdas.open_*` function to use (@atrabattoni, @yetinam).
- Add automatic engine detection to every `xdas.open_*` function (@atrabattoni, @yetinam).
- Add `pathlib.Path` support as input for all xdas file-related functions and methods (@atrabattoni).
- Add `xdas.io.compressed` that compresses a specific dataset in an HDF5 file while preserving the rest of the file structure and metadata (@marbail).
- Add `xdas.concat_coords` to merge coordinates. Also `Coordinate.append` is now `Coordinate.concat` (avoid in-place confusion) and `xdas.concatenate` has now a preferred alias `xdas.concat` (@atrabattoni).

### Improvements
- All `to_netcdf` methods now have a `create_dirs` argument to create intermediate directories if necessary (@aurelienfalco).
- Make `DataArray.sel` handle overlaps when slicing, and `xdas.split` can split on overlaps or gaps now (@atrabattoni).
- New `io.Engine` backend system to register different file formats (@atrabattoni).
- Make `open_mfdataarray` raise `RuntimeError` when opening all files fails (@asladen). 
- Add "prodml" engine (@atrabattoni) and make "optasense" and "sintela" aliases of it (@atrabattoni). 
- Add the `component_strategy` argument to the `xdas.atoms.MLPicker` to choose whether to use the same component on the 3 channels or to use one channel and set the others to 0 based (@marbail). 
- Make `DataArray.rename` capable of renaming `dims` and `coords` (@atrabattoni). 
- Add `parallel` argument to most `open*` functions to let the user choose the file opening strategy (@atrabattoni). 

### Bug Fixes
- Fix **memory accumulation** when slicing multiple times data arrays, e.g. when using atoms (@atrabattoni).
- Fix **non-terminating loaders and writers** in `xdas.processing` (@atrabattoni).
- Fix/improve distance handling for: "apsensing", "febus", "optasense", "silixa", and "sintela" (@atrabattoni).
- Add dim swapping handling for the "prodml" based engines with the `swapped_dims=False` kwarg (@atrabattoni).
- Fix ASN ROI handling (@asladen).
- Use the `annotate_batch_pre` model's function to normalize in `xdas.atoms.MLPicker` (@marbail).
- Fix the RuntimeError encountered when using `open_mf*` functions in scripts due to the use of multiprocessing by using the loky library (@atrabattoni)

## 0.2.5
- Add SampleCoordinate for more SEED-like coordinates and refactor the coordinate backend (@atrabattoni).
- Add `xdas.picking.tapered_selection` to extract windows around picks (@atrabattoni).
- Add `create_dirs` to `.to_netcdf` methods to create intermediate directories (@aurelienfalco).
- Add support for multiple ROI for ASN engine (@martijnende).
- `tolerance` can now be passed as seconds for datetime64 coordinates (@martijnende, @atrabattoni)
- Add support for python 3.14, numpy 2.4 and obspy 1.4.2 incompatibilities and add `xdas.__version__` (@atrabattoni).

## 0.2.4
- Add StreamWriter to write long time series to miniSEED (@marbail).
- Fix OptaSense engine's wrong axis attribution (@smouellet).
- Fix ASN (OptoDAS) engine: handling of roiDec (@AndresLaurine).
- Fix NaN handling for several methods (@ClaudioStrumia).
- Fix `InterpCoordinate.get_availabilities` (@AMordret).

## 0.2.3
- Fix Febus engine (round timestamps to closest us).
- Faster `xdas.concatenate` (faster linking for efficient reading of Febus files).

## 0.2.2
- Add support for Python 3.13
- Fix bugs and dependency issues

## 0.2.1
- Add `xdas.signal.stft`.
- Add inverse Fourier transforms `xdas.fft.ifft` and `xdas.fft.irfft`.
- Add support for APSensing format.
- Improve overlap error message.
- Fix decimation of freshly opened multi-file datasets.
- Fix `zerophase` keyword argument for `xdas.signal.filter`.
- Fix applying FFT functions in presence of non-dimensional coordinates.

## 0.2
- Add Dask virtualization backend for non-HDF5 formats (@atrabattoni).
- Add support for miniSEED format (@atrabattoni, @chauvetige).
- Add support for Silixa (TDMS) format (@atrabattoni, @Stutzmann).

## 0.1.2
- Add ZeroMQ streaming capabilities (@atrabattoni).
- Add support of Terra15 format (@chauvetige).
- Fix Febus engine (@ClaudioStrumia).

## 0.1.1
- Add support for `hdf5plugin` compression schemes.
- Drop `netCDF4` dependency and only use `h5netcdf` to fix incompatibilities.
- Drop useless `dask` dependency.

## 0.1
Initial stable version.