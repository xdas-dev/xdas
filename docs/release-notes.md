# Release notes

## 0.2.9 (unreleased)

### New Features
- **Tile-backed virtual arrays.** The new `xdas.virtual.tiles` module exposes a file archive as one lazy `TileArray`: slicing (any step), integer indexing, `np.newaxis`, concatenation and the numpy manipulation routines stay lazy, reductions stream one tile row at a time, and a read touches only the tiles the selection overlaps. Select it with `vtype="tiles"` on any HDF5 engine — Febus defaults to it, Silixa and the ObsPy formats always use it. Tile-backed arrays round-trip through the native netCDF format as a compact `__tiles__` group, relocatable by editing the single root path of its header. Custom engines opt in by implementing `Engine.load_tile(path, selection, **params)` (@atrabattoni).
- **The `obspy` engine**, named for the library rather than for a format: decoding is `obspy.read`, so miniSEED, SAC, GSE2, SEG-2 and everything else ObsPy supports goes through it. Each contiguous `Trace` becomes one lazy `DataArray` and the collection mirrors the `Stream`, nested `network / station / location / channel`; files the miniseed engine rejected (two sampling rates, duplicated ids, interleaved acquisitions) are now readable (@atrabattoni).
- **`xdas.trim_overlaps`.** Resolve the overlaps of a data array or collection by dropping the duplicated samples, keeping the later copy by default (ObsPy's `merge(method=1, interpolation_samples=0)`) or the earlier one with `keep="first"`. Trimming stays at the manifest level, so a lazy array stays lazy; `xdas.split(da, "overlaps")` still keeps every copy (@atrabattoni).
- **`DataCollection.select`**, with `obspy.Stream.select` semantics: `dc.select(station="SX00*", channel="HH?")`. It is an alias of `query`, which now applies an indexer wherever its level sits in the tree rather than only at the root (@atrabattoni).

### Improvements
- **Scanning scales to archives of any size.** `open_mfdataarray` fuses scan results every 100 000 files instead of holding one data array per file, so memory no longer grows with the archive. With `vtype="tiles"` the file-count ceiling is lifted and constant tile geometry costs one element instead of one per tile: a 23-million-tile archive opens in 1.11 GB instead of 1.67 GB (@atrabattoni).
- **Explicit engine configuration.** Every open function declares `engine`, `vtype` and `ctype`, and `engine` also accepts a configured `xdas.io.Engine` instance. Format-specific parameters (`overlaps`/`offset` for febus, `ignore_last_sample` for miniseed, `swapped_dims` for prodml, `tz` for terra15, `group` for the native format) are engine constructor arguments, validated up front — a misspelled keyword now raises instead of being silently ignored (@atrabattoni).
- **`xdas.sortby`.** Sort a tile- or stack-backed data array along a dimension by coordinate value, lazily: the blocks are permuted through the manifest without reading any data (@atrabattoni).
- `xdas.concat` opening a *new* dimension now checks that the inputs agree on their other coordinates and promotes the scalar ones that vary. Stacking the components of a station is `xd.concat(traces, "channel")`, lazily, with `channel` becoming a real coordinate (@atrabattoni).
- Acquisitions interleaved in time now group by compatibility, one array each, instead of splitting at every alternation (@atrabattoni).
- Saving and opening a data collection is linear in its size again (#81): saving 1300 events went from ~53 min to ~35 s (@atrabattoni).
- `simplify` runs in linear time whatever the number of gaps. The deviation guarantee is unchanged, though the surviving tie points may differ slightly on jittery axes (@atrabattoni).
- When no engine can open a file, the error now lists what each engine that recognised it said, instead of only reporting that none succeeded (@atrabattoni).

### Breaking Changes
- Python 3.10 support is dropped and the numpy requirement is raised to 2.3 (@atrabattoni).
- **Dask virtualization is removed**, reader and writer alike, along with the `xdas.dask` module: a `__dask_array__` graph can no longer be read. A Dask array remains valid `DataArray` data — it is now computed on write like any other eager array, and `virtual=True` rejects it (@atrabattoni).
- **Opening a seismological file without naming an engine returns a nested collection, not a stacked array.** `xd.open(file)` on a three-component file used to return a `(3, 100)` array by guessing the traces were synchronized; it now returns the `network / station / location / channel / acquisition` tree. `xd.concat(traces, "channel")` is the one-liner back, and `engine="miniseed"` still gives the old shape. More generally, `xd.open` now combines whether it opened one file or many, so the shape it returns no longer depends on the file count (@atrabattoni).
- `DataCollection.query` raises a `KeyError` on an indexer naming no level of the collection, instead of silently returning everything unchanged. `dc.query(time=slice(0, 5))`, which used to be a no-op, now raises: use `sel` to trim inside the leaves (@atrabattoni).
- Custom engines must subclass `xdas.io.Engine`: passing a bare read function as `engine` now raises a `TypeError` (see the data-formats documentation) (@atrabattoni).

### Bug Fixes
- Fix a STEIM-compressed `int32` miniSEED file being scanned as `float64`: the element type now comes from the file's encoding rather than from the empty array `headonly=True` returns (@atrabattoni).
- Fix the miniseed `ctype` argument being ignored: the reader always built interpolated time coordinates. The default is unchanged (@atrabattoni).
- Fix a data collection keyed by zero-padded codes — a SEED location such as `"00"`, say — reading back from netCDF as a sequence with its keys lost (@atrabattoni).
- Fix miniSEED scans being forced to a single process (@atrabattoni).
- Fix `sel` raising on a string-labelled coordinate: the overlap guard differenced the coordinate values, which numpy cannot do on strings, so `da.sel(phase="P")` failed before any selection happened. Label selection — scalar, list, reordering list and slice — now works, and `isel` with hard-coded positions is no longer the only option (@atrabattoni).
- Fix `sel` refusing an exact label look-up on a coordinate whose values are not sorted, such as a categorical axis like `["P", "S", "N"]`. The overlap guard covered every kind of selection, but only ordered look-ups — a slice, or `method="nearest"` and friends — need a sorted axis; naming a label does not. Those stay guarded, `da.sel(phase=["S", "P"])` now works and returns the labels in the requested order (@atrabattoni).

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