# Release notes

## 0.2.9 (unreleased)

### New Features
- **Tile manifests carry an axis map.** Manifests gain two optional entries — a 1-D `axes` variable (which stored geometry axis each virtual axis presents) and a 0-d `source_ndim` — that make transpose-like operations (`transpose`, `permute_dims`, `matrix_transpose`, `swapaxes`, `moveaxis`), `expand_dims`/`stack`/`np.newaxis` at any position, `squeeze`, and integer indexing all lazy rewrites of the map. Engines now always receive exactly one slice per source axis, in source order — custom `load_tile` implementations no longer need any rank-padding logic — and freshly scanned manifests store neither entry (absent means identity), so existing files are unaffected (@atrabattoni).
- **Lazy numpy manipulation routines on tile arrays.** The `split` family (`split`, `array_split`, `vsplit`, `hsplit`, `dsplit`), the `stack` family (`stack`, `vstack`, `hstack`, `dstack`, `column_stack`) and `atleast_1d`/`atleast_2d`/`atleast_3d`, plus `roll`, `tile`, `delete`, and `append`/`insert` between tile arrays, now dispatch on `TileArray` as rewrites of the tile geometry and stay lazy. Cases the tile grid cannot express — axis fusion, element repetition, eager operands — keep materializing as before (@atrabattoni).
- **Explicit engine configuration.** The open functions (`open`, `open_dataarray`, `open_mfdataarray`, `open_mfdatatree`) now declare `engine`, `vtype` and `ctype` explicitly, and `engine` accepts a configured `xdas.io.Engine` instance as well as a name. Format-specific parameters (`overlaps`/`offset` for febus, `ignore_last_sample` for miniseed, `swapped_dims` for prodml, `tz` for terra15, `group` for the native format) are engine constructor parameters, validated up front; passing them next to the engine name keeps working and now raises a `TypeError` on misspelled or unsupported keywords instead of silently ignoring them (@atrabattoni).

### Breaking Changes
- Passing a bare read function as `engine` is no longer supported: subclass `xdas.io.Engine` instead (see the data-formats documentation). Combining a configured engine instance with `vtype`, `ctype` or extra engine keywords raises a `ValueError` (@atrabattoni).
- The miniseed `ctype` argument is now honored: it previously routed to an unused attribute and the reader always built interpolated time coordinates. The default is unchanged (@atrabattoni).
- **Tile-backed virtual arrays.** The new `xdas.tiles` module (ported from the 0.3 line) exposes multi-file archives as one lazy `TileArray`: positive-step slicing, concatenation (including along a new dimension), and whole-array reductions all stay lazy, and reads touch only the tiles a selection overlaps. The Silixa TDMS and MiniSEED engines now emit tile-backed data arrays, replacing the serialized-dask-graph fallback; Silixa reads gained time-axis push-down (@atrabattoni).
- Tile-backed data arrays round-trip through the native xdas netCDF format: the tile manifest is stored as a `__tiles__` sibling group (@atrabattoni).
- **Compact manifest path storage.** Tile manifests split the common directory of their source paths into a single 0-d `root` variable and keep only the root-relative rest per tile, and manifest strings are written as fixed-width char arrays instead of variable-length HDF5 strings: manifests shrink in memory and on disk and open faster, and relocating an archive amounts to editing one stored value. Arrays rooted in different directories still concatenate (the fusion is rebased under the deepest directory containing every root), and manifests written by earlier versions reopen unchanged (@atrabattoni).
- **Optional `tiles` vtype for every HDF5 engine.** `open_dataarray`/`open_mfdataarray` with `vtype="tiles"` back the returned array with a lazy `TileArray` instead of an HDF5 virtual source, for the asn, febus, terra15, apsensing, prodml, and native xdas engines. Saved tile views are directly readable by the 0.3 line. Custom engines add tile support by implementing the `load_tile(path, selection, **params)` static half of `xdas.io.Engine` (@atrabattoni).
- **Febus now defaults to `tiles`.** A tile view describes a Febus file as a single tile — the overlap trimming lives in the reader — whereas the HDF5 backing needs one mapping per block, so its manifest grew with the block count as well as the file count. Every other HDF5 engine still defaults to `hdf5` (@atrabattoni).
- `open_mfdataarray` no longer refuses more than 100 000 paths regardless of backing: the ceiling is now taken from the engine's resolved vtype and is far higher for `tiles`, which does not build one HDF5 mapping per file. The error explains the remaining limit — the scan holds one data array per file in memory until they are combined (@atrabattoni).

### Deprecations
- Writing dask-backed virtual arrays (`__dask_array__` attribute) is deprecated and emits a `FutureWarning`; existing files still open. No engine emits them any more: the tile-backed engines replace this mechanism (@atrabattoni).

### Bug Fixes
- Fix data collections holding more than one tile-backed data array being impossible to reopen. The tile manifest lives in a `__tiles__` sibling group, which the reader counted when deciding whether a group held a data array or a nested collection, so every tile-backed array looked one level too deep (@atrabattoni).

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