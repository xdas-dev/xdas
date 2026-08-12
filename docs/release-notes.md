# Release notes

## 0.2.9 (unreleased)

### Tiles Backend
- **Tile-backed virtual arrays.** The new `xdas.virtual.tiles` module exposes a file archive as one lazy `TileArray`: slicing, integer indexing, concatenation and the numpy manipulation routines stay lazy, reductions stream, and a read touches only the tiles the selection overlaps. Select it with `vtype="tiles"` on any HDF5 engine — Febus defaults to it, Silixa and the ObsPy formats always use it — and custom engines opt in by implementing `Engine.load_tile` (@atrabattoni).
- Tile-backed arrays round-trip through the native netCDF format as a compact `__tiles__` group, relocatable by editing the single root path of its header (@atrabattoni).
- Scanning scales to archives of any size: memory no longer grows with the file count and constant tile geometry is stored once — a 23-million-file archive opens in about 1 GB (@atrabattoni).
- **`xdas.sortby`** sorts a virtual data array along a dimension by coordinate value without reading any data (@atrabattoni).

### Atoms
- **Every atom has a user-friendly functional form, and every function an atom form.** This is the norm now rather than the exception. The function is the face one writes — `xdas.resample(da, 50.0)` applies straight away — and seeding it with `...` gives the atom behind it, `xdas.resample(..., 50.0)`, ready to compose with `>>` and to be streamed. So the same code runs on a slice in memory and, chunk by chunk, on an archive that does not fit in one. Ordinary numpy expressions join in under the same seed: `20 * np.log10(np.abs(atom))` extends the pipeline instead of computing (@atrabattoni).
- **A new default processing vocabulary, in physical units.** The top-level functions no longer take the parameters of the SciPy routine underneath — an output sample count, a decimation factor, a normalised frequency — but the quantities of the measurement: `xdas.resample(da, 50.0)` names a target rate in Hz, `xdas.filter(da, (1.0, 10.0))` its corner frequencies, `xdas.medfilt(da, {"time": 0.5})` its kernel in seconds. A parameter then keeps its meaning when the sampling rate changes, which is what lets one pipeline serve whatever it is given. The new set is `xdas.filter`, `xdas.resample`, `xdas.decimate`, `xdas.integrate`, `xdas.differentiate`, `xdas.stft`, `xdas.detrend`, `xdas.taper`, `xdas.hilbert`, `xdas.medfilt`, `xdas.sliding_mean_removal`, `xdas.rechunk`, and — for picking — `xdas.annotate`, `xdas.trigger` and `xdas.pick`. The SciPy-shaped functions stay in `xdas.signal` (@atrabattoni).
- **Gap-aware chunked processing.** Stateful atoms judge the seams of their input: state carries across continuous chunks and is flushed and restarted at gaps and rate changes, and eager calls split gappy input the same way — so a result no longer depends on how the data was chunked. The new `flush()` lifecycle drains buffered tails, atoms that cannot answer chunk by chunk now raise instead of answering wrong, and `xdas.testing.assert_chunk_invariant` asserts the whole property on a pipeline of your own (@atrabattoni).
- **`process()` on every atom, with source and sink auto-dispatch.** `pipeline.process(source, out=...)` infers both ends — an array, a virtual array, a path, a `DataCollection`, `xdas.watch(dir)` or a ZeroMQ address in; a directory, a `.csv`, an address, a writer or `None` out — so one line covers everything from a slice in memory to a growing archive. Walking a collection, each leaf's result is labelled with its tree path, and an eager call or an accumulation beyond the `"memory_limit"` configuration entry (8 GiB by default) raises rather than filling the machine (@atrabattoni).
- **Picking, end to end.** `Annotate` (replacing `MLPicker`) drives a SeisBench model with everything its weight set declares — window overlap, stacking, blinding, preprocessing — and overlaps GPU compute with transfers; `Trigger` gains per-phase thresholds and no longer loses the last pick of a record; `Picker(model)` assembles the whole chain from the weight set, so `xdas.pick(dc, model)` turns a network tree of waveforms into one flat pick table (@atrabattoni).
- **The exact machine-parameter atoms become an expert layer.** `LFilter`, `SOSFilter`, `DownSample`, `UpSample` and the new fused `Polyphase` move to `xdas.atoms.kernel`. They remain public and importable from `xdas.atoms`, but they are no longer what one reaches for: the vocabulary above designs them from the data at the first call. Resampling rides `Polyphase`, which is 2.6–8.7× faster than the chain it replaces (@atrabattoni).

### Seismological Data
- **The `obspy` engine**, named for the library rather than for a format: decoding is `obspy.read`, so miniSEED, SAC, GSE2, SEG-2 and everything else ObsPy supports goes through it. Each contiguous `Trace` becomes one lazy `DataArray` and the collection mirrors the `Stream`, nested `network / station / location / channel`; files the miniseed engine rejected (two sampling rates, duplicated ids, interleaved acquisitions) are now readable (@atrabattoni).
- **`xdas.stack`** collapses a level of a collection into an array dimension — `xd.stack(dc, "channel")` turns each station's traces into one `(channel, time)` array — lazily on virtual arrays, with `tolerance` snapping near-identical sampling grids and `join="inner"/"outer"` handling leaves that disagree (@atrabattoni).
- **`xdas.trim_overlaps`** resolves the overlaps of a data array or collection by dropping the duplicated samples, keeping the later copy by default or the earlier one with `keep="first"`, at the manifest level so lazy arrays stay lazy (@atrabattoni).
- **`DataCollection.select`**, with `obspy.Stream.select` semantics: `dc.select(station="SX00*", channel="HH?")` (@atrabattoni).

### Improvements
- **Explicit engine configuration.** Every open function declares `engine`, `vtype` and `ctype`, and `engine` also accepts a configured `xdas.io.Engine` instance. Format-specific parameters are engine constructor arguments, validated up front — a misspelled keyword now raises instead of being silently ignored (@atrabattoni).
- **Process pools for chunk ingress and egress.** `DataArrayLoader` and `DataArrayWriter` accept `pool="processes"`, which reads and writes chunks in worker processes instead of threads — on compressed archives, an order of magnitude faster. Ray is an optional dependency (`pip install xdas[ray]`); `pool="threads"` remains the default (@atrabattoni).
- `xdas.concat` can open a *new* dimension, checking that the inputs agree on their other coordinates and promoting the scalar ones that vary: stacking the components of a station is `xd.concat(traces, "channel")` (@atrabattoni).
- `sel` works on string and categorical coordinates: exact labels, lists and reordering no longer require a sorted axis (@atrabattoni).
- Acquisitions interleaved in time now group by compatibility, one array each, instead of splitting at every alternation (@atrabattoni).
- Saving and opening a data collection is linear in its size again (#81): saving 1300 events went from ~53 min to ~35 s (@atrabattoni).
- `simplify` runs in linear time whatever the number of gaps (@atrabattoni).
- When no engine can open a file, the error now lists what each engine that recognised it said (@atrabattoni).

### Deprecations
- `MLPicker` and `xdas.mlpicker` are deprecated in favour of `Annotate` and `Picker`; they remain as aliases until 0.4 (@atrabattoni).

### Breaking Changes
- Python 3.10 support is dropped and the numpy requirement is raised to 2.3 (@atrabattoni).
- **Dask virtualization is removed**, along with the `xdas.dask` module. A Dask array remains valid `DataArray` data — it is now computed on write — but `virtual=True` rejects it (@atrabattoni).
- **Opening a seismological file without naming an engine returns a nested collection, not a stacked array.** `xd.open(file)` no longer guesses that the traces of a file are synchronized; `xd.concat(traces, "channel")` is the one-liner back, and `engine="miniseed"` still gives the old shape (@atrabattoni).
- `DataCollection.query` raises a `KeyError` on an indexer naming no level of the collection, instead of silently returning everything unchanged (@atrabattoni).

### Bug Fixes
- Fix a STEIM-compressed `int32` miniSEED file being scanned as `float64`, the miniseed `ctype` argument being ignored, and miniSEED scans being forced to a single process (@atrabattoni).
- Fix chunked `DownSample` dropping its trailing samples when the stream length is not a multiple of the factor (@atrabattoni).
- Fix resampling losing track of the *other* coordinates of the dimension it resamples: decimating a DAS acquisition left its `station` coordinate at full length, labelling every lane with the code of the lane at its own index. Labels now follow the samples they name (@atrabattoni).
- Fix `DataCollection` coercing a `pandas.DataFrame` leaf into a broken `DataArray`: a table is now a leaf of its own kind (@atrabattoni).
- Fix a data collection keyed by zero-padded codes — a SEED location such as `"00"` — reading back from netCDF with its keys lost (@atrabattoni).
- Fix a directory sink joining its chunks along the wrong dimension when the pipeline's output does not lead with the chunked one (@atrabattoni).
- Fix the numpy dispatch overriding explicitly passed arguments with its registered defaults: `np.cumsum(da, 0)` accumulated along the last axis whatever the caller said (@atrabattoni).

### Refactoring
- Custom engines must subclass `xdas.io.Engine`: passing a bare read function as `engine` now raises a `TypeError` (@atrabattoni).
- `Trigger` moved to `xdas.atoms.detect` with a lowercase twin `xdas.trigger`; the `xdas.trigger` module remains importable and `find_picks` is unchanged. Its `dim` default is now `"time"` instead of `"last"` (@atrabattoni).

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