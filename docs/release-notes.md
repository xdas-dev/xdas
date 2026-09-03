# Release notes

## Unreleased

### Atoms
- **`xdas.stalta`**, the short-term over long-term average characteristic
  function, feeding the existing `Trigger` so that detection no longer needs a
  model. Both windows are given in the units of the dimension, `mode="causal"`
  (default) or `"centered"` places them relative to the sample they produce,
  and the atom is stateful: chunked processing returns what a single call on
  the whole record returns, `flush()` releasing the tail `"centered"` mode
  holds back (@amaggi).

## 0.2.9

### Tiles Backend
- **Tile-backed virtual arrays.** The new `xdas.virtual.tiles` module exposes a file archive as one lazy `TileArray`: slicing, concatenation and the numpy routines stay lazy, reductions stream, and a read touches only the tiles it overlaps. Select it with `vtype="tiles"` on any HDF5 engine — the default for Febus, always used by Silixa and ObsPy — or implement `Engine.load_tile` (@atrabattoni).
- Tile-backed arrays round-trip through the native netCDF format as a compact `__tiles__` group, relocatable by editing its header's single root path (@atrabattoni).
- Scanning no longer grows in memory with the file count: a 23-million-file archive opens in ~1 GB (@atrabattoni).
- **`xdas.sortby`** sorts a virtual data array along a dimension by coordinate value without reading data (@atrabattoni).

### Atoms
- **Every atom has a functional form, and every function an atom form.** `xdas.resample(da, 50.0)` applies straight away; `xdas.resample(..., 50.0)`, seeded with `...`, is the atom behind it, ready to compose with `>>` and to be streamed. Numpy expressions extend a pipeline under the same seed: `20 * np.log10(np.abs(atom))` (@atrabattoni).
- **A processing vocabulary in physical units.** The top-level functions take the quantities of the measurement — a rate in Hz, corner frequencies, a kernel in seconds — so a parameter survives a change of sampling rate: `xdas.filter`, `resample`, `integrate`, `differentiate`, `stft`, `detrend`, `taper`, `hilbert`, `medfilt`, `sliding_mean_removal`, `rechunk`, `annotate`, `trigger` and `pick`. The SciPy-shaped ones stay in `xdas.signal` (@atrabattoni).
- **One `xdas.resample` for every way of changing a sampling rate**, `xdas.decimate` gone into it. The target is a `rate`, an `interval` or an `up`/`down` ratio, and `method="fir"` (default), `"iir"` or `"fft"` chooses how — `"fft"` alone reaches an arbitrary interval. The solver picks the simplest ratio within `tolerance`, not the closest; `snap=True` stays on the original grid (@atrabattoni).
- **Gap-aware chunked processing.** State carries across continuous chunks and is flushed at gaps and rate changes, eager calls splitting gappy input the same way, so a result no longer depends on the chunking. `flush()` drains buffered tails, and `xdas.testing.assert_chunk_invariant` asserts the property on a pipeline of your own (@atrabattoni).
- **`process()` on every atom, with source and sink auto-dispatch.** `pipeline.process(source, out=...)` infers both ends — an array, a path, a `DataCollection`, `xdas.watch(dir)` or a ZeroMQ address in; a directory, a `.csv`, an address or a writer out. An accumulation beyond the `"memory_limit"` entry raises rather than filling the machine (@atrabattoni).
- **Picking, end to end.** `Annotate` (replacing `MLPicker`) drives a SeisBench model with everything its weight set declares (overlap, stacking, blinding, preprocessing), overlapping GPU compute with transfers; `Trigger` gains per-phase thresholds; `xdas.pick(dc, model)` turns a tree of waveforms into one flat pick table (@atrabattoni).
- **The machine-parameter atoms become an expert layer.** `LFilter`, `SOSFilter`, `DownSample`, `UpSample` and the new fused `Polyphase` move to `xdas.atoms.kernel`, still importable from `xdas.atoms`: the vocabulary above designs them from the data. Resampling rides `Polyphase`, 2.6–8.7× faster than the chain it replaces (@atrabattoni).

### Seismological Data
- **The `obspy` engine**, named for the library, not a format: decoding is `obspy.read`, so miniSEED, SAC, GSE2, SEG-2 and the rest go through it. Each contiguous `Trace` becomes one lazy `DataArray`, nested `network / station / location / channel`; files the miniseed engine rejected are now readable (@atrabattoni).
- **`xdas.stack`** collapses a level of a collection into an array dimension — `xd.stack(dc, "channel")` — lazily, with `tolerance` snapping near-identical sampling grids and `join="inner"/"outer"` for leaves that disagree (@atrabattoni).
- **`xdas.trim_overlaps`** drops the duplicated samples of an array or collection, keeping the later copy, or the earlier with `keep="first"`, at the manifest level so lazy arrays stay lazy (@atrabattoni).
- **`DataCollection.select`**, with `obspy.Stream.select` semantics: `dc.select(station="SX00*", channel="HH?")` (@atrabattoni).

### Improvements
- **A data collection prints as a table of its leaves**: one row per leaf with the keys that address it, its shape and its size in memory, plus a summary line (@atrabattoni).
- **An atom prints as one line, and a pipeline as one line per stage**, showing the parameters its constructor was given with the defaults left out — `Filter(freq=(None, 10.0), ftype='fir')`. A model or a set of coefficients is summarised as `<PhaseNet>` or `<ndarray>` (@atrabattoni).
- **Explicit engine configuration.** Every open function declares `engine`, `vtype` and `ctype`, and `engine` also accepts a configured `xdas.io.Engine` instance. Format-specific parameters are constructor arguments, validated up front: a misspelled keyword now raises (@atrabattoni).
- **Process pools for chunk ingress and egress.** `DataArrayLoader` and `DataArrayWriter` accept `pool="processes"` — on compressed archives, an order of magnitude faster than threads. Chunks travel through shared memory and arrive read-only; `pool="threads"` remains the default (@atrabattoni).
- **Parallelism sized to the work, not to the machine.** Both were `os.cpu_count()`, which made small work slower. Threads are now sized against the array they are given, and a small scan stays in the calling process. `xdas.config` gains `"scan_workers"` beside `"n_workers"` (@atrabattoni).
- **Subscribing to a stream no longer races it.** A publisher answers each new subscription with a greeting, and `ZMQSubscriber.wait_until_subscribed()` returns when it arrives, so nothing published from then on is missed. A `timeout` makes both subscribers raise on a stream gone quiet, and `ZMQPublisher.wait_for_subscribers()` holds a replay back until its listeners have joined (@atrabattoni).
- `xdas.concat` can open a *new* dimension, checking the inputs agree elsewhere and promoting the scalar coordinates that vary: `xd.concat(traces, "channel")` (@atrabattoni).
- `sel` works on string and categorical coordinates: exact labels, lists and reordering no longer need a sorted axis (@atrabattoni).
- Acquisitions interleaved in time now group by compatibility, one array each, instead of splitting at every alternation (@atrabattoni).
- Saving and opening a data collection is linear in its size again (#81): 1300 events went from ~53 min to ~35 s (@atrabattoni).
- **`simplify` is linear in the number of gaps, and no longer asks a coordinate for precision it cannot hold**: a zero tolerance was a no-op on any rate that is not a whole number of ticks, the budget now carrying the half tick the coordinate rounds by (@atrabattoni).
- When no engine can open a file, the error lists what each engine that recognised it said (@atrabattoni).

### Deprecations
- `MLPicker` and `xdas.mlpicker` are deprecated in favour of `Annotate` and `Picker`; they remain as aliases until 0.4 (@atrabattoni).
- `ResamplePoly` is deprecated in favour of `Resample` (`method="fir"`, its `maxfactor` renamed `maxup`); it remains as a thin subclass until 0.3 (@atrabattoni).
- `Coordinate.isdense`, `isinterp` and `issampled` are deprecated in favour of `isinstance(coord, DenseCoordinate)` / `InterpCoordinate` / `SampledCoordinate`, and `AxisCoordinate.get_value(index)` in favour of `coord[index].values`; all remain as aliases (@atrabattoni).

### Breaking Changes
- Python 3.10 support is dropped and the numpy requirement is raised to 2.3 (@atrabattoni).
- **Dask virtualization is removed**, along with the `xdas.dask` module. A Dask array remains valid `DataArray` data — computed on write — but `virtual=True` rejects it (@atrabattoni).
- **Opening a seismological file without naming an engine returns a nested collection, not a stacked array.** `xd.concat(traces, "channel")` is the one-liner back, and `engine="miniseed"` still gives the old shape (@atrabattoni).
- **The innermost level of a collection is named `record`**, whatever it holds: `dc.query(trace=0)` becomes `dc.query(record=0)`. A collection saved before this reads back with its old level names (@atrabattoni).
- `DataCollection.query` raises a `KeyError` on an indexer naming no level of the collection, instead of silently returning everything unchanged (@atrabattoni).
- Custom engines must subclass `xdas.io.Engine`: passing a bare read function as `engine` now raises a `TypeError` (@atrabattoni).
- `Trigger` moved to `xdas.atoms.detect`, with a lowercase twin `xdas.trigger`; the module remains importable and `find_picks` is unchanged. Its `dim` default is now `"time"` instead of `"last"` (@atrabattoni).
- The `xdas.atoms.signal` module is gone, split into `xdas.atoms.kernel` and `xdas.atoms.operations`; its atoms are still importable from `xdas.atoms` (@atrabattoni).

### Bug Fixes
- **`get_discontinuities` reports the discontinuity, not the sample after it.** Since 0.2.5 every row was shifted one sample forward: `delta` gave the sampling interval instead of the jump, and `type` could never say `overlap`. `plot_availability` is fixed with it, and detection was never affected (@atrabattoni).
- **A coordinate is sorted when its values increase, not when it has no overlaps.** A smoothly decreasing axis was declared sorted and handed to a binary search that assumes otherwise. `get_split_indices` gains a `"reversals"` kind, where an unsorted slice is now cut (@atrabattoni).
- **ZeroMQ publishers and subscribers release their sockets**, until now held for the life of the process. Both ends gain `close()` and work as context managers (@atrabattoni).
- Fix `Trigger` losing the last pick of a record (@atrabattoni).
- Fix a file handle leaking on every failed TDMS auto-detection attempt (@atrabattoni).
- Fix a STEIM-compressed `int32` miniSEED file being scanned as `float64`, and miniSEED scans forced to a single process (@atrabattoni).
- Fix chunked `DownSample` dropping its trailing samples when the length is not a multiple of the factor (@atrabattoni).
- Fix resampling losing track of the *other* coordinates of the dimension it resamples: a DAS acquisition's `station` coordinate was left at full length, labelling every lane with the code of the lane at its own index (@atrabattoni).
- Fix `DataCollection` coercing a `pandas.DataFrame` leaf into a broken `DataArray`: a table is now a leaf of its own kind (@atrabattoni).
- Fix a data collection keyed by zero-padded codes (a SEED location such as `"00"`) losing them through netCDF (@atrabattoni).
- Fix `ScalarCoordinate` raising `TypeError` on `+`/`-`, unlike every other coordinate class (@atrabattoni).
- Fix a directory sink joining its chunks along the wrong dimension when the output does not lead with the chunked dimension (@atrabattoni).
- Fix the numpy dispatch overriding explicitly passed arguments: `np.cumsum(da, 0)` accumulated along the last axis (@atrabattoni).

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