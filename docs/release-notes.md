# Release notes

## 0.2.9 (unreleased)

### New Features
- **Tile-backed virtual arrays.** The new `xdas.virtual.tiles` module exposes a file archive as one lazy `TileArray`: slicing (any step), integer indexing, `np.newaxis`, concatenation and the numpy manipulation routines stay lazy, reductions stream one tile row at a time, and a read touches only the tiles the selection overlaps. Select it with `vtype="tiles"` on any HDF5 engine — Febus defaults to it, Silixa and the ObsPy formats always use it. Tile-backed arrays round-trip through the native netCDF format as a compact `__tiles__` group, relocatable by editing the single root path of its header. Custom engines opt in by implementing `Engine.load_tile(path, selection, **params)` (@atrabattoni).
- **The `obspy` engine**, named for the library rather than for a format: decoding is `obspy.read`, so miniSEED, SAC, GSE2, SEG-2 and everything else ObsPy supports goes through it. Each contiguous `Trace` becomes one lazy `DataArray` and the collection mirrors the `Stream`, nested `network / station / location / channel`; files the miniseed engine rejected (two sampling rates, duplicated ids, interleaved acquisitions) are now readable (@atrabattoni).
- **`xdas.trim_overlaps`.** Resolve the overlaps of a data array or collection by dropping the duplicated samples, keeping the later copy by default (ObsPy's `merge(method=1, interpolation_samples=0)`) or the earlier one with `keep="first"`. Trimming stays at the manifest level, so a lazy array stays lazy; `xdas.split(da, "overlaps")` still keeps every copy (@atrabattoni).
- **`DataCollection.select`**, with `obspy.Stream.select` semantics: `dc.select(station="SX00*", channel="HH?")`. It is an alias of `query`, which now applies an indexer wherever its level sits in the tree rather than only at the root (@atrabattoni).
- **`xdas.stack`.** Collapse a level of a collection into an array dimension — the inverse of `combine_by_coords`, which concatenates *along* an existing one: `xd.stack(dc, "channel")` turns each station's traces into one `(channel, time)` array, keyed by the level's own keys. The new dimension is named after the level it collapsed, so nothing is renamed behind your back (`dim=` chooses otherwise), everything below the level is merged in lock-step, and tile-backed leaves stay tile-backed, so a collection of virtual arrays stacks without reading a byte. Leaves that do not share their other coordinates raise, naming what disagreed; `join="inner"` trims them to what they all have and `join="outer"` pads the rest with NaN. Agreement is judged on the sampling grid, not on tie points: leaves that describe one grid to within `tolerance` — by default a hundredth of a sample, and only where a nominal sampling interval is declared — are snapped onto the first leaf's coordinate, so three components whose start times were rounded a nanosecond apart stack without a join. `tolerance=False` restores strict equality, and a join that would interleave two grids into an array longer than their span can hold now raises instead of returning it (@atrabattoni).

- **`process()` with source and sink auto-dispatch.** `process()` is now a method on every atom and a dispatch boundary: `pipeline.process(da, out="results/")` infers both ends. Sources dispatch on the input value — an in-memory `DataArray` runs eagerly (or chunk by chunk with `chunks=`), a virtual one streams through a loader with storage-aligned `chunks="auto"`, a path/directory/glob opens with `open_mfdataarray`, `"tcp://..."` subscribes over ZeroMQ, and any iterable of chunks (a generator, a loader) is consumed as is. Sinks dispatch on the out spec crossed with the first output chunk, so writer creation is deferred to what the pipeline actually emits: a directory stores `DataArray` chunks joined along the chunked dimension (or an SDS archive for `Stream` chunks), `*.csv` appends DataFrames, `"tcp://..."` publishes, `out=None` accumulates and returns the joined result, and a configured writer instance passes through. A chunked source with discontinuities announces them upfront — one warning with the count, read off the source coordinate before any data. The historical `process(atom, loader, writer)` form keeps working unchanged (@atrabattoni).
- **`xdas.watch` and unbounded sources.** Realtime is now *named*: `pipeline.process(xd.watch("/incoming", engine=...), out=...)` watches a directory forever, and a bare directory path always means "process what is there". Unbounded sources (watch, ZMQ subscriptions) get streaming semantics — throughput-style progress, a clean `KeyboardInterrupt` that flushes the pipeline and returns the writer result, `until=` to stop at a coordinate value (inclusive, truncating the last chunk), and a warning at each seam as it arrives, since a realtime source cannot be inspected upfront (@atrabattoni).
- **Memory guards.** The new `"memory_limit"` configuration entry (default 8 GiB) makes footguns loud: an eager call on a huge virtual array and an `out=None` accumulation that outgrows the limit both raise with the estimated size and a pointer to `.process(out=...)` (@atrabattoni).
- **`STFT`.** The spectral vocabulary joins the task-atom route: `STFT` streams complex frames with window length and hop in physical units — both are snapped, the window to the next fast FFT size of the target and the hop to a whole sample count — with an expert `nfft` to zero-pad and a `scaling=` of `"spectrum"` or `"psd"`, so `np.abs(stft)**2` composes to an exact spectrogram. Only fully computable frames are ever emitted: the unconsumed tail is buffered across chunks and dropped at gaps, so chunked processing emits exactly the eager frames and no frame ever spans a discontinuity. Built on `scipy.signal.ShortTimeFFT` internally, with the `xdas.stft` function form at the top level (@atrabattoni).
- The `xdas.fft` functions (`fft`, `rfft`, `ifft`, `irfft`) now declare whole-record semantics: used as atoms in a chunked pipeline they raise along the transformed dimension instead of silently computing one transform per chunk. Transforming along another dimension than the chunked one keeps working (@atrabattoni).
- **`xdas.testing.assert_chunk_invariant`.** The chunk-safety story in one call: run a pipeline eagerly and streamed and assert the two agree — values, coordinates and all. The invariant is quantified over *cuts* (the same stream re-chunked at derived non-divisor sizes, so boundaries land elsewhere) and over *gaps* (`xdas.testing.inject_gaps` places real discontinuities in the input first, so seam resets are exercised at boundaries that do not line up with them). It is both the CI harness for every stateful atom xdas ships and the tool to run on your own pipelines before trusting them chunked (@atrabattoni).
- **Continuous-run semantics.** Stateful atoms now understand gaps: every atom judges the seams of its own input stream from the chunk coordinates — a continuous chunk carries state across, a gap or rate change flushes the previous run and restarts (redesigning coefficients on rate changes), an overlap raises, and the `on_discontinuity="reset"|"raise"` policy makes strict runs opt-in. Eager calls auto-split gappy input into runs, process each with a fresh state and re-join the outputs with the gaps kept in the coordinates, so filters never cross discontinuities — and the split is announced: a warning states how many discontinuities the source has and that state is flushed and reset at each. Sequence collections fold through the same seam-aware machinery — `concat(atom(split(da, anywhere)))` equals `atom(da)` for arbitrary split points — and mapping collections map over their leaves. Chunked processing along a dimension now requires a regular coordinate (a declared `sampling_interval`) on it, raising with a pointer to `to_regular()` instead of silently carrying state across unverifiable seams (@atrabattoni).
- **`flush()` lifecycle and the transducer contract.** `call()` now maps one input chunk to zero or more output chunks, and the new `Atom.flush()` drains what remains: buffering atoms emit their tail at the end of the stream, at every seam and at the end of every eager call (`Sequential.flush` cascades codec-drain style, and `process()` drains the pipeline at the end of the stream). Reductions fall out for free: a `call()` that accumulates and returns nothing plus a `flush()` that emits the result gives constant-memory streaming statistics. `Atom.iter_chunks(source)` exposes the whole machinery as a plain generator — the manual chunk loop with buffering, seams and flushing handled inside — and writers now silently drop empty chunks (@atrabattoni).
- **`Rechunk` kernel atom.** `Rechunk({"time": n})` (and its function form `xdas.rechunk`) merges and splits streaming chunks to a target size in samples — a performance knob, e.g. to restore a workable cadence after a decimation shrank the chunks — without ever merging across a discontinuity (@atrabattoni).
- Chunked `DownSample` (and thus the stateful decimation path) no longer drops its trailing samples when the stream length is not a multiple of the factor: the buffered remainder is emitted by the new `flush()` lifecycle. The `"first"`/`"last"` dimension aliases are now resolved against the data before being compared with the chunked dimension, so a kernel built with its documented default no longer skips allocating its seam state, and `UpSample` handles one-sample chunks (@atrabattoni).
- **Task atoms with physical units.** A new public processing vocabulary where every parameter keeps its meaning when the sampling rate changes: `Filter` (one atom for all bands — a `(low, high)` corner pair in Hz with `None` opening one end, `ftype="iir"/"fir"`, `zerophase`), `Decimate` and `Resample` (target rate in Hz, both riding the polyphase kernel — the filter-at-full-rate-then-discard chain is never taken), `Integrate` and `Differentiate` (chunk-correct, carrying state across seams), plus whole-record `detrend`, `taper`, `hilbert`, `sliding_mean_removal` and `medfilt` (kernel lengths now in seconds/meters), each refusing chunked execution along its working dimension instead of silently answering wrong. Task atoms default to `dim="time"` and live in `xdas.atoms.tasks` (@atrabattoni).
- **Function forms at the top level.** Every task atom generates a top-level function with a synthesized signature and docstring: `xdas.decimate(da, 50.0)` applies eagerly, `xdas.decimate(..., 50.0)` returns the atom, and passing an atom extends a pipeline — so the same code runs eagerly on a slice and chunked on an archive by seeding it with `...` (@atrabattoni).
- **Polyphase resampling in a kernel layer.** The exact machine-parameter atoms move to the expert layer `xdas.atoms.kernel` (`LFilter`, `SOSFilter`, `DownSample`, `UpSample`, still importable from `xdas.atoms`), joined by the new `Polyphase` kernel: upsample, FIR filter and downsample fused into a single `scipy.signal.upfirdn` pass that computes only the output samples surviving the decimation and never materialises the zero-stuffed signal (which for `up=4` allocated a four times larger, mostly-zero array). `FIRFilter` is born with `up=`/`down=` and `ResamplePoly` rides it, so the upsample/filter/downsample trio collapses to one child atom — on a 254 MiB chunk that is 2.6× on a decimation by two along distance and 8.7× on a 62.5 → 50 Hz resampling. The taps are cast down to the data precision, so float32 stays float32 instead of being promoted by the filter; a target rate the coordinate resolution cannot represent exactly (100 Hz → 30 Hz is 10/3 ns per sample) declares its residual drift as jitter instead of rejecting its own sampling interval (@atrabattoni).
- **`>>` composition and operator tracing.** Atoms compose into pipelines with `>>`/`>>=` (bare callables auto-wrap, `da >> atom` applies), and ordinary numpy expressions trace under the `...` seed: `20 * np.log10(np.abs(atom))` appends `absolute → log10 → multiply` to the pipeline instead of computing. Tracing covers ufuncs exactly — a traced expression involving two atoms (fan-in) raises at the line that wrote it rather than silently computing. Composition has value semantics: passing a `Sequential` to an atomized function returns a new extended pipeline instead of mutating (and aliasing) the input — the mutating form also returned `None`, breaking chained composition. `xdas.atoms.as_function` generates the function form of any atom class, and atoms gain `fresh()` (a stateless clone whose config is shared by reference) while `initialized` now recurses into nested atoms (@atrabattoni).

### Improvements
- **Scanning scales to archives of any size.** `open_mfdataarray` fuses scan results every 100 000 files instead of holding one data array per file, so memory no longer grows with the archive. With `vtype="tiles"` the file-count ceiling is lifted and constant tile geometry costs one element instead of one per tile: a 23-million-tile archive opens in 1.11 GB instead of 1.67 GB (@atrabattoni).
- **Explicit engine configuration.** Every open function declares `engine`, `vtype` and `ctype`, and `engine` also accepts a configured `xdas.io.Engine` instance. Format-specific parameters (`overlaps`/`offset` for febus, `ignore_last_sample` for miniseed, `swapped_dims` for prodml, `tz` for terra15, `group` for the native format) are engine constructor arguments, validated up front — a misspelled keyword now raises instead of being silently ignored (@atrabattoni).
- **Process pools for chunk ingress and egress.** `DataArrayLoader` and `DataArrayWriter` accept `pool="processes"`, which reads and writes chunks in worker processes instead of threads: compressed HDF5 decodes and compresses under the global HDF5 lock, so extra *threads* only contend, while processes each hold their own lock. What crosses to a worker on the read side is the manifest of the chunk — a sliced virtual array, kilobytes — so each worker reads its own files, and the loaded chunk comes back through Ray's shared-memory object store: written once by the worker, mapped zero-copy by the parent, arriving read-only (the immutability convention atoms already follow). End to end on a compressed ZFP archive at 16 workers, ingest goes from 137 to 1378 MiB/s and egress from 151 to 1556. Ray is an optional dependency (`pip install xdas[ray]`); `pool="threads"` remains the default (@atrabattoni).
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
- `xdas.concat` opening a *new* dimension over `VirtualSource`-backed arrays no longer raises `TypeError: only VirtualSource object can be provided`. Whether the result can stay virtual was decided before `expand_dims`, which no virtual source can follow — a stack of sources is a longer axis, never an extra one — so a `VirtualStack` was promised over arrays that had already been loaded (@atrabattoni).
- Fix a data collection keyed by zero-padded codes — a SEED location such as `"00"`, say — reading back from netCDF as a sequence with its keys lost (@atrabattoni).
- Fix miniSEED scans being forced to a single process (@atrabattoni).
- Fix `sel` raising on a string-labelled coordinate: the overlap guard differenced the coordinate values, which numpy cannot do on strings, so `da.sel(phase="P")` failed before any selection happened. Label selection — scalar, list, reordering list and slice — now works, and `isel` with hard-coded positions is no longer the only option (@atrabattoni).
- Fix `DataCollection` coercing a `pandas.DataFrame` leaf into a `DataArray`, producing collections whose leaves raised on `repr`. A table is now a leaf of its own kind (@atrabattoni).
- Fix a directory sink joining its chunks along the wrong dimension when the pipeline's output does not lead with the chunked one — `(distance, time)` chunks written from a time-chunked source were stacked along `distance`. `DataArrayWriter` now takes the dimension as a `dim` argument (still `"first"` when left unsaid) (@atrabattoni).
- Fix the numpy dispatch overriding explicitly passed arguments with its registered defaults: `np.cumsum(da, 0)` accumulated along the last axis whatever the caller said. Registered defaults now fill in only when the caller says nothing (@atrabattoni).
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