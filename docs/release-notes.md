# Release notes

## 0.2.6

### New features
- Add `xdas.open` that automatically infer which `xdas.open_*` function to use (@atrabattoni, @yetinam).
- Add automatic engine detection to every `xdas.open_*` functions (@atrabattoni, @yetinam).
- Add `pathlib.Path` support as input for all Xdas file-related functions and methods (@atrabattoni).

### Improvements
- New `io.Engine` backend system to register different file formats (@atrabattoni).
- Most `xdas.open_*` functions now have a `create_dirs` argument to creating intermediate directories if necessary (@aurelienfalco).
- Make `open_mfdataarray` raise `RuntimeError` when opening all files fails (@asladen). 
- Add "prodml" engine (@atrabattoni) and make "optasense" and "sintela" aliases of it (@atrabattoni). 

### Bugs Fixed
- Fix **memory accumulation** when slicing multiple times data arrays, e.g. when using atoms (@atrabattoni) (@atrabattoni).
- Fix **non-terminating loaders and writers** in `xdas.processing` (@atrabattoni).
- Fix/improve distance handling for: "apsensing", "febus", "optasense", "silixa", and "sintela" (@atrabattoni).
- Add dim swapping handling for the "prodml" based engines with the `swapped_dims=False` kwarg (@atrabattoni).
- Fix ASN ROI handling (@asladen).

## 0.2.5
- Add SampleCoordinate for more SEED-like coordinates and refactor the coordinate backend (@atrabattoni).
- Add `xdas.picking.tapered_selection` to extract windows around picks (@atrabattoni).
- Add `create_dirs` to `.to_netcdf` methods to create intermediate directories (@aurelienfalco).
- Add support for multiple ROI for ASN engine (@martijnende).
- `tolerance` can now be passed as seconds for datetime64 coordinates (@martijnende, @atrabattoni)
- Add suppport for python 3.14, numpy 2.4 and obspy 1.4.2 incompatibilities and add `xdas.__version__` (@atrabattoni).

## 0.2.4
- Add StreamWriter to write long time series to miniSEED (@marbail).
- Fix OptaSense engine wrong axis attribution (@smouellet).
- Fix ASN (OptoDAS) engine: handling of roiDec (@AndresLaurine).
- Fix nan handling for several methods (@ClaudioStrumia).
- Fix `InterpCoordinate.get_availabilities` (@AMordret).

## 0.2.3
- Fix Febus engine (round timestamps to closest us).
- Faster `xdas.concatenate` (faster linking for efficient reading of Febus files).

## 0.2.2
- Add support for python 3.13
- Fix bugs and dependencies issues

## 0.2.1
- Add `xdas.signal.stft`.
- Add inverse Fourier transforms `xdas.fft.ifft` and `xdas.fft.irfft`.
- Add support for APSensing format.
- Improve overlap error message.
- Fix decimation of freshly opened multi-file datasets.
- Fix `zerophase` keyword argument for `xdas.signal.filter`.
- Fix applying fft functions in presence of non-dimensional coordinates.

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