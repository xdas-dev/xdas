# Release notes

## 0.2.1
- Add `xdas.signal.stft`. 
- Add inverse fourrier transforms `xdas.fft.ifft` and `xdas.fft.irfft`.
- Add support for APSensing format.
- Fix decimation of freshly openned multi-file datasets.
- Fix zerophase kwarg for `xdas.signal.filter`.
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