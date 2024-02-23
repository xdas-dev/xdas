import scipy.signal as sp

def medfilt(da, kernel_size):
    """
    Median filter data along given dimensions

    Parameters
    ----------
    da : DataArray
        The data to detrend.
    kernel_size : array_like, optional
        A scalar or an N-length list giving the size of the median filter window 
        in each dimension. Elements of kernel_size should be odd. If kernel_size 
        is a scalar, then this scalar is used as the size in each dimension. 
        Default size is 3 for each dimension.

    Returns
    -------
    DataArray
        The median filtered data.

    Examples
    --------
    This example is made to apply median filtering at a randomly generated dataarray
    by selecting a size of 7 for the median filtering along the time dimension
    and a size of 3 for the median filtering along the space dimension.
    The database is synthetic data.
    >>> from xdas.synthetics import generate
    >>> da = generate()
    >>> filtered_da = medfilt(da, [7,3])
    """
    data = sp.medfilt(da.values, kernel_size)
    return da.copy(data=data)