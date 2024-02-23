import pytest 

from xdas.synthetics import generate

def medfilt(self): 
    da = generate()
    da_filtered = medfilt(da, [1,1])

    assert np.allclose(da_filtered, da) 
    with pytest.raises(ValueError):  # check it raise the correct error
        my_function(-1)