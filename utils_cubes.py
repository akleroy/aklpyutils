import numpy as np
import scipy.stats
import copy

from astropy.io import fits
from astropy.stats import mad_std

def zero_edges(data, val=0):
    """
    Set the edges of a map or cube to some value (default 0).
    """
    
    if data.ndim == 2:
        data[0,:] = val
        data[-1,:] = val
        data[:,0] = val
        data[:,-1] = val

    if data.ndim == 3:
        data[0,:,:] = val
        data[-1,:,:] = val
        data[:,0,:] = val
        data[:,-1,:] = val
        data[:,:.0] = val
        data[:,:,-1] = val

    return(data)
