import numpy as np
import copy

from scipy.ndimage import maximum_filter

from astropy.io import fits
from astropy.stats import mad_std

from aklpyutils.utils_cubes import zero_edges

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Clumpfind
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def seeded_clumpfind(
        data=None,
        mask=None,
        levs=None,
        kern_coords=None,
        kern_id=None,
        corners=False,
        verbose=True,
):
    """Do a simple version of seeded clumpfind and return an assignment
    mask that looks like the map.
    
    data : input data, can be a map or a cube

    mask : only assign pixels within this mask

    levs : levels used to contour the data

    kern_coords : (x, y) or (x, y, z) tuple giving location of seeds

    kern_id : identifiers placed into assignment for that kernel

    corners : connect via diagonals (default False)

    verbose : print feedback

    """

    # Define connectivity
    footprint = simple_connectivity(ndim=data.ndim, corners=corners)
    
    # Sort the levels to descending order
    sorted_levs = (np.sort(levs))[::-1]
    min_lev_val = np.min(sorted_levs)
    a_low_value = min_lev_val-1.
    
    # Copy the data
    working_data = copy.deepcopy(data)
    
    # Replace nans with values below the minimum level
    nan_mask = (np.isfinite(working_data) == False)
    working_data[nan_mask] = a_low_value

    # Make a default mask
    if mask is None:
        mask = np.isfinite(working_data)*(working_data >= min_lev_val)

    # Zero the mask edges (in case) to prevent wrapping
    mask = zero_edges(mask)
        
    # Initialize the assignment mask with the kernels
    assign = np.zeros_like(working_data, dtype=int)
    if data.ndim == 2:        
        assign[kern_coords[1],kern_coords[0]] = kern_id
    if data.ndim == 3:        
        assign[kern_coords[2],kern_coords[1],kern_coords[0]] = kern_id

    # Only keep assignments in the mask
    assign = assign*mask

    # Loop over levels
    for ii, this_lev in enumerate(levs):

        if verbose:
            print("Level ", ii, this_lev)
            
        # Make a mask appropriate for this level
        this_mask = (working_data >= this_lev)*mask
        
        # Initialize counter and change marker
        counter = 0
        delta = -1.0

        # Make masks of assigned and unassigned pixels in this level
        assigned = (this_mask)*(assign != 0)
        to_assign = (this_mask)*(assign == 0)
        count_to_assign = np.sum(to_assign)

        # Continually apply a small maximum filter until the
        # assignment converges for this level        
        while delta != 0.0:
            
            counter += 1

            # These are assigned pixels in the current mask
            curr_assign = assign*assigned

            # Calculate local maximum assignment over the footprint
            local_maxval = maximum_filter(
                curr_assign, footprint=footprint,
                mode='constant', cval=a_low_value)

            # For pixels without a previous assignment, assign the
            # expanded assignment to the assignment map
            assign[to_assign] = local_maxval[to_assign]

            # Remake the masks showing which pixels have and which
            # need assignment
            assigned = (this_mask)*(assign != 0)
            to_assign = (this_mask)*(assign == 0)

            # Check for convergence
            old_count_to_assign = count_to_assign
            count_to_assign = np.sum(to_assign)
            delta = count_to_assign - old_count_to_assign

        if verbose:
            print("Rolls to converge: ", counter)
            
    # Return the assignment mask
    return(assign)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Connectivity kernels
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def simple_connectivity(
    ndim=3,
    skip_axes=None, 
    corners=False):
    """
    Return simple connectivity either with or without corners for n
    dimensions (default 3). Can suppress connectivity along one or
    more axes.

    ndim : 2 or 3 dimensionality of the data

    skip_axes : note one or more axes to skip

    corners : connect via corners? Default False

    """

    # Make an ndim array of 3 elements each
    connect = np.ones(np.ones(ndim,dtype=int)*3,dtype=int)
    
    if corners == False:
        indices = np.indices(connect.shape)
        has_a_one = np.sum((indices==1), axis=0) >= 1
        connect[has_a_one==False] = 0

    # Suppress connect along specified axes
    if skip_axes != None:
    
        if not isinstance(skip_axes, list):
            skip_axes = [skip_axes]

        ind = np.indices(connect.shape)
        for this_axis in skip_axes:
            blank = (ind[this_axis] == 0) + (ind[this_axis] == 2)
            connect[blank] = 0                        

    return(connect)
