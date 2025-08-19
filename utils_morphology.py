import numpy as np
import copy

from scipy.ndimage import maximum_filter, watershed_ift, \
    binary_dilation, label
from skimage.segmentation import watershed

from astropy.io import fits
from astropy.stats import mad_std

import matplotlib.pyplot as plt

from aklpyutils.utils_cubes import zero_edges

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Mask Manipulation
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def seeded_mask(
        data=None, kern_coords=None, thresh=None,        
        prior_mask=None, support=None,
        verbose=True):
        
    mask = np.zeros_like(data, dtype=bool)

    if prior_mask is None:
        prior_mask = np.isfinite(data)

    if support is not None:
        if data.ndim == 2:
            y_ind,x_ind = np.indices((data.shape[0],data.shape[1]))*1.0
        else:
            z_ind, y_ind,x_ind = \
                np.indices((data.shape[0],data.shape[1],data.shape[2]))*1.0
            
    for ii in np.arange(len(kern_coords[0])):
        print("Kernel ... ", ii)
        
        x = kern_coords[0][ii]
        y = kern_coords[1][ii]

        if len(thresh) > 1:
            this_thresh = thresh[ii]
        else:
            this_thresh = thresh

        if support is not None:
            if len(support) > 1:
                this_support = support[ii]
            else:
                this_support = support
        else:
            this_support = None
            
        if data.ndim == 3:
            z = kern_coords[2][ii]

        # If a support kernel is provided then use this to build the
        # prior mask.
        
        if this_support is not None:
            
            this_mask = np.zeros_like(data, dtype=bool)
            if data.ndim == 2:
                this_mask[y,x] = True
            if data.ndim == 3:
                this_mask[z,y,x] = True

            if data.ndim == 2:
                this_mask = ((x_ind - x)**2 + \
                             (y_ind - y)**2) <= this_support**2
            else:
                this_mask = ((x_ind - x)**2 + \
                             (y_ind - y)**2 + \
                             (z_ind - z)**2) <= this_support**2                

            this_mask *= prior_mask

        else:

            this_mask = prior_mask

        # Identify and keep the region that contains the local maximum
        regions, regct = label(
            this_mask*(data >= this_thresh))

        if data.ndim == 2:
            this_reg = regions[y,x]
        if data.ndim == 3:
            this_reg = regions[z,y,x]

        if this_reg > 0:
            this_mask = (regions == this_reg)

        # Join to existing mask
        mask = np.logical_or(this_mask, mask)

        if verbose:
            #if (ii % 100) == 0:
            if False:
                plt.clf()
                plt.imshow(mask, origin='lower')
                plt.show()
        
    return(mask)
    
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Clumpfind
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def seeded_clumpfind(
        data=None, levs=None,
        kern_coords=None, kern_id=None,
        mask=None, corners=False, verbose=True,
):
    """Do a simple version of seeded clumpfind and return an assignment
    mask that looks like the map.
    
    Parameters:

    data : input data, can be a map or a cube

    levs : levels used to contour the data

    Keywords:

    mask : only assign pixels within this mask

    kern_coords : (x, y) or (x, y, z) tuple giving location of seeds

    kern_id : identifiers placed into assignment for that kernel

    corners : connect via diagonals (default False)

    verbose : print feedback

    """

    # Define connectivity
    connectivity = simple_connectivity(ndim=data.ndim, corners=corners)
    
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

            # Calculate local maximum assignment over the connectivity
            local_maxval = maximum_filter(
                curr_assign, footprint=connectivity,
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
# Watershed
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def seeded_watershed(
        data=None,
        mask=None,
        levs=None,
        kern_coords=None,
        kern_id=None,
        corners=False,
        compactness=False,
        verbose=True,
):
    """Do a seeded watershed and return an assignment mask that looks like
    the map.
    
    data : input data, can be a map or a cube

    mask : only assign pixels within this mask

    levs : levels used to contour the data

    kern_coords : (x, y) or (x, y, z) tuple giving location of seeds

    kern_id : identifiers placed into assignment for that kernel

    corners : connect via diagonals (default False)

    compactness : compactness parameter fed to the skimage watershed

    verbose : print feedback

    """

    # Define connectivity
    connectivity = \
        simple_connectivity(ndim=data.ndim, corners=corners)

    if levs is not None:
        if verbose:
            print("Digitizing ...")
        # Sort the levels to descending order
        sorted_levs = (np.sort(levs))
        working_data = np.digitize(
            data, bins=sorted_levs, right=False).astype(np.uint16)
        a_low_value = np.nanmin(levs)
    else:
        working_data = copy.deepcopy(data)            
        a_low_value = np.nanmin(data)
        
    # Replace nans with values below the minimum level
    nan_mask = (np.isfinite(data) == False)
    working_data[nan_mask] = a_low_value

    # Make a default mask
    if mask is None:
        mask = np.isfinite(working_data)*(working_data > a_low_value)

    if verbose:
        plt.clf()
        plt.imshow(working_data, origin='lower')
        plt.show()
        
    # Initialize the assignment mask with the kernels

    image_of_peaks = np.zeros_like(working_data, dtype=int)
    if data.ndim == 2:        
        image_of_peaks[kern_coords[1],kern_coords[0]] = kern_id
    if data.ndim == 3:        
        image_of_peaks[kern_coords[2],kern_coords[1],kern_coords[0]] = kern_id
        
    # Assign
    
    assign = watershed(
        working_data, markers=image_of_peaks,
        connectivity=connectivity, compactness=compactness,
        mask=mask)
    
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


def ellipse_connectivity(
    major=None,
    minor=None,
    posang=None,
    ):
    """
    ...
    """

    # ------------------------------------------------------------
    # Error Checking on Inputs
    # ------------------------------------------------------------

    if major==None:
        print("Requires a major axis.")
        return

    if minor==None:
        minor = major

    if posang==None:
        posang=0.0

    if minor > major:
        print("Minor axis must be <= major axis.")
        return


    # ------------------------------------------------------------
    # Build the ellipse
    # ------------------------------------------------------------
    
    npix = int(2*np.ceil(major/2.0)+1)
    y,x = np.indices((npix, npix))*1.0
    y -= np.mean(y)*1.
    x -= np.mean(x)*1.
    dtor = np.pi/180.
    xp = x*np.cos(posang*dtor) - y*np.sin(posang*dtor)
    yp = x*np.sin(posang*dtor) + y*np.cos(posang*dtor)
    
    return (((xp/(major/2.0))**2 + (yp/(minor/2.0))**2) <= 1.0)


def rectangle_connectivity(
    major=None,
    minor=None,
    posang=None,
    ):
    """
    ...
    """

    # ------------------------------------------------------------
    # Error Checking on Inputs
    # ------------------------------------------------------------

    if major==None:
        print("Requires a major axis.")
        return

    if minor==None:
        minor = major

    if posang==None:
        posang=0.0

    if minor > major:
        print("Minor axis must be <= major axis.")
        return

    # ------------------------------------------------------------
    # Build the rectangle
    # ------------------------------------------------------------
    
    npix = int(2*np.ceil(major/2.0)+1)
    y,x = np.indices((npix, npix))
    y -= np.mean(y)*1.
    x -= np.mean(x)*1.
    dtor = np.pi/180.
    xp = x*np.cos(posang*dtor) - y*np.sin(posang*dtor)
    yp = x*np.sin(posang*dtor) + y*np.cos(posang*dtor)
    
    return ((np.abs(xp) <= (major/2.0))*(np.abs(yp) <= (minor/2.0)))

# TBD - routine to add depth to a two-d connectivity
