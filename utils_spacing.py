# Module to help with spacing calculations.

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Imports
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import astropy.constants as c

from astropy.table import Table

import pandas as pd
import math
import random as r
import scipy.stats
import os.path
import sys

from utils_deproject import deproject

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Distance calculations
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def dist_matrix(
        x1, y1, x2=None, y2=None,
        blank_upper=False, blank_diagonal=False):
    """Accepts either one or two paired vectors of x+y corresponding to
    (x,y) coordinates of objects and calculates a matrix filled with
    distances between entries.

    x1 : x coordinates of first set of objects

    y1 : y coordinates of first set of objects

    x2 : x coordinates of second set of objects. Default: None in
    which case distances are calculated between pairs of (x1,y1).

    y2 : y coordinates of second set of objects. Default: None in
    which case distances are calculated between pairs of (x1,y1).

    blank_upper : set the upper right triangle defined as
    dist_matrix[ii,0:ii] to nans after calculating the
    distances. Useful to avoid double counting distances in the
    matrix. Default: False.

    blank_diagonal : set the diagonal [ii,ii] measurements to nan
    after calculating the distances. Useful to avoid including
    distances from an object to itself in the calculation. Default: False.

    returns : an N x M matrix of distances where N is the length of
    (x1,y1) and M is the length of (x2,y2). Entry (i,j) in the matrix
    is the pairwise distance between (x1[i],y1[i]) and (x2[i],y2[i]).

    """

    # Duplicate the x1, y1 vector if no second vector is supplied
    if x2 is None or y2 is None:
        x2 = np.copy(x1)
        y2 = np.copy(y1)

    # Note length of vectors
    nx1 = len(x1)
    nx2 = len(x2)
    
    # Make a grid with x and y fixed along columns
    x1_grid = np.tile(x1, (nx2,1))
    y1_grid = np.tile(y1, (nx2,1))

    # Make a grid with x and y fixed along rows
    x2_grid = np.transpose(np.tile(x2, (nx1,1)))
    y2_grid = np.transpose(np.tile(y2, (nx1,1)))

    # Simple Euclidean distance
    dist_grid = np.sqrt((x1_grid-x2_grid)**2 + (y1_grid-y2_grid)**2)    

    # Blank the diagonal or upper right corner
    if blank_diagonal or blank_upper:
        if nx1 <= nx2:
            diag_len = nx1
        else:
            diag_len = nx2

        for ii in range(diag_len):
            if blank_upper:
                dist_grid[ii,0:ii] = np.nan
            if blank_diagonal:
                dist_grid[ii,ii] = np.nan

    # Return teh matrix
    return(dist_grid)

def sorted_dist_matrix(
        x1, y1, x2=None, y2=None,
        blank_upper=True, blank_diagonal=True):
    """Calculate a distance matrix using the function *dist_matrix* and
    then sort and return each row.

    x1 : x coordinates of first set of objects

    y1 : y coordinates of first set of objects

    x2 : x coordinates of second set of objects. Default: None in
    which case distances are calculated between pairs of (x1,y1).

    y2 : y coordinates of second set of objects. Default: None in
    which case distances are calculated between pairs of (x1,y1).    

    blank_upper : set the upper right triangle defined as
    dist_matrix[ii,0:ii] to nans after calculating the
    distances. Useful to avoid double counting distances in the
    matrix. Default: True.

    blank_diagonal : set the diagonal [ii,ii] measurements to nan
    after calculating the distances. Useful to avoid including
    distances from an object to itself in the calculation. Default: True.

    returns : an N x M matrix of distances where N is the length of
    (x1,y1) and M is the length of (x2,y2). Now each row has been
    sorted by the values in that row (NaNs go last) so that
    dist_matrix[i,:] gives the sorted distances of other peaks to the
    (x1[i],y1[i]).

    """

    # Calculate the distance matrix    
    all_dists = dist_matrix(
        x1, y1, x2=x2, y2=y2,
        blank_upper=blank_upper, blank_diagonal=blank_diagonal)

    # Note the number of points
    npts = len(x1)

    # Loop over rows and sort each row
    for ii in range(npts):
        
        this_vec = all_dists[ii,:]
        this_ind = np.argsort(this_vec)
        all_dists[ii,:] = this_vec[this_ind]

    # Return sorted matrix
    return(all_dists)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Calculate number of nearby points
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def n_neighbors(x,y,radius,matrix=None,blank_diagonal=True):
    """Calculate the number of neighbors within a specified radius for a
    set of points (x, y).

    Inputs

    x : x coordinates of the points to consider

    y : y coordinates of the points to consider

    radius : the radius within which to count other points

    matrix : if supplied, gives the pre-computed sorted distance
    matrix, e.g., from a previous call. Allows for each calculation of
    neighbor counts within various radii.

    blank_diagonal : if True then does not count the local source
    itself. Default: True.

    Returns

    A vector holding the number of neighbors within "radius" of each
    point (x,y).

    """

    # Calculate the sorted distance matrix if none is supplied.
    if matrix is None:
        matrix = sorted_dist_matrix(x,y, blank_upper=True, blank_diagonal=True)

    # Flag entries within the matrix with distance less than the
    # radius of interest
    masked_matrix = matrix <= radius

    # Count the neighbors for each entry.
    neighbor_vec = np.sum(masked_matrix, axis=1)

    # Return vector of neighbor counts.
    return(neighbor_vec)

def nth_neighbor(x,y,n=0,matrix=None,blank_diagonal=True):
    """Calculate the distance to the nth nearest neighbor for a set of
    points (x, y).

    Inputs

    x : x coordinates of the points to consider

    y : y coordinates of the points to consider

    n : the nth nearest neighbor. Note that the meaning of this
    changes according to whether the diagonal has been blanked or
    not. Assuming blank diagonal is true then the first nearest
    neighbor is n=0. Default n=0.

    radius : the radius within which to count other points

    matrix : if supplied, gives the pre-computed sorted distance
    matrix, e.g., from a previous call. Allows for each calculation of
    neighbor counts within various radii.

    blank_diagonal : if True then does not count the local source
    itself. Default: True.

    """
    
    if matrix is None:
        matrix = sorted_dist_matrix(x,y,blank_diagonal=blank_diagonal)
    
    return(matrix[:,n])

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Two point correlation
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def calc_twopoint(
        x_data, y_data,
        x_rand, y_rand,
        bins_lo, bins_hi,
        chunk=True, verbose=True):

    """
    """

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_rand = np.array(x_rand)
    y_rand = np.array(y_rand)
    
    n_bins = len(bins_lo)
    n_data = len(x_data)
    n_rand = len(x_rand)    
    if verbose:
        print("... number of bins: ", n_bins)
        print("... number of data: ", n_data)
        print("... number of randoms: ", n_rand)
    if chunk:
        n_chunk = int(np.floor(n_rand/n_data))
        if verbose:
            print("... number of data chunks: ", n_chunk)

    # ..........................................
    # Calculate the data-data distance matrix
    # ..........................................
        
    dd_norm = 1./(n_data*(n_data-1)/2.)    
    dist_data_data = dist_matrix(
        x_data, y_data,
        x2=np.copy(x_data), y2=np.copy(y_data),
        blank_upper=True, blank_diagonal=True)
    #print(dist_data_data)
    #test = input('hold')

    dd_bins = np.zeros(n_bins)*np.nan
    for ii in range(n_bins):        
        dd_bins[ii] = np.nansum(
            (dist_data_data >= bins_lo[ii]) \
            * (dist_data_data < bins_hi[ii])) \
            * dd_norm
    
    # ..........................................
    # Calculate the rand rand matrix
    # ..........................................

    # This step works differently depending on whether the
    # chunking of data is requested.
    
    if chunk:
        
        # ------------------------------------------------------
        # In this case run the RR matrix one piece at a time.
        # ------------------------------------------------------
        
        counter = 0
        chunk_rr_bins = np.zeros((n_chunk, n_bins))*np.nan
        chunk_rr_norm = np.zeros(n_chunk)*np.nan
        for jj in range(n_chunk):

            # Extract the next chunk of the random vectors
            this_x_rand = x_rand[counter:counter+n_data]
            this_y_rand = y_rand[counter:counter+n_data]
            this_n_rand = n_data
            counter += n_data

            # Note the normalization
            this_rr_norm = 1./(this_n_rand*(this_n_rand-1)/2.)    
            chunk_rr_norm[jj] = this_rr_norm

            # Expand the distance matrix
            this_dist_rand_rand = dist_matrix(
                this_x_rand, this_y_rand,
                x2=np.copy(this_x_rand), y2=np.copy(this_y_rand),
                blank_upper=True, blank_diagonal=True)
            
            # Bin the data
            for ii in range(n_bins):

                chunk_rr_bins[jj,ii] = np.nansum(
                    (this_dist_rand_rand >= bins_lo[ii]) \
                    * (this_dist_rand_rand < bins_hi[ii])) \
                    * this_rr_norm

        # Now collapse to form the average data-rand DR term
        rr_bins = np.nansum(chunk_rr_bins,axis=0)/(1.0*n_chunk)
                
    else:

        # ------------------------------------------------------
        # In this case the full RR matrix is calculated at once.
        # ------------------------------------------------------

        # Note the normalization
        rr_norm = 1./(n_rand*(n_rand-1)/2.)

        # Expand the distance matrix
        dist_rand_rand = dist_matrix(
            x_rand, y_rand,
            x2=np.copy(x_rand), y2=np.copy(y_rand),
            blank_upper=True, blank_diagonal=True)

        # Bin the data
        rr_bins = np.zeros(n_bins)*np.nan
        for ii in range(n_bins):        
            rr_bins[ii] = np.nansum(
                (dist_rand_rand >= bins_lo[ii]) \
                * (dist_rand_rand < bins_hi[ii])) \
                * rr_norm

    # ..........................................
    # Calculate the data rand matrix
    # ..........................................

    # This step works differently depending on whether the
    # data chunking is requested.

    if chunk:
        
        # ------------------------------------------------------
        # In this case run the DR matrix one piece at a time.
        # ------------------------------------------------------

        counter = 0
        chunk_dr_bins = np.zeros((n_chunk, n_bins))*np.nan
        chunk_dr_norm = np.zeros(n_chunk)*np.nan
        for jj in range(n_chunk):

            # Extract the next chunk of the random vectors
            this_x_rand = x_rand[counter:counter+n_data]
            this_y_rand = y_rand[counter:counter+n_data]
            this_n_rand = n_data
            counter += n_data

            # Note the normalization
            this_dr_norm = 1./(n_data*this_n_rand)    
            chunk_dr_norm[jj] = this_dr_norm

            # Expand the distance matrix
            this_dist_data_rand = dist_matrix(
                x_data, y_data,
                x2=this_x_rand, y2=this_y_rand,
                blank_upper=False, blank_diagonal=False)

            # Bin the data
            for ii in range(n_bins):

                chunk_dr_bins[jj,ii] = np.nansum(
                    (this_dist_data_rand >= bins_lo[ii]) \
                    * (this_dist_data_rand < bins_hi[ii])) \
                    * this_dr_norm

        # Now collapse to form the average data-rand DR term
        dr_bins = np.nansum(chunk_dr_bins,axis=0)/(1.0*n_chunk)
        
    else:
        
        # ------------------------------------------------------
        # In this case the full DR matrix is calculated at once.
        # ------------------------------------------------------
        
        #  Note the normalization and note that this is different than
        #  the other two because we do not blank the lower triangle
        #  and we do not blank diagonals.
        
        dr_norm = 1./(n_data*n_rand)

        # Expand the distance matrix
        
        dist_data_rand = dist_matrix(
            x_data, y_data,
            x2=x_rand, y2=y_rand,
            blank_upper=False, blank_diagonal=False)

        # Bin the data
        
        for ii in range(n_bins):
            dr_bins[ii] = np.nansum(
                (dist_data_rand >= bins_lo[ii]) \
                * (dist_data_rand < bins_hi[ii])) \
                * dr_norm

    # ..........................................            
    # Calculate the output
    # ..........................................    

    # Combine into the clustering estimator
    #print(dd_bins[0], dr_bins[0], rr_bins[0])
    
    clustering = 1/rr_bins*(dd_bins - 2.*dr_bins + rr_bins) + 1.
        
    return(clustering)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Add spacing related value to a pyCPROPS catalog
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def add_spacings_to_catalog(
        cat_file, out_file,
        ra_ctr, dec_ctr, pa, incl, dist_mpc,
        min_clouds=6,
        use_peak=True,
        overwrite=True):
    """
    Add spacing and neighbor information to a pyCPROPS style catalog.
    """

    cat = Table.read(cat_file)

    if use_peak:
        ra = np.array(cat['XMAX_DEG'])
        dec = np.array(cat['YMAX_DEG'])
    else:
        ra = np.array(cat['XCTR_DEG'])
        dec = np.array(cat['YCTR_DEG'])

    n_clouds = len(ra)
        
    # Note the center
    center_coords = [ra_ctr, dec_ctr]

    # Deproject ra and dec to dx and dy and cylindrical coords        
    rgal, theta, dx, dy = deproject(
        center_coord=center_coords, incl=incl, pa=pa,
        ra=ra, dec=dec, return_offset=True)

    # Note the deprojected coords into the new catalog
    cat['DELTAX_DEG'] = dx
    cat['DELTAY_DEG'] = dy    
    cat['RGAL_DEG'] = rgal
    cat['THETA_DEG'] = theta

    # Convert to physical distance
    deg_to_pc = np.pi/180.*dist_mpc*1e6
    
    dx_pc = dx*deg_to_pc
    dy_pc = dy*deg_to_pc
    rgal_pc = rgal*deg_to_pc

    # Add the location in physical offset to the catalog
    cat['DELTAX_PC'] = dx_pc
    cat['DELTAY_PC'] = dy_pc 
    cat['RGAL_PC'] = rgal_pc

    # Calculate matrix of deprojected distances
    matrix = sorted_dist_matrix(dx_pc,dy_pc, blank_upper=True, blank_diagonal=True)
    
    if n_clouds < min_clouds:
        
        cat['DIST_NN1_PC'] = np.nan
        cat['DIST_NN3_PC'] = np.nan
        cat['DIST_NN5_PC'] = np.nan
        
        cat['NEIGHBORS_300PC'] = np.nan
        cat['NEIGHBORS_500PC'] = np.nan
        cat['NEIGHBORS_700PC'] = np.nan

    else:
    
        # Distance to 1st (n=0), 3rd (n=2), 5th (n=4) neighbor
        dist_nn1 = nth_neighbor(dx_pc,dy_pc,n=0,matrix=matrix)
        dist_nn3 = nth_neighbor(dx_pc,dy_pc,n=2,matrix=matrix)
        dist_nn5 = nth_neighbor(dx_pc,dy_pc,n=4,matrix=matrix)
        
        cat['DIST_NN1_PC'] = dist_nn1
        cat['DIST_NN3_PC'] = dist_nn3
        cat['DIST_NN5_PC'] = dist_nn5

        # Number of neighbors in 250, 500, 750pc
        nn_250pc = n_neighbors(dx_pc,dy_pc,250.0,matrix=matrix)
        nn_500pc = n_neighbors(dx_pc,dy_pc,500.0,matrix=matrix)
        nn_750pc = n_neighbors(dx_pc,dy_pc,750.0,matrix=matrix)
        
        cat['NEIGHBORS_250PC'] = nn_250pc
        cat['NEIGHBORS_500PC'] = nn_500pc
        cat['NEIGHBORS_750PC'] = nn_750pc
    
    cat.write(out_file, format='fits', overwrite=overwrite)    

    return()


def add_maps_to_catalog(
        cat_file, out_file,
        field_dict,
        use_peak=True,
        verbose=False,
        overwrite=True):
    """
    Read a series of maps and add their values to the cloud catalog.
    """

    cat = Table.read(cat_file)

    if use_peak:
        ra = np.array(cat['XMAX_DEG'])
        dec = np.array(cat['YMAX_DEG'])
    else:
        ra = np.array(cat['XCTR_DEG'])
        dec = np.array(cat['YCTR_DEG'])
    
    for this_key in field_dict.keys():
        this_map_file = field_dict[this_key]

        if verbose:
            print('... adding: ', this_key)
            print('... from: ', this_map_file)
        
        if os.path.isfile(this_map_file) == False:
            print("Missing map ", this_map_file)

        hdulist = fits.open(this_map_file)
        this_map = hdulist[0].data
        this_header = hdulist[0].header
        this_wcs = WCS(this_header).celestial
        pix_x, pix_y = this_wcs.wcs_world2pix(ra, dec, 0)
        #print(pix_x, pix_y)

        # Need to add some error checking here
        pix_x = pix_x.astype('int')
        pix_y = pix_y.astype('int')
        cat[this_key] = this_map[pix_y,pix_x]
            
    cat.write(out_file, format='fits', overwrite=overwrite)
    
    return()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Randomly draw clouds from a probability image to make a mock catalog
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def place_nclouds_in_map(
        image_file, n_clouds_to_place, outfile_name,        
        ra_ctr, dec_ctr, pa, incl,
        mask_file=None,mask_thresh=1.0,
        overwrite=True):
    """Generate a monte carlo catalog from a probability image.

    Inputs

    image_file : probability map. Read in an normalized and negatives
    set to zero, then used as a probability field to generate random
    placements. No default.

    n_clouds_to_place : number of items to place in the map. No
    default.

    outfile_name : name of file to write CSV results to. No default.

    ra_ctr, dec_ctr, pa, incl : orientation of the galaxy used to
    calculate the location of the placed clouds in the galactocentric
    frame.

    overwrite : overwrite previous output results.

    """
    
    # Open the image
    hdulist = fits.open(image_file)

    # Read the data, interpreted as a probability field
    prob_map = hdulist[0].data

    # Open the mask if requested
    if mask_file is not None:
        hdumask = fits.open(mask_file)
        mask = hdumask[0].data >= mask_thresh
    else:
        mask = np.isfinite(prob_map)

    # Extract the WCS
    wcs = WCS(hdulist[0].header, naxis=2)
    naxis = wcs._naxis # size of image naxis[0] = x and [1] = y

    # Make a grid of RA and Dec coordinates
    grid  = np.indices((naxis[1],naxis[0]))
    ra_map, dec_map  = wcs.wcs_pix2world(grid[1],grid[0],0)

    # Note the center
    center_coords = [ra_ctr, dec_ctr]

    # Deproject ra and dec to dx and dy and cylindrical coords
    radius_map, theta_map, dx_map, dy_map = deproject(
        center_coord=center_coords, incl=incl, pa=pa,
        ra=ra_map, dec=dec_map, return_offset=True)

    #flatten data structures
    f_prob = prob_map.flatten()
    f_mask = mask.flatten()
    f_ra   = ra_map.flatten()
    f_dec  = dec_map.flatten()
    f_dx   = dx_map.flatten()
    f_dy   = dy_map.flatten()
    f_rad  = radius_map.flatten()
    f_theta  = theta_map.flatten()

    #remove nans
    keep    = np.where(np.isfinite(f_prob)*(f_mask))
    f_prob  = f_prob[keep]
    f_ra    = f_ra[keep]
    f_dec   = f_dec[keep]
    f_dx    = f_dx[keep]
    f_dy    = f_dy[keep]
    f_rad   = f_rad[keep]
    f_theta = f_theta[keep]
    
    # Set negative probabilities to zero
    if np.any(f_prob < 0):
        f_prob[np.where(f_prob < 0)] = 0

    # Normalize the probabilities
    total_prob = np.sum(f_prob)
    f_prob  = f_prob/total_prob

    #Generates index for each coordinate pair
    n_pts = len(f_prob)
    indices = np.arange(n_pts, dtype=int)
    
    #Probability weighted random placement
    rand_indices = np.random.choice(
        indices, p=f_prob,
        size=int(n_clouds_to_place))

    rand_dx = f_dx[rand_indices]
    rand_dy = f_dy[rand_indices]
    rand_ra = f_ra[rand_indices]
    rand_dec = f_dec[rand_indices]
    rand_rad = f_rad[rand_indices]
    rand_theta = f_theta[rand_indices]

    # Make an astropy table
    output_tab = Table()

    output_tab['ra'] = rand_ra
    output_tab['dec'] = rand_dec    
    output_tab['dx'] = rand_dx
    output_tab['dy'] = rand_dy
    output_tab['rad'] = rand_rad
    output_tab['theta'] = rand_theta
    
    output_tab.write(outfile_name, format='fits', overwrite=overwrite)

    return()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Physical spacing quantities
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def jeans_len(sigma, rho):
    """Calculate Jeans length given a velocity dispersion (sigma) and
    density (rho).

    Inputs

    sigma: velocity dispersion in km/s

    rho: density of the medium in M_sun / pc^3

    Returns

    lambda_j : Jeans length in pc

    """
    G = c.G.to(u.pc**3 / u.solMass / u.s**2)
    
    lambda_j = np.sqrt((np.pi * vel.to(u.pc/u.s)**2) / (G * rho))
    
    return(lambda_j)

def kappa2(beta, v_circ, r_gal):
    """
    Calculate epicyclic frequency.
    """
    kappa = 2 * (1+beta) * (v_circ/r_gal)**2
    return(kappa)

def lambda_T(sigma_gal, kappa2):
    """
    Calculate Toomre length.
    """
    G = c.G.to(u.pc**3 / u.Msun / u.s**2)    
    lT = 4*np.pi*G * sigma_gal / kappa2
    return(lT)
