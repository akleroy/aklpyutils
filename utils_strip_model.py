# Module to use Richardson-Lucy deconvolution to infer a surface
# density profile from a measured set of "strip integrals." Follows
# Warmels et al. 1988 and is appropriate/useful for optically thin
# tracers targeting edge-on systems (it was developped for HI).

# A "strip" or "stripe" integral is a sum over a series of strips
# perpendicular to the major axis. Each strip covers the minor axis of
# the galaxy and the sum has units of intensity*area. As a practical
# matter, area needs the same units as the strip spacing. So if the
# strips are defined by steps in arcsec, then integral needs to have
# intensity*arsec^2 units.

# The program assumes a model in which the the galaxy is a disk
# described by an axisymmetric surface density profile. Inclination is
# irrelevant as long as the strips cover the full minor axis
# extent. The disk is taken to be thin, but moderate thickness should
# have only a second-order effect, moving a little power between
# adjacent rings. A bigger issue will be the breakdown of the assumed
# disk geometry.

# The procedure is Richardson-Lucy style iteration on a model. The
# program calculates the cross-linkage between rings and strips and
# then begins with a model surface density profile. It predicts the
# observed strip integral from that surface density profile, contrasts
# the model and prediction, adjusts the model accordingly, and repeats
# for either a fixed number of iterations or until a convergence
# criteria is satisfied.

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Imports
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Helper programs for geometric calculations
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def theta_for_xr(x,r):
    # arc cosine call with a small error trap
    if x > r:
        return(np.nan)
    return(np.arccos(x/r))

def area_circseg_theta(r,theta,default_zero=True):
    # area of a circular segment subtended by theta for a circle with
    # radius r.
    if r == 0:
        return(0.0)
    if theta < 0. or theta > np.pi:
        if default_zero:
            return(0.0)
        else:
            return(np.nan)
    return((r**2)/2.0*(theta-np.sin(theta)))

def area_circseg_x(r,x,default_zero=True):
    # area of a circular segment defined by a chord at distance x from
    # the center of a circle of radius r
    if r == 0:
        return(0.0)
    half_theta = theta_for_xr(x,r)
    if np.isfinite(half_theta) == False:
        if default_zero:
            return(0.0)
        else:
            return(np.nan)
    return(area_circseg_theta(r,half_theta*2.0))

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Calculate matrices linking stripes and rings
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def calc_pxr(r_min, r_max):
    """Calculate the probability given r that a bit of emission lies at
    projected distance x from the center of a system. The resulting
    matrix is used as part of the deconvolution.

    Inputs:

    r_min --- the lower boundary of both the x-axis strips/bins and the
    radial profile rings.

    r_max --- the upper boundary  of both the x-axis strips/bins and the
    radial profile rings.

    Returns:

    pxr --- an n x n matrix where n is the number of bins. For ring
    ii, the matrix contains the fraction of the full area of that ring
    contributing to strip jj. So for each row ii, corresponding to a
    ring, the vector pxr[ii,:] gives the normalized distribution of
    area across strip integral bins.

    The matrix is used during deconvolution to map deviations between
    the model and observation into refinemenets to the model.
    """

    # Parse the input bins
    n_prof = len(r_min)
    step = r_max[1]-r_max[0]
    width = r_max[0]-r_min[0]
    if step != width:
        norm = step/width
    else:
        norm = 1.0

    # Initialize a grid that cross-links contributions from surface
    # profiles at different radii
    pxr_grid = np.zeros((n_prof,n_prof))

    # Loop over the profile bins and calculate the cross-linkage
    for ii in range(n_prof):

        this_rmin = r_min[ii]
        if this_rmin < 0.0:
            this_rmin = 0.0
        this_rmax = r_max[ii]

        # The full area of the annulus divided by 2 because we work
        # only at positive x.
        full_area = (np.pi*this_rmax**2-np.pi*this_rmin**2) / 2.0

        # Loop over all stripes and calculate the fraction in this x
        # range.
        for jj in range(n_prof):

            this_xmin = r_min[jj]
            if this_xmin < 0.0:
                this_xmin = 0.0
            this_xmax = r_max[jj]

            # Segment 1 - big R, low x
            area_bigr_lowx = area_circseg_x(this_rmax,this_xmin)
            # Segment 2 - big R, high x
            area_bigr_bigx = area_circseg_x(this_rmax,this_xmax)
            # Segment 3 - small R, low x            
            area_lowr_lowx = area_circseg_x(this_rmin,this_xmin)            
            # Segment 4 - small R, high x
            area_lowr_bigx = area_circseg_x(this_rmin,this_xmax)            

            if (np.isfinite(area_bigr_lowx) == False) or \
               (area_bigr_lowx == 0.0):
                pxr_grid[ii,jj] = 0.0
                continue

            big_circle_area = area_bigr_lowx - area_bigr_bigx
            small_circle_area = area_lowr_lowx - area_lowr_bigx

            pxr_grid[ii,jj] = \
                (big_circle_area - small_circle_area) / full_area * norm

            if np.isfinite(pxr_grid[ii,jj]) == False:
                print("Diagnosing unexpected non-finite grid element.")
                print(pxr_grid[ii,jj])
                print(this_rmax, this_rmin, this_xmin, this_xmax)
                print(full_area, big_circle_area, small_circle_area)
                raise Exception('Non-finite cross-link grid')
            
    # Return the cross-linkage grid
    return(pxr_grid)

def calc_arx(x_min, x_max):
    """Calculate a matrix giving the area contributed by ring jj to stripe
    ii. The resulting matrix is used to predict strip integrals from a
    surface density profile.

    Inputs:

    x_min --- the lower boundary of both the x-axis strips/bins and the
    radial profile rings.

    x_max --- the upper boundary  of both the x-axis strips/bins and the
    radial profile rings.

    Returns:

    arx --- an n x n matrix where n is the number of bins. The first
    index ii, refers to the strip number and the second jj to the ring
    number in the surface density profile.

    Thus, for strip ii from x=x_min[ii] to x=x_max[ii], ring jj from r
    = x_min[jj] to r=x_max[jj] contributes an area arx[ii,jj] to that
    strip in the same units as x_min (e.g., if x_min is in arcsec then
    arx has units of arcseconds squared).
    """

    step = x_max[1]-x_max[0]
    width = x_max[0]-x_min[0]
    
    # Note number of bins
    n_prof = len(x_min)

    # Initialize the grid
    arx_grid = np.zeros((n_prof,n_prof))

    # Loop over the stripes (ii) ...
    for ii in range(n_prof):

        this_xmin = float(x_min[ii])
        this_xmax = float(x_max[ii])
        if this_xmin < 0.0:
            scaleby = (np.abs(this_xmin)+this_xmax)/this_xmax
            this_xmin = 0.0
        else:
            scaleby = 1.0

        # Loop over all of the rings
        for jj in range(n_prof):

            # Note that we use the middle of the bin and the STEP (not
            # width) to avoid issues with oversampling. In the
            # oversamled case each ring just contributes near the ring
            # center.
            
            this_rmid = float((x_max[jj]+x_min[jj])*0.5)

            this_rmin = this_rmid-step*0.5
            this_rmax = this_rmid+step*0.5
            
            if this_rmin < 0.0:
                this_rmin = 0.0

            # Segment 1 - big R, low x
            area_rmax_xmin = area_circseg_x(this_rmax,this_xmin)
            # Segment 2 - big R, high x
            area_rmax_xmax = area_circseg_x(this_rmax,this_xmax)
            # Segment 3 - small R, low x            
            area_rmin_xmin = area_circseg_x(this_rmin,this_xmin)            
            # Segment 4 - small R, high x
            area_rmin_xmax = area_circseg_x(this_rmin,this_xmax)            

            if (np.isfinite(area_rmax_xmin) == False) or \
               (area_rmax_xmin == 0.0):
                arx_grid[ii,jj] = 0.0
                continue

            big_circle_area = area_rmax_xmin - area_rmax_xmax
            small_circle_area = area_rmin_xmin - area_rmin_xmax

            arx_grid[ii,jj] = \
                (big_circle_area - small_circle_area)* \
                scaleby

            if np.isfinite(arx_grid[ii,jj]) == False:
                print("Diagnosing unexpected non-finite grid element.")
                print(arx_grid[ii,jj])
                print(this_rmax, this_rmin, this_xmin, this_xmax)
                print(full_area, big_circle_area, small_circle_area)
                raise Exception('Non-finite cross-link grid')
            
    # Return the area grid
    return(arx_grid)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines to handle stripes and profiles
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def pred_strip_fromsd(
        r_min, r_max, sd_prof,
        arx_grid=None):
    """Translate from a surface density profile into a strip integral.

    Inputs:

    r_min --- the inner/lower edge of both the rings in the disk and
    the strips to consider.

    r_max --- the outer/upper edge of both the rings in the disk and
    the strips to consider.

    sd_prof --- a surface density profile for the face-on, thing,
    axisymmetric disk to be modeled.

    Keywords:

    arx_grid --- if supplied gives the matrix cross-linking the strips
    and rings as calculated by calc_arx above. arx_grid[ii,jj] gives
    the area of the jj ring that lies in the ii strip. 

    If not supplied this is calculated, but since this is by far the
    most expensive part, it's better to pre-calculate this if doing a
    lot of fits.

    Outputs:

    strip --- an integral with units matching those of sd_prof times
    those of r_min squared (e.g., MJy/sr for sd_prof, arcseconds for
    r_min yields MJy/sr*arcseconds^2 for strip). The ii element
    corresponds to the integral between r_min[ii] and r_max[ii].

    """

    # If missing, calculate the area that each ring contributes to
    # each strip.
    if arx_grid is None:
        arx_grid = calc_arx(r_min, r_max)

    # Nore the number of strips.
    n_prof = len(r_min)
        
    # collapse into a strip using the matrix calculated above
    strip = r_min*np.nan
    for ii in range(n_prof):
        strip[ii] = np.nansum(sd_prof*arx_grid[ii,:])

    return(strip)

def fold_centered_prof(xlo_in, xhi_in, y_in, e_in=None):
    """Fold a strip or radial profile about zero to make it one
    sided. Useful for cases where the profile has been measured from
    negative to positive but axisymmetry is assumed and you want to
    simplify.

    Averaging is used to combine overlapping y values..

    Inputs:

    xlo_in --- the lower boundary of each bin of the profile
    
    xhi_in --- the upper boundary of each bin of the profile

    y_in --- the value of the profie in each bin

    Keyword parameters:

    e_in --- the uncertainty on y_in. If supplied the uncertainty is
    also returned as part of the output. Propagated via simplistic
    error propagation.q

    Output:

    x_lo, x_hi, y and e (if e_in supplied) --- the lower and upper
    edges of the bins, the mean y value, and the error on y.

    """
    
    # Assume a regular pattern
    xmid = 0.5*(xhi_in+xlo_in)
    step = xmid[1]-xmid[0]
    width = xhi_in[0]-xlo_in[0]
    half_width = 0.5*width
    # ... give the bins integer labels
    bin_ind = np.round(xmid/step)

    # Build new bins
    nlo = np.min(np.abs(bin_ind))
    nhi = np.max(np.abs(bin_ind))
    fold_ind = np.arange(nlo,nhi+1,1,dtype=np.int)
    x_lo = fold_ind*step-half_width
    x_hi = x_lo + width
    y = x_lo*np.nan
    if e_in is not None:
        e = x_lo*np.nan
    
    # Take the mean of all relevant bins
    for ii in fold_ind:
        mask = (np.abs(bin_ind) == ii)
        norm = np.sum(mask)*1.0
        y[ii] = np.sum(y_in[mask])/(1.0*norm)
        if e_in is not None:
            e[ii] = np.sqrt(np.sum(e_in[mask]**2)/(1.0*norm))

    # Return
    if e_in is not None:
        return(x_lo, x_hi, y, e)
    else:
        return(x_lo, x_hi, y)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Deconvolution routine
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def fit_sdprof_to_strip(
        x_lo, x_hi, strip, e_strip=None,
        clip_s2n=1.0, max_iter=100):    
    """
    Fit a radial model to a strip integral profile.
    """
    
    # Fold the strip (should be non-destructive even if the strip is
    # one sided)
    print("... folding the profile...")
    if e_strip is not None:
        x_lo, x_hi, strip, e_strip = \
            fold_centered_prof(
                x_lo, x_hi, strip, e_in=e_strip)
    else:
        x_lo, x_hi, strip = \
            fold_centered_prof(
                x_lo, x_hi, strip)

    n_bins = len(strip)
        
    # Ensure positive definite
    if e_strip is not None:
        use_stripes = (strip >= e_strip*clip_s2n)
    else:
        use_stripes = (strip >= 0.0)
        
    strip = strip*use_stripes
    strip[np.where(np.isfinite(strip)==False)] = 0.0
    
    # Estabish a first guess    
    full_sum = np.nansum(strip)
    full_area = np.pi*(np.nanmax(x_hi))**2
    mean_sd = full_sum/full_area

    # Calculate the cross-linkage matrix
    print("... calculating the p(x,r) grid...")
    pxr_grid = calc_pxr(x_lo, x_hi)

    #plt.imshow(pxr_grid)
    #plt.show()
    
    print("... calculating the a(r,x) grid...")
    arx_grid = calc_arx(x_lo, x_hi)

    #print(arx_grid)
    #plt.imshow(arx_grid)
    #plt.show()
    
    # Initialize the iterative solution
    curr_sd = mean_sd + 0.0*strip
    prev_sd = curr_sd

    # Iterate
    print("... iterating...")
    tol = 0.01
    converged = False
    eps = np.nan
    for jj in range(max_iter):        
        if converged:
            continue
        
        # Check the fractional difference between model bins against
        # the tolerance. If things have converged, break out of the loop.
        diff = (curr_sd-prev_sd)
        
        eps = np.nanmax(np.abs(diff/curr_sd))
        
        if eps < tol and jj != 0:
            converged = True
            #print("... ... converged, eps = ", eps)
            continue
        else:
            #print("... ... continuing, eps = ", eps)
            pass

        #print("... ... iteration ", jj, " epsilon ", eps)
            
        # Make a predicted strip profile from the current model. The
        # arx_grid is optional but pre-calculating it saves a lot of
        # time. This grid arx_grid[ii,jj] for x strip ii and ring jj
        # gives the area of ring jj that contributes to strip ii so
        # that the summed strip ii is curr_sd*arx_grid[ii,:] .

        pred = pred_strip_fromsd(
            x_lo, x_hi, curr_sd,
            arx_grid = arx_grid)

        # Save the current model
        prev_sd = np.array(curr_sd)
                
        # Now use the cross-linkage matrix to refine the model. The
        # idea is that pxr[ii,jj] gives the fractional contribution of
        # strip jj to ring ii, so that the vector pxr[ii,:] is the
        # fractional contributions of each strip to that ring and sums
        # to 1.0. We multiply the ratio of (real data/model data) for
        # each strip by the fractionl contribution of that strip to
        # the current ring and then we sum the changes and scale the
        # current model surface density in this ring.
        
        for ii in range(n_bins):
            delta = np.nansum(strip/pred*pxr_grid[ii,:])
            #print(delta)
            #plt.plot(0.5*(x_lo+x_hi), delta)
            #plt.show()
            curr_sd[ii] = prev_sd[ii] * delta

        #print(curr_sd)
        #plt.plot(0.5*(x_lo+x_hi), curr_sd/prev_sd)
        #plt.plot(0.5*(x_lo+x_hi), curr_sd)
        #plt.show()
        
        #plt.plot(0.5*(x_lo+x_hi), np.log10(strip))
        #plt.plot(0.5*(x_lo+x_hi), np.log10(pred))
        #plt.show()
        
    # Return the results
    return(x_lo, x_hi, curr_sd)


