from collections import OrderedDict
import numpy as np
import scipy.stats
from scipy.odr import ODR, Model, Data, RealData
import scipy.optimize
from scipy.stats import spearmanr, kendalltau

from astropy.stats import mad_std
from astropy.table import Table

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines related to describing distributions
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def perc_w_lim(vals,lims=None,perc=50):
    """Calculate a percentile in the presence of limits. This is only
    approximate when limits are present.

    Parameters: 

    vals --- measured values

    Keywords:

    lims --- measured upper limits

    perc --- percentile to calculate (e.g., 50, 16, 84)

    Returns: 

    value --- estimate of the percentile value of that distribution

    is_value_a_limit --- boolean indicating if the value is an upper
    limit

    Notes:

    Reverse the sign on the array (i.e., * -1) to use with lower
    instead of upper limits.

    The approach in the presence of limits is approximte. It tries to
    bracket the true values by averaging the results using all limits
    that might be relevant with the result using all limits that are
    sure to be relevant.

    """

    if len(vals) == 0:
        return((np.nan,False))

    # focus on finite values
    fin_ind = np.isfinite(vals)

    if np.sum(fin_ind) == 0:
        return((np.nan,False))

    vals = vals[fin_ind]
    
    # if there are no limits, use the numpy function without modification
    if lims is None or len(lims) == 0:
        return((np.percentile(vals,perc),False))
  
    # If there are limits restrict to the finite cases
    fin_lims = np.isfinite(lims)
    lims = lims[fin_lims]
    if len(lims) == 0:
        return((np.percentile(vals,perc),False))
    
    # Make a first guess from the non-limits
    first_guess = np.percentile(vals,perc)

    # ... identify which limits fall below this first guess
    relevant_lims = lims < first_guess
    
    # ... if the number of limits exceeds the number of values below
    # the first guess then return the maximum relevant limit and an
    # upper value flag.
    relevant_vals = vals < first_guess
    if np.sum(relevant_lims) > np.sum(relevant_vals):
        return((np.max(lims[relevant_lims]),True))
    
    # Make a second guess, forcing all relevant limits to the lowest
    # value in the data set.
    placeholders = lims[relevant_lims]*0.0+np.nanmin(vals)
    new_vals = np.concatenate((vals,placeholders))
    second_guess = np.percentile(new_vals,perc)
    
    # ... now identify which limits are certain to be relevant because
    # they affect the first and second estimate.
    certain_lims = lims < second_guess

    # Make a third estimate that includes only the certain limits,
    # again replacing them with the minimum value in the data set.
    placeholders = lims[certain_lims]*0.0+np.nanmin(vals)
    new_vals = np.concatenate((vals,placeholders))
    third_guess = np.percentile(new_vals,perc)
    
    # Return the average of the second and third guess
    return((0.5*(second_guess+third_guess),False))

def lo_perc(data):
    """
    Convenience function to return the 16th percentile, can be fed to scipy binned stats, etc.
    """
    return(np.percentile(data,16))

def hi_perc(data):
    """
    Convenience function to return the 84th percentile, can be fed to scipy binned stats, etc.
    """
    return(np.percentile(data,84))

def calc_stats(
        vec, err=None, oversamp_fac=None,
        lims=None, lims_are_lowerlims=False,
        fidval_for_lims=None,
        doprint=True, return_empty=False):
    """Calculate a statistics on a distribution, trying to take into
    account errors and limits.

    Parameters:

    vec --- the measurements to be described

    Keywords:

    err --- error values, can be a single value or a matched vector or
    None. If none, error fields are not calculated. Default: None and
    errors are not calculated.
    
    oversamp_fac --- sample by which the measurements are oversampled
    relative to statistically independent data. Used to make ad hoc
    corrections to uncertainties on the sum, mean, etc. Default: 1.0
    and oversampling is not considered.

    lims --- if present, limits on values that will be incorporated
    into calculations. Default: None and limits are not used.

    lims_are_lowerlims --- boolean indicating whether the limits are
    lower limits. If not, then the limits are treated as upper
    limits. Default: False.

    fidval_for_lims --- value to replace limits when calculating vaues
    like the mean, standard deviation, etc.. In many circumstances 0
    might be an appropriate choice. Default: None and limits are neglected
    from mean, sum, rms, etc.

    do_print --- if True then the dictionary is printed to the
    console before returning. Default: False.

    return_empty --- if True then no calculations are done and an
    empty structure is returned. Default: False.

    Outputs:

    Notes:

    """

    # Initialize an empty dictionary
    
    stat_dict = OrderedDict()

    stat_dict['counts_meas'] = 0
    stat_dict['indep_meas'] = np.nan
    stat_dict['counts_limits'] = 0
    stat_dict['indep_limits'] = np.nan
    stat_dict['counts_total'] = 0
    stat_dict['indep_total'] = np.nan

    stat_dict['mean'] = np.nan
    stat_dict['e_mean'] = np.nan
    stat_dict['sum'] = np.nan
    stat_dict['e_sum'] = np.nan
    stat_dict['mad_std'] = np.nan
    stat_dict['stddev'] = np.nan
    stat_dict['scatt_hi'] = np.nan
    stat_dict['scatt_lo'] = np.nan
    
    stat_dict['05'] = np.nan
    stat_dict['05islim'] = False
    stat_dict['16'] = np.nan
    stat_dict['16islim'] = False
    stat_dict['50'] = np.nan
    stat_dict['50islim'] = False
    stat_dict['84'] = np.nan
    stat_dict['84islim'] = False
    stat_dict['95'] = np.nan
    stat_dict['95islim'] = False

    # If requested just return this empty dictionary
    
    if return_empty:
        if doprint:
            print_stat_dict(stat_dict)

        return(stat_dict)

    # Pare to only the finite measurements. If there are none,
    # return. Debatable whether the routine should do something when
    # there are limits but not values.
    
    fin_ind = np.isfinite(vec)
    if np.sum(fin_ind) == 0:
        return(stat_dict)

    # Initialize the oversampling factor to 1.0 if it is not supplied
    # by the user.
    
    if oversamp_fac is None:
        oversamp_fac = 1.0

    # If a single value is supplied as the error estimate, then
    # replicate this to a size matched to the data.
        
    if type(err) == type(1.0):
        err = np.tile(err,len(vec))

    # Percentiles
    if lims_are_lowerlims:
        if lims is not None:
            lims = -1.*lims

        this_stat_dict = (perc_w_lim(vals=-1.*vec, lims=lims, perc=95))
        stat_dict['05'] = -1.*this_stat_dict[0]
        stat_dict['05islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=-1.*vec, lims=lims, perc=84))
        stat_dict['16'] = -1.*this_stat_dict[0]
        stat_dict['16islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=-1.*vec, lims=lims, perc=50))
        stat_dict['50'] = -1.*this_stat_dict[0]
        stat_dict['50islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=-1.*vec, lims=lims, perc=16))
        stat_dict['84'] = -1.*this_stat_dict[0]
        stat_dict['84islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=-1.*vec, lims=lims, perc=5))
        stat_dict['95'] = -1.*this_stat_dict[0]
        stat_dict['95islim'] = this_stat_dict[1]

    else:
        this_stat_dict = (perc_w_lim(vals=vec, lims=lims, perc=5))
        stat_dict['05'] = this_stat_dict[0]
        stat_dict['05islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=vec, lims=lims, perc=16))
        stat_dict['16'] = this_stat_dict[0]
        stat_dict['16islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=vec, lims=lims, perc=50))
        stat_dict['50'] = this_stat_dict[0]
        stat_dict['50islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=vec, lims=lims, perc=84))
        stat_dict['84'] = this_stat_dict[0]
        stat_dict['84islim'] = this_stat_dict[1]

        this_stat_dict = (perc_w_lim(vals=vec, lims=lims, perc=95))
        stat_dict['95'] = this_stat_dict[0]
        stat_dict['95islim'] = this_stat_dict[1]

    stat_dict['scatt_lo'] = stat_dict['50'] - stat_dict['16']
    stat_dict['scatt_hi'] = stat_dict['84'] - stat_dict['50']
        
    # Sums and means
    
    vec_for_mean = vec
    if err is None:
        err_for_mean = np.nan*vec
    elif len(err) == len(vec):
        err_for_mean = err
    elif len(err) == 1:
        err_for_mean = np.tile(err,len(vec))
    else:
        print()
        print(len(err), len(vec))
        print()
        raise Exception("Mismatched error and data vectors.")

    # Fill in limit values if requested
    if (fidval_for_lims is not None) and \
       (lims is not None):
        vec_for_mean = np.concatenate(
            [vec_for_mean,np.tile(fidval_for_lims,n_lims)])
        median_err = np.median(err_for_mean)
        err_for_mean = np.concatenate(
            [err_for_mean,np.tile(median_err,n_lims)])

    # Record counts, limits, and independent measurements
    
    stat_dict['counts_meas'] = len(vec)    
    stat_dict['indep_meas'] = len(vec)/oversamp_fac
    if lims is None:
        stat_dict['counts_limits'] = 0
        stat_dict['indep_limits'] = 0
    else:
        stat_dict['counts_limits'] = len(lims)
        stat_dict['indep_limits'] = stat_dict['counts_limits']/oversamp_fac

    stat_dict['counts_total'] = stat_dict['counts_meas']+stat_dict['counts_limits']
    stat_dict['indep_total'] = stat_dict['counts_total']/oversamp_fac
        
    # ... don't let the independent measurements be less than 1
    # regardless of how few data we have.
    if (stat_dict['counts_meas'] > 0) and \
       (stat_dict['indep_meas'] < 1.0):
        stat_dict['indep_meas'] = 1.0
    if (stat_dict['counts_limits'] > 0) and \
       (stat_dict['indep_limits'] < 1.0):
        stat_dict['indep_limits'] = 1.0
    if (stat_dict['counts_total'] > 0) and \
       (stat_dict['indep_total'] < 1.0):
        stat_dict['indep_total'] = 1.0
        
    # Calculate means and sums

    # ... independent measurements
    n_for_mean_err = (1.0*len(vec_for_mean))/oversamp_fac
    if n_for_mean_err < 1.0:
        n_for_mean_err = 1.0

    # ... mean
    stat_dict['mean'] = np.sum(vec_for_mean)/(1.0*len(vec_for_mean))
    stat_dict['e_mean'] = np.sqrt(np.sum(err_for_mean**2))/n_for_mean_err

    # ... sum
    stat_dict['sum'] = np.sum(vec_for_mean)
    if (len(vec_for_mean) > oversamp_fac):
        corr_fac_for_sum = np.sqrt(oversamp_fac)
    else:
        corr_fac_for_sum = 1.0
    stat_dict['e_sum'] = np.sqrt(np.sum(err_for_mean**2))*corr_fac_for_sum

    # ... mad-based rms
    stat_dict['mad_std'] = mad_std(vec_for_mean)
    
    # ... standard deviation
    stat_dict['stddev'] = np.std(vec_for_mean)

    # Print the output
    if doprint:
        print_stat_dict(stat_dict)

    return(stat_dict)

def print_stat_dict(stat_dict):
    # Helper routine
    for key in stat_dict.keys():
        print("... "+key+": ", stat_dict[key])

    return

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Routines related to binning data
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def bin_edges_from_xloxhi(
        xlo, xhi, check=True
        ):
    """Return an array of bin edges from a low and high array. This sort
    of single array is used by a number of scipy and numpy
    routines. But note that this will not work for oversampled bins.

    Parameters:

    xlo --- low edge of bins
    xhi --- high edge of bins

    Keywords:

    check --- check validity by comparing xlo and xhi

    Returns:

    A single array that has all bin edges.

    """
    if check:
        invalid = np.sum(xlo[1:]!=xhi[:-1]) > 0
        if invalid:
            raise Exception("Mismatched bin edge vectors. Bins can't be merged.")
        
    return(np.append(xlo,xhi[-1]))

def make_centered_bins(
        ctr, step, extent,
        width=None, round_down=False):
    """Calculate edges for bins centered on some value with defined
    extent, step size, and (optionally) width.

    Parameters:

    ctr --- middle of the center bin
    
    step --- step size between bin centers
    
    extent --- one-sided extent of the bins FROM THE CENTER. So that
    the full extent is two times this value.
    
    Keywords:

    width --- width of the bins. If None then defaults to step and
    data will not be oversampled. Default: None.

    round_down --- round the number of bins down so that the bins do
    not reach beyond the extent. Otherwise make sure that the full
    extent is covered, even if the last bin extends beyond
    this. Default: False.

    Returns: (xlo, xhi, xmid, nbin)

    xlo --- lower edge of bins
    
    xhi --- upper edge of bins

    xmid --- center value of bins
    
    nbin --- number of bins

    The last two are degenerate with the first two, but provided for convenience.

    """
    
    if width is None:
        width = step
    
    half_width = width/2.0
    last_step = (extent-half_width)
    
    if round_down:
        half_nbin = int(np.floor(last_step / step))
    else:
        half_nbin = int(np.ceil(last_step / step))
    
    nbin = int(2.0*half_nbin+1.0)

    bin_ind = np.arange(-1*half_nbin, half_nbin+1,1)

    xmid = bin_ind*step
    xlo = xmid-half_width
    xhi = xmid+half_width
    
    return((xlo,xhi,xmid,nbin))

def make_spanning_bins(
        xmin, xmax, step,
        width=None, round_down=False,
        stick_to_min=True):
    """Calculate edges for bins centered on some value with defined
    extent, step size, and (optionally) width.

    Parameters:

    xmin --- minimum value

    xmax --- maximum value
    
    step --- step size between bin centers
    
    Keywords:

    width --- width of the bins. If None then defaults to step and
    data will not be oversampled. Default: None.

    round_down --- round the number of bins down so that the bins do
    not reach beyond the nominal range. Otherwise make sure that the
    full extent is covered, even if the last bin extends beyond
    this. Default: False.

    stick_to_min --- if True stick to the xmin (minimum value) as the
    anchor. Otherwise stick to the maximum as the anchor. Default: True.

    Returns: (xlo, xhi, xmid, nbin)

    xlo --- lower edge of bins
    
    xhi --- upper edge of bins

    xmid --- center value of bins
    
    nbin --- number of bins

    The last two are degenerate with the first two, but provided for convenience.

    """
    
    if width is None:
        width = step
    
    half_width = width/2.0
    extent = xmax - xmin

    # Subtract 0.5 width from each side then calculate number of
    # steps, either ensuring totally-within-range or
    # covers-whole-range according to keywords.
    
    if round_down:
        nbin = int(np.floor((extent-width) / step))
    else:
        nbin = int(np.ceil((extent-width) / step))

    if stick_to_min:
        bin_ind = np.arange(0,nbin,1)
        xmid = bin_ind*step+(xmin+width*0.5)
    else:
        bin_ind = np.arange(-1*nbin+1,1,1)
        xmid = bin_ind*step+(xmax-width*0.5)

    xlo = xmid-half_width
    xhi = xmid+half_width
    
    return((xlo,xhi,xmid,nbin))

def bin_data(
        x, y,
        yerr=None, oversample_fac=None,
        x_of_ylim=None, y_of_ylim=None,
        ylim_are_lowerlims=False, fidval_for_ylim=None,
        use_scatt_as_err=False, # might move this to calc_stats
        xlo=None, xhi=None):        
    """
    Calculate statistics in bins.
    """
    
    # Details on x edges
    if xlo is None or xhi is None:
        raise Exception("Need bin edges!")
    
    # Identify the limits we work with (if any)
    if y_of_ylim is not None:
        using_lims = True
        lim_fin_ind = np.isfinite(x_of_ylim)*np.isfinite(y_of_ylim)
        x_of_ylim = x_of_ylim[lim_fin_ind]
        y_of_ylim = y_of_ylim[lim_fin_ind]
    else:
        using_lims = False
        
    # Make sure we have an appropriate error array
    if yerr is None:
        yerr = x*np.nan
    if len(yerr) == 1:
        yerr = x*0.0+yerr
    if len(yerr) != len(x):
        print()
        print("Length of error, vector: ", len(yerr), len(x))
        print()
        raise Exception("Mismatched error and data arrays.")

    # Identify the measurements we work with
    fin_ind = np.isfinite(x)*np.isfinite(y)
    x = x[fin_ind]
    y = y[fin_ind]
    yerr = yerr[fin_ind]
    
    # Initialize the output
    bins = OrderedDict()
    
    stats_for_bins = []
    
    # Loop and calculate statistics in each bin
    for ii, low_edge in enumerate(xlo):

        # note boundaries 
        this_xlo = xlo[ii]
        this_xhi = xhi[ii]
        this_xctr = (this_xlo+this_xhi)*0.5
        
        # for this bin get the data ...        
        bin_ind = (x >= this_xlo)* \
            (x < this_xhi)

        this_x = x[bin_ind]
        this_y = y[bin_ind]
        this_yerr = yerr[bin_ind]
        
        # ... and the limits
        if using_lims:
            lim_ind = (x_of_ylim >= this_xlo)* \
                (x_of_ylim < this_xhi)
            this_lim_x = x_of_ylim[lim_ind]
            this_lim_y = y_of_ylim[lim_ind]

        # Calculate the stats for this bin
        if using_lims:
            
            stat_dict = calc_stats(
                this_y, err=this_yerr,
                lims=this_lim_y,
                lims_are_lowerlims=ylim_are_lowerlims,
                doprint=False)
        else:

            stat_dict = calc_stats(
                this_y, err=this_yerr,
                lims=None, doprint=False)                

        # Add it to the list
        stat_dict['xmid'] = this_xctr
        stat_dict['xlo'] = this_xlo
        stat_dict['xhi'] = this_xhi
        stats_for_bins.append(stat_dict)

    bin_table = Table(stats_for_bins)

    return(bin_table)

# N.B. the routine below needs to be refactored to use the newer
# distribution descriptions.

def running_med(
        x, y, yerr=None,
        halfwin=10,
        x_of_ylim=None, y_of_ylim=None,        
        ylim_are_lowerlims=False, fidval_for_ylim=None):
    """Conduct a running percentile calculation, potentially with limits,
    in y vs x.
    """   
    
    # Make sure x and y are arrays
    x = np.array(x)
    y = np.array(y)

    # Make sure we have an appropriate error array
    if yerr is None:
        yerr = x*np.nan
    else:
        yerr = np.array(yerr)
        
    if len(yerr) == 1:
        yerr = x*0.0+yerr
        
    if len(yerr) != len(x):
        print()
        print("Length of error, vector: ", len(yerr), len(x))
        print()
        raise Exception("Mismatched error and data arrays.")
    
    # Keep only finite elements
    fin_ind = np.isfinite(x)*np.isfinite(y)
    x = x[fin_ind]
    y = y[fin_ind]
    yerr = yerr[fin_ind]
    
    # Identify the limits we work with (if any)
    if y_of_ylim is not None:
        using_lims = True
        lim_fin_ind = np.isfinite(x_of_ylim)*np.isfinite(y_of_ylim)
        x_of_ylim = x_of_ylim[lim_fin_ind]
        y_of_ylim = y_of_ylim[lim_fin_ind]
    else:
        using_lims = False

    # Initialize a limit flag
    
    # ... for the data
    y_is_lim = y*0.0
    # ... for the limits
    if using_lims:
        ylim_is_lim = y_of_ylim*0.0+1.0
        
    # Merge the limits and measurements if needed
    if using_lims:
        x = np.concatenate((x,x_of_ylim))
        y = np.concatenate((y,y_of_ylim))
        y_is_lim = np.concatenate((y_is_lim,ylim_is_lim))
        
    # Sort the data by x
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]
    yerr = yerr[ind]
    y_is_lim = y_is_lim[ind]

    # Note vector length
    n = len(x)
    
    # Initialize output
    stats_for_bins = []
    
    # Loop over array
    for ii in range(n):
        
        # Skip if we can't fill the window with data
        if ii < halfwin:
            continue
        if ii > n-1-halfwin:
            continue
    
        # Extract working vectors
        this_x = x[ii]
        this_xlo = x[ii-halfwin]
        this_xhi = x[ii+halfwin]
        this_y = y[(ii-halfwin):(ii+halfwin)]
        this_yerr = yerr[(ii-halfwin):(ii+halfwin)]
        this_yislim = y_is_lim[(ii-halfwin):(ii+halfwin)]

        # Differentiate measurements and limits
        this_yvals = this_y[this_yislim == False]
        this_ylims = this_y[this_yislim == True]

        # Calculate the stats for this bin
        if using_lims:
            
            stat_dict = calc_stats(
                this_yvals, err=this_yerr,
                lims=this_ylims,
                lims_are_lowerlims=ylim_are_lowerlims,
                doprint=False)
        else:
            
            stat_dict = calc_stats(
                this_yvals, err=this_yerr,
                lims=None, doprint=False)                

        # Add info on the x coordinate
        stat_dict['xmid'] = this_x
        stat_dict['xlo'] = this_xlo
        stat_dict['xhi'] = this_xhi
        
        stats_for_bins.append(stat_dict)

    # Downselect to finite values
    bin_table = Table(stats_for_bins)

    return(bin_table)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Related to non-parameteric correlatinos
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def corr_with_repair(
        x, y,
        n_iter = 1000,
        method = 'KENDALL'):
    """Calculate a correlation coefficient, dropping NaNs, and then
    calculate an error bar and p-value from randomly repairing the
    data.

    x, y : the input vectors

    n_iter : integer, number of random repairings

    method : string specifying statistic. Options are KENDALL SPEARMAN

    returns : coefficient, error bar, p_value

    """

    if method.upper() not in ['KENDALL','SPEARMAN']:
        print("... unrecognized method, defaulting to SPEARMAN")
        method = 'SPEARMAN'

    use = np.isfinite(x)*np.isfinite(y)
    use_x = x[use]
    use_y = y[use]

    if method == 'SPEARMAN':
        rho, p_rho = spearmanr(use_x, use_y)
        fig_of_merit = rho
        
    if method == 'KENDALL':
        tau, p_tau = kendalltau(use_x, use_y)
        fig_of_merit = tau

    if n_iter > 0:

        # Initialize output
        vals = np.zeros(int(n_iter))*np.nan

        for ii in range(n_iter):
            
            # re-pair the data by randomizing y vector
            this_x = use_x
            ind = np.argsort(np.random.random(len(use_y)))            
            this_y = use_y[ind]
            
            # calculate the stat
            if method == 'SPEARMAN':
                rho, p_rho = spearmanr(this_x, this_y)
                vals[ii] = rho

            if method == 'KENDALL':
                tau, p_tau = kendalltau(this_x, this_y)
                vals[ii] = tau

    # calculate an effective p value and a scatter

    p_value = np.sum(np.abs(fig_of_merit) > np.abs(vals))/(1.0*n_iter)
    scatter = np.std(vals)
    
    # return the stat, the error bar, and the effective p_value
    return(fig_of_merit, scatter, p_value)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Related to two dimensional fields
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def calc_local_density(
        x,y,w=None,
        support=(0.01,0.01),
        calc_cdf=False,
        smoothing_box=None):
    """
    From Jiayi
    """
    
    n = len(x)
    dens = np.zeros_like(x)*np.nan
    if w is None:
        w = np.isfinite(x)*np.isfinite(y)*1.0
    
    for ii in range(n):
        dens[ii] = \
            np.sum((x>=(x[ii]-support[0]))* \
                   (x<(x[ii]+support[0]))* \
                   (y>=(y[ii]-support[1]))* \
                   (y<(y[ii]+support[1]))* \
                   w)*1.0
    
    if calc_cdf:        
        cdf = np.zeros_like(dens).ravel()        
        for i, density in enumerate(dens.ravel()):
            cdf[i] = dens[dens >= density].sum()
        cdf = (cdf/cdf.max()).reshape(dens.shape)
        dens = cdf

    new_dens = dens
    if (smoothing_box is not None):
        for ii in range(n):
            condition = \
                (x>=(x[ii]-smoothing_box[0]))* \
                (x<(x[ii]+smoothing_box[0]))* \
                (y>=(y[ii]-smoothing_box[1]))* \
                (y<(y[ii]+smoothing_box[1]))
            new_dens[ii] = np.nanmedian(dens[condition])

    return(new_dens)

def calc_local_mean(
        x,y,z,w=None,
        support=(0.01,0.01),
        calc_cdf=False,
        smoothing_box=None):
    """
    Modified from above.
    """
    
    n = len(x)
    dens = np.zeros_like(x)*np.nan
    if w is None:
        w = np.isfinite(x)*np.isfinite(y)*1.0

    zfield = z*np.nan
    for ii in range(n):
        ind = (x>=(x[ii]-support[0]))* \
            (x<(x[ii]+support[0]))* \
            (y>=(y[ii]-support[1]))* \
            (y<(y[ii]+support[1]))
        zvec = z[ind]
        wvec = w[ind]
        zfield[ii] = np.sum(zvec*wvec)/np.sum(wvec)
    
    new_zfield = zfield
    if (smoothing_box is not None):
        for ii in range(n):
            condition = \
                (x>=(x[ii]-smoothing_box[0]))* \
                (x<(x[ii]+smoothing_box[0]))* \
                (y>=(y[ii]-smoothing_box[1]))* \
                (y<(y[ii]+smoothing_box[1]))
            new_zfield[ii] = np.nanmedian(zfield[condition])

    return(new_zfield)
