from collections import OrderedDict
import numpy as np
import scipy.stats
from scipy.odr import ODR, Model, Data, RealData
from scipy import optimize
from scipy.stats import spearmanr, kendalltau

from astropy.stats import mad_std
from astropy.table import Table

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Helper functions
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def orth_dist(x, y, m, b):
    """
    Orthogonal distance between a point and a line.
    """
    aa = -1.0*m
    bb = 1.0
    cc = -1.*b
    num = np.abs(aa*x+bb*y+cc)
    denom = np.sqrt(aa**2+bb**2)
    return(num/denom)

def broken_law(xval=np.nan,
               xmin=np.nan, xmax=np.nan, 
               minval=np.nan, maxval=np.nan):

    yval = \
           (xval <= xmax)*minval + \
           (xval > xmin)*(xval < xmax)*(xval - xmin)* \
           (maxval-minval)/(xmax - xmin) + \
           (xval > xmax)*maxval
        
    return(yval)

def line_func(beta, x):
    y = beta[0]+beta[1]*x
    return(y)

def line_func_curvefit(x,b,m):
    y = b + m*x
    return(y)

def neg_log_likelihood_for_a_line(parms, x, e_x, y, e_y, x0=0.0):
    """
    Negative log likelihood for a line with scatter.
    """

    slope = parms[0]
    intercept = parms[1]
    scatter = parms[2]
    resid = y - (slope*(x-x0) + intercept)

    weight = 1./(scatter**2 + e_y**2 + (slope*e_x)**2)
    log_likelihood = np.sum(np.log(weight) - weight*(resid**2))
    return(-1.0*log_likelihood)

def clean_up_data_for_fitting(x, y, e_x=None, e_y=None, x0=None):
    """
    Recenter, initialize error vecotrs, pare to finite data.
    """
    
    if x0 is not None:
        x = x - x0
    
    if e_x is None:
        e_x = 1.0+x*0.0
    elif type(e_x) == type(1.0):
        e_x = e_x+x*0.0     
    elif len(e_x) == 1:
        e_x = e_x+x*0.0
    
    if e_y is None:
        e_y = 1.0+y*0.0
    elif type(e_y) == type(1.0):
        e_y = e_y+y*0.0     
    elif len(e_y) == 1:
        e_y = e_y+y*0.0     

    fin_ind = np.isfinite(x)*np.isfinite(y)*np.isfinite(e_x)*np.isfinite(e_y)
    x = x[fin_ind]
    y = y[fin_ind]
    e_x = e_x[fin_ind]
    e_y = e_y[fin_ind]

    return(x, y, e_x, e_y)
    
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Fitting routings
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def iterate_ols(x, y, e_y=None, x0=None,
                guess=[0.0,1.0], s2nclip=3., iters=3,
                doprint=False, min_pts=3):
    """
    Iterate an ordinary least squares fit.    
    """

    x, y, e_x, e_y = \
        clean_up_data_for_fitting(x, y, e_x=None, e_y=e_y, x0=x0)    
       
    use = np.isfinite(x)
    if np.sum(use) < min_pts:
        return((np.nan,np.nan,np.nan))
    
    first = True

    for ii in range(iters):
        if s2nclip is None:
            if first is False:
                continue
            
        popt, pcurve = optimize.curve_fit(
            line_func_curvefit, x[use], y[use],
            sigma = e_y[use], p0 = guess)
            
        intercept, slope = popt
        resid = y - (intercept + slope*x)
        rms = mad_std(resid)
        
        if s2nclip is not None:            
            use = np.abs(resid < s2nclip*rms)

        first = False
        
    if doprint:
        print("Fit results:")
        print("... slope: ", slope)
        print("... intercept: ", intercept)
        print("... scatter: ", rms)
        print("... kept/rejected: ", np.sum(use), np.sum(use==False))
     
    return((slope,intercept,rms))
    
def iterate_odr(x, y, e_x=None, e_y=None, 
                x0=None, s2nclip=3., iters=3, guess=[0.0,1.0],
                doprint=False):  
    """
    Iterate an orthogonal distance regression fit.
    """
      
    x, y, e_x, e_y = \
        clean_up_data_for_fitting(x, y, e_x=e_x, e_y=e_y, x0=x0)    
    
    use = np.isfinite(x)
    for ii in range(iters):
        if s2nclip is None:
            if ii > 0:
                continue
        
        data = RealData(x[use], y[use], e_x[use], e_y[use])
        model = Model(line_func)
        odr = ODR(data, model, guess)
        odr.set_job(fit_type=0)
        output = odr.run()
        
        intercept, slope = output.beta
        resid = orth_dist(x, y, slope, intercept)
        rms = mad_std(resid)
        
        if s2nclip is not None:            
            use = np.abs(resid < s2nclip*rms)        
        
    if doprint:
        print("Fit results:")
        print("... slope: ", slope)
        print("... intercept: ", intercept)
        print("... scatter: ", rms)
        print("... kept/rejected: ", np.sum(use), np.sum(use==False))
     
    return((slope,intercept,rms))

def iterate_loglikelihood_linear_fit(
        x, y, e_x=None, e_y=None, 
        x0=None, guess=None,
        s2nclip=3., iters=1,
        doprint=False, min_pts=3):
    """
    Fit a line with scatter by minimizing the negative log
    likelihood. Use iterative outlier rejection if requested.
    """

    x, y, e_x, e_y = \
        clean_up_data_for_fitting(x, y, e_x=e_x, e_y=e_y, x0=x0)

    if guess is None:
        guess = [np.median(y), 1.0, 0.0]
    
    # Iterate fits
    use = np.isfinite(x)
    for ii in range(iters):
        if s2nclip is None:
            if ii > 0:
                continue

        parms = guess
        output = \
            optimize.minimize(
                neg_log_likelihood_for_a_line, parms, args=(x, e_x, y, e_y, 0.0)
                , method='BFGS', options={'maxiter':99,'xrtol':1e-9})
                                          
        slope = output.x[0]
        intercept = output.x[1]
        scatter = output.x[2]
        
        resid = y - (slope*x+intercept)
        rms = mad_std(resid)
        
        if s2nclip is not None:            
            use = np.abs(resid < s2nclip*rms)        
        
    if doprint:
        print("Fit results:")
        print("... slope: ", slope)
        print("... intercept: ", intercept)
        print("... fit scatter: ", scatter)
        print("... calculated scatter: ", rms)        
        print("... kept/rejected: ", np.sum(use), np.sum(use==False))
     
    return((slope,intercept,scatter,rms))
